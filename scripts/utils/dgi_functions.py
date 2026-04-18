"""
dgi_functions.py
================
All logic for notebook P3 · Drug–Gene Interaction Collection.

Functions
---------
load_dgi_inputs     — load hub genes + survival targets
collect_interactions — query selected databases + curated fallback
build_dgi_dataframe — clean, deduplicate, and compute composite score
build_gnn_edge_list — add GNN feature columns and export
plot_dgi_dashboard  — 5-panel summary figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Colours per source (used in dashboard)
SRC_COL = {
    "DGIdb"      : "#534AB7",
    "ChEMBL"     : "#1D9E75",
    "OpenTargets": "#D85A30",
    "Curated"    : "#888780",
}
PHASE_COL = {
    0: "#D3D1C7",
    1: "#B5D4F4",
    2: "#378ADD",
    3: "#185FA5",
    4: "#1D9E75",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_dgi_inputs(tables_dir):
    """
    Load hub gene list and survival target set from previous steps.

    Parameters
    ----------
    tables_dir : Path

    Returns
    -------
    gene_list : list
        Hub gene symbols (input to database queries).
    hub_score_map : dict
        gene → hub_score (used in composite scoring).
    surv_genes : set
        Genes with significant survival association (get score bonus).
    """
    hub_df    = pd.read_csv(tables_dir / "hub_genes.csv")
    gene_list = hub_df.gene.dropna().unique().tolist()
    hub_score_map = (hub_df.set_index("gene")["hub_score"].to_dict()
                     if "hub_score" in hub_df.columns else {})

    surv_file = tables_dir / "survival_filtered_genes.csv"
    surv_genes = (set(pd.read_csv(surv_file)["gene"].dropna())
                  if surv_file.exists() else set())

    print(f"Hub genes        : {len(gene_list)}")
    print(f"Survival targets : {len(surv_genes)}")
    return gene_list, hub_score_map, surv_genes


# ─────────────────────────────────────────────────────────────────────────────
def collect_interactions(gene_list,
                         use_dgidb=True,
                         use_chembl=True,
                         use_opentargets=True,
                         use_curated=True):
    """
    Query selected databases and merge results.

    Each flag independently enables or disables a data source.
    The curated fallback automatically fills genes not covered by live APIs.

    Parameters
    ----------
    gene_list : list
        Gene symbols to query.
    use_dgidb, use_chembl, use_opentargets : bool
        Enable/disable each live API.
    use_curated : bool
        Enable/disable the built-in curated fallback dataset.

    Returns
    -------
    all_edges : list of dict
        Combined raw interaction records from all enabled sources.
    apis_ok : list of str
        Names of sources that returned at least one result.
    """
    # Import here so the function works in Colab without the full repo
    import sys
    from pathlib import Path

    # Try to find and add scripts/ to path if not already there
    for candidate in [Path(__file__).resolve().parent,
                      Path.cwd() / "scripts",
                      Path.cwd()]:
        if (candidate / "utils" / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    from utils.api_clients import (query_dgidb, query_chembl,
                                   query_opentargets, get_curated_fallback)

    all_edges, apis_ok = [], []

    if use_dgidb:
        print("Querying DGIdb...")
        edges = query_dgidb(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("DGIdb")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_chembl:
        print("Querying ChEMBL...")
        edges = query_chembl(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("ChEMBL")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_opentargets:
        print("Querying OpenTargets...")
        edges = query_opentargets(gene_list)
        if edges:
            all_edges.extend(edges)
            apis_ok.append("OpenTargets")
            print(f"  → {len(edges)} interactions")
        else:
            print("  → No results (API unreachable or no matches)")

    if use_curated:
        if not all_edges:
            print("No live API results — loading full curated fallback...")
            all_edges = get_curated_fallback(gene_list)
            print(f"  → {len(all_edges)} curated interactions")
        else:
            covered = {e["gene"] for e in all_edges}
            missing = [g for g in gene_list if g not in covered]
            if missing:
                curated = get_curated_fallback(missing)
                all_edges.extend(curated)
                print(f"Curated: {len(curated)} interactions added "
                      f"for {len(missing)} uncovered genes")
    elif not all_edges:
        print("\nWarning: no databases selected and curated fallback is disabled.")

    print(f"\nSources used    : {apis_ok or ['curated fallback']}")
    print(f"Raw interactions: {len(all_edges)}")
    return all_edges, apis_ok


# ─────────────────────────────────────────────────────────────────────────────

DRUG_FEAT_COLS = [
    "approved", "immunotherapy", "anti_neoplastic", "clinical_phase",
    "interaction_score", "n_publications", "source_DGIdb", "source_ChEMBL",
    "source_OpenTargets", "type_inhibitor", "type_agonist", "type_antagonist",
    "type_antibody", "type_binder", "type_activator",
]


def build_dgi_dataframe(all_edges, hub_score_map, surv_genes, W):
    """
    Convert raw interaction records into a clean, scored dataframe.

    Steps:
      1. Ensure all feature columns exist (fill missing with 0 / False)
      2. Standardise types (bool, int, float)
      3. Deduplicate gene-drug pairs (keep highest interaction_score)
      4. Compute composite score as weighted sum

    Parameters
    ----------
    all_edges : list of dict
        Raw records from collect_interactions().
    hub_score_map : dict
        gene → hub_score.
    surv_genes : set
        Genes with survival association (get +0.10 bonus).
    W : dict
        Scoring weights: interaction, publications, phase, approved, hub.

    Returns
    -------
    dgi_df : pd.DataFrame
        Clean edge dataframe sorted by composite_score descending.
    """
    dgi_df = pd.DataFrame(all_edges)

    for col in DRUG_FEAT_COLS:
        if col not in dgi_df.columns:
            dgi_df[col] = 0
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        dgi_df[col] = dgi_df[col].fillna(False).astype(bool)
    dgi_df["interaction_score"] = pd.to_numeric(
        dgi_df.interaction_score, errors="coerce").fillna(0)
    dgi_df["n_publications"] = pd.to_numeric(
        dgi_df.n_publications, errors="coerce").fillna(0).astype(int)
    dgi_df["clinical_phase"] = pd.to_numeric(
        dgi_df.clinical_phase, errors="coerce").fillna(0).astype(int)
    dgi_df["drug"] = dgi_df.drug.str.strip().str.title()
    dgi_df["gene"] = dgi_df.gene.str.strip().str.upper()

    dgi_df = (dgi_df
              .sort_values("interaction_score", ascending=False)
              .drop_duplicates(["gene", "drug"], keep="first")
              .reset_index(drop=True))

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    dgi_df["composite_score"] = (
        W["interaction"]  * norm(dgi_df.interaction_score) +
        W["publications"] * norm(dgi_df.n_publications.clip(0, 30)) +
        W["phase"]        * (dgi_df.clinical_phase / 4) +
        W["approved"]     * dgi_df.approved.astype(float) +
        W["hub"]          * norm(dgi_df.gene.map(hub_score_map).fillna(0)) +
        dgi_df.gene.isin(surv_genes).astype(float) * 0.10
    ).clip(0, 1).round(4)

    dgi_df = dgi_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    print(f"Edges (deduplicated): {len(dgi_df)}")
    print(f"Unique genes        : {dgi_df.gene.nunique()}")
    print(f"Unique drugs        : {dgi_df.drug.nunique()}")
    print(f"Approved drugs      : {dgi_df.approved.sum()}")
    return dgi_df


# ─────────────────────────────────────────────────────────────────────────────
def build_gnn_edge_list(dgi_df, hub_score_map, surv_genes, tables_dir):
    """
    Add one-hot GNN feature columns and export dgi_edges_gnn.csv.

    Adds:
      - hub_score, survival_target
      - source_DGIdb, source_ChEMBL, source_OpenTargets  (one-hot)
      - type_inhibitor, type_agonist, … (one-hot by interaction_type)

    Parameters
    ----------
    dgi_df : pd.DataFrame
        Output of build_dgi_dataframe().
    hub_score_map : dict
    surv_genes : set
    tables_dir : Path

    Returns
    -------
    gnn_df : pd.DataFrame
        GNN-ready edge dataframe (also saved to disk).
    """
    gnn_df = dgi_df.copy()
    gnn_df["hub_score"]       = gnn_df.gene.map(hub_score_map).fillna(0)
    gnn_df["survival_target"] = gnn_df.gene.isin(surv_genes).astype(int)

    for src in ["DGIdb", "ChEMBL", "OpenTargets"]:
        gnn_df[f"source_{src}"] = (gnn_df.source == src).astype(int)
    for it in ["inhibitor", "agonist", "antagonist", "antibody", "binder", "activator"]:
        gnn_df[f"type_{it}"] = (gnn_df.interaction_type.str.lower() == it).astype(int)

    gnn_cols = (
        ["gene", "drug", "composite_score", "approved", "immunotherapy",
         "anti_neoplastic", "clinical_phase", "interaction_score", "n_publications"] +
        [f"source_{s}" for s in ["DGIdb", "ChEMBL", "OpenTargets"]] +
        [f"type_{t}" for t in ["inhibitor", "agonist", "antagonist",
                                "antibody", "binder", "activator"]] +
        ["hub_score", "survival_target", "interaction_type", "directionality", "source"]
    )
    gnn_df[[c for c in gnn_cols if c in gnn_df.columns]].to_csv(
        tables_dir / "dgi_edges_gnn.csv", index=False)

    print(f"Saved: dgi_edges_gnn.csv")
    print(f"  Edges  : {len(gnn_df)}")
    print(f"  Genes  : {gnn_df.gene.nunique()}")
    print(f"  Drugs  : {gnn_df.drug.nunique()}")
    print(f"  → Ready for notebook P4 (GNN)")
    return gnn_df


# ─────────────────────────────────────────────────────────────────────────────
def plot_dgi_dashboard(dgi_df, figures_dir):
    """
    5-panel summary dashboard: interactions per gene, type donut,
    approval by source, clinical phase, score heatmap.

    Parameters
    ----------
    dgi_df : pd.DataFrame
        Output of build_dgi_dataframe().
    figures_dir : Path

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # A — interactions per gene (stacked bar)
    ax1 = fig.add_subplot(gs[0, :2])
    gc  = dgi_df.groupby(["gene", "source"]).size().unstack(fill_value=0)
    gc  = gc.loc[gc.sum(axis=1).sort_values(ascending=True).index]
    bot = np.zeros(len(gc))
    for src in list(SRC_COL):
        if src in gc.columns:
            v = gc[src].values
            ax1.barh(gc.index, v, left=bot, color=SRC_COL[src],
                     label=src, alpha=0.88, height=0.65)
            bot += v
    ax1.set_xlabel("Interactions")
    ax1.set_title("A  Interactions per gene",
                  fontsize=11, fontweight="bold", loc="left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9)

    # B — interaction type donut
    ax2  = fig.add_subplot(gs[0, 2])
    tc   = dgi_df.interaction_type.str.lower().value_counts().head(6)
    cols6 = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#888780", "#B5D4F4"]
    ax2.pie(tc.values, labels=tc.index, colors=cols6[:len(tc)],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white"})
    ax2.set_title("B  Interaction types",
                  fontsize=11, fontweight="bold", loc="left")

    # C — approval by source
    ax3  = fig.add_subplot(gs[1, 0])
    appr = dgi_df.groupby(["source", "approved"]).size().unstack(fill_value=0)
    x, w = np.arange(len(appr)), 0.35
    ax3.bar(x - w/2,
            appr.get(True,  pd.Series(0, index=appr.index)).values,
            width=w, color="#1D9E75", alpha=0.85, label="Approved")
    ax3.bar(x + w/2,
            appr.get(False, pd.Series(0, index=appr.index)).values,
            width=w, color="#D3D1C7", alpha=0.85, label="Not approved")
    ax3.set_xticks(x); ax3.set_xticklabels(appr.index, fontsize=9)
    ax3.set_ylabel("Count")
    ax3.set_title("C  Approval by source",
                  fontsize=11, fontweight="bold", loc="left")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(fontsize=8)

    # D — clinical phase
    ax4 = fig.add_subplot(gs[1, 1])
    pm  = {0:"Preclinical",1:"Phase 1",2:"Phase 2",3:"Phase 3",4:"Approved"}
    po  = ["Preclinical","Phase 1","Phase 2","Phase 3","Approved"]
    pv  = [dgi_df.clinical_phase.map(pm).value_counts().get(p, 0) for p in po]
    bars = ax4.bar(po, pv,
                   color=[PHASE_COL[k] for k in range(5)],
                   alpha=0.88, edgecolor="white")
    for b, v in zip(bars, pv):
        if v:
            ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                     str(v), ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.set_title("D  Clinical phase",
                  fontsize=11, fontweight="bold", loc="left")
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.tick_params(axis="x", labelsize=8)

    # E — top drug-gene score heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    t15 = dgi_df.head(15)[["drug", "gene", "composite_score"]]
    piv = t15.pivot_table(index="drug", columns="gene",
                          values="composite_score", fill_value=0)
    im  = ax5.imshow(piv.values, cmap="YlOrRd", aspect="auto",
                     vmin=0, vmax=1)
    ax5.set_xticks(range(len(piv.columns)))
    ax5.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
    ax5.set_yticks(range(len(piv.index)))
    ax5.set_yticklabels(piv.index, fontsize=7)
    plt.colorbar(im, ax=ax5, shrink=0.8, label="Score")
    ax5.set_title("E  Score heatmap",
                  fontsize=11, fontweight="bold", loc="left")

    fig.suptitle("Drug–Gene Interaction Analysis — HCC Hub Genes",
                 fontsize=14, y=1.02, fontweight="bold")
    fig.savefig(figures_dir / "dgi_summary_dashboard.png",
                dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {figures_dir}/dgi_summary_dashboard.png")
    return fig
