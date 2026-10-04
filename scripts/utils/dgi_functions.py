"""
dgi_functions.py
================
All logic for notebook P3 · Drug–Gene Interaction Collection.

Functions
---------
load_dgi_inputs     — load all candidates including network isolates
collect_interactions — collect auditable live evidence and explicit failure status
build_dgi_dataframe — clean, deduplicate, and compute composite score
build_gnn_edge_list — add GNN feature columns and export
plot_dgi_dashboard  — 5-panel summary figure
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

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
    Load all network candidates, including isolates; survival is not a gate.

    Parameters
    ----------
    tables_dir : Path

    Returns
    -------
    gene_list : list
        Hub gene symbols (input to database queries).
    hub_score_map : dict
        gene → hub_score (used in composite scoring).
    """
    hub_df    = pd.read_csv(tables_dir / "hub_genes.csv")
    gene_list = hub_df.gene.dropna().unique().tolist()
    hub_score_map = (hub_df.set_index("gene")["hub_score"].to_dict()
                     if "hub_score" in hub_df.columns else {})

    print(f"Candidates for drug queries: {len(gene_list)}")
    return gene_list, hub_score_map


# ─────────────────────────────────────────────────────────────────────────────
def collect_interactions(gene_list, use_dgidb=True, use_chembl=False,
                         use_opentargets=False, use_curated=False, return_status=False):
    """Fetch live evidence; no hardcoded scientific fallback.

    Source failures are explicit. DGIdb discards failed partial batches.
    ChEMBL/OpenTargets legacy clients require an additional mapping/pagination
    audit before scientific use, so they cannot be silently enabled here.
    """
    from utils.api_clients import query_dgidb
    if use_curated:
        raise ValueError("Unverified hardcoded fallback disabled; supply cited records separately")
    if use_chembl or use_opentargets:
        raise ValueError("ChEMBL/OpenTargets clients need mapping/pagination validation before enabling")
    all_edges, apis_ok, statuses = [], [], {}
    if use_dgidb:
        try:
            records = query_dgidb(gene_list)
            all_edges.extend(records)
            apis_ok.append("DGIdb")
            statuses["DGIdb"] = dict(status="completed", n_records=len(records))
        except Exception as exc:
            statuses["DGIdb"] = dict(status="failed", reason=str(exc), n_records=0)
            print(f"DGIdb unavailable: {exc}")
    else:
        statuses["DGIdb"] = dict(status="disabled", n_records=0)
    return (all_edges, apis_ok, statuses) if return_status else (all_edges, apis_ok)


DRUG_FEAT_COLS = [
    "approved", "immunotherapy", "anti_neoplastic", "clinical_phase",
    "interaction_score", "n_publications", "source_DGIdb", "source_ChEMBL",
    "source_OpenTargets", "type_inhibitor", "type_agonist", "type_antagonist",
    "type_antibody", "type_binder", "type_activator",
]


def _tokens(value):
    return {v.strip() for v in str(value).split(" | ") if v.strip() and v.strip() not in {"nan", "None", "<NA>"}}


def build_dgi_dataframe(all_edges, hub_score_map, W):
    """Evidence ranking, preserving provenance; unknown metadata stays missing.

    Interaction scores are normalized ONLY for DGIdb. Other source scores are
    not assumed comparable. Default weights put publications/phase at zero
    to avoid double-counting and inference of clinical maturity from approval.
    Drug IDs define identity when present; names alone are a flagged fallback.
    """
    expected = {"interaction", "publications", "phase", "approved", "hub"}
    if set(W) != expected or any(v < 0 for v in W.values()) or not np.isclose(sum(W.values()), 1):
        raise ValueError("Scoring weights must be nonnegative and sum to one")
    columns = ["gene", "drug", "drug_id", "source", "interaction_type", "directionality",
               "publication_ids", "evidence_sources", "reference_urls", "approved", "immunotherapy",
               "anti_neoplastic", "clinical_phase", "interaction_score", "n_publications", "phase_scope",
               "identity_basis", "record_count", "composite_score", "hub_score"]
    if not all_edges:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(all_edges).copy()
    for col in columns:
        if col not in frame:
            frame[col] = np.nan
    if frame.gene.isna().any() or frame.drug.isna().any():
        raise ValueError("Interactions require gene and drug identifiers")
    frame["gene"] = frame.gene.str.strip().str.upper()
    frame["drug"] = frame.drug.str.strip()
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        def boolean(value):
            if pd.isna(value): return pd.NA
            if value in [True, 1, "True", "true"]: return True
            if value in [False, 0, "False", "false"]: return False
            raise ValueError(f"Invalid Boolean metadata: {value!r}")
        frame[col] = frame[col].map(boolean).astype("boolean")
    for col in ["interaction_score", "n_publications", "clinical_phase"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["drug_id"] = frame.drug_id.fillna("").astype(str).str.strip()
    frame["_identity"] = np.where(frame.drug_id != "", "id:" + frame.drug_id,
                                   "name:" + frame.drug.str.casefold())
    records = []
    for (_, identity), group in frame.groupby(["gene", "_identity"], sort=True):
        record = group.iloc[0].drop(labels="_identity").to_dict()
        for col in ["source", "drug_id", "interaction_type", "directionality", "publication_ids",
                    "evidence_sources", "reference_urls", "phase_scope"]:
            record[col] = " | ".join(sorted(set().union(*( _tokens(v) for v in group[col]))))
        for col in ["approved", "immunotherapy", "anti_neoplastic"]:
            known = group[col].dropna().unique()
            record[col] = bool(known[0]) if len(known) == 1 else pd.NA
        record["metadata_conflict"] = any(group[c].dropna().nunique() > 1 for c in ["approved", "clinical_phase"])
        record["clinical_phase"] = group.clinical_phase.max() if group.clinical_phase.dropna().nunique() <= 1 else np.nan
        dgidb = group[group.source == "DGIdb"].interaction_score
        record["interaction_score"] = dgidb.max()
        record["n_publications"] = len(_tokens(record["publication_ids"])) if record["publication_ids"] else group.n_publications.max()
        record["identity_basis"] = "database_id" if identity.startswith("id:") else "name_only_requires_review"
        record["record_count"] = len(group)
        records.append(record)
    result = pd.DataFrame(records)
    for col in ["approved", "immunotherapy", "anti_neoplastic"]:
        result[col] = result[col].astype("boolean")
    result["hub_score"] = result.gene.map(hub_score_map).fillna(0).clip(0, 1)
    def norm(series):
        series = pd.to_numeric(series, errors="coerce")
        # Zero evidence has zero contribution even if all observed scores agree.
        return series.fillna(0).clip(lower=0) / max(series.max() if series.notna().any() else 0, 1e-9)
    result["score_interaction"] = norm(result.interaction_score)
    result["score_publications"] = norm(result.n_publications.clip(upper=30))
    result["score_phase"] = result.clinical_phase.fillna(0).clip(0, 4) / 4
    result["score_approved"] = result.approved.fillna(False).astype(float)
    result["score_hub"] = result.hub_score
    result["composite_score"] = sum(W[key] * result["score_" + key] for key in W)
    result["score_scope"] = "Within this contrast and database snapshot; heuristic evidence ranking"
    result["mechanism_review"] = "Original evidence and therapeutic direction require review"
    return result.sort_values(["composite_score", "gene", "drug"], ascending=[False, True, True]).reset_index(drop=True)


def score_weight_sensitivity(all_edges, hub_score_map, baseline_weights, alternatives=None):
    """Compare edge ranks under explicitly documented heuristic weights."""
    scenarios = {"baseline": baseline_weights, **(alternatives or {
        "less_hub": dict(interaction=0.75, publications=0, phase=0, approved=0.15, hub=0.10),
        "more_hub": dict(interaction=0.55, publications=0, phase=0, approved=0.15, hub=0.30)})}
    rows = []
    for scenario, weights in scenarios.items():
        frame = build_dgi_dataframe(all_edges, hub_score_map, weights)
        for rank, record in enumerate(frame.to_dict("records"), 1):
            rows.append({"scenario": scenario, "gene": record["gene"], "drug": record["drug"],
                         "drug_id": record["drug_id"], "rank": rank, "score": record["composite_score"]})
    return pd.DataFrame(rows, columns=["scenario", "gene", "drug", "drug_id", "rank", "score"])


def build_gnn_edge_list(dgi_df, hub_score_map, tables_dir):
    """Preserve biological/provenance columns; encode multi-valued evidence.

    Missing numeric metadata is zero encoded only in this model input, with
    accompanying missingness flags. It remains missing in the evidence table.
    """
    gnn = dgi_df.copy()
    gnn["hub_score"] = gnn.gene.map(hub_score_map).fillna(0)
    for source in ["DGIdb", "ChEMBL", "OpenTargets"]:
        gnn["source_" + source] = gnn.source.map(lambda v: int(source in _tokens(v)))
    for kind in ["inhibitor", "agonist", "antagonist", "antibody", "binder", "activator"]:
        gnn["type_" + kind] = gnn.interaction_type.map(lambda v: int(kind in {t.lower() for t in _tokens(v)}))
    for col in ["approved", "immunotherapy", "anti_neoplastic", "clinical_phase", "interaction_score", "n_publications"]:
        gnn[col + "_missing"] = gnn[col].isna().astype(int)
        gnn[col] = gnn[col].astype("Float64").fillna(0).astype(float)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    gnn.to_csv(tables_dir / "dgi_edges_gnn.csv", index=False)
    return gnn


_TYPE_COLS = [
    "#534AB7","#1D9E75","#D85A30","#BA7517","#888780","#B5D4F4",
    "#E07B54","#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
]
 
 
def _as_path(p):
    return Path(p)
def _save_panel(ax, figures_dir: Path, filename: str, dpi: int = 200):
    """Save a single axes as an individual PNG via tight bbox extraction."""
    fig      = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    extent   = ax.get_tightbbox(renderer)
    if extent is None:
        return
    bbox_in = extent.transformed(fig.dpi_scale_trans.inverted())
    # Use a very small pad (1.02) to avoid picking up content from
    # adjacent axes while still capturing axis labels and titles
    fig.savefig(figures_dir / filename, dpi=dpi,
                bbox_inches=bbox_in.expanded(1.02, 1.02))
 
 
def plot_dgi_dashboard(dgi_df: pd.DataFrame, figures_dir,
                       top_genes: int = 30,
                       top_heatmap_drugs: int = 20,
                       max_heatmap_genes: int = 12):
    """
    5-panel summary dashboard.  Saves combined + 5 individual panels.
 
    Reported trial phases and approval are separate fields; missing phase
    is displayed as Unknown. Approval is not specific to HCC.
    """
    figures_dir = _as_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if dgi_df.empty:
        raise ValueError("No interaction evidence to plot")
 
    # ── Pre-compute gene counts ───────────────────────────────────────────────
    gc_full = dgi_df.groupby(["gene", "source"]).size().unstack(fill_value=0)
    gc_full["_total"] = gc_full.sum(axis=1)
    gc_full = gc_full.sort_values("_total", ascending=False)
    gc = gc_full.head(top_genes).drop(columns="_total")
    gc = gc.loc[gc.sum(axis=1).sort_values(ascending=True).index]
    n_genes = len(gc)
 
    # ── Layout ────────────────────────────────────────────────────────────────
    panel_a_height = max(4.5, min(11.0, n_genes * 0.30))
    # Extra height for bottom row to give room for Panel B legend below donut
    fig = plt.figure(figsize=(18, panel_a_height + 7.0), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[panel_a_height, 7.0],
        hspace=0.65, wspace=0.45,
    )
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — interactions per gene
    # ══════════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :2])
    bot = np.zeros(n_genes)
    for src in list(SRC_COL):
        if src in gc.columns:
            v = gc[src].values
            ax1.barh(gc.index, v, left=bot, color=SRC_COL[src],
                     label=src, alpha=0.88, height=0.72)
            bot += v
    totals = gc.sum(axis=1)
    for i, (_, tot) in enumerate(totals.items()):
        ax1.text(tot + totals.max() * 0.01, i, f"{int(tot):,}",
                 va="center", fontsize=7.5, color="#333")
    ax1.set_xlabel("Number of drug interactions", fontsize=10)
    ax1.set_title(
        f"A  Interactions per gene  (top {top_genes} of {gc_full.shape[0]})",
        fontsize=11, fontweight="bold", loc="left")
    ax1.tick_params(axis="y", labelsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax1.margins(y=0.01)
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel B — interaction type donut
    # FIX: legend placed BELOW the donut (not beside it touching Panel A)
    # ══════════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 2])
 
    tc_raw  = dgi_df["interaction_type"].str.lower().fillna("unknown").value_counts()
    pct_raw = tc_raw / tc_raw.sum() * 100
    keep    = pct_raw >= 3.0
    tc_kept = tc_raw[keep].copy()
    other   = tc_raw[~keep].sum()
    if other > 0:
        tc_kept["other"] = other
    n_t   = len(tc_kept)
    tcols = _TYPE_COLS[:n_t]
    pct_k = tc_kept / tc_kept.sum() * 100
 
    ax2.pie(
        tc_kept.values,
        labels=None,                    # all labels go in the legend below
        colors=tcols,
        autopct=lambda p: f"{p:.0f}%" if p >= 6.0 else "",
        pctdistance=0.72,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 8, "fontweight": "bold"},
    )
 
    # Legend BELOW the donut — bbox_to_anchor y < 0 puts it under the axes
    legend_handles = [
        mpatches.Patch(color=tcols[i],
                       label=f"{lbl}  ({pct_k.iloc[i]:.0f}%)")
        for i, lbl in enumerate(tc_kept.index)
    ]
    ax2.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.05),   # centred, below the axes
        ncol=2,                          # two columns so it doesn't get too tall
        fontsize=8, framealpha=0.9,
        title="Interaction type", title_fontsize=8.5,
        borderpad=0.8, handlelength=1.2,
    )
    ax2.set_title("B  Interaction types",
                  fontsize=11, fontweight="bold", loc="left")
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel C — stacked 100% horizontal bars
    # FIX: legend placed ABOVE the bars (outside the data area)
    # ══════════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 0])
 
    approval_status = dgi_df.approved.map(
        lambda v: "Unknown" if pd.isna(v) else "Approved" if v else "Not approved")
    appr = (dgi_df.assign(approval_status=approval_status)
            .groupby(["source", "approval_status"]).size().unstack(fill_value=0))
    for col in ["Approved", "Not approved", "Unknown"]:
        if col not in appr.columns:
            appr[col] = 0
    appr     = appr[appr.sum(axis=1) > 0].copy()
    tot_src  = appr.sum(axis=1)
    pct_a    = appr["Approved"]     / tot_src * 100
    pct_na   = appr["Not approved"] / tot_src * 100
    pct_unknown = appr["Unknown"] / tot_src * 100
    y_pos    = np.arange(len(appr))
    bh       = 0.45
 
    bars_a  = ax3.barh(y_pos, pct_a.values, height=bh,
                       color="#1D9E75", alpha=0.88, label="Approved")
    bars_na = ax3.barh(y_pos, pct_na.values, height=bh, left=pct_a.values,
                       color="#D3D1C7", alpha=0.88, label="Not approved")
    ax3.barh(y_pos, pct_unknown.values, height=bh, left=(pct_a + pct_na).values,
             color="#888888", alpha=0.88, label="Unknown")
 
    # Segment labels
    for i in range(len(appr)):
        pa, pn = pct_a.iloc[i], pct_na.iloc[i]
        if pa > 6:
            ax3.text(pa / 2, i, f"{pa:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        if pn > 6:
            ax3.text(pa + pn / 2, i, f"{pn:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="#444", fontweight="bold")
        # Total count to the right of the 100% mark
        ax3.text(103, i, f"n={int(tot_src.iloc[i]):,}",
                 va="center", fontsize=7.5, color="#333")
 
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(appr.index, fontsize=9)
    ax3.set_xlabel("Percentage (%)", fontsize=9)
    ax3.set_xlim(0, 120)
    ax3.axvline(100, color="#ccc", lw=0.7, ls="--")
    ax3.set_title("C  Approval by source",
                  fontsize=11, fontweight="bold", loc="left")
    ax3.spines[["top", "right"]].set_visible(False)
 
    # Legend ABOVE the bars — completely outside the data area
    ax3.legend(
        fontsize=8, framealpha=0.90,
        loc="upper right",
        # bbox_to_anchor=(0.0, 1.02),     # above the axes
        ncol=2, borderpad=0.7,
    )
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel D — clinical phase (log scale)
    # Do not infer phase from approval. Preserve an explicit Unknown category.
    # ══════════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 1])
 
    pm = {
        0: "Reported 0",
        1: "Phase 1",
        2: "Phase 2",
        3: "Phase 3",
        4: "Reported 4",
    }
    po = [*pm.values(), "Unknown"]
    pv = [dgi_df["clinical_phase"].map(pm).value_counts().get(p, 0) for p in po[:-1]] + [int(dgi_df["clinical_phase"].isna().sum())]
    bars = ax4.bar(po, pv, color=[*[PHASE_COL[k] for k in range(5)], "#888888"],
                   alpha=0.88, edgecolor="white", zorder=3)
 
    ax4.set_yscale("symlog", linthresh=10)
    ax4.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else "0"))
 
    fig.canvas.draw()
    y_top = ax4.get_ylim()[1]
    for b, v in zip(bars, pv):
        if v > 0:
            y_ann = min(v * 1.12, y_top * 0.88)
            ax4.text(b.get_x() + b.get_width() / 2, y_ann, f"{v:,}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
 
    ax4.axvline(0.5, color="#aaa", lw=1.0, ls="--", zorder=2)
    ax4.set_ylabel("Count (log scale)", fontsize=9)
    ax4.set_title("D  Clinical phase",
                  fontsize=11, fontweight="bold", loc="left")
    # Two-line tick labels need slightly more space — reduce font
    ax4.tick_params(axis="x", labelsize=7.5, rotation=0)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", ls=":", alpha=0.4, zorder=0)
 
    # ══════════════════════════════════════════════════════════════════════════
    # Panel E — score heatmap
    # ══════════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 2])
 
    d_gc  = dgi_df.groupby("drug")["gene"].nunique()
    multi = d_gc[d_gc >= 2].index
    top_m = (dgi_df[dgi_df["drug"].isin(multi)].drop_duplicates("drug")
             .nlargest(top_heatmap_drugs, "composite_score")["drug"].tolist())
    if len(top_m) < top_heatmap_drugs:
        rem = (dgi_df[~dgi_df["drug"].isin(top_m)].drop_duplicates("drug")
               .nlargest(top_heatmap_drugs - len(top_m), "composite_score")
               ["drug"].tolist())
        sel = top_m + rem
    else:
        sel = top_m
 
    hdf = (dgi_df[dgi_df["drug"].isin(sel)]
           .pivot_table(index="drug", columns="gene",
                        values="composite_score", aggfunc="max", fill_value=0))
    hdf = hdf.loc[:, (hdf > 0).any()]
    hdf = hdf.loc[hdf.max(axis=1).sort_values(ascending=False).index]
    if hdf.shape[1] > max_heatmap_genes:
        col_fill = (hdf > 0).sum().sort_values(ascending=False)
        hdf = hdf[col_fill.head(max_heatmap_genes).index]
 
    im = ax5.imshow(hdf.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax5.set_xticks(range(len(hdf.columns)))
    ax5.set_xticklabels(hdf.columns, rotation=90, ha="center", fontsize=7.5)
    ax5.set_yticks(range(len(hdf.index)))
    ax5.set_yticklabels(hdf.index, fontsize=7.5)
    plt.colorbar(im, ax=ax5, shrink=0.75, pad=0.02, label="Composite score")
    ax5.set_title("E  Score heatmap — top drugs",
                  fontsize=11, fontweight="bold", loc="left")
 
    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Drug–Gene Interaction Evidence — HCC Candidates\n"
        f"{gc_full.shape[0]} genes · {dgi_df['drug'].nunique():,} unique drugs "
        f"· {int(dgi_df['approved'].sum()):,} approved",
        fontsize=13, fontweight="bold", y=1.01,
    )
 
    # ── Save combined ─────────────────────────────────────────────────────────
    combined_out = figures_dir / "dgi_summary_dashboard.png"
    fig.savefig(combined_out, dpi=200, bbox_inches="tight")
    print(f"Saved (combined): {combined_out}")
 
    # ── Save each panel individually ──────────────────────────────────────────
    fig.canvas.draw()
    for ax, fname in [
        (ax1, "dgi_panel_A_interactions.png"),
        (ax2, "dgi_panel_B_interaction_types.png"),
        (ax3, "dgi_panel_C_approval.png"),
        (ax4, "dgi_panel_D_clinical_phase.png"),
        (ax5, "dgi_panel_E_score_heatmap.png"),
    ]:
        _save_panel(ax, figures_dir, fname, dpi=200)
        print(f"Saved (panel)  : {figures_dir / fname}")
 
    plt.show()
    return fig
