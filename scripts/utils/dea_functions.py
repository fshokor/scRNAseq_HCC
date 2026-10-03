"""
dea_functions.py
================
All logic for notebook 04 · Differential Expression Analysis.

Functions
---------
run_wilcoxon        — Wilcoxon rank-sum test via Scanpy
plot_volcano        — volcano plot with gene labels
export_dea          — save dea_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
from pathlib import Path
import json
import re


# ─────────────────────────────────────────────────────────────────────────────
def run_wilcoxon(adata, groupby="sample", group="tumor (HCC2)",
                 padj_thresh=0.05, log2fc_thresh=1.0, reference=None,
                 layer=None, min_detection_fraction=0.01):
    """
    Run Wilcoxon rank-sum differential expression test.

    Compare an explicit pair using log-normalized expression. Return all
    tested genes as well as the significant shortlist. Does not modify adata.
    These cell-level tests are exploratory when donors are not replicated.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (adata.obs must contain groupby column).
    groupby : str
        obs column to group by (default "sample").
    group : str
        The reference group label (tumor side, e.g. "tumor (HCC2)").
    padj_thresh : float
        Adjusted p-value cutoff.
    log2fc_thresh : float
        Minimum absolute log2 fold change.
    reference : str or None
        Explicit comparison side. If omitted, exactly one other group is required.
    layer : str or None
        Log-normalized layer; defaults to logcounts when available, else X.
    min_detection_fraction : float
        Pre-test detection threshold in either side, independent of significance.

    Returns
    -------
    sig : pd.DataFrame
        Significant DEGs with columns: gene, log2FC, adj_pvalue, regulation.
    de_results : pd.DataFrame
        Full (unfiltered) results from Scanpy.
    """
    if groupby not in adata.obs:
        raise KeyError(f"Missing grouping column: {groupby}")
    if not 0 <= min_detection_fraction <= 1:
        raise ValueError("min_detection_fraction must be between 0 and 1")
    labels = adata.obs[groupby].astype(str)
    if reference is None:
        alternatives = [x for x in labels.unique() if x != group]
        if len(alternatives) != 1:
            raise ValueError("Specify reference explicitly for a pairwise contrast")
        reference = alternatives[0]
    if group == reference:
        raise ValueError("group and reference must differ")
    pair = labels.isin([group, reference]).to_numpy()
    pair_labels = labels.loc[pair]
    for name in [group, reference]:
        if (pair_labels == name).sum() < 2:
            raise ValueError(f"Need at least two cells in {name!r}")
    if layer is None and "logcounts" in adata.layers:
        layer = "logcounts"
    subset = adata[pair]
    expression = subset.layers[layer] if layer else subset.X
    if layer == "counts" or np.issubdtype(expression.dtype, np.integer):
        raise ValueError("DE requires log-normalized expression, not integer raw counts")
    # Filter using expression detection, never DE significance. Retain a gene
    # if detected in the configured fraction of either side of this contrast.
    fractions = [np.asarray((expression[(pair_labels == name).to_numpy()] > 0)
                            .mean(axis=0)).ravel() for name in [group, reference]]
    keep = ((np.maximum(*fractions) >= min_detection_fraction)
            & (np.maximum(*fractions) > 0))
    if not keep.any():
        raise ValueError("No expressed genes pass the detection filter")
    # Only copy the required matrix, not counts, embeddings, or old DE objects.
    work = sc.AnnData(X=expression[:, keep].copy(),
                      obs=subset.obs[[groupby]].copy(),
                      var=subset.var.loc[keep].copy())
    work.obs[groupby] = pd.Categorical(pair_labels.to_numpy())
    if "log1p" in adata.uns:
        work.uns["log1p"] = dict(adata.uns["log1p"])
    sc.tl.rank_genes_groups(
        work, groupby=groupby, groups=[group], reference=reference,
        method="wilcoxon", tie_correct=True, use_raw=False,
        n_genes=work.n_vars, pts=True, corr_method="benjamini-hochberg",
        key_added="sample_contrast",
    )
    de_results = sc.get.rank_genes_groups_df(work, group=group, key="sample_contrast")
    de_results = de_results.rename(columns={
        "names": "gene", "logfoldchanges": "log2FC",
        "pvals": "pvalue", "pvals_adj": "adj_pvalue",
    })
    # Scanpy versions do not consistently export reference detection fractions
    # for an explicit pairwise reference. Attach both from the tested matrix.
    for column, fraction in zip(["pct_nz_group", "pct_nz_reference"], fractions):
        detection = pd.Series(fraction[keep], index=work.var_names)
        de_results[column] = de_results["gene"].map(detection)
    de_results["regulation"] = np.where(de_results["log2FC"] > 0, "up", "down")

    sig = de_results[
        (de_results["adj_pvalue"] < padj_thresh) &
        (de_results["log2FC"].abs() > log2fc_thresh)
    ].copy()

    print(f"Total DEGs    : {len(sig)}")
    print(f"Upregulated   : {(sig.regulation=='up').sum()}")
    print(f"Downregulated : {(sig.regulation=='down').sum()}")
    return sig, de_results


# ─────────────────────────────────────────────────────────────────────────────
def plot_volcano(sig, figures_dir, n_labels=10, padj_thresh=0.05,
                 log2fc_thresh=1.0, title=None, show=True):
    """
    Volcano plot: log2FC (x) vs -log10(adj_pvalue) (y).

    Upregulated genes are shown in coral, downregulated in teal.
    The top n_labels genes by significance are labelled.

    Parameters
    ----------
    sig : pd.DataFrame
        Significant DEGs (gene, log2FC, adj_pvalue, regulation).
    figures_dir : Path
        Directory to save volcano_plot.png.
    n_labels : int
        Number of gene names to annotate.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    significant = ((sig["adj_pvalue"] < padj_thresh)
                   & (sig["log2FC"].abs() > log2fc_thresh))
    colors = np.where(significant, np.where(sig["log2FC"] > 0,
                                            "#D85A30", "#1D9E75"), "#bbbbbb")
    ax.scatter(sig["log2FC"], -np.log10(sig["adj_pvalue"] + 1e-300),
               c=colors, alpha=0.7, s=20, linewidths=0)

    ax.axvline(log2fc_thresh, color="#888", lw=0.8, ls="--")
    ax.axvline(-log2fc_thresh, color="#888", lw=0.8, ls="--")
    ax.axhline(-np.log10(padj_thresh), color="#888", lw=0.8, ls="--")

    for _, r in sig.loc[significant].nsmallest(n_labels, "adj_pvalue").iterrows():
        ax.text(r["log2FC"],
                -np.log10(r["adj_pvalue"] + 1e-300) + 0.3,
                r["gene"], fontsize=7, ha="center")

    ax.set_xlabel("log2 fold change (tumor / normal)", fontsize=11)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=11)
    ax.set_title(
        title or f"HCC2 vs HCC1: {int(significant.sum())} shortlisted genes",
        fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#D85A30", label="Up in tumor"),
        mpatches.Patch(facecolor="#1D9E75", label="Down in tumor"),
    ], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "volcano_plot.png", dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved: {figures_dir}/volcano_plot.png")
    return fig


def run_sample_contrasts(adata, proc_dir, figures_dir, groupby="sample",
                         group="tumor (HCC2)", reference="normal (HCC1)",
                         celltype_col="manual_celltype", min_cells=20,
                         padj_thresh=0.05, log2fc_thresh=1.0,
                         min_detection_fraction=0.01, layer=None,
                         exclude_clusters=(), cluster_col="leiden_res_0.50"):
    """OmicSage-style result dictionary, summary, provenance and pairwise tests.

    Run pooled HCC2 versus HCC1, then the SAME contrast within each cell type.
    min_cells is a practical screening rule, not biological replication.
    Keep the master object unchanged. Optional cluster exclusions apply to
    pooled and within-type contrasts and are recorded in provenance.
    """
    if min_cells < 2:
        raise ValueError("min_cells must be at least 2")
    for column in [groupby, celltype_col]:
        if column not in adata.obs:
            raise KeyError(f"Missing required column: {column}")
    if not adata.var_names.is_unique:
        raise ValueError("Gene identifiers must be unique")
    if group == reference:
        raise ValueError("group and reference must differ")
    labels = adata.obs[groupby].astype(str)
    for sample in [group, reference]:
        if sample not in labels.values:
            raise ValueError(f"Sample {sample!r} not found")
    eligible = labels.isin([group, reference])
    if exclude_clusters:
        if cluster_col not in adata.obs:
            raise KeyError(f"Missing cluster column: {cluster_col}")
        eligible &= ~adata.obs[cluster_col].astype(str).isin(map(str, exclude_clusters))
    types = adata.obs[celltype_col].astype(str)
    entries = [("pooled", "All cells", eligible)]
    for celltype in sorted(types.loc[eligible].unique()):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", celltype)
        entries.append((f"celltype_{slug}", celltype, eligible & (types == celltype)))
    if len({entry[0] for entry in entries}) != len(entries):
        raise ValueError("Cell-type names produce duplicate output directory names")
    proc_dir, figures_dir = Path(proc_dir), Path(figures_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for contrast, celltype, mask in entries:
        n_group = int((mask & (labels == group)).sum())
        n_reference = int((mask & (labels == reference)).sum())
        row = dict(contrast=contrast, cell_type=celltype, n_tumor=n_group,
                   n_adjacent=n_reference, status="completed", reason="",
                   n_tested=0, n_significant=0, n_up=0, n_down=0)
        if celltype in {"Needs_review", "Myeloid_unresolved", "Unknown", "nan"}:
            row.update(status="skipped", reason="Unresolved annotation")
        elif min(n_group, n_reference) < min_cells:
            row.update(status="skipped", reason=f"Need >= {min_cells} cells per side")
        if row["status"] == "skipped":
            rows.append(row)
            continue
        row["interpretation"] = (
            "Composition-sensitive pooled sample contrast" if contrast == "pooled"
            else "Within broad cell type; subtype composition may differ"
        )
        row["small_or_imbalanced"] = (min(n_group, n_reference) < 50
                                      or max(n_group, n_reference) / min(n_group, n_reference) > 10)
        sig, full = run_wilcoxon(
            adata[mask.to_numpy()], groupby=groupby, group=group, reference=reference,
            padj_thresh=padj_thresh, log2fc_thresh=log2fc_thresh, layer=layer,
            min_detection_fraction=min_detection_fraction,
        )
        for frame in [sig, full]:
            frame["contrast"] = contrast
            frame["cell_type"] = celltype
            frame["group"] = group
            frame["reference"] = reference
        directory = proc_dir / contrast
        directory.mkdir(parents=True, exist_ok=True)
        full.to_csv(directory / "de_all_genes.csv", index=False)
        sig.to_csv(directory / "de_significant.csv", index=False)
        plot_volcano(full, figures_dir / contrast, padj_thresh=padj_thresh,
                     log2fc_thresh=log2fc_thresh,
                     title=f"{celltype}: {group} vs {reference}", show=False)
        results[contrast] = {"all_genes": full, "significant": sig,
                             "proc_dir": directory, "figures_dir": figures_dir / contrast}
        row.update(n_tested=len(full), n_significant=len(sig),
                   n_up=int((sig["log2FC"] > 0).sum()), n_down=int((sig["log2FC"] < 0).sum()))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(proc_dir / "contrast_summary.csv", index=False)
    provenance = dict(groupby=groupby, group=group, reference=reference,
                      celltype_col=celltype_col, min_cells=min_cells,
                      min_detection_fraction=min_detection_fraction,
                      padj_thresh=padj_thresh, log2fc_thresh=log2fc_thresh,
                      expression_source=layer or ("logcounts" if "logcounts" in adata.layers else "X"),
                      excluded_clusters=list(map(str, exclude_clusters)),
                      method="wilcoxon", tie_correct=True,
                      limitation="One sample per group; exploratory cell-level statistics",
                      scanpy_version=sc.__version__)
    (proc_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"results": results, "summary_df": summary, "provenance": provenance}


# ─────────────────────────────────────────────────────────────────────────────
def export_dea(sig, proc_dir):
    """
    Save significant DEGs to dea_results.csv.

    Parameters
    ----------
    sig : pd.DataFrame
        Columns: gene, log2FC, adj_pvalue, regulation.
    proc_dir : Path
        data/processed/ directory.
    """
    out = proc_dir / "dea_results.csv"
    sig[["gene", "log2FC", "adj_pvalue", "regulation"]].to_csv(out, index=False)
    print(f"Saved: {out}  ({len(sig)} DEGs)")
    print("\nTop 5 upregulated:")
    print(sig[sig.regulation == "up"]
          .nlargest(5, "log2FC")[["gene", "log2FC", "adj_pvalue"]]
          .to_string(index=False))
    print("\nTop 5 downregulated:")
    print(sig[sig.regulation == "down"]
          .nsmallest(5, "log2FC")[["gene", "log2FC", "adj_pvalue"]]
          .to_string(index=False))
