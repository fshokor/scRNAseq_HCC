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


# ─────────────────────────────────────────────────────────────────────────────
def run_wilcoxon(adata, groupby="sample", group="tumor (HCC2)",
                 padj_thresh=0.05, log2fc_thresh=1.0):
    """
    Run Wilcoxon rank-sum differential expression test.

    Compares the specified group against all other cells.
    Filters to significant DEGs by adjusted p-value and fold change.

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

    Returns
    -------
    sig : pd.DataFrame
        Significant DEGs with columns: gene, log2FC, adj_pvalue, regulation.
    de_results : pd.DataFrame
        Full (unfiltered) results from Scanpy.
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")
    de_results = sc.get.rank_genes_groups_df(adata, group=group)

    sig = de_results[
        (de_results["pvals_adj"] < padj_thresh) &
        (de_results["logfoldchanges"].abs() > log2fc_thresh)
    ].copy()
    sig.columns = ["gene","scores", "log2FC", "pvalue", "adj_pvalue"]
    sig["regulation"] = (sig["log2FC"] > 0).map({True: "up", False: "down"})

    print(f"Total DEGs    : {len(sig)}")
    print(f"Upregulated   : {(sig.regulation=='up').sum()}")
    print(f"Downregulated : {(sig.regulation=='down').sum()}")
    return sig, de_results


# ─────────────────────────────────────────────────────────────────────────────
def plot_volcano(sig, figures_dir, n_labels=10):
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
    colors = sig["regulation"].map({"up": "#D85A30", "down": "#1D9E75"})
    ax.scatter(sig["log2FC"], -np.log10(sig["adj_pvalue"] + 1e-300),
               c=colors, alpha=0.7, s=20, linewidths=0)

    ax.axvline( 1,  color="#888", lw=0.8, ls="--")
    ax.axvline(-1,  color="#888", lw=0.8, ls="--")
    ax.axhline(-np.log10(0.05), color="#888", lw=0.8, ls="--")

    for _, r in sig.nsmallest(n_labels, "adj_pvalue").iterrows():
        ax.text(r["log2FC"],
                -np.log10(r["adj_pvalue"] + 1e-300) + 0.3,
                r["gene"], fontsize=7, ha="center")

    ax.set_xlabel("log2 fold change (tumor / normal)", fontsize=11)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=11)
    ax.set_title(
        f"DEA: HCC2 (tumor) vs HCC1 (normal) — {len(sig)} significant DEGs",
        fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(facecolor="#D85A30", label="Up in tumor"),
        mpatches.Patch(facecolor="#1D9E75", label="Down in tumor"),
    ], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures_dir / "volcano_plot.png", dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {figures_dir}/volcano_plot.png")
    return fig


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
