"""
gsea_functions.py
=================
All logic for notebook 05 · Gene Set Enrichment Analysis.

Functions
---------
prepare_ranked_list   — build ranked gene list from DEA results
run_gsea_r            — run clusterProfiler GSEA via rpy2
print_gsea_summary    — print top terms per ontology from saved CSVs
"""

import pandas as pd
import numpy as np


# HCC-relevant biological themes (used for targeted reporting)
HCC_THEMES = {
    "Lipid metabolism"    : "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis / energy" : "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K-AKT / Wnt"     : "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune regulation"   : "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB",
}

# R script template — executed via ro.r() after setting variables in globalenv
_R_GSEA_SCRIPT = r"""
suppressPackageStartupMessages({
    library(clusterProfiler); library(org.Hs.eg.db)
    library(enrichplot);      library(ggplot2)
})

rnk <- read.table(paste0(PROC_DIR, "/ranked_genes_log2fc.tsv"),
                  sep="\t", header=FALSE, stringsAsFactors=FALSE)
colnames(rnk) <- c("SYMBOL", "log2FC")
rnk <- rnk[order(rnk$log2FC, decreasing=TRUE), ]

mapping  <- bitr(rnk$SYMBOL, fromType="SYMBOL",
                 toType="ENTREZID", OrgDb=org.Hs.eg.db)
rnk2     <- merge(rnk, mapping, by="SYMBOL")
rnk2     <- rnk2[order(rnk2$log2FC, decreasing=TRUE), ]
geneList <- rnk2$log2FC
names(geneList) <- rnk2$ENTREZID

cat(sprintf("Genes mapped to Entrez: %d / %d\n",
            length(geneList), nrow(rnk)))

PARAMS <- list(minGSSize=15, maxGSSize=500,
               pvalueCutoff=0.05, verbose=FALSE)
set.seed(42)

cat("Running GO-BP...\n")
gsea_bp <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="BP", keyType="ENTREZID"), PARAMS))

cat("Running GO-MF...\n")
gsea_mf <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="MF", keyType="ENTREZID"), PARAMS))

cat("Running GO-CC...\n")
gsea_cc <- do.call(gseGO, c(
    list(geneList=geneList, OrgDb=org.Hs.eg.db,
         ont="CC", keyType="ENTREZID"), PARAMS))

cat("Running KEGG...\n")
gsea_kegg <- do.call(gseKEGG, c(
    list(geneList=geneList, organism="hsa"), PARAMS))

cat(sprintf("\nGO-BP : %d terms\n",   nrow(as.data.frame(gsea_bp))))
cat(sprintf("GO-MF : %d terms\n",   nrow(as.data.frame(gsea_mf))))
cat(sprintf("GO-CC : %d terms\n",   nrow(as.data.frame(gsea_cc))))
cat(sprintf("KEGG  : %d pathways\n", nrow(as.data.frame(gsea_kegg))))

# ── Dot plots ─────────────────────────────────────────────────────────────
safe_dot <- function(obj, title, showCat=15) {
    df <- as.data.frame(obj)
    if (nrow(df) == 0) {
        cat(sprintf("No results for '%s'\n", title)); return()
    }
    p <- dotplot(obj, showCategory=min(showCat, nrow(df))) +
         ggtitle(title) + theme_bw(base_size=11)
    fname <- paste0(FIGURES_DIR, "/gsea_",
                    gsub(" ", "_", tolower(title)), ".png")
    ggsave(fname, p, width=10, height=7, dpi=200)
    cat(sprintf("Saved: %s\n", fname))
    print(p)
}
safe_dot(gsea_bp,   "GO Biological Process")
safe_dot(gsea_mf,   "GO Molecular Function")
safe_dot(gsea_cc,   "GO Cellular Component")
safe_dot(gsea_kegg, "KEGG Pathways")

# ── Ridge plot ────────────────────────────────────────────────────────────
bp_df <- as.data.frame(gsea_bp)
if (nrow(bp_df) > 0) {
    p <- ridgeplot(gsea_bp, showCategory=min(15, nrow(bp_df))) +
         ggtitle("GO-BP enrichment distribution") + theme_bw(base_size=10)
    ggsave(paste0(FIGURES_DIR, "/gsea_ridgeplot_bp.png"),
           p, width=10, height=8, dpi=200)
    cat(sprintf("Saved: %s/gsea_ridgeplot_bp.png\n", FIGURES_DIR))
}

# ── HCC theme plots ───────────────────────────────────────────────────────
themes <- list(
    "Lipid_metabolism"  = "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis_energy" = "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K_AKT_Wnt"     = "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune_regulation" = "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB"
)
for (theme_name in names(themes)) {
    hits <- grep(themes[[theme_name]], bp_df$Description, ignore.case=TRUE)
    if (length(hits) == 0) next
    term_id <- bp_df$ID[hits[1]]
    tryCatch({
        p <- gseaplot2(gsea_bp, geneSetID=term_id,
                       title=paste0(theme_name, ": ",
                                    bp_df$Description[hits[1]]))
        ggsave(paste0(FIGURES_DIR, "/gsea_theme_", theme_name, ".png"),
               p, width=10, height=6, dpi=200)
        cat(sprintf("Saved theme: %s\n", theme_name))
    }, error=function(e) {
        cat(sprintf("Could not plot theme '%s': %s\n",
                    theme_name, conditionMessage(e)))
    })
}

# ── Export CSV tables ─────────────────────────────────────────────────────
write.csv(as.data.frame(gsea_bp),
          paste0(TABLES_DIR, "/gsea_go_bp.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_mf),
          paste0(TABLES_DIR, "/gsea_go_mf.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_cc),
          paste0(TABLES_DIR, "/gsea_go_cc.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_kegg),
          paste0(TABLES_DIR, "/gsea_kegg.csv"),    row.names=FALSE)
cat("CSV tables saved.\n")
"""


# ─────────────────────────────────────────────────────────────────────────────
def prepare_ranked_list(dea_path, proc_dir):
    """
    Build and save a ranked gene list from DEA results.

    Ranks all genes by log2FC (not just significant ones).
    Saves to data/processed/ranked_genes_log2fc.tsv for R.

    Parameters
    ----------
    dea_path : Path
        Path to dea_results.csv.
    proc_dir : Path
        Output directory (data/processed/).

    Returns
    -------
    ranked : pd.DataFrame
        Columns: gene, log2FC — sorted descending by log2FC.
    """
    dea = pd.read_csv(dea_path)
    ranked = (dea[["gene", "log2FC"]].dropna()
              .groupby("gene", as_index=False).mean()
              .sort_values("log2FC", ascending=False))

    out = proc_dir / "ranked_genes_log2fc.tsv"
    ranked.to_csv(out, sep="\t", index=False, header=False)
    print(f"Ranked list: {len(ranked)} genes → {out}")
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
def run_gsea_r(ro, proc_dir, figures_dir, tables_dir):
    """
    Execute the full GSEA pipeline in R via rpy2.

    Passes Python path variables into R's globalenv, then runs
    the embedded R script (_R_GSEA_SCRIPT) which:
      - Maps gene symbols → Entrez IDs
      - Runs gseGO (BP/MF/CC) and gseKEGG
      - Saves dot plots, ridge plot, and theme-specific GSEA plots
      - Exports CSV tables

    Parameters
    ----------
    ro : rpy2.robjects module
        Must already be imported with %load_ext rpy2.ipython.
    proc_dir, figures_dir, tables_dir : Path
        Project paths passed into R as string variables.
    """
    ro.globalenv["PROC_DIR"]    = str(proc_dir)
    ro.globalenv["FIGURES_DIR"] = str(figures_dir)
    ro.globalenv["TABLES_DIR"]  = str(tables_dir)

    try:
        ro.r(_R_GSEA_SCRIPT)
        print("\n✓ GSEA complete")
    except Exception as e:
        print(f"\n✗ R error: {e}")
        print("Check that clusterProfiler and org.Hs.eg.db are installed.")
        print("Run: Rscript env/r_packages.R")


# ─────────────────────────────────────────────────────────────────────────────
def print_gsea_summary(tables_dir):
    """
    Print a concise summary of GSEA results from the saved CSV tables.

    Shows the count of enriched terms per ontology and the top 5 terms
    by NES for each HCC-relevant biological theme.

    Parameters
    ----------
    tables_dir : Path
        results/tables/ directory.
    """
    print("── GSEA results summary ──\n")
    for label, fname in [("GO-BP",  "gsea_go_bp.csv"),
                          ("GO-MF",  "gsea_go_mf.csv"),
                          ("GO-CC",  "gsea_go_cc.csv"),
                          ("KEGG",   "gsea_kegg.csv")]:
        fpath = tables_dir / fname
        if not fpath.exists():
            print(f"  {label}: file not found (R step may have failed)")
            continue
        df = pd.read_csv(fpath)
        print(f"  {label}: {len(df)} enriched terms")
        if len(df) > 0 and "Description" in df.columns and "NES" in df.columns:
            top = df.nlargest(3, "NES")[["Description", "NES", "p.adjust"]]
            print(top.to_string(index=False))
        print()

    bp_file = tables_dir / "gsea_go_bp.csv"
    if not bp_file.exists():
        return

    bp_df = pd.read_csv(bp_file)
    print("── HCC-relevant pathway hits (GO-BP) ──\n")
    for theme, pattern in HCC_THEMES.items():
        hits = bp_df[bp_df["Description"].str.contains(
            pattern, case=False, na=False, regex=True)]
        print(f"  {theme}: {len(hits)} terms")
        if len(hits) > 0:
            top = hits.nlargest(3, "NES")[["Description", "NES"]].to_string(index=False)
            for line in top.split("\n")[1:]:
                print(f"    {line.strip()}")
        print()
