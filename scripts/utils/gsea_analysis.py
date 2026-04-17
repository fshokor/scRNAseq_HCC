"""
gsea_analysis.py
================
Step 05 — Gene Set Enrichment Analysis (GSEA)

Runs ranked GSEA via clusterProfiler (R) for GO-BP, GO-MF, GO-CC, and KEGG.
Produces dot plots, ridge plots, and per-theme pathway tables for the four
HCC-relevant biological themes from Wang et al. (2025).

Inputs (auto-resolved via paths.py)
------------------------------------
  data/processed/dea_results.csv        — gene, log2FC, adj_pvalue, regulation

Outputs
-------
  data/processed/ranked_genes_log2fc.tsv  — ranked gene list for R
  results/tables/gsea_go_bp.csv
  results/tables/gsea_go_mf.csv
  results/tables/gsea_go_cc.csv
  results/tables/gsea_kegg.csv
  results/figures/gsea_go_bp.png
  results/figures/gsea_go_mf.png
  results/figures/gsea_go_cc.png
  results/figures/gsea_kegg.png
  results/figures/gsea_ridgeplot_bp.png
  results/figures/gsea_theme_*.png        — one per HCC theme

Requirements
------------
  Python : rpy2>=3.5
  R      : clusterProfiler, org.Hs.eg.db, enrichplot, DOSE, ggplot2
           Install with: Rscript env/r_packages.R
"""

# ── Import shared paths ───────────────────────────────────────────────────────
import sys
from pathlib import Path

def _find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / "paths.py").exists():
            return p
    return start.parent

_repo = _find_repo_root(Path(__file__).resolve().parent)
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

try:
    from paths import REPO_ROOT, PROC_DIR, FIGURES_DIR, TABLES_DIR
except ImportError:
    REPO_ROOT   = Path(__file__).resolve().parent.parent
    PROC_DIR    = REPO_ROOT / "data" / "processed"
    FIGURES_DIR = REPO_ROOT / "results" / "figures"
    TABLES_DIR  = REPO_ROOT / "results" / "tables"
    for d in [PROC_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DEA_FILE    = PROC_DIR / "dea_results.csv"
RANKED_FILE = PROC_DIR / "ranked_genes_log2fc.tsv"
PADJ_THRESH = 0.05
LOG2FC_THRESH = 1.0
GSEA_P_CUTOFF = 0.05
MIN_GS_SIZE   = 15
MAX_GS_SIZE   = 500
RANDOM_SEED   = 42

# HCC-relevant biological themes for targeted plots
HCC_THEMES = {
    "Lipid metabolism"    : "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis / energy" : "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K-AKT / Wnt"     : "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune regulation"   : "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB",
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Prepare ranked gene list
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ranked_list():
    print("── Step 1: Preparing ranked gene list ──")
    dea = pd.read_csv(DEA_FILE)
    print(f"  DEGs loaded: {len(dea)}")

    ranked = (dea[["gene", "log2FC"]].dropna()
              .groupby("gene", as_index=False).mean()
              .sort_values("log2FC", ascending=False))
    ranked.to_csv(RANKED_FILE, sep="\t", index=False, header=False)
    print(f"  Ranked list saved: {len(ranked)} genes → {RANKED_FILE}")
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run GSEA via rpy2
# ─────────────────────────────────────────────────────────────────────────────

R_GSEA_SCRIPT = r"""
suppressPackageStartupMessages({
    library(clusterProfiler)
    library(org.Hs.eg.db)
    library(enrichplot)
    library(DOSE)
    library(ggplot2)
})

# ── Load ranked gene list ────────────────────────────────────────────────────
rnk <- read.table(RANKED_FILE, sep="\t", header=FALSE, stringsAsFactors=FALSE)
colnames(rnk) <- c("SYMBOL", "log2FC")
rnk <- rnk[order(rnk$log2FC, decreasing=TRUE), ]

mapping  <- bitr(rnk$SYMBOL, fromType="SYMBOL", toType="ENTREZID", OrgDb=org.Hs.eg.db)
rnk2     <- merge(rnk, mapping, by="SYMBOL")
rnk2     <- rnk2[order(rnk2$log2FC, decreasing=TRUE), ]
geneList <- rnk2$log2FC
names(geneList) <- rnk2$ENTREZID

cat(sprintf("Genes mapped to Entrez: %d / %d\n", length(geneList), nrow(rnk)))

PARAMS <- list(
    minGSSize    = MIN_GS_SIZE,
    maxGSSize    = MAX_GS_SIZE,
    pvalueCutoff = GSEA_P_CUTOFF,
    verbose      = FALSE
)

set.seed(RANDOM_SEED)

cat("Running GO Biological Process...\n")
gsea_bp <- do.call(gseGO, c(list(geneList=geneList, OrgDb=org.Hs.eg.db,
                                  ont="BP", keyType="ENTREZID"), PARAMS))

cat("Running GO Molecular Function...\n")
gsea_mf <- do.call(gseGO, c(list(geneList=geneList, OrgDb=org.Hs.eg.db,
                                  ont="MF", keyType="ENTREZID"), PARAMS))

cat("Running GO Cellular Component...\n")
gsea_cc <- do.call(gseGO, c(list(geneList=geneList, OrgDb=org.Hs.eg.db,
                                  ont="CC", keyType="ENTREZID"), PARAMS))

cat("Running KEGG...\n")
gsea_kegg <- do.call(gseKEGG, c(list(geneList=geneList, organism="hsa"), PARAMS))

cat(sprintf("\nGO-BP  : %d terms\n",  nrow(as.data.frame(gsea_bp))))
cat(sprintf("GO-MF  : %d terms\n",  nrow(as.data.frame(gsea_mf))))
cat(sprintf("GO-CC  : %d terms\n",  nrow(as.data.frame(gsea_cc))))
cat(sprintf("KEGG   : %d pathways\n", nrow(as.data.frame(gsea_kegg))))

# ── Save CSV tables ───────────────────────────────────────────────────────────
write.csv(as.data.frame(gsea_bp),   paste0(TABLES_DIR, "/gsea_go_bp.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_mf),   paste0(TABLES_DIR, "/gsea_go_mf.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_cc),   paste0(TABLES_DIR, "/gsea_go_cc.csv"),   row.names=FALSE)
write.csv(as.data.frame(gsea_kegg), paste0(TABLES_DIR, "/gsea_kegg.csv"),    row.names=FALSE)
cat("CSV tables saved.\n")

# ── Dot plots ─────────────────────────────────────────────────────────────────
safe_dotplot <- function(obj, title, outfile, showCat=15) {
    df <- as.data.frame(obj)
    if (nrow(df) == 0) { cat(sprintf("No results for '%s'\n", title)); return() }
    p <- dotplot(obj, showCategory=min(showCat, nrow(df))) + ggtitle(title) +
         theme_bw(base_size=11)
    ggsave(outfile, p, width=10, height=7, dpi=200)
    cat(sprintf("Saved: %s\n", outfile))
}

safe_dotplot(gsea_bp,   "GO Biological Process — HCC DEGs",
             paste0(FIGURES_DIR, "/gsea_go_bp.png"))
safe_dotplot(gsea_mf,   "GO Molecular Function — HCC DEGs",
             paste0(FIGURES_DIR, "/gsea_go_mf.png"))
safe_dotplot(gsea_cc,   "GO Cellular Component — HCC DEGs",
             paste0(FIGURES_DIR, "/gsea_go_cc.png"))
safe_dotplot(gsea_kegg, "KEGG Pathways — HCC DEGs",
             paste0(FIGURES_DIR, "/gsea_kegg.png"))

# ── Ridge plot ────────────────────────────────────────────────────────────────
bp_df <- as.data.frame(gsea_bp)
if (nrow(bp_df) > 0) {
    p <- ridgeplot(gsea_bp, showCategory=min(15, nrow(bp_df))) +
         ggtitle("GO-BP enrichment distribution") + theme_bw(base_size=10)
    ggsave(paste0(FIGURES_DIR, "/gsea_ridgeplot_bp.png"), p, width=10, height=8, dpi=200)
    cat(sprintf("Saved: %s/gsea_ridgeplot_bp.png\n", FIGURES_DIR))
}

# ── HCC theme plots ───────────────────────────────────────────────────────────
themes <- list(
    "Lipid_metabolism"    = "lipid|fatty.acid|cholesterol|PPAR|lipoprotein",
    "Glycolysis_energy"   = "glycolysis|gluconeogenesis|glucose|TCA|oxidative.phosphorylation",
    "PI3K_AKT_Wnt"       = "PI3K|AKT|Wnt|beta.catenin|mTOR|MAPK",
    "Immune_regulation"   = "immune|inflamm|cytokine|T.cell|B.cell|interferon|NF.kB"
)

for (theme_name in names(themes)) {
    pattern <- themes[[theme_name]]
    hits    <- grep(pattern, bp_df$Description, ignore.case=TRUE)
    if (length(hits) == 0) {
        cat(sprintf("No GO-BP hits for theme '%s'\n", theme_name))
        next
    }
    term_id <- bp_df$ID[hits[1]]
    tryCatch({
        p <- gseaplot2(gsea_bp, geneSetID=term_id,
                       title=paste0(theme_name, ": ", bp_df$Description[hits[1]]))
        outfile <- paste0(FIGURES_DIR, "/gsea_theme_", theme_name, ".png")
        ggsave(outfile, p, width=10, height=6, dpi=200)
        cat(sprintf("Saved: %s\n", outfile))
    }, error=function(e) {
        cat(sprintf("Could not plot theme '%s': %s\n", theme_name, conditionMessage(e)))
    })
}

cat("\nGSEA analysis complete.\n")
"""


def run_gsea_in_r(ranked: pd.DataFrame):
    """Execute GSEA in R via rpy2, passing Python paths as R variables."""
    print("\n── Step 2: Running GSEA in R ──")

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        print("  ✗ rpy2 not installed. Install with: pip install rpy2")
        print("  Falling back to CSV export only — no R visualisations.")
        return

    # Pass Python paths into R global environment
    ro.globalenv["RANKED_FILE"]  = str(RANKED_FILE)
    ro.globalenv["TABLES_DIR"]   = str(TABLES_DIR)
    ro.globalenv["FIGURES_DIR"]  = str(FIGURES_DIR)
    ro.globalenv["MIN_GS_SIZE"]  = MIN_GS_SIZE
    ro.globalenv["MAX_GS_SIZE"]  = MAX_GS_SIZE
    ro.globalenv["GSEA_P_CUTOFF"] = GSEA_P_CUTOFF
    ro.globalenv["RANDOM_SEED"]  = RANDOM_SEED

    try:
        ro.r(R_GSEA_SCRIPT)
        print("  ✓ GSEA complete")
    except Exception as e:
        print(f"  ✗ R error: {e}")
        print("  Check that clusterProfiler and org.Hs.eg.db are installed.")
        print("  Run: Rscript env/r_packages.R")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Print summary from saved CSV tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n── Step 3: Results summary ──")
    for label, fname in [("GO-BP",  "gsea_go_bp.csv"),
                          ("GO-MF",  "gsea_go_mf.csv"),
                          ("GO-CC",  "gsea_go_cc.csv"),
                          ("KEGG",   "gsea_kegg.csv")]:
        fpath = TABLES_DIR / fname
        if not fpath.exists():
            print(f"  {label}: not found (R step may have failed)")
            continue
        df = pd.read_csv(fpath)
        print(f"\n  {label}: {len(df)} enriched terms/pathways")
        if len(df) > 0 and "Description" in df.columns and "NES" in df.columns:
            top = df.nlargest(5, "NES")[["Description","NES","p.adjust"]]
            print(top.to_string(index=False))

    # HCC theme hits
    bp_file = TABLES_DIR / "gsea_go_bp.csv"
    if bp_file.exists():
        bp_df = pd.read_csv(bp_file)
        print("\n  HCC-relevant pathway hits:")
        for theme, pattern in HCC_THEMES.items():
            import re
            hits = bp_df[bp_df["Description"].str.contains(
                pattern, case=False, na=False, regex=True)]
            print(f"    {theme}: {len(hits)} terms")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HCC Drug Discovery — Step 05: GSEA")
    print("=" * 60)
    print(f"  DEA input : {DEA_FILE}")
    print(f"  Tables    : {TABLES_DIR}")
    print(f"  Figures   : {FIGURES_DIR}")
    print()

    ranked = prepare_ranked_list()
    run_gsea_in_r(ranked)
    print_summary()

    print("\n" + "=" * 60)
    print("  GSEA complete. Outputs:")
    print(f"    {TABLES_DIR}/gsea_go_bp.csv")
    print(f"    {TABLES_DIR}/gsea_kegg.csv")
    print(f"    {FIGURES_DIR}/gsea_go_bp.png")
    print(f"    {FIGURES_DIR}/gsea_ridgeplot_bp.png")
    print("=" * 60)
