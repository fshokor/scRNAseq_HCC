
suppressPackageStartupMessages({
    library(clusterProfiler); library(org.Hs.eg.db)
    library(enrichplot);      library(ggplot2)
})

rnk <- read.table(RANK_FILE,
                  sep="\t", header=FALSE, stringsAsFactors=FALSE)
colnames(rnk) <- c("SYMBOL", "rank_metric")
rnk <- rnk[is.finite(rnk$rank_metric), ]
rnk <- rnk[order(rnk$rank_metric, decreasing=TRUE), ]

mapping  <- bitr(rnk$SYMBOL, fromType="SYMBOL",
                 toType="ENTREZID", OrgDb=org.Hs.eg.db)
mapping <- unique(mapping)
ambiguous <- names(which(table(mapping$SYMBOL) > 1))
mapping <- mapping[!mapping$SYMBOL %in% ambiguous, ]
rnk2 <- merge(rnk, mapping, by="SYMBOL")
# Omit ambiguous symbol mappings; resolve shared Entrez IDs deterministically.
rnk2 <- rnk2[order(-abs(rnk2$rank_metric), rnk2$SYMBOL), ]
duplicate_entrez <- sum(duplicated(rnk2$ENTREZID))
rnk2 <- rnk2[!duplicated(rnk2$ENTREZID), ]
rnk2 <- rnk2[order(rnk2$rank_metric, decreasing=TRUE), ]
if (nrow(rnk2) < 15) stop("Too few uniquely mapped genes for GSEA")
write.csv(rnk2, paste0(TABLES_DIR, "/rank_mapping.csv"), row.names=FALSE)
write.csv(data.frame(input_genes=nrow(rnk), mapped_unique_entrez=nrow(rnk2),
                     ambiguous_symbols=length(ambiguous),
                     duplicate_entrez_removed=duplicate_entrez),
          paste0(TABLES_DIR, "/mapping_summary.csv"), row.names=FALSE)
geneList <- rnk2$rank_metric
names(geneList) <- rnk2$ENTREZID

cat(sprintf("Genes mapped to Entrez: %d / %d\n",
            length(geneList), nrow(rnk)))

PARAMS <- list(minGSSize=15, maxGSSize=500,
               pvalueCutoff=1, pAdjustMethod="BH", verbose=FALSE, eps=0)
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
    df <- df[!is.na(df$p.adjust) & df$p.adjust <= GSEA_FDR, ]
    if (nrow(df) == 0) {
        cat(sprintf("No results for '%s'\n", title)); return()
    }
    filtered <- obj
    filtered@result <- df
    p <- dotplot(filtered, showCategory=min(showCat, nrow(df))) +
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
bp_df <- bp_df[!is.na(bp_df$p.adjust) & bp_df$p.adjust <= GSEA_FDR, ]
if (nrow(bp_df) > 0) {
    filtered_bp <- gsea_bp
    filtered_bp@result <- bp_df
    p <- ridgeplot(filtered_bp, showCategory=min(15, nrow(bp_df))) +
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
export_gsea <- function(obj) {
    df <- as.data.frame(obj)
    if (nrow(df) == 0) {
        df <- data.frame(ID=character(), Description=character(), NES=numeric(),
                         pvalue=numeric(), p.adjust=numeric(),
                         core_enrichment=character(), leading_edge=character())
    }
    # Entrez IDs and readable symbols remain separate. leading_edge contains
    # tags/list/signal statistics, NOT leading-edge gene identifiers.
    if (nrow(df) > 0) {
        id_symbols <- setNames(rnk2$SYMBOL, rnk2$ENTREZID)
        df$core_enrichment_symbols <- vapply(df$core_enrichment, function(ids) {
            symbols <- unname(id_symbols[strsplit(ids, "/", fixed=TRUE)[[1]]])
            paste(unique(symbols[!is.na(symbols)]), collapse="/")
        }, character(1))
    } else {
        df$core_enrichment_symbols <- character(0)
    }
    df$contrast <- rep(CONTRAST_ID, nrow(df))
    df$significant <- !is.na(df$p.adjust) & df$p.adjust <= GSEA_FDR
    df
}
write.csv(export_gsea(gsea_bp),
          paste0(TABLES_DIR, "/gsea_go_bp.csv"),   row.names=FALSE)
write.csv(export_gsea(gsea_mf),
          paste0(TABLES_DIR, "/gsea_go_mf.csv"),   row.names=FALSE)
write.csv(export_gsea(gsea_cc),
          paste0(TABLES_DIR, "/gsea_go_cc.csv"),   row.names=FALSE)
write.csv(export_gsea(gsea_kegg),
          paste0(TABLES_DIR, "/gsea_kegg.csv"),    row.names=FALSE)
cat("CSV tables saved.\n")
