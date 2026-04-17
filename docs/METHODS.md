# Methods

Detailed computational methods for the HCC drug discovery pipeline,
corresponding to Wang et al. (2025) *npj Precision Oncology* 9:309.
[doi:10.1038/s41698-025-00952-3](https://doi.org/10.1038/s41698-025-00952-3)

---

## Dataset

Single-cell RNA sequencing data were obtained from NCBI GEO accession
[GSE166635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635).
The dataset contains 25,189 cells from two conditions: HCC1 (tumor-adjacent
normal tissue) and HCC2 (tumor tissue), profiled on the 10x Genomics Chromium
platform and sequenced on Illumina NovaSeq 6000. Raw count matrices (MTX
format) were downloaded programmatically via `scripts/data_download.py`.

---

## Preprocessing (notebook 01)

Raw count matrices were loaded with Scanpy (v≥1.10). Quality control metrics
were computed for each cell: number of detected genes (`n_genes_by_counts`),
total RNA counts (`total_counts`), and mitochondrial gene percentage
(`pct_counts_mt`). Cells were filtered using the following thresholds:

| Filter | Threshold |
|--------|-----------|
| Minimum genes per cell | 200 |
| Maximum genes per cell | 2,500 |
| Maximum mitochondrial % | 5% |

Raw counts were saved to `adata.layers["counts"]` before normalisation.
Library sizes were normalised to 10,000 counts per cell with
`sc.pp.normalize_total`, followed by log(x+1) transformation with
`sc.pp.log1p`. The top 2,000 highly variable genes were selected using
`sc.pp.highly_variable_genes` with `batch_key="sample"` to account for
batch differences between HCC1 and HCC2.

---

## Dimensionality reduction & clustering (notebook 02)

Principal Component Analysis (PCA) was applied to the HVG-subset expression
matrix. The top 10 principal components were retained based on the elbow of
the variance-explained scree plot. A k-nearest-neighbour graph was constructed
with k=15 neighbours in 10 PC dimensions using `sc.pp.neighbors`. UMAP
embeddings were computed with default parameters for visualisation.

Leiden clustering was applied at four resolutions (0.3, 0.5, 1.0, 2.0) using
the `igraph` flavour. Resolution 0.5 (11 clusters) was selected for downstream
annotation based on biological interpretability.

---

## Cell-type annotation (notebook 03)

Three automated annotation methods were applied and reconciled via a majority
vote:

**CellTypist** — `Immune_All_High.pkl` and `Immune_All_Low.pkl` models with
majority voting at the cluster level.

**ScType** — Liver-specific marker database sourced from the ScType GitHub
repository. The `sctype_score_` function was applied to log-normalised
expression, and the top-scoring cell type per Leiden cluster was recorded.
ScType was given double weight in the majority vote for parenchymal cell types
(hepatocytes, fibroblasts, endothelial cells) because it is the only tool
using a liver-specific reference.

**SingleR** — Human Primary Cell Atlas (HPCA) reference dataset via the
`celldex` package. Spearman correlation scores were computed against all
reference profiles; pruned labels (NA for low-confidence calls) were used.

**Manual scoring** — Mean log-normalised expression of canonical marker gene
sets (12 cell types, 6–10 markers each) per Leiden cluster. The top-scoring
type per cluster served as the fourth vote.

Final labels were stored in `adata.obs["manual_celltype"]`.

---

## Differential expression analysis (notebook 04)

Wilcoxon rank-sum test was applied to compare tumor (HCC2) vs normal-adjacent
(HCC1) cells using `sc.tl.rank_genes_groups`. Genes were retained as
significant DEGs if: adjusted p-value < 0.05 (Benjamini-Hochberg correction)
and |log2 fold change| > 1. This produced 1,178 DEGs (926 upregulated in
tumor, 252 downregulated).

---

## Gene set enrichment analysis (notebook 05)

A ranked gene list sorted by log2FC was prepared from all DEGs and passed to
`clusterProfiler::gseGO` and `clusterProfiler::gseKEGG` via `rpy2`. Gene
symbols were mapped to Entrez IDs using `bitr` with `org.Hs.eg.db`. GSEA
parameters: `minGSSize=15`, `maxGSSize=500`, `pvalueCutoff=0.05`, `seed=42`.
Four ontologies were analysed: GO Biological Process (BP), GO Molecular
Function (MF), GO Cellular Component (CC), and KEGG. HCC-relevant themes
(lipid metabolism, glycolysis, PI3K-AKT/Wnt, immune regulation) were
highlighted by pattern-matching pathway descriptions.

---

## PPI network analysis (notebook 09)

The 1,178 significant DEGs were submitted to the STRING API (functional
network, minimum interaction score ≥400) in batches of 500 genes. A NetworkX
undirected graph was constructed from the returned interactions, with edge
weights set to STRING combined scores. Isolated nodes (no interactions) were
removed. Four centrality measures were computed per node: degree, betweenness,
closeness, and eigenvector centrality. A composite hub score was calculated as
the normalised mean of all four measures. The top 20 hub genes were exported as
`hub_genes.csv` for downstream analysis.

---

## Survival analysis (notebook 11)

Clinical and gene expression data for TCGA-LIHC (n=374 HCC patients) were
downloaded from the UCSC Xena public hub. For each DEG with available
expression data, patients were stratified at the median expression level into
high and low groups. Log-rank tests assessed Kaplan-Meier survival
differences. Cox proportional-hazards regression (penalizer=0.1) computed
hazard ratios with 95% confidence intervals. Genes were retained as survival
biomarkers if: log-rank p<0.05, Cox p<0.05, and HR outside [0.8, 1.2].

---

## Drug–gene interaction collection (notebook 12)

Candidate drug-gene interactions were retrieved from three databases:

- **DGIdb** (GraphQL API) — interaction type, directionality, FDA approval status,
  clinical indication
- **ChEMBL** (REST API) — mechanism of action, maximum clinical trial phase
- **OpenTargets** (GraphQL API) — approved status, clinical phase, mechanism

A curated fallback dataset (37 literature-based interactions for 16 hub genes)
was used to supplement genes not covered by live APIs.

Interactions were deduplicated by gene-drug pair, retaining the highest
interaction score. A composite score was computed as a weighted sum:

| Component | Weight |
|-----------|--------|
| Interaction score (normalised) | 0.35 |
| Publication count (normalised, capped at 30) | 0.20 |
| Clinical phase (0–4, normalised to 0–1) | 0.20 |
| FDA approval (binary) | 0.15 |
| Hub score | 0.10 |
| Survival target bonus | +0.10 |

---

## GNN model (notebooks 13–14)

A heterogeneous drug-gene graph was constructed with PyTorch Geometric. Nodes
represent either genes (features: hub score, survival target flag) or drugs
(15 binary/numeric features from DGI collection). Edges are bidirectional,
weighted by composite score. Node features were standardised with
`StandardScaler` (scaler saved to `models/feature_scaler.pkl`).

Three architectures were trained and compared:

| Model | Key design |
|-------|------------|
| GCN | 2-layer graph convolutional network, mean aggregation |
| GAT | Graph attention network, 4 attention heads |
| GraphSAGE | Inductive node embedding, mean neighbourhood aggregation |

All models share the same edge prediction head: concatenated source-destination
embeddings → Linear → ReLU → Dropout(0.3) → Linear → Sigmoid. Training used
Adam optimiser (lr=0.005, weight_decay=1e-4), ReduceLROnPlateau scheduler
(patience=15), and early stopping (patience=40). Train/val/test split: 70/15/15.

The best-performing model (by test R²) was used to score all drug-gene pairs,
producing the final ranked drug candidate list.

---

## Macrophage sub-cluster analysis (notebook 06)

This analysis extends the Wang et al. paper. Cells annotated as macrophages
were subset and re-clustered (Leiden, resolution=0.4). Module scores were
computed for four macrophage phenotypes using `sc.tl.score_genes`:

- M1 (pro-inflammatory): TNF, IL1B, CXCL9, CD86, STAT1, NOS2, CXCL10
- M2 (anti-inflammatory): CD163, MRC1, IL10, VEGFA, APOE, ARG1, TGFB1
- Kupffer cells: VSIG4, TIMD4, C1QA, C1QB, MARCO, CLEC4F, FOLR2
- TAMs: SPP1, TREM2, GPNMB, CD9, MMP9, VCAN, SLC40A1

Differential expression between normal and tumor macrophages was computed by
Wilcoxon rank-sum test (padj<0.05, |log2FC|>1).

---

## Software versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| scanpy | ≥1.10 |
| anndata | ≥0.10 |
| celltypist | ≥1.6 |
| rpy2 | ≥3.5 |
| torch | ≥2.2 |
| torch-geometric | ≥2.5 |
| lifelines | ≥0.28 |
| networkx | ≥3.2 |
| scikit-learn | ≥1.4 |
| R | ≥4.3 |
| Seurat | ≥5.0 |
| SingleR | ≥2.0 |
| clusterProfiler | ≥4.0 |
