# Methods

Detailed computational methods for the HCC drug discovery pipeline.

---

## Dataset

Single-cell RNA sequencing data were obtained from NCBI GEO accession
[GSE166635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635).
The dataset contains cells from two conditions profiled on the 10x Genomics
Chromium platform:

| Sample | Condition | Cells (raw) |
|--------|-----------|-------------|
| HCC1 | Tumor-adjacent normal tissue | 16,077 |
| HCC2 | Tumor tissue | 9,112 |

Raw count matrices (MTX format) are downloaded automatically by
`scripts/data_download.py` from the NCBI GEO FTP server.

---

## scRNA Analysis (`01_scrna_analysis.ipynb`)

### 01 · Preprocessing

Raw count matrices were loaded with Scanpy (v≥1.10) using `sc.read_10x_mtx`.
The two samples were merged with `concatenate`, preserving sample identity in
`adata.obs["sample"]`.

**QC metrics** were computed per cell: number of detected genes
(`n_genes_by_counts`), total RNA counts (`total_counts`), and percentage of
mitochondrial gene expression (`pct_counts_mt`). Ribosomal (RPS/RPL) and
haemoglobin (HB*) gene flags were also added for reference.

**Cell filtering** applied three thresholds simultaneously:

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Minimum genes | 200 | Remove empty droplets |
| Maximum genes | 2,500 | Remove potential doublets |
| Max mitochondrial % | 5% | Remove dying / low-quality cells |

Raw counts were preserved in `adata.layers["counts"]` before normalisation.
**Normalisation** scaled each cell to 10,000 counts with `sc.pp.normalize_total`,
followed by log(x+1) transformation with `sc.pp.log1p`.

**Highly variable gene (HVG) selection** used `sc.pp.highly_variable_genes` with
`n_top_genes=2000` and `batch_key="sample"` to account for batch differences
between HCC1 and HCC2. The 2,000 HVGs capture ~85% of the total variance while
suppressing noise from lowly expressed genes.

---

### 02 · Clustering 

**Dimensionality reduction** applied PCA to the HVG-subset expression matrix.
The top 10 principal components were retained based on the elbow of the
variance-ratio scree plot.

**Neighbor graph construction** used `sc.pp.neighbors` with k=15 nearest
neighbors in 10 PC dimensions. UMAP embeddings were computed with
`sc.tl.umap` for visualisation.

**Leiden clustering** was run at four resolutions (0.3, 0.5, 1.0, 2.0) using
the igraph implementation (`flavor="igraph"`). Resolution 0.5 produced 11
clusters and was selected for downstream annotation based on biological
interpretability and alignment with known liver cell-type proportions.

---

### 03 · Cell-type annotation 

Three automated annotation methods were applied independently and then reconciled:

#### CellTypist
`Immune_All_High.pkl` (coarse) and `Immune_All_Low.pkl` (fine) models with
majority voting at the Leiden cluster level. Input: normalised log-counts.

#### ScType
Liver-specific marker database (`ScTypeDB_full.xlsx`) sourced from the
[ScType GitHub repository](https://github.com/IanevskiAleksandr/sc-type).
Run via rpy2 in R. The `sctype_score_` function was applied to log-normalised
expression; the highest-scoring cell type per Leiden cluster was recorded.

#### SingleR
Human Primary Cell Atlas (HPCA) reference dataset (`celldex::HumanPrimaryCellAtlasData()`).
Spearman correlation scores were computed against all reference profiles; pruned
labels (NA for low-confidence calls) were used.

#### Marker gene scoring (manual)
Mean log-normalised expression of canonical marker gene sets per Leiden cluster.
11 cell types were scored using 6–10 markers each (defined in
`scrna_functions.MARKER_SETS`).

#### 4-way majority vote
The four evidence sources were combined per cluster:

- CellTypist fine label
- ScType label
- SingleR HPCA label
- Best label by marker score

**ScType received double weight** for parenchymal cell types (Hepatocyte,
Fibroblast, Endothelial) because it is the only tool using a liver-specific
reference database. Final labels were stored in `adata.obs["manual_celltype"]`.

---

### 04 · Differential expression analysis 

The Wilcoxon rank-sum test was applied via `sc.tl.rank_genes_groups` comparing
tumor tissue (HCC2) against normal-adjacent tissue (HCC1). DEGs were retained if:

- Adjusted p-value < 0.05 (Benjamini-Hochberg correction)
- |log2 fold change| > 1

This produced **1,385 significant DEGs**: 335 upregulated in tumor, 1,050 downregulated.
Results exported to `data/processed/dea_results.csv`.

---

### 05 · GSEA 

A ranked gene list sorted descending by log2FC was prepared from all DEGs and passed to clusterProfiler in R via rpy2.

**Gene symbol → Entrez ID mapping** used `bitr` from `org.Hs.eg.db`.

**GSEA** was run with `gseGO` and `gseKEGG` using:

| Parameter | Value |
|-----------|-------|
| minGSSize | 15 |
| maxGSSize | 500 |
| pvalueCutoff | 0.05 |
| Seed | 42 |

Four ontologies were analysed: GO Biological Process (BP), GO Molecular Function
(MF), GO Cellular Component (CC), and KEGG. Results exported as CSV tables and
dot/ridge plots. Four HCC-relevant themes were highlighted by pattern-matching
pathway descriptions: lipid metabolism, glycolysis/energy, PI3K-AKT/Wnt, and
immune regulation.

---
## Target Prioritisation (`02_target_prioritisation.ipynb`)
### PPI Network

The 1,385 significant DEGs were submitted to the
[STRING API](https://string-db.org/api/json/network) in batches of 500 genes.

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Species | Homo sapiens (taxon 9606) |
| Network type | Functional |
| Minimum combined score | 400 (medium confidence) |

A NetworkX undirected graph was constructed from returned interactions, with
edge weights set to STRING combined scores. Isolated nodes were removed.

**Hub score** = normalised mean of four centrality measures per gene:

| Measure | Function |
|---------|----------|
| Degree centrality | `nx.degree_centrality` |
| Betweenness centrality | `nx.betweenness_centrality(weight="weight")` |
| Closeness centrality | `nx.closeness_centrality` |
| Eigenvector centrality | `nx.eigenvector_centrality(max_iter=500, weight="weight")` |

Each measure was normalised to [0, 1] before averaging. Hub scores are used as
node features in the GNN.

---

### Drug–gene interactions

Drug-gene interactions were queried from up to three databases:

| Database | Protocol | Data retrieved |
|----------|----------|----------------|
| [DGIdb](https://dgidb.org) | GraphQL API | Interaction type, directionality, approval, publications |
| [ChEMBL](https://www.ebi.ac.uk/chembl) | REST API | Mechanism of action, max clinical phase |
| [OpenTargets](https://platform.opentargets.org) | GraphQL API | Approval status, phase, mechanism |

A curated fallback dataset (37 literature-based interactions for 16 hub genes,
compiled from Wang et al. 2025) supplements any genes not covered by live APIs.
All sources are selectable independently via boolean flags in the notebook.

**Composite score** (0–1) per drug-gene pair:

```
score = 0.35 × interaction_score_norm
      + 0.20 × publications_norm (capped at 30)
      + 0.20 × clinical_phase / 4
      + 0.15 × approved (binary)
      + 0.10 × hub_score_norm
      + 0.10 × survival_target (binary bonus)
```

Gene-drug pairs were deduplicated (highest interaction score kept per pair)
before scoring.

---

## Graph Neural Network — drug candidate ranking (`03_gnn_drug_ranking.ipynb`)
 
 
### 1. Objective
 
The GNN performs **link-weight regression** on a bipartite drug–gene interaction graph. Its goal is to predict a continuous interaction score for every possible drug–gene pair — including pairs with little or no direct evidence — by learning from the structural properties of the interaction network rather than from direct evidence alone.
 
This goes beyond the composite score from step 7, which ranks each drug–gene pair in isolation. The GNN propagates information through the graph so that a drug's score reflects not just its own properties but the properties of all the genes it is connected to, and transitively the other drugs those genes interact with.
 
### 2. Graph construction — `build_graph()`
 
A **bipartite graph** is constructed with two node types and no edges within a type:
 
```
Gene nodes (547)  ←──── interaction edges (8,027) ────→  Drug nodes (4,978)
                         Total nodes: 5,525
```
 
Edges are made **bidirectional** (undirected), so the edge index contains 16,054 directed entries (8,027 × 2). The composite score from step 7 is used as the **regression target label** for each edge.
 
The graph is represented as a `torch_geometric.data.Data` object with:
- `x`: node feature matrix (5,525 × 17)
- `edge_index`: adjacency in COO format (2 × 16,054)
### 3. Node features — 17 dimensions
 
Every node carries a 17-dimensional feature vector. The two node types occupy **different dimensions** of the same vector (unused dimensions are zero), which allows the model to implicitly distinguish node types:
 
| Dimensions | Node type | Features |
|---|---|---|
| 1–15 | Drug nodes | `approved`, `immunotherapy`, `anti_neoplastic`, `clinical_phase`, `interaction_score`, `n_publications`, `source_DGIdb`, `source_ChEMBL`, `source_OpenTargets`, `type_inhibitor`, `type_agonist`, `type_antagonist`, `type_antibody`, `type_binder`, `type_activator` |
| 16–17 | Gene nodes | `hub_score` (PPI network centrality), `survival_target` (binary: survival-significant gene) |
 
All 17 features are standardised to zero mean and unit variance using `sklearn.preprocessing.StandardScaler` fitted on the training nodes. The fitted scaler is saved alongside the model weights for inference.
 
### 4. Model architectures
 
Three GNN architectures are trained and compared. All share the same **two-layer encoder + prediction head** structure but aggregate neighbourhood information differently.
 
#### GCNModel — Graph Convolutional Network (Kipf & Welling, 2017)
 
```python
conv1 = GCNConv(in_dim=17, out_channels=128)   # BatchNorm → ReLU → Dropout
conv2 = GCNConv(in_channels=128, out_channels=64)  # BatchNorm → ReLU
head  = Linear(128→64) → ReLU → Dropout → Linear(64→1) → Sigmoid
```
 
Each node's representation is updated by **averaging its neighbours' features**, weighted by the inverse square root of the degree product (degree normalisation). All neighbours contribute equally — the model cannot learn that some connections are more informative than others.
 
#### GATModel — Graph Attention Network (Veličković et al., 2018)
 
```python
conv1 = GATConv(in_dim=17, out_channels=32, heads=4, concat=True)  # → 128 dim
conv2 = GATConv(in_channels=128, out_channels=64, heads=1, concat=False)
head  = Linear(128→64) → ELU → Dropout → Linear(64→1) → Sigmoid
```
 
Learns an **attention coefficient** for every neighbour relationship. The coefficient is computed as a learned function of the two connected nodes' features, allowing the model to weight more informative neighbours more heavily. Four parallel attention heads in layer 1 are concatenated (32 × 4 = 128 dimensions), then reduced to 64 in layer 2.
 
#### SAGEModel — GraphSAGE (Hamilton et al., 2017) ★ Best model
 
```python
conv1 = SAGEConv(in_dim=17, out_channels=128, aggr="mean")  # BatchNorm → ReLU → Dropout
conv2 = SAGEConv(in_channels=128, out_channels=64, aggr="mean")  # BatchNorm → ReLU
head  = Linear(128→64) → ReLU → Dropout → Linear(64→1) → Sigmoid
```
 
Instead of summing or averaging neighbours directly, SAGE **concatenates the node's own feature vector with the mean of its neighbours**, then applies a linear projection. This preserves the node's own identity during aggregation — critical here because drug nodes and gene nodes have fundamentally different feature profiles, and naively mixing them would lose that distinction.
 
SAGE is also **inductive**: it learns a function over neighbourhoods rather than memorising node-specific embeddings. This makes it more robust to high-degree hub genes (e.g. GAPDH with 135 connections) where averaging many neighbours can otherwise dilute the signal.
 
### 5. Forward pass
 
For all three architectures, inference follows the same pattern:
 
```
Input:
  X          — node feature matrix  (5,525 × 17)
  edge_index — adjacency            (2 × 16,054)
  src_idx    — gene node indices for query pairs
  dst_idx    — drug node indices for query pairs
 
Step 1 — Layer 1 convolution:
  Each node aggregates its neighbours → 128-dimensional hidden representation
  BatchNorm → ReLU (or ELU for GAT) → Dropout(p=0.3)
 
Step 2 — Layer 2 convolution:
  Aggregation in the 128-dim space → 64-dimensional embedding Z
  BatchNorm → ReLU
 
Step 3 — Prediction head:
  For each query pair (gene i, drug j):
  → Concatenate embeddings: [Z[src_i] ‖ Z[dst_j]] → 128-dim vector
  → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→1) → Sigmoid
  → Output: predicted score ∈ [0, 1]
```
 
The Sigmoid output constrains predictions to [0, 1], matching the composite score range.
 
### 6. Training
 
| Hyperparameter | Value |
|---|---|
| Hidden dimension | 128 |
| Embedding dimension | 64 |
| Dropout rate | 0.3 |
| Learning rate | 0.005 |
| Weight decay | 1 × 10⁻⁴ |
| Max epochs | 300 |
| Early stopping patience | 40 epochs |
| GAT attention heads | 4 (layer 1 only) |
| Random seed | 42 |
 
**Loss function:** MSE between predicted and composite score labels.
 
**Optimiser:** Adam with gradient clipping (max norm 1.0) to prevent exploding gradients.
 
**Learning rate scheduler:** `ReduceLROnPlateau` — halves the learning rate if validation loss does not improve for 15 consecutive epochs, down to a minimum of 1×10⁻⁵.
 
**Data split:** Edges are randomly split 70% train / 15% validation / 15% test. The model never sees test edges during training or early stopping. Best weights (lowest validation MSE) are restored before evaluation.
 
### 7. Model comparison results
 
| Model | R² | MSE | MAE |
|---|---|---|---|
| GCN | 0.970 | 0.001 | 0.020 |
| GAT | 0.976 | 0.001 | 0.015 |
| **GraphSAGE ★** | **0.993** | **0.000** | **0.007** |
 
GraphSAGE is selected as the best model. All three models achieve R² > 0.97, confirming that the graph structure carries substantial information about interaction scores. The progressive improvement from GCN → GAT → GraphSAGE reflects the increasing expressiveness of each aggregation strategy.
 
### 8. Drug ranking — `rank_drugs()`
 
After training, the best model scores **all possible drug–gene pairs** in the bipartite graph. For each pair, the GNN score is computed as described in the forward pass above. The resulting table is sorted descending by GNN score and exported as `results/tables/gnn_drug_ranking.csv`.
 
The output contains:
 
| Column | Description |
|---|---|
| `rank` | Global rank by GNN score |
| `drug` | Drug name |
| `gene` | Target gene |
| `gnn_score` | GNN-predicted interaction strength [0–1] |
| `original_score` | Composite score from step 7 |
| `score_delta` | `gnn_score − original_score` — positive means GNN promotes this pair above its rule-based rank |
| `approved` | FDA approval status |
| `clinical_phase` | Highest clinical phase |
| `interaction_type` | Mechanism (inhibitor, agonist, etc.) |
| `source` | Database of origin |
 
### 9. What the GNN detects that the composite score misses
 
The composite score ranks each drug–gene pair **in isolation**: it uses only the properties of that single interaction. It cannot know that Deferoxamine also appears adjacent to three other high-centrality hub genes, or that Minocycline connects to a cluster of genes whose neighbours are all pharmacologically rich.
 
The GNN's neighbourhood aggregation propagates signals through the graph. After two convolutional layers:
 
- A **drug node's 64-dim embedding** encodes not just its own pharmacological properties but the structural properties of all genes it is connected to (layer 1), and transitively the other drugs those genes interact with (layer 2)
- A **gene node's embedding** encodes its own hub score and survival significance plus the aggregated profile of its entire drug neighbourhood
When these embeddings are concatenated and passed through the prediction head, the score for a drug–gene pair implicitly incorporates **global network position**. A drug connected to many high-centrality hub genes will have an enriched embedding even on a low-evidence pair, because its neighbourhood is pharmacologically coherent.
 
This explains concretely why the GNN ranking differs from the composite score ranking. For example, Deferiprone had a lower composite score than Cerliponase Alfa (0.46 vs 0.70), but the GNN raised Deferiprone to rank 4 because its iron chelation mechanism connects it structurally to a dense cluster of well-connected hub genes (FTL, GAPDH, TFRC), making its embedding more informative. Cerliponase Alfa's connection to TPP1 is a more isolated edge in the network, so the GNN assigns it a lower structural relevance than the direct evidence alone suggested.
 
### 10. Node embeddings
 
The 64-dimensional node embeddings learned by the best model are exported to `results/tables/gnn_node_embeddings.csv`. These can be used for downstream analysis — for example, clustering drugs by embedding similarity to identify pharmacological families, or computing drug–drug similarity in the learned representation space independently of the original interaction data.
 
### 11. Google Colab / GPU compatibility
 
Notebook 03 auto-detects the Colab environment and installs the correct PyTorch Geometric wheels for the available CUDA version. The notebook can be run end-to-end in under 10 minutes on a T4 GPU. On CPU, training takes approximately 3–5 minutes per model depending on hardware.

---

## Software versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| scanpy | ≥1.10 |
| anndata | ≥0.10 |
| celltypist | ≥1.6 |
| rpy2 | ≥3.5 |
| anndata2ri | ≥1.3 |
| torch | ≥2.2 |
| torch-geometric | ≥2.5 |
| lifelines | ≥0.28 |
| networkx | ≥3.2 |
| scikit-learn | ≥1.4 |
| R | ≥4.3 |
| Seurat | ≥5.0 |
| SingleR | ≥2.0 |
| celldex | ≥1.0 |
| clusterProfiler | ≥4.0 |
| org.Hs.eg.db | ≥3.17 |
