# OmicSage spatial case study: verified work and interview boundaries

Audited the run configuration, implementations, HTML reports and HDF5 checkpoint metadata. This audit does not rerun or modify OmicSage. Updated interview slide 9 describes the executed workflow and its current validation limits.

## Dataset and scope

Kuppe et al., **Spatial multi-omic map of human myocardial infarction**, Nature 608, 766–777 (2022), [paper](https://doi.org/10.1038/s41586-022-05060-x). OmicSage used the **control subset**, not the full infarction atlas: [Visium control dataset](https://figshare.com/articles/dataset/Kuppe_Visium_Human_Heart_2022_Control/22132958), file 39347357, with the corresponding snRNA reference file 39347573.

The ingested Visium checkpoint contains four patient-region IDs: control_P1 (4,279 spots), control_P17 (2,049), control_P7 (2,936), control_P8 (2,461). Total: 11,725 spots and 36,588 features. Visium spots contain mixed expression from multiple cells; they are not single cells. There is no disease-versus-control contrast in this run, and this is not spatial HCC analysis.

## What actually ran

| Step | Executed method and saved output | Interpretation and changes needed |
|---|---|---|
| Ingest | Prebuilt h5ad; preserved counts, tissue coordinates/images and patient_region_id | Existing processed public data, not fresh Space Ranger processing. |
| QC | Counts 500–100,000; genes 200–10,000; MT cutoff 20%. 19 spots removed, 11,706 retained. | No MT- symbols exist in the ingested feature names. QC explicitly records zero matching MT genes: zero MT% means the screen was unavailable, not exceptionally healthy tissue. Low-count failures=19, low-gene failures=6, overlapping. |
| Reduction | Normalize to 10,000, log1p, 3,000 Seurat-flavour HVGs, 50 PCs; top 10 explain 18.36% variance. | Saved counts support normalization. HVGs/PCA summarize expression variation; sensitivity and patient structure remain to assess. |
| Expression clustering | 15-neighbour expression KNN, 30 PCs, Leiden resolution 0.5. Five unannotated clusters: 4,199/2,866/2,383/1,971/287 spots. | These are expression clusters, not five established cell types or histological niches. Expression KNN is separate from the physical-neighbour graph. |
| Spatial graph / Moran I | Six-neighbour spatial graph; report counts 30,510 edges and 102 of 3,000 tested genes at FDR<0.05. | Critical graph error below makes the current spatial significance claims unreliable. |
| Deconvolution | **NNLS**, 11 reference cell types, 26,698 shared genes, cell_type_original labels and reference counts. Coefficients normalized to sum one per spot. | NNLS proportions are relative fitted weights, not calibrated cell counts. cell2location was available in code/config but was not the completed method. Despite per_sample=true metadata, NNLS code pools reference signatures and solves spots individually; the per-library model loop belongs to cell2location. |
| Composition regions | Leiden on deconvolved mixtures, resolution 0.5; 49 reported regions. | Depends on questionable deconvolution. These are algorithmic composition clusters, not 49 validated anatomical niches. |
| Marker correlation / type-enriched SVG | Spearman correlations across spots; Moran I in subsets above median fitted abundance. Eleven marker lists and five type-enriched SVG lists reported. | Correlated mixed-spot expression is not cell-specific transcription. Code ranks absolute correlations, allowing negative associations to appear as markers; constant abundance yields arbitrary tied rankings. Retain signed positive associations, skip constant types and validate against reference markers. |
| Neighbourhoods / co-occurrence | Squidpy neighbourhood permutation test (1,000 permutations) and distance co-occurrence completed. | Rebuild physical graphs and compute coordinate-based analyses separately by tissue section. A plot-rendering failure is also reported for neighbourhood enrichment. |
| Ligand–receptor | sq.gr.ligrec, dominant_cell_type groups, OmniPath-compatible symbols, 1,000 permutations. Report: 8,356 tests at raw p<0.001; 21,452 at raw p<0.05. | This call does **not** use physical adjacency or explicitly restrict to colocalised pairs. These are expression-based interaction hypotheses, not demonstrated spatial communication. Apply multiple-testing control and spatial restrictions; dominant mixed-spot labels are uncertain. |
| Pathways | GO_Biological_Process_2023 prerank using Moran I; 1,004 pathways, one reported q<0.05. | Ranking describes relative spatial autocorrelation, not up/down regulation or pathway activation. Current graph invalidates spatial interpretation. Negative NES does not mean suppressed differentiation. |
| Mapping / projection | **Tangram clusters mode**, 2,000 genes projected. Reported Spearman rho=0.292 for means across 50 overlapping genes. | gimVI did not run. This is neither held-out gene validation nor a test of spatial pattern accuracy. Per-spot scores are unavailable; “0 poor spots” is not evidence of success. Projected genes are not necessarily absent from the original assay. |

## Critical findings from the actual checkpoints

1. **Physical graph crosses tissue sections.** In both 03_reduced and 06_downstream, spatial_connectivities has 61,021 stored adjacency entries; 49,893 (81.76%) link different patient_region_id values. Separate tissue sections cannot be physical neighbours. spatial_reduce.py calls spatial_neighbors without library_key. Rebuild a block-diagonal graph per patient_region_id; check explicitly that cross-section adjacency count is zero. Rerun global and type-enriched Moran tests, neighbourhood analyses and associated pathway rankings. Co-occurrence must also respect separate coordinate systems.
2. **Deconvolution has implausible/degenerate fits.** Adipocyte, Cycling cells and Fibroblast weights are zero for every spot. Mean Endothelial weight=0.5695, Cardiomyocyte=0.3371, Lymphoid=0.0867; dominant types are Endothelial 8,000 spots and Cardiomyocyte 3,533. These outputs require validation of signature informativeness, assay scaling, reference labels and fitting residuals. Do not announce absence of fibroblasts or endothelial predominance as a biological discovery.
3. **Checkpoints are from mixed runs.** Latest ingest/reduce/cluster provenance is June 16; deconvolve/downstream/impute embed June 11–12 upstream provenance. Same cell count and matching summarized PCA results do not establish identical inputs. The runner caches by file existence. After corrections, invalidate descendants and use input/config fingerprints or force a consistent full run.
4. **MT QC unavailable and mapping validation weak.** Explain these limitations directly rather than turning the report's zeros or correlation into successful validation.

## What to say in the interview

“I implemented a checkpointed spatial workflow in OmicSage using four control-heart Visium sections and a single-nucleus reference. It runs QC, expression clustering, NNLS deconvolution, spatial analyses and Tangram projection. Auditing the results exposed cross-section neighbour connections and weak reference-based fits, so I treat the current outputs as a workflow benchmark and defer biological claims until those checks are corrected.”

Prepare to explain: why spots require deconvolution; expression KNN versus physical adjacency; why tissue sections need separate graphs; NNLS versus probabilistic abundance models; why zero MT% can be uninformative; why Moran I is not differential expression; why ligand–receptor coexpression is a hypothesis; and how held-out gene/pattern validation differs from comparing expression means.

## Evidence locations

OmicSage: `config/runs/kuppe_heart.yaml`, `run_spatial_pipeline.py`, `pipeline/modules/scripts/spatial/`, `reports/kuppe_heart/`, `data/processed/kuppe_heart/01_ingested.h5ad` through `07_imputed.h5ad`.

Local audit extracts: `tmp/spatial_audit/checkpoint_audit.txt` and report text in `tmp/spatial_audit/kuppe_heart/`. These are audit support, not replacement analysis outputs.
