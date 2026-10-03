# OmicSage versus the HCC scRNA workflow

Reviewed 3 October 2026. This comparison does not replace or rerun either analysis.

## Evidence and recommendation

Reviewed the combined OmicSage report at `\\wsl.localhost\Ubuntu\home\shoko\OmicSage\reports\HCC_scRNA-seq_Wang_et_al_2025\00_combined_report.html`, the matching saved annotated object, and relevant current OmicSage implementation files. The report is a historical run; current source inspection explains implementation choices but does not establish that every source line was identical when that run was made.

OmicSage provides better QC reporting and retains substantially more cells than the original HCC run with 5% mitochondrial filtering. It is not demonstrably better than the revised HCC run with 18,309 cells. Its automatic PC selection, cluster-resolution selection, and cell annotation require validation. Its differential-expression and enrichment outputs answer a different question from the HCC sample contrast and must not be substituted directly into the existing target-prioritization pipeline.

Recommendation: retain the revised HCC run as the working analysis for now. Use OmicSage as a sensitivity comparison and a source of reusable expression data, rather than adopting all of its outputs as a validated replacement.

## Preprocessing comparison

| Step | Original HCC workflow | Revised HCC workflow discussed in this audit | OmicSage report |
|---|---|---|---|
| Input cells | 25,189 | 25,189 | 25,189 |
| Minimum detected genes | 200 | 200 | 200 |
| Maximum detected genes | 2,500 | 7,500 | 6,000 |
| Mitochondrial cutoff | 5% | 15% | 20% |
| Retained cells | 2,795 | 18,309 | 18,568 |
| PCs used in neighbors | Historical configuration | 20, provisional | 5 of 50 computed |
| Leiden resolution | 0.5 | 0.5, provisional | 0.8; 20 clusters |

The revised HCC thresholds/count were supplied in the ongoing audit; this comparison does not independently certify the current notebook execution state. Exact boundary operators should be checked before attempting identical reproduction.

OmicSage retained 13,060 HCC1 and 5,508 HCC2 cells. The revised threshold audit retained 13,499 HCC1 and 4,810 HCC2 cells. Similar total counts therefore conceal different sample retention, and do not prove that the same cells passed both workflows.

OmicSage's QC diagnostics, including joint count/gene plots and doublet scores, improve auditability. Its individual criterion removal counts overlap and should not be summed as sequential losses. The reported single detected doublet is not evidence that the remaining dataset is doublet-free. Current code applies Scrublet to the combined input; evaluate doublets separately by capture/library, inspect score distributions and thresholds, and review mixed-lineage clusters. Do not choose 20% mitochondrial filtering simply because it retains more cells. Compare 15% and 20% using sample-specific QC distributions, marker coherence, and retained cell populations.

The annotated export has 18,568 cells by 33,694 genes, with both `counts` and `logcounts` layers and no `.raw` snapshot. The absence of `.raw` is not an error if subsequent functions explicitly use the appropriate layers. OmicSage reports normalization to 10,000 counts followed by log transformation, with 2,000 HVGs. A sum of log-transformed expression is not expected to equal 10,000.

## PCs and clustering

Five PCs explain approximately 37.1% of variance. That is a descriptive result, not validation that five PCs preserve all useful cell-type structure. Compare neighbors/clustering using 5, 15, 20, and 30 PCs on the same filtered cells and preprocessing; evaluate broad lineage markers, smaller populations, and technical associations. Retain 20 PCs provisionally in the existing HCC workflow until this comparison supplies a reason to change.

The reported automatic resolution-selection metric is based on changes in cluster count between resolutions. Current source does not establish membership stability through repeated seeds, subsampling, or related comparisons. Resolution 0.8 and 20 clusters may be useful, but the word "stability" should not be interpreted as an independent robustness test. Decide resolution from marker-supported distinctions and reproducibility, rather than the number of visually separated UMAP groups.

## Annotation review

The actual annotation methods run were CellTypist and marker scoring, followed by voting. ScType and SingleR appear as configuration options but were not run in this report. Two CellTypist models share an annotation framework; their agreement is not two independent biological confirmations. Voting is an aggregation step, not a third classifier. A median confidence of 0.5 is not a calibrated 50% probability of a correct label.

Specific disagreements require inspection:

- Clusters 8 and 9: CellTypist epithelial labels versus hepatocyte marker labels. Check hepatocyte and epithelial programs before assigning lineage; an epithelial label alone cannot identify malignant cells.
- Cluster 6: 2,485 cells called regulatory T cells, with only a generic T-cell marker assignment. Confirm a coherent regulatory T-cell program rather than accepting the subtype label automatically.
- Cluster 13: endothelial prediction versus mast-cell marker assignment. Inspect endothelial versus mast-cell marker coexpression and doublet/QC signals.
- Cluster 10: only 19 cells, with plasma-cell versus hepatocyte disagreement. Review individually before a biological interpretation.
- Cluster 3: DC2 versus monocyte disagreement. Distinguish dendritic from monocyte/macrophage programs using multiple markers.

Strong sample segregation also warrants review: regulatory T cells are 2,480 HCC1 versus 5 HCC2, while helper effector/memory T cells are 0 versus 1,332; the other cytotoxic subtypes are likewise overwhelmingly HCC2. This may contain genuine sample biology, but sample-associated technical effects and reference-label behavior remain alternatives. Broad T-cell lineage annotation should precede claims about these subtypes. Do not automatically integrate away all sample differences when batch and biological sample are confounded.

Recovered fibroblast and endothelial candidates are useful hypotheses for populations lost under stringent QC. They do not, by themselves, prove that the new labels are correct. Validate with marker plots and cluster-level QC/sample composition, then harmonize broad labels before comparing annotation methods.

## Why the downstream tables are not interchangeable

OmicSage uses `cell_type_vote` for differential expression, comparing each predicted cell type against the rest. This identifies lineage-associated markers. It does not estimate HCC2-versus-HCC1 changes or disease-associated changes within a lineage. The reported 5,498 significant records are gene/group entries, not necessarily 5,498 unique genes. The configuration retains the top 500 ranked genes per group, which also limits the exported list. For a ranked enrichment analysis, recover a full signed ranking rather than this truncated list.

The step labeled GSEA uses Enrichr over-representation analysis on selected upregulated genes. That is a legitimate enrichment method when its gene selection and background are appropriate, but it is different from ranked GSEA. It must not be described as evidence of ranked pathway activation or treated as equivalent to the HCC GSEA outputs. Likewise, 6,860 significant entries represent pathway/group records, not independent biological discoveries. Immune or extracellular-matrix enrichment supports annotation plausibility; heat-shock enrichment can also reflect dissociation stress.

Neither workflow resolves the sample-provenance discrepancy between the 2025 paper and GEO metadata. Use neutral HCC1/HCC2 names until tissue/donor mapping is verified. Even a confirmed one-sample-per-condition comparison does not provide biological replication for population disease inference.

## Safe reuse and the next fix

The matching `05_annotated.h5ad` is a reusable technical input. Preserve barcodes, sample identity, and the full `counts`/`logcounts` matrices. Its annotations and embedding should remain provisional. Starting from counts permits a clean rerun; importing existing normalized values requires checking that the HCC notebook will not normalize/log them again.

Next, validate broad lineage labels with cluster marker plots, sample-by-cluster counts, and QC/doublet overlays. Complete the ongoing ScType/SingleR work and compare harmonized labels to OmicSage. Then assess whether the additional cells retained at 20% mitochondrial content contribute coherent populations. Only after selecting a supported cell set and annotation should differential expression be rebuilt for the intended contrast. Regenerate dependent enrichment, PPI, target scores, and GNN inputs together; these outputs must correspond to the same upstream data and definitions. Previously identified downstream statistical and implementation problems still need correction independently of which preprocessing workflow is used.

Interview defense: "I compared two preprocessing workflows. OmicSage improved QC transparency, but automated parameter choices and classifier agreement were not sufficient validation. I checked marker coherence and sample effects, and distinguished lineage-marker enrichment from disease contrasts before reusing downstream inputs."
