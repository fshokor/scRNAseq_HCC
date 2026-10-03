# QC sensitivity check for GSE166635

Date: 2 October 2026. This diagnostic recomputes QC from the two deposited matrix triplets. It does not change the main pipeline, filtered objects, or original reports. More retained cells are not automatically higher-quality cells.

## Paper comparison

Wang et al. 2025, Results pp. 2–3 and Methods p. 13, specifies removal of cells with fewer than 200 or more than 2,500 genes and more than 5% mitochondrial content. It reports approximately 2,794 retained cells. The reviewed implementation uses inclusive gene limits 200–2,500 and mitochondrial percentage strictly below 5%, retaining 2,795. It closely reproduces the paper's filtering. The strict versus inclusive mitochondrial boundary is a minor implementation difference; its effect has not been established here.

The paper describes a retained mitochondrial mean around 3.2% as evidence for absence of excessive stressed/apoptotic cells. Low mitochondrial content after explicitly filtering on mitochondrial content is expected and does not independently establish viability, absence of apoptosis, or unbiased lineage retention. Figure 2B shows the post-filter restricted ranges, whereas Figure 2A shows the broad input ranges.

## Calculation

Each raw matrix was read as genes × cells, explicit zeros removed, and counts summed per cell. Detected genes were the number of nonzero features per cell. Mitochondrial percentage was 100 × mitochondrial UMI counts / total UMI counts, with gene symbols beginning MT- (13 features in each sample), matching the pipeline's definition. These are count fractions, not separately measured fractions of aligned sequencing reads. Matrix column order was aligned with deposited barcode order. The resulting baseline counts exactly reproduce the reported 2,118 and 677 retained cells.

`per_cell_qc.csv` contains metrics for all 25,189 deposited cells. `threshold_sensitivity.csv` evaluates mitochondrial cutoffs 5/10/15/20% and upper detected-gene limits 2,500/5,000/7,500/none, always with minimum 200 genes. All mitochondrial comparisons use a strict less-than cutoff, matching the code. `qc_diagnostics.png` shows distributions, joint metrics, and retention sensitivity. The histogram and joint plots restrict their displayed axes; the CSV metrics and sensitivity counts include the full distributions.

| Mitochondrial cutoff, with 200–2,500 genes | HCC1 | HCC2 | Total |
|---|---:|---:|---:|
| <5% | 2,118 | 677 | 2,795 |
| <10% | 7,229 | 3,372 | 10,601 |
| <15% | 8,230 | 4,602 | 12,832 |
| <20% | 8,513 | 5,092 | 13,605 |

After the gene limits and before mitochondrial filtering, HCC1 has 9,811 cells and median mitochondrial percentage 7.16%; HCC2 has 7,170 cells and median 10.50%. The 5% filter removes 78.4% and 90.6%, respectively, from those gene-filtered sets.

The cells in the 5–15% band within the original gene limits have median detected genes / total counts of 1,618 / 4,474.5 in HCC1 and 1,010 / 2,670 in HCC2. This supports examining their expression quality; it does not certify those cells as healthy.

With mitochondrial percentage <15%, 5,399 HCC1 cells and 216 HCC2 cells have more than 2,500 detected genes. Removing the upper gene limit retains 13,629 and 4,818 cells (18,447 total) before any additional complexity filters or doublet detection. This reveals sensitivity to the gene cap, particularly in HCC1; these are not final singlet counts.

## Recommended revision

Preserve the paper-matched analysis as a reproduction baseline. For a revised exploratory analysis, use <15% mitochondrial content as a provisional candidate, <10% as a stricter comparison, and <20% as a more permissive comparison. The choice of 15% is an analyst judgment based on these distributions, not a validated optimum or a universal liver threshold.

Keep minimum 200 genes as an initial screen, then jointly inspect low counts, low gene complexity, and high mitochondrial fraction within each sample. Do not automatically discard every cell above 2,500 genes: retain high-complexity candidates for library-specific doublet detection and marker review, with high-count/gene outlier flags. Avoid a replacement arbitrary gene cap solely to recover more cells.

Assess whether recovered 5–15% cells show coherent lineage markers, normal complexity, and robust annotations or instead form poor-quality clusters characterized by high mitochondrial fraction and weak markers. Compare broad lineage recovery and within-lineage expression programs across QC variants, including the original 5% baseline. Do not select thresholds to force matching sample proportions or a desired cancer conclusion. High mitochondrial fraction has both technical and biological causes; adaptive thresholds require that the reference distribution is not predominantly damaged cells.

Only after that assessment should a final QC rule be chosen and downstream normalization, HVGs, embedding, clustering, annotation, DE, and enrichment rerun. Altered DE inputs also require regenerating downstream network and drug/GNN outputs before presenting them as one consistent revised run. QC changes do not resolve the separate donor/tissue provenance and biological-replication limitations.

Method references: [Single-cell best practices](https://www.sc-best-practices.org/preprocessing-visualization/quality-control/), [Bioconductor OSCA QC](https://www.bioconductor.org/books/3.19/OSCA.basic/quality-control.html), [OSCA doublet detection](https://www.bioconductor.org/books/3.19/OSCA.advanced/doublet-detection.html).
