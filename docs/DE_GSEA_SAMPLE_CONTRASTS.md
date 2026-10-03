# Pooled and within-cell-type DE/GSEA

Notebook 01 now runs an explicit HCC2 (tumor) versus HCC1 (adjacent tissue)
contrast across all retained cells, then repeats that contrast within each
reviewed broad cell type. The organization follows OmicSage's pairwise tests,
result dictionaries, group summaries and provenance; it does not copy its
cell-type-versus-rest comparison or its significant-only enrichment input.

## Running the updated section

Keep the live reviewed `adata`, or load its saved annotated checkpoint. The
object must have `sample`, `manual_celltype`, and log-normalized expression in
`logcounts` or `X`; counts are preserved separately. Do not normalize again.
Run the configuration cell, then start at section 5. That section reloads the
edited utilities, so annotation need not be repeated. R setup must already be
available for section 6. The R package cell installs missing packages only
when the user runs it, following the notebook's existing behavior.

Default configuration:

- Explicit HCC2 group and HCC1 reference, independent of category order.
- Minimum 20 cells on each side. This is a practical exploratory screen, not
  a guarantee of statistical reliability or biological replication.
- Genes detected in at least 1% of either side enter DE; all-zero genes are
  omitted. No selection by DE significance is used for GSEA.
- Wilcoxon with tie correction, BH adjustment per contrast, full gene output,
  expression-detection fractions, and approximate Scanpy log fold changes.
- Significant shortlist: adjusted cell-level p-value < 0.05 and |log2FC| > 1.
- GSEA: full signed Wilcoxon ranking, GO-BP/MF/CC and KEGG, set sizes 15–500,
  seed 42, with adjusted pathway p-value <= 0.05 used for reporting.

Given the current counts, pooled, B cells, fibroblasts, T cells, monocytes,
endothelial cells and macrophages are eligible. Small/imbalanced groups are
flagged. Hepatocytes, cholangiocytes, mast cells and DCs lack sufficient cells
on one side; unresolved myeloid annotations are skipped. Skipped groups are
listed in the manifest rather than treated as failed biological comparisons.

## Outputs and historical compatibility

`data/processed/sample_contrasts/<contrast>/` contains `de_all_genes.csv`,
`de_significant.csv` and `ranked_genes.tsv`. The root contains a contrast
summary and JSON provenance. Each table retains contrast, cell type, group
and reference. `results/figures/sample_contrasts/<contrast>/` contains plots;
`results/tables/sample_contrasts/<contrast>/` contains GSEA tables, mapping
diagnostics and a run-status JSON. A GSEA failure is recorded explicitly.

Gene mapping omits ambiguous symbols and selects the entry with the greatest
absolute ranking statistic when multiple symbols share an Entrez identifier.
The mapped vector has unique Entrez IDs. Mapping tables make this choice
reviewable. `core_enrichment` retains Entrez IDs and
`core_enrichment_symbols` contains symbols. The separate `leading_edge`
statistics column must never be parsed as genes. Full pathway tables retain
both significant and nonsignificant results; summaries filter by adjusted
p-value and retain both NES directions.

The historical `data/processed/dea_results.csv` is NOT overwritten. Notebook
02 still reads that older file until a contrast and candidate-selection rule
are deliberately selected. Do not mix newly generated pathway evidence with
old target-ranking inputs. The scRNA HTML report includes pooled results and
the per-contrast DE/GSEA overview when regenerated.

For a sensitivity run excluding clusters 9 and 11, set
`DE_EXCLUDE_CLUSTERS = ("9", "11")` and choose a different `DE_RUN_NAME`, such
as `sample_contrasts_without_9_11`. The master AnnData is not modified.
Substantial cell removal followed by biological interpretation may also
warrant rebuilding the graph/clustering; this optional comparison simply
assesses the existing annotation's sensitivity to these cells.

## Interpretation boundaries

There is one sample per condition. Cell-level significance and preranked
pathway enrichment do not estimate between-donor variation or establish
population-level HCC changes. Aggregating counts or downsampling cells cannot
create missing biological replication. Pooled DE is composition-sensitive;
broad-cell-type DE may still reflect subtype/state composition, particularly
for T cells and stromal cells. Positive scores/NES point toward HCC2 and
negative values toward HCC1; enrichment does not by itself prove pathway
activation, causal cancer drivers, or therapeutic effectiveness. Annotation
and doublet uncertainty remain separate from this computational correction.
