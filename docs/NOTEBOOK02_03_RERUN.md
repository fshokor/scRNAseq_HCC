# Rerunning target prioritisation and evidence-score regression

Restart the kernel and run notebook 02, then restart the kernel and run notebook 03.
Keep CONTRAST="pooled" for the first comparison. Outputs remain contrast-specific.
Saved notebook outputs were cleared; existing result files are historical until rerun.

## Notebook 02

The primary score preserves the original weights and max normalization. New
score_scaling_sensitivity.csv compares max versus log1p/max normalization with the
same weights. score_component_contributions.csv shows median and 90th-percentile
actual contributions; nominal weights alone do not describe ranking influence.
No rule is automatically chosen to favor a desired drug or an HCC-looking result.

drug_identity_audit.csv flags suspected aliases. Database IDs remain distinct unless
DRUG_IDENTITY_MAP_CSV points to a confirmed mapping with drug_id,
canonical_drug_id, confirmed=True and evidence_reference. An empty template is
exported; it need not be populated to run. Do not infer identity from names alone.
Original IDs and aliases are preserved when confirmed mappings consolidate records.

The dashboard now distinguishes association rows, database IDs and drug names.
Approval counts describe rows marked approved, not distinct drugs or HCC indications.
Heatmap selection prefers drugs with at least two associated genes, then fills with
other drugs if needed. Grey means absent database records, rather than score zero.
Columns are limited for display; missing associations do not imply inactivity.

REUSE_SAVED_DGIDB=True reuses a completed saved DGIdb snapshot for the same DE
input when available, preserving the database snapshot for this sensitivity comparison.
Set it False for a fresh query. PPI and survival sections still run normally.
Survival failure remains unavailable evidence; no simulated data are substituted.

## Notebook 03

Candidate aliases for the same gene stay together in a label split, removing the
same-name/different-ID overlap observed in the previous run. This grouping is
conservative: it does not prove chemical equivalence. The known graph remains
visible, and the task remains constructed-score approximation rather than novel
drug–gene link prediction, drug response or efficacy prediction.

GNN architecture selection and learned comparator selection use validation only.
The report includes Ridge, mean, MLP and no-neighbor baselines, test score bands,
ranking stability and disagreement diagnostics. composite_drug_ranking.csv is the
primary evidence ranking. gnn_drug_ranking.csv is an experimental approximation.

## Biological review after rerunning

Use biological_review_queue.csv to review direct target evidence, mechanism,
desired action, HCC-specific evidence, cell-type support and the final review decision.
These fields are intentionally unfilled: a database association is not an HCC
therapeutic conclusion. Review the cited primary evidence, not only the ranking.

Keep pooled results composition-sensitive. A downregulated pooled gene is not
automatically a gene to inhibit, and network centrality is not a causal target claim.
Repeat notebook 02 for eligible cell-type contrasts before assigning cell-type
support; T-cell, B-cell and fibroblast contrasts have less severe cell-count imbalance
here. They still have one sample per group and can reflect subtype composition.
Unsupported malignant-hepatocyte comparisons cannot be rescued by a pooled score.
