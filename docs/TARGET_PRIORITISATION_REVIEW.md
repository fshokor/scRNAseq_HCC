# Revised target prioritisation workflow

Notebook 02 processes one explicit tumour-versus-adjacent contrast from notebook 01 at a time. It does not combine genes from different cell types into an unexplained union. All candidates are exploratory: there is one sample per condition, and cell-level DE does not provide population-level replication.

## Running the notebook

1. Restart the kernel and run from the beginning with `CONTRAST = "pooled"`. The notebook displays the available completed contrasts and their sample cell counts. It reads `data/processed/sample_contrasts/<contrast>/de_all_genes.csv` and applies the configured FDR and absolute log2FC thresholds.
2. For another comparison, change `CONTRAST` to a completed cell-type identifier, such as `celltype_T_cell`, and rerun from the configuration cell onward. Notebook 01 currently exports eligible B-cell, endothelial, fibroblast, macrophage, monocyte and T-cell contrasts. Endothelial/macrophage/monocyte contrasts have limited or imbalanced cells; their eligibility does not establish robustness.
3. Results are separated under `results/tables/target_prioritisation/<contrast>/`, with corresponding figure/report directories. Historical top-level tables and reports remain unchanged. A rerun replaces the selected contrast's generated output; use a different `TARGET_RUN_NAME` to preserve sensitivity runs.
4. Notebook 03 must subsequently read the selected new `dgi_edges_gnn.csv` path printed by notebook 02. Its existing top-level input remains historical until that notebook is updated. Do not run a GNN when the new edge file is empty.

The notebook validates the selected contrast against the DE manifest and records its input hash, sample counts, parameters, package versions, UTC timestamps and source statuses in `provenance.json`. `status=running` means the run is incomplete. Review that manifest before using files from an interrupted run; a failed PPI query is recorded explicitly.

## STRING associations

The STRING API is pinned to version 12.5. Symbols are mapped to protein IDs first; unresolved or ambiguous/many-to-one mappings are exported to `string_mapping.csv`. These genes remain in the candidate ledger, but ambiguous mappings are excluded from edges.

Every pair of protein blocks is queried together. This preserves between-block associations omitted by the old disjoint-batch implementation. API/schema failures stop PPI analysis instead of ranking an incomplete graph. The default uses medium confidence (400), `network_type=functional`, and no added neighbours. Functional associations are not necessarily physical binding. A 700-confidence sensitivity run can use another output run name.

Unweighted degree, betweenness and closeness avoid treating confidence as a distance. Confidence-weighted eigenvector centrality is calculated per connected component and scaled by component size; this is an explicit heuristic. Four normalized measures receive equal weight in the hub score. Isolates have zero hub score and remain available to drug queries. Centrality depends on the submitted DEG set, knowledge-base coverage and topology; it does not establish essentiality or therapeutic benefit. Querying all block pairs and exact betweenness can take substantial time for the pooled shortlist.

## Optional survival evidence

TCGA download failure never triggers simulation. The previous hardcoded gene-specific effects in simulated survival data were removed. Explicit demonstration data are marked simulated and rejected by scientific survival analysis.

Real Xena data undergo primary-tumour sample selection, endpoint validation and patient-level deduplication. Multiple primary expression aliquots are averaged per patient; conflicting survival endpoints cause validation failure. A validated local CSV can be supplied through `SURVIVAL_CSV`: unique `patient_id`, positive `OS_time` in days, binary `OS_event` and gene expression columns. Its source and endpoint definition must be independently checked.

Continuous-expression Cox models report HR per expression standard deviation, confidence intervals and BH-adjusted p-values across successfully tested genes within this contrast. Missing/untestable/failed genes remain in the results ledger. Minimum patient/event counts and events-per-parameter rules are screening choices, not guarantees of reliable inference. Convergence warnings prevent a supported call. Rank-transformed Schoenfeld residual tests flag possible PH problems; passing is not proof of proportional hazards, and flagged models need graphical/clinical follow-up. KM plots use median splits descriptively and do not independently validate Cox results.

`SURVIVAL_COVARIATES=[]` explicitly means unadjusted exploratory models. Validated age/stage or other clinical columns can be selected; categorical columns are dummy encoded. Covariates are not guessed from ambiguous metadata. Survival provides an annotation, not a gate for drug queries. It cannot establish cell-intrinsic mechanism or therapeutic direction from bulk-tissue correlations. FDR is not corrected across all contrasts/project hypotheses, and there is no independent survival validation cohort.

## Drug evidence and ranking

DGIdb is the enabled source. Failed batches discard partial results and are recorded as failures. Successful queries with zero records are distinguished from failures. ChEMBL/OpenTargets legacy clients remain disabled and guarded pending identifier/pagination validation; the unverified hardcoded fallback is retired.

Raw records are saved in `dgi_raw_records.json`. The evidence table retains identifiers, all reported interaction types/directions, PMID unions and originating evidence sources. Deduplication uses drug IDs when available. Name-only records are flagged for identity review; different identifier namespaces/synonyms are not assumed to identify the same compound. Conflicting approval/phase metadata is flagged and left unknown. A drug–gene association may be indirect: direct target engagement and mechanism require the original reference, not just a database label.

DGIdb does not supply clinical phase, so that field remains unknown rather than being inferred from approval. Approval is not an HCC indication. The evidence dashboard displays unknown phase and approval separately.

Default score weights are interaction 0.65, approval 0.15 and hub 0.20, with publications and phase zero. These are heuristic choices, not optimized or biologically validated weights. DGIdb scores already incorporate publication/source evidence, so publications are not rewarded twice by default. Interaction scores are scaled against the maximum DGIdb score within the current contrast; scores from other databases are not treated as equivalent. Hub scores contribute their stated weight without the old extra multiplication by 0.10. `score_weight_sensitivity.csv` compares baseline, lower-hub and higher-hub rankings. Rankings/scores across contrasts and database snapshots are not directly calibrated to one another.

`dgi_evidence.csv` preserves missing metadata and DE direction/context. `dgi_edges_gnn.csv` adds multi-valued source/type encodings and numeric missingness flags, preserving the evidence fields. Zero imputation applies only to model input; clinical phase zero with `clinical_phase_missing=1` means unknown. Notebook 03 still needs a separate audit of how it consumes those features and interprets the score.

## Validation and remaining limits

Focused regression checks mock HTTP and exercise the actual notebook code through export/report generation, including unavailable evidence, mapping, cross-block edges, retained isolates, deduplicated citations, scoring weights, continuous Cox models, FDR correction, patient conflicts and simulated-data rejection. They do not validate live API availability, the real TCGA endpoint distribution or biological candidate rankings. External analyses must be rerun in notebook 02 before interpreting the revised shortlist.

Method references: [STRING API](https://string-db.org/help/api/), [NetworkX weighted betweenness](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html), [DGIdb interaction scores](https://dgidb.org/about/overview/interaction-score), [lifelines PH diagnostics](https://lifelines.readthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html).
