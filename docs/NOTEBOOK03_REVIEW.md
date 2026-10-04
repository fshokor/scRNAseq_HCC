# Notebook 03: known-association evidence-score regression

The project retains evidence-based target prioritisation. The model approximates
notebook 02's rule-derived composite score. It does not estimate sensitivity,
affinity, clinical efficacy, or independently established therapeutic relevance.
Because inputs overlap with score ingredients, even excellent held-out metrics
are score-distillation performance rather than biological validation.

## Changes

- Read the selected contrast's current notebook 02 export and require its manifest
  to be completed. Record input SHA256, upstream provenance, software versions,
  configuration and label split. Historical top-level inputs are not used.
- Give genes and drugs separate identity namespaces. Prefer stable drug IDs;
  record name-based fallbacks. Reject duplicated pair identities and mixed contrasts.
  Group same-gene candidate aliases before splitting: identical normalized names
  and shared database IDs link suspected aliases conservatively. Keep their labels
  in the same split, without asserting chemical equivalence or silently merging
  distinct database IDs. Confirmed upstream mappings require an evidence reference.
- Keep interaction scores, publications, sources and interaction types on pairs,
  rather than copying the first interaction onto every edge of a drug. Use drug
  properties and gene features on their respective nodes. Restore missingness
  flags; audit conflicting drug properties as unknown instead of taking the first.
- Fit feature scalers on training pairs and their incident nodes only. Neither
  targets nor upstream `score_*` components are feature inputs. Raw ingredients
  overlap with the constructed target; that limitation remains explicit.
- Compare GCN, GAT and GraphSAGE with a fixed label split and three initialization
  seeds. Choose the architecture with minimum mean validation MSE, never test R².
  Use validation early stopping for each initialization. The approximate 70/15/15
  proportions refer to gene/candidate-alias groups; association row counts can differ.
- After freezing the selection, compare its ensemble against an MLP using the
  same feature information, Ridge with fixed alpha=1, the training-label mean,
  and the selected architecture trained without neighbor connections. The latter
  retains self transforms. Test results are assessed after model selection.
- Preserve all original evidence columns in the pair ranking. Export unrounded
  scores, initialization variability, stable pair IDs, split ledger, per-seed
  predictions, fitted preprocessing, node identities, graph tensors and architecture-
  named checkpoints. Each ranked row is a known association, not a unique drug.
- Compare all learned approximators, including Ridge, on validation data before
  examining test metrics. Export the transparent composite ranking as the primary
  prioritisation output, the GNN ranking as an experimental approximation, and a
  biological review queue. Examine errors by score band and largest disagreements.
- Generate a separate contrast-specific report with correct interpretation,
  baselines, initialization stability and figures. Clear old notebook outputs.

## Scope and interpretation

All known, unweighted graph connections remain available, including pairs whose
labels are held out. This is an explicitly transductive known-association task,
not a novel-link or unseen-drug evaluation. No score-derived edge weights are used.
Full-data graph visibility is therefore part of the declared input, not evidence
of inductive generalization. Initialization standard deviations and top-20 rank
overlap measure computational stability, not biological confidence intervals.

Keep the original composite ranking as the transparent reference. A learned
ranking differing from it is not automatically an improvement. If graph models
do not add defensible value over simpler models, report that finding and use the
composite ranking for prioritisation. Independent experimentally supported labels
would be needed to evaluate therapeutic prediction. The single-cell evidence has
one sample per tissue group and remains exploratory.

## Running

Finish notebook 02 first, then restart and run notebook 03. `CONTRAST` must match
the upstream run. Results go under `evidence_score_regression/<contrast>/` in
tables, figures, reports and models. Colab requires the helper and the completed
input CSV/manifest in the same project structure, plus its Python dependencies.
Notebook 03 requires the updated notebook 02 manifest and verifies its exported
input hash. Its helper also requires `drug_identity_functions.py` beside it.

Validation uses offline synthetic associations to check missingness, typed IDs,
pair-feature placement, scaler boundaries, rejection of invalid inputs, invariance
of model selection to altered test labels, real training of each comparator,
checkpoint prediction equivalence, and the complete notebook/export/report path.
These checks do not substitute for running the full experiment on the new export.
