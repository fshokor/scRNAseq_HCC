# HCC single-cell project: methods audit and interview study guide

Audit date: 2 October 2026. This is a first audit of the materials currently in the repository, not a presentation or a corrected analysis.

## Evidence and overall verdict

Reviewed: all three HTML reports; code and stored outputs in the three notebooks; the scRNA, DEA, GSEA, PPI, drug-database, survival, and GNN functions; METHODS.md and data_sources.md; saved annotated AnnData metadata and expression matrix, DEG and enrichment tables, PPI tables, and GNN input edge table. External checks covered the GEO series and sample records, the dataset's original publication record, the 2025 reference paper, and method documentation.

I did not rerun the complete pipeline, verify the raw sequencing alignment, experimentally validate drugs, or reproduce GNN training. Current code, stored notebook outputs, and historical reports are different evidence sources and do not always agree. Statements below identify those differences rather than treating a report as proof of the current implementation.

The project demonstrates a substantial computational workflow, but its present outputs support exploratory sample comparisons and evidence-based candidate listing. They do not establish general HCC-versus-normal expression changes, causal cancer drivers, drug efficacy, or new drug-target interactions.

The most consequential findings are:

1. The HCC1 = adjacent-normal assignment is unsupported by the checked source metadata and conflicts with the GEO summary describing tumors from two patients. This compromises every downstream disease interpretation.
2. QC retains 2,795 of 25,189 input cells (11.1%), with different retention between samples.
3. The two retained samples have markedly different annotated cell compositions and sex-chromosome expression patterns.
4. DE tests cells as independent observations without biological replication and pools cell types.
5. GSEA uses the significant DEG list, despite a function comment claiming all genes are used.
6. Independent STRING requests on nonoverlapping gene batches omit interactions between batches.
7. Curated drug records obscure their manual origin, and the composite-score implementation differs from the methods description.
8. The GNN learns a constructed evidence score from its ingredients; held-out connections remain in the training graph, and model selection uses test R².
9. Ranking scores only existing edges; it does not enumerate novel pairs. The historical GNN report used different composite labels from the current input table.

## 1. Dataset provenance and study design

**What and why.** The downloader obtains the two 10x matrix triplets from GSE166635. The loading function labels HCC1 as normal and HCC2 as tumor to define a disease comparison.

**Verdict.** Reading deposited count matrices is appropriate; assigning biological conditions requires a verified sample map. GEO identifies GSM5076749/HCC1 and GSM5076750/HCC2 as liver-cancer samples. The series summary says the 25,189 transcriptomes come from tumors of two patients. Its overall-design field mentions tumor and adjacent tissue, so there is an ambiguity at series level; this does not establish HCC1 as adjacent normal. Treat the current assignment as unsupported and likely erroneous until resolved. [GEO series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635), [HCC1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5076749), [HCC2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5076750).

The dataset-generating study is distinct from the 2025 paper that inspired this pipeline: [original publication record](https://pubmed.ncbi.nlm.nih.gov/33619115/).

**Improve.** Build a manifest with accession, donor, tissue, disease, sex, processing batch, and supporting source. Resolve donor/tissue mapping using original supplementary information or submitter clarification. Use neutral HCC1/HCC2 labels meanwhile. If these are two tumor donors, reframe this dataset as a small heterogeneity case study. A population tumor-versus-adjacent comparison requires a dataset with verified controls and biological replication.

**Biology allowed / excluded.** Differences between these deposited samples can be described. Normal-to-tumor changes cannot currently be assigned. Even a verified one-sample-per-condition comparison would confound condition with donor and batch.

**Interview questions.** What is your experimental unit? Are these paired tissues or different donors? Which primary metadata support the condition labels? How would you handle a published paper that conflicts with its source dataset?

**Defensible answer.** “The audit exposed an unsupported sample-label assumption inherited from the reference workflow. I would resolve provenance first and treat the existing results as sample contrasts until that is done.”

## 2. Loading, merging, and preserving measurements

**What and why.** Scanpy loads gene-symbol matrices, makes variable names unique, concatenates samples, and preserves counts in a layer before normalization. This supplies a shared expression object and sample identity.

**Verdict.** These are reasonable operations. The saved annotated matrix has 2,795 cells × 33,694 genes and a counts layer. It has no `.raw` snapshot. HVGs are flagged, rather than the object being reduced to 2,000 genes, so the retained expression matrix still contains the full gene set. Counts are preserved after cell filtering; “raw counts” here means unnormalized counts for retained cells, not an unfiltered experiment.

**Improve.** Preserve original Ensembl IDs alongside symbols; inspect duplicate mappings and feature types; record genes lost through the merge's join behavior; align R-returned annotations by cell barcode instead of only row position. Save input checksums and a sample manifest.

**Biology allowed / excluded.** The matrix supports expression inspection of retained cells. Successful concatenation does not establish comparable tissues or eliminate technical effects.

**Interview questions.** Why retain counts? Why use gene IDs as well as symbols? What is the difference between a counts layer and AnnData `.raw`? How do you verify Python–R cell ordering?

## 3. Quality control and filtering

**What and why.** The workflow computes detected genes, counts, mitochondrial percentage, and ribosomal/hemoglobin metrics. It removes cells with fewer than 200 or more than 2,500 genes and mitochondrial percentage at least 5%. The purpose is to reduce empty droplets, damaged cells, and possible multiplets.

**Verified result.** Stored notebook output shows 25,189 → 24,832 after the minimum-gene filter → 16,981 after the maximum-gene filter → 2,795 after the mitochondrial filter. The last filter removes 14,186 of the 16,981 remaining cells. HCC1 retains 2,118/16,077 (13.2%); HCC2 retains 677/9,112 (7.4%). The input barcodes should be called deposited input cells, since GEO describes an initial quality-control stage before deposition.

**Verdict.** The metrics are appropriate; the thresholds are insufficiently justified for this dataset. A 5% mitochondrial threshold is a choice requiring data-specific evidence, not a universal rule. Removing almost 89% of input cells demands scrutiny. High gene counts can reflect valid cell biology as well as doublets; the upper limit does not replace doublet detection. No dedicated doublet or ambient-RNA correction appears in the reviewed main workflow.

**Improve.** Examine QC jointly by sample and preliminary lineage; show retention after each filter per sample. Compare reasonable mitochondrial and gene-count thresholds, inspecting whether coherent populations disappear. Use a dedicated doublet method per library and investigate ambient RNA. Evaluate stability of clusters, composition, DE, and enrichment under alternative thresholds rather than merely maximizing cell retention.

**Biology allowed / excluded.** Describe the filtered subset. Do not claim retained proportions measure the original tissue or that excluded cells were all dying cells. Differences in retention can create apparent biological differences.

**Interview questions.** Why 5%? What accounts for the large loss? How can a fixed upper-gene cutoff remove real hepatocytes? How would you demonstrate robustness to QC choices?

## 4. Normalization and highly variable genes

**What and why.** Library sizes are normalized to 10,000 counts per cell and log1p transformed. Batch-aware selection flags 2,000 HVGs to focus the embedding on informative variation.

**Verdict.** This is a conventional exploratory normalization. It assumes library-size scaling is suitable; RNA-content differences and dominant transcripts can still distort comparisons. `batch_key="sample"` makes feature selection batch-aware; it does not correct the embedding or estimate a disease effect separately from donor. The documentation's assertion that 2,000 HVGs capture about 85% of total variance is not substantiated by the inspected output.

**Improve.** Document the exact HVG flavor and input representation, quantify HVG overlap between samples, and inspect influential genes. Keep count-based inference separate from the normalized embedding. Consider integration only for a defined objective; with sample, donor, and claimed condition confounded, automatic integration may erase biological variation rather than solve the design.

**Biology allowed / excluded.** Relative expression patterns and clustering features can be explored. HVG selection does not demonstrate disease specificity or remove confounding.

**Interview questions.** Why 10,000 counts? Why log1p? Does batch-aware HVG selection equal batch correction? Would you integrate these samples, and what evidence would guide that decision?

## 5. PCA, neighborhood graph, UMAP, and Leiden

**What and why.** PCA provides a compact representation; the neighbor graph uses 15 neighbors and 10 PCs; UMAP visualizes it. Leiden is run at resolutions 0.3, 0.5, 1.0, and 2.0, with 0.5 selected for annotation.

**Verified result.** The report and saved object show 14 clusters at the selected resolution, whereas METHODS.md says 11. The saved first 10 PCs explain about 40.2% of variance in the PCA representation, not 85% of whole-transcriptome variance. The `n_pcs` argument in `run_pca` controls the displayed scree plot; it is not passed to the PCA computation. The downstream graph explicitly uses 10 PCs.

**Verdict.** These methods are appropriate for exploratory grouping. The choice of 10 PCs and resolution needs stability and marker evidence. No explicit expression scaling appears before PCA; this is a modeling choice worth testing, not an automatic error. UMAP separation does not establish independent disease subtypes, developmental trajectories, or population-level differences.

**Improve.** Compare nearby PC counts, neighbor settings, resolutions, and seeds; inspect marker coherence, cluster sizes, sample occupancy, and QC/cell-cycle associations. Assess stability in the higher-dimensional representation. Document the actual output and exact versions.

**Biology allowed / excluded.** Identify transcriptionally distinguishable groups in retained cells. Avoid inferring malignancy, temporal progression, or treatment sensitivity from UMAP positions.

**Interview questions.** Why 10 PCs and resolution 0.5? What does UMAP distance mean? How would you distinguish a donor-specific cluster from a cell type? Is scaling necessary for your chosen PCA objective?

## 6. Cell-type annotation and consensus

**What and why.** CellTypist immune models, liver ScType markers, SingleR HPCA pruned labels, and mean marker expression are reconciled at cluster level. Multiple tools can expose inconsistent annotations.

**Verdict.** Cross-checking is useful, but this vote does not constitute four independent validations. Tools have overlapping markers, unequal coverage, and different label granularity. An immune reference should not decide whether epithelial cells are malignant.

Confirmed implementation problems:

- ScType receives log-normalized expression with `scaled=TRUE`, although the reviewed pipeline does not gene-standardize that matrix. Its documented scaled workflow uses scaled expression; this mismatch warrants correction and reannotation. [ScType instructions](https://github.com/IanevskiAleksandr/sc-type).
- Votes compare literal strings without harmonizing ontology: “Macrophages,” “Macrophage,” and “Kupffer cells” are different tokens; broad “T_cells” cannot vote directly for a specific T-cell subtype.
- Ties follow input order because `Counter(...).most_common(1)` chooses the first tied label. A four-way disagreement can therefore become a CellTypist label without consensus.
- Extra ScType weight applies only to exact “Hepatocyte,” “Fibroblast,” or “Endothelial.” Observed ScType labels include “Hepatocytes,” “Hepatic stellate cells,” and “Cholangiocytes,” so the intended liver-specific weighting does not apply to those observed labels.
- Cluster-level assignment masks heterogeneous cells and does not provide uncertainty. SingleR output includes `NA_character_`, which should be excluded or treated explicitly as unknown.
- Mean expression across marker sets favors highly expressed marker programs; it is not a calibrated or background-adjusted score.

**Improve.** First harmonize broad lineages, then annotate subtypes with suitable references and coherent positive/negative markers. Use confidence margins, unknown labels, disagreement tables, and barcode-based alignment. Review clusters 5 and 10 in particular: ScType calls them hepatocytes and stellate cells, while other tools favor plasma/B-cell identities. Separate lineage identity from malignant status; assess epithelial/hepatocyte populations with patient-specific CNV inference and orthogonal markers, recognizing CNV inference is also imperfect.

**Biology allowed / excluded.** Broad immune and epithelial lineages are plausible working labels. The exact T-cell states, Kupffer-versus-TAM identity, and malignant status are not established by this vote.

**Interview questions.** Why these references? What happens in a tie? Are the votes independent? How do you identify malignant hepatocytes? What would make you assign “unknown”?

## 7. Cell composition and expression confounding

**What and why.** Cluster labels are summarized to characterize cellular heterogeneity.

**Verified counts in the saved object.** Labels below are the workflow's provisional labels; sample names are neutralized here.

| Annotated type | HCC1 | HCC2 |
|---|---:|---:|
| Hepatocytes | 207 | 0 |
| Macrophage | 905 | 5 |
| Monocyte | 233 | 15 |
| Helper/effector T labels | 585 | 220 |
| Cytotoxic/resident T labels | 78 | 264 |
| Plasma cells | 70 | 54 |
| B cells | 40 | 21 |
| Epithelial cells | 0 | 98 |
| Total | 2,118 | 677 |

Macrophage labels account for approximately 42.7% of retained HCC1 and 0.7% of retained HCC2. This is a sample-subset observation, not proof of macrophage depletion in HCC.

Sex-chromosome signals are especially informative: XIST is detected in 0% of HCC1 and 59.8% of HCC2 cells. RPS4Y1 is detected in 57.2% of HCC1 and 0% of HCC2; DDX3Y and KDM5D also occur only in HCC1. The combined pattern strongly suggests sex-related donor differences, but donor sex is not confirmed from the reviewed metadata and tumor chromosome changes are another possible contributor.

ALB and APOA2 are detected in approximately 94.3% and 96.3% of HCC1 cells, respectively, despite only 207 hepatocyte-labelled cells. Both are undetected in retained HCC2 cells. This motivates inspection of ambient hepatocyte RNA, sample-specific expression, and annotation. Broad detection alone does not prove contamination.

**Verdict and improve.** Report sample-stratified fractions with QC and sampling qualifications. Distinguish cell recovery from tissue abundance and compositional shifts from within-lineage regulation. Verify donor metadata and investigate widespread liver-secretory transcripts. More cells do not supply more independent donors.

**Biology allowed / excluded.** These retained samples differ in recovered composition. They do not demonstrate disease-driven infiltration, immune evasion, or a temporal transition between cell states.

**Interview questions.** How does cell composition affect DE? Why do sex genes dominate? Could abundant liver transcripts in immune cells be ambient RNA? What is the denominator in a composition plot?

## 8. Differential expression and volcano plot

**What and why.** Scanpy Wilcoxon compares cells grouped by sample. BH-adjusted p < 0.05 and absolute reported log2FC > 1 retain 1,385 genes: 335 higher and 1,050 lower in HCC2.

**Verdict.** This is a descriptive cell-level sample contrast, not a valid replicated disease comparison. Cells share donor/library effects; treating them as biological replicates inflates confidence. BH correction does not repair the wrong experimental unit. Mixing lineages makes differences in composition part of the measured contrast. Biological-replicate-aware approaches are supported by empirical benchmarking. [Squair et al.](https://www.nature.com/articles/s41467-021-25960-2).

The function tests all groups and retrieves HCC2 versus the rest; with two samples this yields the intended sample contrast. It does not specify a cell-type comparison. The inspected saved object has log-normalized X and no `.raw`, consistent with the displayed analysis, although the historical runtime cannot be reconstructed completely.

Extreme values such as XIST +31.18, ALB −33.18, and APOA2 −33.59 arise where the other sample has no detected expression. They should not be advertised as precise billion-fold changes. Scanpy's approximate logFC derived from log-expression summaries and near-zero handling require care. Zero displayed adjusted p-values are numerical underflow, not literal zero probability. The volcano plot contains only significant genes, omitting the nonsignificant background.

**Improve.** Resolve labels, annotate reliable broad types, inspect detection fractions and effect sizes, and use pseudobulk counts per donor × cell type with a replicated design when available. With only one donor per proposed condition, aggregating counts cannot create replication; random cell splits are not independent pseudobulks. Export every tested gene and statistic. Plot the full volcano with significant genes highlighted and investigate zero-expression genes separately.

**Biology allowed / excluded.** HCC2 has less liver-secretory and several myeloid-associated expression programs in the retained mixture. The present data do not identify tumor-cell suppression of ALB/APOE, oncogenic XIST activation, or causal drivers. In particular, sex-linked expression is a confounding signal before it is a cancer hypothesis.

**Interview questions.** Why Wilcoxon? What does BH control under your model? Why not pseudobulk? Can pseudobulk work with one donor per condition? What does a logFC of 33 mean when the reference has zero counts?

## 9. GSEA, pathway summaries, and gene membership

**What and why.** Genes are ranked by log2FC, symbols mapped to Entrez IDs, and clusterProfiler gseGO/gseKEGG run with gene-set sizes 15–500. Enrichment aims to organize expression differences into programs.

**Verdict.** Ranked enrichment is appropriate in principle, but the actual input is the exported significant-only DEG table. The notebook confirms a 1,385-gene ranked list. GSEA normally evaluates a suitable full tested-gene ranking; selecting on significance and fold change removes the middle of that distribution and changes the enrichment question. [GSEA guidance](https://docs.gsea-msigdb.org/GSEA/GSEA_FAQ/).

The mapping code does not explicitly resolve duplicate Entrez IDs after symbol conversion. Large near-zero-reference fold changes can distort ranking. Keyword-selected themes are an interpretation aid, not independent evidence for a named cancer pathway. GO terms overlap, so many significant terms need not imply many independent discoveries.

**Verified saved result.** All retained enrichment results are negative: 120 BP terms, 10 MF terms, 16 CC terms, and 2 KEGG pathways. Cholesterol metabolism has NES −2.098, adjusted p 0.001062; PPAR signaling has NES −2.012, adjusted p 0.002554. These are enriched toward lower HCC2 values in the selected ranking. The saved KEGG results do not establish PI3K–AKT/Wnt or increased glycolysis.

Two additional confirmed reporting defects:

- Saved `core_enrichment` contains Entrez IDs, but `query_gene_pathways` searches it for gene symbols. A negative ALB lookup can therefore reflect incompatible identifiers, not lack of membership.
- `pathway_summary_table.csv` lists text such as “tags=54%, list=16%, signal=48%” as “Key genes.” The summary code conflates `leading_edge` metadata with `core_enrichment`, renaming both to the same field. This table cannot support gene-level interpretation in its present form.

**Improve.** Export the complete tested ranking; prefer a stable signed test statistic and compare sensible ranking choices. Deduplicate identifiers, report mapping loss, and convert leading-edge IDs back to symbols. Separate leading-edge statistics from member genes. Recompute enrichment after fixing provenance and DE; inspect NES sign and actual leading-edge drivers. For over-representation of selected DEGs, use an explicit tested/expressed background and label it as ORA.

**Biology allowed / excluded.** Lipid transport and liver metabolic functions are relatively higher in retained HCC1 in this analysis. This can reflect different cell mixtures and sampling. It does not demonstrate inhibition of PPAR signaling within malignant cells, pathway flux, immune suppression, or causal metabolic adaptation.

**Interview questions.** GSEA versus ORA? Why use all tested genes? What does negative NES mean? What is a leading-edge gene? How do gene-ID duplicates and overlapping GO terms affect the result?

## 10. STRING network construction and hub prioritization

**What and why.** Significant genes are sent to human STRING functional-network queries at score 400 in batches of 500. An undirected graph removes isolates and averages normalized degree, betweenness, closeness, and eigenvector centrality to rank hubs.

**Verified result.** The notebook submits 1,385 genes, receives 7,759 unique candidate interactions, and retains 1,162 nodes with 7,551 edges. The report's “DEGs queried 1162” is a reporting mismatch: that is the surviving network-node count. Top hubs include GAPDH, CD4, IL1B, ALB, and APOE.

**Verdict.** Functional networks are useful for context. They are not HCC-specific physical binding maps or causal regulatory networks. The network endpoint returns interactions within the submitted set; querying disjoint chunks and concatenating results omits cross-chunk edges. Because the DEG list has an ordering, the partition can systematically alter network structure and hub ranks. [STRING API](https://string-db.org/help/api/).

Betweenness passes confidence directly as a distance weight. NetworkX interprets these weights as path lengths, so a higher-confidence edge becomes a longer route, contrary to the intended stronger-connection interpretation. [NetworkX documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html).

Centrality measures are correlated and their equal-weight mean is a heuristic. Removing isolates discards genes with sparse database coverage. Housekeeping proteins and well-studied immune genes may rank highly due to annotation coverage rather than tumor dependence.

**Improve.** Retrieve the complete induced network through a supported whole-set request or a species network download filtered to the gene set. Separate confidence from distance, justifying any conversion; compare unweighted results. Test score thresholds and evidence channels, record unmapped/isolate genes, and examine rank stability. Combine topology with cell-type-specific expression, selectivity, dependency, mechanism, and tractability.

**Biology allowed / excluded.** These genes are connected in the retrieved functional graph. A high hub score does not establish a cancer driver, a selective vulnerability, or a reason to inhibit the gene. A downregulated marker of a missing lineage is not automatically a therapeutic target.

**Interview questions.** Functional association versus physical interaction? What is lost by batching? Why combine four centralities? What does the edge weight mean in each algorithm? Would targeting a housekeeping hub harm normal cells?

## 11. Survival: code present, results not established

**What and why.** Notebook 02 includes a TCGA-LIHC survival block using median-split Kaplan–Meier/log-rank tests and penalized univariable Cox models with standardized continuous expression. Prognosis could provide independent clinical context.

**Evidence status.** The inspected survival cells have no stored execution output, no survival result tables are present, and the current downstream survival bonus/feature is commented out. Do not claim this project has demonstrated prognostic validation. This does not prove the block was never run elsewhere.

**Verdict.** Continuous Cox expression is reasonable for an exploratory association; median splits lose information. Testing many genes needs multiplicity handling. Age, sex, stage, liver function, treatment, tumor purity, and other available prognostic variables require consideration; assumptions and event counts must be checked. KM and Cox use the same patients and do not constitute independent validations. A prognostic association does not prove a tumor suppressor or an actionable target.

The automatic download-failure fallback simulates expression and outcomes and explicitly engineers associations for APOE/ALB and XIST/FTL. Those outputs can test software only; they cannot validate biology. Such simulation must never enter biological evidence scoring.

**Improve.** Fail explicitly for a biological run if real data are unavailable. Validate clinical fields, units, sample/patient deduplication, joins, gene mapping, censoring, and events. Use prespecified continuous-expression Cox models, multiplicity correction, assumption checks, sensible covariates, and external validation; show numbers at risk on KM plots.

**Biology allowed / excluded.** No verified project-specific survival conclusion is available in this checkout. Published survival claims must be identified as external evidence with their own design limitations.

**Interview questions.** Prognostic versus predictive biomarkers? Why continuous Cox rather than median split? What is proportional hazards? What does a simulated fallback demonstrate? Why is an HR below 1 insufficient to call a gene a tumor suppressor?

## 12. Drug-database retrieval and manual curation

**What and why.** The workflow queries DGIdb for network genes and supplements genes without results using a hard-coded interaction list. ChEMBL and OpenTargets live queries are disabled in the stored notebook configuration.

**Verified result.** Stored retrieval output shows 8,069 DGIdb records plus 6 manual additions, deduplicated to 8,027 pairs across 547 genes and 4,978 drugs. Current source labels are DGIdb 8,022, ChEMBL 4, and OpenTargets 1. The latter labels come from manual fallback records, not enabled live queries. “Approved drugs 2795” counts approved interaction rows; the current table contains 1,223 unique drugs marked approved.

**Verdict.** Database retrieval is useful hypothesis generation. The fallback's manual origin is hidden because it reuses database source names without per-record citations or confidence. Manually supplied interaction scores and publication counts lack traceable evidence in the reviewed code. These records are not proof of direct target inhibition. Examples requiring primary-record verification include TYROBP–sorafenib/regorafenib, AIF1–minocycline, and FTL–iron-chelator assignments. Drug–gene associations can also involve response biomarkers, downstream effects, or carrier binding. DGIdb interaction evidence should be followed back to its source. [DGIdb publication](https://doi.org/10.1093/NAR/GKAA1084).

The client retains only the first interaction type, discards publication identifiers from exported records, and assigns DGIdb clinical phase from approval alone: approved → 4, otherwise → 0. Thus phase is not an independently retrieved clinical-stage measurement for those records. String title-casing is not reliable chemical-identity harmonization. Unknown direction should remain unknown; the disabled ChEMBL/OpenTargets clients otherwise infer “activating” whenever a mechanism does not contain “inhibit,” which is unsafe for future use.

**Improve.** Track retrieval source separately from evidence source and manual curation; preserve stable drug IDs, all mechanisms, PMIDs, indication, disease/genotype context, retrieval date, and primary evidence. Require a source for every curated record and remove unsupported edges from biological ranking. Keep missing clinical phase distinct from preclinical status. Harmonize salt forms, aliases, and drug identities. Preserve conflicts and provenance when deduplicating.

**Biology allowed / excluded.** The table is a candidate evidence catalog. An edge does not establish direct binding, favorable direction, achievable exposure, HCC efficacy, or suitability for combination therapy. “Approved” is neither synonymous with FDA approval without jurisdictional verification nor with HCC approval.

**Interview questions.** What exactly does an interaction mean? Were all three APIs queried? How are curated records sourced? Does albumin binding make ALB an anticancer target? How would you distinguish a drug-response biomarker from the inhibited protein?

## 13. Composite evidence scoring

**What and why.** A weighted formula combines normalized interaction evidence, publication count, clinical phase, approval, and hub score to rank known pairs.

**Verdict.** An explicit heuristic is interpretable if its assumptions are disclosed. It is an evidence-priority index, not measured affinity, therapeutic efficacy, or a calibrated probability. Weights are unvalidated, publication counts favor well-studied relationships, and approval and assigned phase partly duplicate the same information.

**Confirmed discrepancy.** With the notebook's hub weight 0.10, the code multiplies the hub term by another 0.10, giving effective coefficient 0.01. Recomputing the formula from the saved edge table exactly matches all saved rounded scores using 0.01; using 0.10 does not. The survival bonus described in METHODS.md is absent. The non-survival coefficients consequently sum to 0.91 rather than the documented 1.00. Min–max normalization also makes scores depend on the current collection of records and outliers.

**Improve.** Align code and description, name the score correctly, and justify or sensitivity-test weights. Assess stability when removing each term and curated record. Avoid double-counting correlated evidence. Include mechanistic compatibility and cell-context evidence explicitly, without assuming that lower expression implies the need for activation or higher expression implies the need for inhibition.

**Biology allowed / excluded.** Higher scores indicate prioritization under chosen rules. They do not rank clinical benefit or tell whether a gene should be inhibited.

**Interview questions.** Why these weights? Does the formula sum to one in code? What changes when the queried dataset changes? How do you assess sensitivity and correlated criteria?

## 14. GNN graph and feature construction

**What and why.** A bidirectional drug–gene graph contains 5,525 nodes and 8,027 observed pairs. GCN, GAT, and GraphSAGE receive 17 feature dimensions and predict the composite edge score using concatenated node embeddings.

**Verdict.** This is a coherent prototype of graph-based score regression. It is not currently a validated predictor of binding, disease response, or resistance. Its target is generated from inputs it receives: approval, phase, interaction score, publications, and hub score. High agreement can reflect recovery of this deterministic rule.

Additional confirmed construction concerns:

- Node features are created before splitting, from all edge rows. `drop_duplicates("drug")` keeps the first row; the table is score-sorted, so pair-specific interaction evidence/publications/type/source from a high-scoring edge become features for the entire drug node.
- A drug may have different mechanisms and evidence for different genes. Encoding its first interaction as a universal drug property is misleading and order-dependent.
- StandardScaler is fitted on all nodes, contrary to the methods claim of fitting on training nodes.
- Both directions of all pairs, including validation/test pairs, remain in the graph used during training. Only supervision indices are split. Known-topology transductive edge-weight regression can allow that setup, but it cannot be described as held-out link discovery or as never exposing test edges.
- `survival_target` is absent from the current edge CSV; the builder silently leaves its feature dimension zero. Seventeen columns therefore do not establish seventeen informative features.
- Gene and drug names share an untyped node-key namespace. Typed IDs would avoid accidental name collisions, even though the reported node count shows no such collision in this saved run.

**Improve.** Define the prediction task first. For rule emulation, compare against the original formula, linear models, and an MLP; the formula itself reconstructs the rounded targets exactly. For new-link prediction, use independent target evidence, separate message-passing from query edges, remove held-out pairs and reverse edges, and avoid features derived from those held-out records. Use typed node identities and stable drug/gene features, with pair-specific properties represented at edge level. Molecular and protein/context descriptors require task-specific validation.

**Biology allowed / excluded.** The model can learn patterns in evidence scores on a known catalog. Its predictions are not evidence of new chemistry or HCC efficacy.

**Interview questions.** What is the ground truth? Can a simple baseline recover it? What information is available at inference? Transductive versus inductive prediction? Why do interaction properties belong on edges rather than nodes?

## 15. GNN training, evaluation, and architecture comparison

**What and why.** The notebook randomly splits edge labels 70/15/15, trains with MSE and Adam, uses dropout/weight decay, validation early stopping and scheduling, and reports R²/MSE/MAE. These are standard optimization tools for bounded score regression.

**Verified historical result.** Stored notebook output gives GraphSAGE R² 0.9928, MSE approximately 0.0002, and MAE 0.0072. The HTML rounds MSE to 0.000; it is not zero. GCN and GAT perform less well against the same constructed labels. The current model checkpoint is named `gcn_best.pt` even though the export routine saves whichever model wins, so the filename alone cannot identify its architecture.

**Verdict.** These metrics measure recovery of historical heuristic labels, not drug-response accuracy. The notebook chooses the best architecture by test-set R², reusing the test set for model selection. The full-graph and feature-provenance problems above further limit generalization claims. One random seed provides no estimate of variability. GraphSAGE has more parameters in the stored comparison, so performance differences do not isolate neighborhood aggregation as the cause.

In the current catalog, 3,594 of 4,978 drugs (72.2%) have just one observed edge. Random edge splitting can therefore place many drugs with no training interaction into validation/test while still supplying their held-out topology and record-derived features. This is particularly different from genuine unseen-drug prediction.

**Improve.** Choose architecture and hyperparameters on validation only and evaluate once on an untouched test set. Compare the direct rule, non-graph baselines, and graph/feature ablations. Use repeated seeds, error distributions, rank metrics, and separate evaluation of top-ranked candidates rather than only global average error. If novelty is the objective, use drug-held-out, gene-held-out, scaffold/temporal or external splits as appropriate, with corresponding restrictions on features and adjacency. Unknown interactions are not automatically experimentally confirmed negatives.

**Biology allowed / excluded.** You demonstrated that several encoders fit a synthetic evidence-index task. The reported R² does not demonstrate clinical accuracy or superiority to the reference paper, and ranking shifts do not prove improved relevance. Explanations involving graph structure need ablations or attribution evidence, rather than narrative plausibility.

**Interview questions.** Why is test-based selection problematic? Why R² rather than AUC? What baseline should a GNN beat here? How would you evaluate new drugs? How can global accuracy coexist with large errors among the top candidates?

## 16. Drug ranking, interpretation, and validation

**What and why.** `rank_drugs` applies the selected model to rows in the existing edge table and sorts predicted scores.

**Verdict.** The implementation re-scores 8,027 known pairs. It does not score all 547 × 4,978 = 2,722,966 possible pairs. It ranks pairs, not a defined drug-level or multitarget objective. Repeated drugs in a top-20 pair list are not twenty unique drugs. The graph's learned embeddings are evidence-structure representations, not validated pharmacological families.

**Historical versus current evidence.** Report 03 ranks minocycline–AIF1 first and regorafenib–TYROBP second. Several reported original scores disagree with current report 02/current input: Cerliponase Alfa–TPP1 is 0.7229 historically versus 0.7023 now; minocycline–AIF1 is 0.5012 versus 0.4548. Report 03 is dated earlier than the other reports. Its exported ranking and embedding CSVs are absent in the inspected tables directory. These are evidence of unmatched run states, not grounds to compare the reports as a single reproducible run.

**Biological interpretation.** The provisional myeloid labels and genes such as APOE, TYROBP, AIF1, and FTL motivate examination of myeloid-related biology. Lipid-related enrichment motivates cell-specific metabolic characterization. The epithelial population unique to retained HCC2 motivates marker and CNV inspection. Each is a research hypothesis; none establishes resistance or drug sensitivity. FTL abundance does not establish iron dependence, direct FTL inhibition by a chelator, or a favorable therapeutic direction. Recognizing an existing HCC medicine in a ranking is not independent validation, especially where its edge was manually inserted.

**Improve.** Match model, scaler, features, input hash, and report to one run. Audit the top candidates' primary mechanisms and indications before interpreting them. Define an explicit drug-level objective if needed, including supported targets, desired directions, evidence strength, off-target liabilities, and uncertainty. Novel combinations require combination experiments; a bipartite graph alone cannot establish synergy.

**Validation sequence.** Verify evidence and disease context; replicate expression/target localization in independent donors; test target dependence with perturbation and rescue; establish target engagement; measure dose-response in relevant malignant and nonmalignant models at plausible exposures; use appropriate immune co-cultures for microenvironment mechanisms. Combination or resistance claims need designs that actually measure those endpoints.

**Biology allowed / excluded.** Present candidates as hypotheses for evidence review and experiments. Do not call them effective therapies, validated repurposing opportunities, resistance solutions, or newly discovered interactions.

**Interview questions.** Are you predicting unknown pairs? How do you rank a drug with multiple targets? Why is minocycline ranked first? What would validate the TYROBP edge? Which experiment would falsify your preferred hypothesis?

## Comparison with the 2025 reference paper

Wang et al. describes HCC1 as adjacent normal and HCC2 as tumor, whereas the checked GEO sample records do not establish that assignment. Your 1,385 DEGs differ from its 1,178; this is a pipeline difference, not proof that either list is correct. The paper also includes trajectory, cell-cycle, infiltration, and survival analyses that the inspected outputs do not independently reproduce. Its GNN methods describe molecular fingerprints and gene-expression context; your features mainly encode evidence metadata and centrality. Its reported model metrics concern a different target scale and evaluation setup, so numerical performance cannot establish improvement. Its survival and drug interpretations should be assessed separately from source-data and experimental evidence. [Reference paper](https://www.nature.com/articles/s41698-025-00952-3).

Call the project an adaptation inspired by the paper unless a matched computational reproduction is demonstrated. A published description is something to audit, not a replacement for provenance or validation.

## What to defend in the interview

Use this as a corrected framing, acknowledging that the audit exposed issues after the initial analysis:

“I implemented a modular workflow connecting single-cell preprocessing, annotation, differential expression, pathway analysis, network prioritization, and graph-based ranking. Auditing it showed that provenance, the experimental unit, and model ground truth were the decisive issues. The source metadata did not justify the assumed normal–tumor labels, pooled expression differences reflected composition and likely donor differences, and the GNN learned an evidence score rather than an experimental drug-response endpoint. I would first resolve sample identity and QC, then use replicated cell-type-specific inference and independent model evaluation. The current results are exploratory hypotheses.”

Prepare concise answers to these five questions first:

1. **What was your strongest result?** A connected, inspectable computational workflow and identification of major sample heterogeneity; distinguish this from validated biological discovery.
2. **What was your biggest mistake?** Accepting condition labels without verifying source metadata. Explain its downstream consequences and correction.
3. **Why are the p-values so small?** Cell-level testing, large numbers of correlated cells, strong mixture differences, and sex-related/sample differences; small p-values do not remove design bias.
4. **Why is GNN R² so high?** The labels are constructed from input evidence, and the evaluation exposes full topology and record-derived features. The number measures rule recovery under that setup.
5. **What would you do next?** Resolve provenance, evaluate QC and annotation, obtain a replicated design for the disease question, then assess target and drug hypotheses with independent evidence.

## Prioritized corrections and outstanding evidence

1. Obtain or reconstruct the donor/tissue manifest and confirm whether any deposited control exists. This is the prerequisite for disease-specific conclusions.
2. Revisit QC with per-sample retention and lineage stability, then correct ScType inputs and consensus-label logic.
3. Reframe current DE descriptively; obtain biological replication for disease inference and rerun enrichment with the full tested ranking.
4. Rebuild the complete STRING induced graph, correct distance weights, and assess hub stability.
5. Trace every curated drug edge and scoring field to evidence; correct the hub coefficient and remove unsupported survival claims.
6. Define a scientifically meaningful GNN endpoint and compare appropriate baselines under a leakage-controlled evaluation.
7. Produce one consistent run with exported full DE, ranking, split indices, provenance, model architecture/configuration, environment versions, input hashes, and matched reports.

Additional materials that would refine this audit: the original dataset's clinical/supplementary sample map; any separate raw-data or annotation notes; the actual GNN export files corresponding to report 03; genuine survival outputs and provenance if they exist elsewhere; and the role's focus and interview duration. The existing presentation PDF was not used as analysis evidence; no presentation was created or edited.
