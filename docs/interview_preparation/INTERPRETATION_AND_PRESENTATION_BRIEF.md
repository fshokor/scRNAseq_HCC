# HCC single-cell case study: interpretation and presentation brief

Prepared from the completed, updated runs saved on 4 October 2026. This is presentation content and an interview study guide, not a slide deck. The provisional structure below assumes a 10–15-minute bioinformatics interview presentation.

## 1. The central story

**Suggested title:** From HCC single-cell profiles to auditable drug–gene evidence prioritisation.

**One-sentence conclusion:** I developed and audited a workflow connecting cell annotation, exploratory expression contrasts, pathway analysis and drug–gene evidence; the strongest findings concern sample composition and immune/stromal expression patterns, while baseline comparisons show that a transparent evidence score is preferable to a GNN approximation for the current target.

The contribution is an end-to-end, reproducible analysis with explicit checks on interpretation. It is not the discovery of an effective HCC drug. The model comparison is a useful result: it identifies where machine learning adds complexity without improving the current scoring task.

### Provenance and experimental unit

- Dataset: [GSE166635](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166635).
- Analysis labels follow the supplied Wang et al. reference: HCC1 is labelled adjacent normal and HCC2 tumour. Say that this is the reference-paper assignment; do not present it as independently verified donor/tissue metadata. The earlier repository audit documents source-metadata ambiguity. Use HCC1/HCC2 in primary results captions.
- There is one sample per analysis group. Do not describe the experiment as replicated or paired without donor-level supporting metadata. Condition, donor and technical effects cannot be separated in this design.
- Cells are observations within samples, not independent biological replicates. Cell-level FDR does not establish population-level HCC differences.
- Primary evidence files: the current annotated AnnData, `data/processed/sample_contrasts/`, and contrast-specific results under `target_prioritisation/pooled/` and `evidence_score_regression/pooled/`. Historical top-level reports and the older presentation PDF are not the evidence base for this brief.

## 2. What was done, why, and what can be defended

| Step | What and why | Assessment and limitation | Interview preparation |
|---|---|---|---|
| Loading | Read the deposited single-cell count matrices, retain sample labels and preserve unnormalised retained-cell counts. | Appropriate input handling; deposited matrices are not a reproduction of alignment or cell calling. Count preservation does not resolve provenance or ambient RNA. | Explain counts versus normalized expression, and how sample labels were obtained. |
| QC | Retain cells with at least 200 and at most 7,500 detected genes, and mitochondrial percentage below 15. Inspect distributions and threshold sensitivity. | Data-informed screening is more defensible than copying a fixed paper threshold. It remains a heuristic; high gene counts are not proof of doublets, and mitochondrial percentage is not a universal viability threshold. | Why these thresholds? What happens at 5%, 10%, 15% and 20% mitochondrial RNA? |
| Normalisation and features | Library-size normalization to 10,000 followed by log1p; sample-aware selection of 2,000 HVGs for dimension reduction. | Appropriate exploratory preprocessing. HVG selection with a sample key is not batch integration. No claim of separating donor and condition effects. | Why preserve counts? Why do DE and GSEA use more than the HVGs? |
| Representation and clustering | Compare PCs and Leiden resolutions; use 20 PCs, 15 neighbors and Leiden resolution 0.5 for the annotated partition. | Reasonable working settings based on structure and marker coherence; UMAP appearance alone cannot select the true partition. | Distinguish PCA, the neighbor graph, UMAP and Leiden. What would a stability check show? |
| Annotation | Harmonize CellTypist, ScType, SingleR HPCA and marker labels; vote, then manually review unresolved clusters. | Multiple concordant tools support broad identity, but they are not independent validation experiments. Marker review resolves lineage more reliably than naming clusters from UMAP location. | Show the markers supporting a label, and explain why unresolved cells were retained as unresolved. |
| Doublet/low-complexity review | Review cluster 9 as hepatocyte-like but doublet-suspect; keep cluster 11 as unresolved myeloid. | Very low predicted doublet rates and overlapping simulated/observed score distributions do not prove singlets. The primary DE run excludes no clusters. | Explain why high counts plus mixed markers warrant review, and how an exclusion sensitivity run would help. |
| DE | Wilcoxon sample contrasts, pooled and within broad cell types; tie correction, BH adjustment, at least 20 cells per side and detection in at least 1% of either side; shortlist at FDR <0.05 and absolute log2FC ≥1. | Appropriate descriptive screening. Minimum cell counts do not create replication. Pooling is composition-sensitive; broad cell-type contrasts still mix subtypes. | Why not claim patient-level significance? What would donor-level pseudobulk require? |
| GSEA | Rank the complete tested gene list by signed Wilcoxon score; GO and KEGG preranked enrichment, seed 42 and FDR <0.05. | Preferable to using only significant DEGs. Mapping loss, tied ranks, overlapping terms and sample effects remain. NES direction is relative to this comparison, not direct pathway activity. | What is the leading edge? Why can a disease-named pathway be enriched without that disease? |
| Network and drug evidence | Query STRING functional associations, retain isolates, compute centrality, query DGIdb and preserve sources, IDs, mechanisms and unknown metadata. | Network connectivity is not causal importance; database associations are not necessarily direct binding or appropriate therapeutic direction. | Explain functional association versus physical interaction, and approval versus HCC indication. |
| Model comparison | Fit GCN, GAT and GraphSAGE plus MLP, no-neighbor, Ridge and mean baselines; grouped label splitting, validation-only selection and repeated initialization seeds. | Valid for approximation of constructed scores on a known graph. It is not novel-link, cold-start, sensitivity or efficacy prediction. | What exactly is the target? Why does Ridge perform almost perfectly? |

### QC and annotation numbers

25,189 input cells became **18,309 retained cells**: 13,499 HCC1 and 4,810 HCC2. Overall retention was approximately 72.7%; sample-specific retention was 84.0% and 52.8%. The differential retention is relevant when interpreting recovered cell proportions.

The selected partition contains 19 clusters and 11 broad annotation labels, including an unresolved label. Cluster 11 contains 53 low-complexity cells labelled `Myeloid_unresolved`; cluster 9 contains 241 hepatocyte-like cells flagged for doublet review. The fibroblast/stellate, biliary and cDC2-like subtype calls remain provisional. Do not call those subtypes experimentally validated.

## 3. Biological result A: recovered cell composition is very different

| Broad population | HCC1 cells | HCC1 fraction | HCC2 cells | HCC2 fraction |
|---|---:|---:|---:|---:|
| T cells | 2,469 | 18.29% | 3,740 | 77.75% |
| Macrophages | 6,975 | 51.67% | 26 | 0.54% |
| Hepatocytes | 2,615 | 19.37% | 1 | 0.02% |
| B cells | 173 | 1.28% | 308 | 6.40% |
| Fibroblasts | 281 | 2.08% | 101 | 2.10% |

**Supported statement:** Among retained annotated cells, HCC2 is dominated by T cells, whereas HCC1 contains many more macrophages and hepatocytes.

**Biological implication:** Differences in recovered lineages can explain substantial pooled expression and enrichment differences. Lower ALB and APOA1 in pooled HCC2 cannot be treated as evidence that matched malignant hepatocytes suppressed liver functions: there is only one annotated HCC2 hepatocyte and the within-hepatocyte comparison was skipped.

**Cannot conclude:** General HCC immune infiltration rates, absolute tissue cell abundance, depletion of macrophages in HCC, or absence of malignant hepatocytes from the tissue. Recovery, dissociation, filtering, annotation and donor/technical differences can change observed fractions.

**Use figure:** `01_cell_composition.png`. Call the denominator retained cells, not tissue abundance.

## 4. Biological result B: the broad T-cell population differs in composition or state

There are 2,873 shortlisted DE genes in the T-cell contrast. These are exploratory sample-associated results, not independent-patient significance.

| Gene | Pooled log2FC | Within-T-cell log2FC | Within-T-cell BH-adjusted p | Interpretation |
|---|---:|---:|---:|---|
| CD8A | +4.469 | +2.495 | 6.64×10⁻¹⁰⁰ | Greater CD8-associated expression; subtype composition is a plausible contributor. |
| NKG7 | +3.337 | +1.440 | 7.25×10⁻⁷⁶ | Cytotoxic-associated expression; does not by itself identify NK cells or prove killing. |
| PDCD1 | +3.541 | +1.700 | 2.24×10⁻⁵⁴ | Higher checkpoint-associated expression; compatible with altered activation/differentiation states. |
| IFNG | +2.329 | +0.116 | 0.382 | Pooled increase is not accompanied by a detectable broad T-cell increase. Composition contributes to the pooled interpretation. |
| TIGIT | +0.802 | −1.708 | 7.68×10⁻⁸⁸ | Checkpoint-associated genes do not all move in the same direction. |
| FOXP3 | −0.755 | −2.991 | 1.20×10⁻¹²² | Could reflect different regulatory-versus-other T-cell proportions; not proof of altered regulatory function per cell. |

**Defensible interpretation:** HCC2's broad T-cell compartment has a different balance of CD8/cytotoxic-associated, checkpoint-associated and regulatory-associated expression. Both cell-state and subtype-composition explanations remain possible.

**Avoid:** “T cells are exhausted,” “anti-tumour immunity is stronger,” “PD-1 blockade will work,” or “Tregs are functionally suppressed.” PDCD1 alone is insufficient for these claims, and TIGIT/HAVCR2 decrease while PDCD1 increases. A coherent subtype analysis, per-cell coexpression, independent samples and functional evidence would be required.

HCC studies have described diverse infiltrating T-cell states, which makes this a plausible biological hypothesis rather than validation of this dataset's state assignments. [Zheng et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28622514/)

**Use figure:** `02_pooled_vs_tcell_markers.png`. The IFNG comparison is a clear example of why cell-type stratification matters.

## 5. Biological result C: pathway patterns must be read through lineage and quality

| Contrast and term | NES | Adjusted p | Interpretation allowed |
|---|---:|---:|---|
| Pooled: regulation of NK-cell-mediated immunity | +3.234 | 2.05×10⁻⁸ | Immune/cytotoxic-associated genes rank toward HCC2; this does not establish a larger annotated NK population. |
| Pooled: cellular respiration | −2.058 | 1.02×10⁻¹⁷ | Respiration-associated genes rank toward HCC1; composition and technical effects are plausible contributors. |
| T cells: chemokine signaling | +1.730 | 0.00367 | A candidate difference in signaling-associated transcription within the broad T-cell group. |
| T cells: FoxO signaling | +1.965 | 0.000254 | Genes in this set rank toward HCC2; no direct assay of pathway activation. |
| T cells: ribosome | −2.781 | 2.56×10⁻²⁰ | Relative translation-associated transcriptional differences; consider technical and subtype effects. |
| T cells: oxidative phosphorylation | −2.170 | 6.27×10⁻⁶ | A metabolic hypothesis, not demonstrated mitochondrial dysfunction or a Warburg effect. |
| Fibroblasts: muscle contraction | +2.085 | 1.05×10⁻⁷ | Contractile/actomyosin-associated program; investigate myofibroblast, stellate, pericyte or smooth-muscle identities. |
| Fibroblasts: antigen processing/presentation | −2.530 | 1.71×10⁻⁹ | Relative antigen-presentation-associated expression differs; does not prove reduced immune function. |

In fibroblasts, ACTA2 rises (log2FC +2.377), while COL1A1 and COL3A1 fall (−1.410 and −2.324). Therefore, “more fibrosis” or “global CAF activation” is too broad. The pattern is more consistent with a possible difference in contractile/stromal subtype balance, pending identity and contamination review.

GSEA identifies rank enrichment, not pathway activation. Disease-named KEGG sets such as cardiomyopathy, viral infection or primary immunodeficiency often share relevant signaling or structural genes; they are not diagnoses. Inspect leading-edge symbols, effect direction and overlapping gene sets before naming a mechanism. [GSEA user guide](https://docs.gsea-msigdb.org/GSEA/GSEA_User_Guide/)

### Quality caveat that materially changes the biology

ALB is detected in **90.0% of HCC1 T cells, 91.9% of HCC1 B cells and 94.3% of HCC1 fibroblasts**. These are detection fractions from the current DE tables, not a newly performed contamination assay. Such widespread liver-transcript detection in non-hepatocyte labels suggests ambient RNA, another technical effect or annotation problems; it does not prove which explanation is correct.

This is a substantial unresolved issue, especially for liver-function and metabolic interpretations. Ambient contamination can create misleading lineage expression; published correction methods explicitly address such effects. [SoupX primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7763177/)

**Use figure:** `03_alb_quality_caveat.png` as an appendix/limitations figure. Do not present the analysis as ambient-RNA corrected. A future sensitivity analysis needs the appropriate raw droplet data or a justified alternative and must assess changes to annotation, DE and enrichment.

## 6. Network and drug evidence: a reviewable hypothesis list

The pooled run shortlisted **7,121 DE genes** (1,321 up, 5,800 down). The STRING network has **167,480 functional associations**, 6,233 connected genes and 888 isolates. GAPDH, AGO2, ACTB, KIF14 and CDC123 lead the centrality ranking. High connectivity can reflect broad cellular function and database coverage; it does not establish HCC dependency or suitability for inhibition.

DGIdb returned **27,452 association rows** spanning 1,746 genes, 12,301 drug IDs and 11,474 displayed names. Names and IDs are not interchangeable counts. Candidate aliases are grouped before label splitting; distinct database IDs are consolidated only with confirmed, cited equivalences. There are no such confirmed mappings in this run.

The reference score is:

`0.65 × max-normalized DGIdb interaction score + 0.15 × recorded approval + 0.20 × hub score`.

Publication and phase weights are zero. All 27,452 clinical-phase entries are unknown. Approval is not HCC approval. DGIdb's score reflects database evidence and drug/gene specificity, rather than binding potency or efficacy. [DGIdb score documentation](https://dgidb.org/about/overview/interaction-score)

The top reference association is TPP1–cerliponase alfa (score 0.8432). This is an illustration of the difference between an evidence rank and an HCC therapeutic candidate: cerliponase alfa replaces deficient TPP1 enzyme in CLN2 disease. That established mechanism does not establish utility against HCC. [FDA pharmacology review](https://www.accessdata.fda.gov/drugsatfda_docs/nda/2017/761052Orig1s000ClinPharmR.pdf)

Fifteen of the top 20 association rows have unknown interaction types, and 17 involve downregulated genes in the pooled contrast. Review directness, pharmacological action, indication, cellular context and cited evidence before selecting any candidate. Lower expression is not automatically a reason to inhibit a gene.

### Sensitivity and missing survival evidence

The interaction term has a nominal weight of 0.65, but its median actual contribution is only **0.00066** under max normalization, versus **0.05365** for hub score. Log1p/max normalization increases the median interaction contribution to **0.01911**. The two normalization methods share **15/20 top-ranked associations**. This shows score-design sensitivity; it does not identify a biologically optimal normalization rule.

The TCGA survival stage was unavailable after download failure. Do not call candidates prognostic, say they survived a survival gate, or interpret unavailable evidence as a negative result. The report correctly leaves survival as unavailable supporting evidence.

## 7. Model result: the simpler baseline wins

The task holds out constructed-score labels on a graph of known associations. All known unweighted connections remain visible. Approximate grouped splits yielded **19,207 training, 4,127 validation and 4,118 test rows**, with same-gene candidate aliases in one split. This does not evaluate novel links or unseen drugs.

| Model | Test MSE | Test MAE | Test R² |
|---|---:|---:|---:|
| GraphSAGE ensemble | 4.80×10⁻⁵ | 0.00511 | 0.99113 |
| MLP ensemble | 7.93×10⁻⁵ | 0.00666 | 0.98533 |
| Selected architecture without neighbors | 6.83×10⁻⁵ | 0.00615 | 0.98736 |
| Ridge | 1.18×10⁻¹¹ | 0.00000283 | approximately 1 |
| Training mean | 0.005414 | 0.06759 | −0.00109 |

GraphSAGE wins among the GNN architectures using mean validation MSE. Ridge wins the validation comparison among learned approximators. It nearly reconstructs the target because the target is a linear combination of information supplied to the model. This is expected, not independent biological validation.

GraphSAGE beats the tested MLP/no-neighbor versions on this split, but that does not demonstrate a necessary advantage from graph structure: Ridge performs better. The original composite ranking remains the primary prioritisation output.

Across three initializations, rank correlations are 0.9973–0.9983 and top-20 membership is identical. This shows computational stability. However, only **one test association has a reference score above 0.5**; performance at the most highly ranked tail is sparsely evaluated. Initialization stability is not patient-level uncertainty, and high overall R² does not validate top-drug efficacy.

**Use figure:** `04_model_comparison.png`. Explain the log MSE axis and the rule-derived target before showing the near-perfect Ridge result.

## 8. How this differs from the Wang paper

This is an audited adaptation, not a faithful reproduction of all claims. The supplied paper reports a 200–2,500 detected-gene filter; this project instead uses a distribution-informed 200–7,500 filter and a 15% mitochondrial cutoff. The revised project uses complete tested-gene GSEA rankings, explicit contrast provenance, guarded real-data survival analysis, auditable database records and validation-only model selection.

The paper describes MACCS drug fingerprints and gene-expression inputs for a GNN. This project uses evidence/metadata features and predicts a constructed evidence score. Do not claim molecular-representation learning, experimentally established drug sensitivity or reproduction of the paper's predictive performance. The experimental target and available labels differ. Reference: [Wang et al., 2025](https://doi.org/10.1038/s41698-025-00952-3), supplied PDF pp. 2 and 15.

## 9. Provisional presentation structure

| Slide | Message | Evidence/visual | Speaker emphasis |
|---|---|---|---|
| 1 | Build an auditable route from single-cell profiles to target hypotheses. | Title and three-stage workflow. | State the prioritisation objective and experimental unit. |
| 2 | Quality choices determine the cells and conclusions retained. | QC distributions; 25,189 → 18,309 cells. | Explain sample-specific retention and why the paper threshold was adapted. |
| 3 | Broad cell identities require marker review. | Current annotated UMAP plus a small marker panel. | Explain harmonized voting, unresolved cells and provisional subtypes. |
| 4 | Sample composition explains much of the pooled contrast. | `01_cell_composition.png`. | Fractions are among recovered cells; matched hepatocyte DE is unavailable. |
| 5 | Cell-type stratification changes the immune interpretation. | `02_pooled_vs_tcell_markers.png`. | Contrast pooled IFNG with within-T-cell IFNG; avoid asserting exhaustion. |
| 6 | Enrichment identifies hypotheses, not diagnoses or activity assays. | Selected pathway table with NES and leading-edge genes. | Cytotoxic-associated pooled programs and provisional contractile stromal pattern. |
| 7 | Network and drug evidence generate a reviewable list. | Pipeline counts and the transparent score equation. | Explain isolates, missing clinical phases, approval scope and TPP1 example. |
| 8 | Baselines prevent overclaiming machine-learning value. | `04_model_comparison.png`. | Define the label, why Ridge wins, and why primary ranking remains transparent. |
| 9 | Conclusions are bounded by design and measurement. | Three limitations: one sample/group; composition/ambient RNA; constructed labels. | Add unavailable survival evidence and normalization sensitivity in spoken notes. |
| 10 | Next experiments follow directly from the limitations. | Validation roadmap. | Verify provenance; address ambient RNA; replicate across donors; validate mechanisms and response experimentally. |

For a shorter talk, combine slides 2–3 and 6–7. For a more technical talk, add appendices on annotation review, GSEA leading edges, alias grouping, score normalization and score-band model errors. Do not use the old top-level reports or historical presentation as current figures without checking them.

## 10. Interview questions and answers to rehearse

**What was your main contribution?** An audited and reproducible integration of single-cell analysis with network/drug evidence, plus a baseline comparison that exposed the limits of the prediction task. Explain which components you implemented or adapted; do not imply ownership of the source dataset or reference-paper methods.

**Why not just copy the paper's QC?** Thresholds were checked against sample distributions and retention. The chosen screen preserves more cells, but sensitivity and unresolved doublet/ambient effects remain part of the interpretation.

**Why can you not claim population-level tumour DE?** Thousands of cells come from two samples. Condition is confounded with sample; cell-level tests do not substitute for independent patients. With replicated donors, I would use an appropriate sample-level model, often donor-level pseudobulk within annotated populations.

**Are the T cells exhausted?** PDCD1 and cytotoxic-associated markers differ, but the broad population mixes subtypes, checkpoint genes are discordant, and no functional test was performed. I would state an altered composition/state hypothesis and test coherent subtype signatures and per-cell coexpression.

**Why is IFNG different in pooled data but not within T cells?** Pooling combines both expression and lineage abundance. The result is consistent with more recovered T cells contributing to pooled IFNG; the broad within-T-cell contrast did not detect a significant increase.

**What does the ALB signal mean in B/T/stromal cells?** It raises a substantial ambient-RNA or technical/annotation concern. I have not established its cause or corrected it, so I would not use these results to claim intrinsic liver-metabolic reprogramming in those populations.

**What does a high network hub score establish?** Connectivity within the queried functional network and database snapshot, not dependency, direct binding or therapeutic suitability.

**Why is a CLN2 therapy at the top of an HCC list?** The rank rewards association evidence, approval and connectivity, without independently established HCC response labels. That is why the candidate needs mechanism/indication review rather than being presented as a discovery.

**Why does Ridge outperform the GNN?** The regression label is an explicit linear score made from available ingredients. A linear model can reconstruct it; a neural graph model approximates it with additional optimization error. This argues for retaining the transparent score for this objective.

**Did you predict novel interactions?** No. The evaluation holds out labels for known associations while retaining known graph connections. Novel-link evaluation would require a different design; efficacy/sensitivity additionally requires appropriate experimental response labels and biological/drug representations.

**What would you do next?** Resolve source metadata, investigate/correct ambient signal, perform flagged-cluster sensitivity, assess subtype stability, obtain independent samples, review candidate mechanisms and measure target perturbation/drug response. Only then could predictive or therapeutic claims be tested.

## 11. Figure and evidence checklist

- New figures and supporting CSVs are saved beside this brief; each figure uses current saved results rather than invented illustrative values.
- `selected_gene_evidence.csv` and `selected_pathway_evidence.csv` preserve the reported directions, adjusted p-values, detection fractions and available leading-edge symbols.
- `evidence_manifest.json` records hashes of primary inputs and exported evidence tables.
- Keep comparison direction explicit: HCC2 minus HCC1. Positive NES favors the HCC2 end of the ranking; negative NES favors HCC1.
- Label all DE/GSEA results exploratory and sample-associated. FDR controls multiplicity in the implemented test, not donor confounding.
- Do not call the top score a probability, a treatment recommendation, a binding measurement or experimental validation.
