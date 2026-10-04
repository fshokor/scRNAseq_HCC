# APOA2, SERPINA1, FTL, APOE and APOC1: expression, enrichment and drug evidence

Checked against the current completed results on 4 October 2026 and the supplied Wang et al. PDF. This audit concerns saved database evidence, not a new exhaustive search for all possible therapeutic connections. No analysis or ranking files were changed.

## Results at a glance

All five are significantly lower in the current pooled HCC2-versus-HCC1 contrast and meet the DE shortlist criteria. In descending strength of the negative Wilcoxon statistic, APOA2, FTL, SERPINA1, APOE and APOC1 occupy positions 1, 3, 4, 5 and 6. This ordering is by test statistic, not absolute fold change. Adjusted p-values are stored as zero because of numerical underflow; do not interpret or display them as literally zero probability.

| Gene | Pooled log2FC | Example significant enrichment containing the gene in its leading edge | NES | Pathway FDR | Current DGIdb association rows |
|---|---:|---|---:|---:|---:|
| APOA2 | −12.20 | KEGG cholesterol metabolism | −2.285 | 9.57 × 10⁻⁹ | 2 |
| SERPINA1 | −6.45 | KEGG complement and coagulation cascades | −2.343 | 4.32 × 10⁻¹³ | 2 |
| FTL | −4.61 | KEGG ferroptosis | −2.086 | 8.63 × 10⁻⁶ | 0 |
| APOE | −7.89 | KEGG cholesterol metabolism | −2.285 | 9.57 × 10⁻⁹ | 39 |
| APOC1 | −8.45 | KEGG cholesterol metabolism | −2.285 | 9.57 × 10⁻⁹ | 1 |

The shared cholesterol result is one enriched gene set containing three selected genes, not three independent pathway findings. Negative NES places the gene set toward the HCC1-associated end of the ranking. It does not directly measure pathway suppression in malignant cells.

All five entered the pooled DE candidate list and the connected STRING network. Their hub scores are APOA2 0.262, SERPINA1 0.300, FTL 0.243, APOE 0.441 and APOC1 0.226. Being a candidate/network node does not establish a validated pharmacological target. FTL has no current drug edge and therefore is not a current drug–gene regression/ranking candidate through these results.

## Gene-by-gene interpretation

### APOA2: lipoprotein-associated expression, with a major contamination caveat

APOA2 is a lipoprotein-associated gene with a role in HDL biology. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/336/)

GSEA confirms its leading-edge membership in cholesterol metabolism and in negatively enriched PPAR signaling (NES −1.976, FDR 8.23 × 10⁻⁶), as well as lipid transport and cholesterol-efflux GO processes. A defensible pooled interpretation is a lower contribution of lipoprotein/liver-associated transcripts in HCC2.

However, APOA2 is detected in 93.3% of HCC1 T cells, 97.1% of HCC1 B cells and 96.4% of HCC1 fibroblasts, versus almost no HCC2 cells in those populations. Together with the ALB issue, this raises a substantial ambient-RNA or other technical/annotation concern. Do not interpret the huge fold changes as confirmed intrinsic metabolic reprogramming in lymphocytes or malignant hepatocytes. Only one HCC2 hepatocyte-labelled cell is available.

The current DGIdb records are **RETINOIC ACID AGENT** and **THERAPEUTIC GLUCOCORTICOID**, both from the NCI Cancer Gene Index, with unknown interaction types. These are broad classes, not two precisely identified compounds with verified direct targets.

Crucially, the retinoic-acid record cites PMID 7918317. Its abstract reports no substantial effect on APOA2 expression/secretion in the studied models. This undermines using that row as positive APOA2-target evidence. The glucocorticoid record cites PMID 8706754; its mechanism was not established in this audit. [Retinoic-acid primary study](https://pubmed.ncbi.nlm.nih.gov/7918317/)

The paper's APOA2 compounds were not recovered in this current DGIdb snapshot. Do not manually restore them without resolving the original IDs, gene mapping and cited evidence.

### SERPINA1: protease inhibition and liver/myeloid-associated transcripts

SERPINA1 encodes alpha-1 antitrypsin, a serine-protease inhibitor produced prominently by the liver. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene?cmd=retrieve&list_uids=5265&rn=1), [NIH gene explanation](https://medlineplus.gov/genetics/gene/serpina1/)

Besides complement/coagulation enrichment, it occurs in negatively enriched **serine-type endopeptidase inhibitor activity** (GO molecular function, NES −2.130, FDR 1.37 × 10⁻⁶). This supports a lower pooled contribution of protease-inhibitor and related secreted-protein transcripts; it does not measure coagulation or protease activity.

Within macrophages its log2FC is −2.329 (FDR 3.48 × 10⁻⁷), but HCC2 has only 26 macrophage-labelled cells versus 6,975 HCC1 cells. Detection in HCC1 T/B cells is also high, adding technical caution to lymphocyte contrasts.

Current DGIdb associations are **IGMESINE** and **ALPHA 1-ANTITRYPSIN**, both from the Therapeutic Target Database. Both have unknown interaction types and no publication IDs in the exported rows. Igmesine repeats a named paper association, but matching a database row does not establish direct binding or HCC efficacy. Alpha-1 antitrypsin is the gene product itself; do not turn that record into an inhibition claim. Mechanism, indication and primary citations remain unresolved for this prioritisation use.

### FTL: iron handling, not a validated iron-chelation vulnerability

FTL encodes ferritin light chain, part of the intracellular iron-storage machinery. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/2512)

FTL occurs in the negatively enriched ferroptosis leading edge and in **intracellular iron ion homeostasis** (pooled NES −1.801, FDR 7.87 × 10⁻⁴). Within broad T cells it remains lower (log2FC −2.213, stored FDR underflows to zero) and occurs in the leading edge of negatively enriched intracellular iron-ion homeostasis (NES −1.879, FDR 0.00623).

This is a useful gene/process example: iron-handling-associated transcripts differ across samples, including within broad cell types. It does not establish iron concentrations, ferritin protein abundance, ferroptosis rates or susceptibility to iron chelators. FTL lowering has no uniquely determined direction of effect on ferroptosis because multiple storage, uptake, export and antioxidant systems contribute.

There are **no FTL rows in the current DGIdb evidence export**. The older deferiprone/deferasirox/deferoxamine–FTL claims are therefore not supported by this current evidence snapshot and should not be presented as reproduced drug discoveries. This absence does not establish that no biological or pharmacological connection exists anywhere in the literature.

### APOE: lipid transport plus heterogeneous drug-association evidence

APOE contributes to lipoprotein transport. Its leading-edge membership supports the cholesterol-metabolism theme; GO lipid transport and lipoprotein-clearance processes are also negatively enriched. The pooled difference is highly sensitive to lineage composition and possible ambient signal.

Within macrophages, APOE log2FC is −1.509 but **FDR is 0.155**, so this comparison does not pass the significant DE threshold. Do not use the pooled result to assert significant APOE reduction within macrophages. In monocytes it is lower (log2FC −1.768, FDR 2.26 × 10⁻⁷), with sample imbalance still relevant.

There are **39 current DGIdb association rows**, including statins, ritonavir, lecanemab, donanemab, aducanumab and other compounds/classes. Many derive from PharmGKB. All 39 exported interaction types are unknown. The database source and cited mechanism matter more than the number of hits.

For example, lecanemab targets amyloid-beta, while APOE genotype informs adverse-event risk. An APOE–lecanemab pharmacogenetic link is not evidence that the drug binds or inhibits APOE. Ritonavir records cite studies of genotype-associated lipid adverse effects. [FDA lecanemab description](https://www.fda.gov/drugs/drug-safety-communications/fda-recommend-additional-earlier-mri-monitoring-patients-alzheimers-disease-taking-leqembi-lecanemab), [Genotype/ritonavir primary study](https://pubmed.ncbi.nlm.nih.gov/15809899/)

The negatively enriched Alzheimer-disease KEGG set also contains APOE; this reflects shared annotated genes and does not diagnose Alzheimer disease in these samples.

### APOC1: cholesterol-associated expression with a questionable ritonavir record

APOC1 appears with APOA2/APOE in the cholesterol-metabolism leading edge. Its macrophage log2FC is −2.896 (FDR 3.55 × 10⁻⁶), but the 26-versus-6,975 imbalance and subtype differences prevent population-level conclusions. The main pooled interpretation is reduced contribution of lipid-associated liver/myeloid transcripts, not proven suppression of a cancer dependency.

The sole current drug association is **RITONAVIR**, from PharmGKB, unknown interaction type, citing PMID 15809899. That study's abstract concerns **APOC3, APOE and TNF**, and ritonavir-associated hyperlipidemia. It does not substantiate APOC1 as a direct drug target. Flag this as a possible mapping/annotation issue requiring original-source review rather than quietly accepting the edge. The audit does not establish at which database/source layer the discrepancy arose. [Cited primary study](https://pubmed.ncbi.nlm.nih.gov/15809899/)

## Comparison with Wang et al.

The supplied paper's Fig. 5 features all five genes; its text highlights APOA2, SERPINA1 and FTL, and associates APOE/APOC1 with cholesterol-related processes. It discusses APOA2, SERPINA1 and APOE expression as higher in HCC1. This is qualitative concordance from reanalysis of the same dataset, not independent replication.

The paper names SERPINA1–igmesine and APOA2–PKR-A/MITZ associations. The current snapshot repeats igmesine but not those APOA2 compounds. Database versions, mappings and processing require review before claiming reproduction of those drug results. Paper prognosis claims for APOE/FTL were not validated by this project's unavailable survival stage. [Wang et al., 2025](https://doi.org/10.1038/s41698-025-00952-3)

## Presentation recommendation

For continuity with the paper, use **APOA2, SERPINA1 and FTL** as three selected pooled DE examples. They span lipoprotein biology, protease inhibition and iron handling, with clear significant GSEA leading-edge links. APOE/APOC1 can join the lipid theme in speaker notes or an appendix.

Suggested title: **Lower lipid-, protease- and iron-associated expression in HCC2**.

Suggested takeaway: “The pooled comparison detects liver/myeloid-associated expression differences and relevant enriched gene sets. Their interpretation depends on the cells recovered and unresolved technical effects. Drug-database associations generate evidence-review tasks, rather than validating HCC treatments.”

Use three rows showing gene, log2FC, named KEGG set and biological role. Keep drug associations on a separate slide or in notes, explicitly distinguishing the igmesine record from absent FTL edges and questionable APOA2/APOC1 records. A downregulated gene is not automatically an inhibition target.

## Saved evidence

- `five_gene_de.csv`: all available pooled and cell-type DE rows for these genes.
- `five_gene_gsea_leading_edges.csv`: significant GO/KEGG results whose leading edges contain a selected gene. This does not enumerate all pathway memberships; only significant leading-edge membership is exported.
- `five_gene_drug_records.csv`: all 44 current association rows across the five genes, preserving source, IDs, scores, unknown mechanisms and citations.
- `five_gene_audit_raw.json`: extracted evidence and paper contexts for traceability.

Sources: current `data/processed/sample_contrasts/*/de_all_genes.csv`, `results/tables/sample_contrasts/*/gsea_*.csv`, `results/tables/target_prioritisation/pooled/{de_candidates,hub_genes,dgi_evidence}.csv`, and the supplied Wang PDF. Original analytical outputs remain unchanged; potentially unreliable database edges still require a separate decision before any future exclusion/rerun.
