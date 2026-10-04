# HCC presentation comparison and recommended revisions

Reviewed 4 October 2026. Source decks: `HCC_Drug_Discovery_Presentation.pptx` (old analysis, eight slides) and `Fatima_Shockor_Interview_Core_Slides_1_8.pptx` (interview deck, eight slides). The HCC section of the interview deck is slides 5–6. Content was extracted from the actual slide XML and compared with the current interpretation brief and completed analysis results. This is a content and file-structure review, not a visual rendering review.

## Recommendation

Use the new interview deck as the starting point. Its HCC section already has the current QC, annotation and biological results, and handles the lack of sample replication much better. Do not copy the old conclusions or drug list into it. The main gap is that notebooks 2–3 are absent: add target/evidence prioritisation and the baseline comparison if presenting the whole HCC project.

## New interview deck: specific changes

### Slide 5: workflow and analytical strategy

Keep the current numbers: 25,189 input cells, 18,309 retained, 200–7,500 detected genes, mitochondrial fraction below 15%, 2,000 HVGs, 20 PCs, 15 neighbors, Leiden 0.5, 19 clusters and 11 broad labels. Keep the explicit experimental-unit caveat.

- Qualify distribution-informed QC as a working choice checked against distributions and retention, rather than implying an objectively optimal threshold. HCC1 retention is approximately 84.0% versus 52.8% for HCC2.
- Keep “multiple references plus marker review,” but do not treat the four annotation methods as independent validation experiments. Preserve the unresolved label and provisional subtypes in the notes.
- Change “used to avoid over-interpreting pooled sample differences” to “used to reduce cell-composition confounding.” Broad T-cell and stromal groups still contain different subtypes, and technical effects remain.
- Add a small source line: “GSE166635; analysis labels follow Wang et al. (2025).” Explain in notes that donor/tissue metadata remains ambiguous and use HCC1/HCC2 in results captions.
- If this slide introduces the complete project, extend the overview to include STRING/DGIdb prioritisation and evidence-score regression. Keep detailed architecture settings in an appendix.

### Slide 6: biological interpretation

Keep the composition percentages, the pooled/within-T-cell contrast, and the ambient-RNA caveat. These match the current results and are stronger evidence than the old gene list.

- State the direction explicitly: “log2FC = HCC2 relative to HCC1.”
- Replace “Pooled increase largely reflects composition” for IFNG with “Pooled increase is consistent with a composition effect; no significant increase within broad T cells.” The data do not quantify how much of the pooled change each cause explains. Within-T-cell IFNG adjusted p = 0.382.
- Explain “ns” as FDR ≥0.05. Add the adjusted p-values to notes or an appendix for the displayed marker examples.
- Present GSEA themes as relative enrichment. Replace upward/downward pathway arrows with “positive/negative NES” or “enriched toward HCC2/HCC1.” These are not pathway-activity assays.
- Retain cautious T-cell wording. Higher CD8A and PDCD1 supports altered composition/state hypotheses, not demonstrated cytotoxic activity, exhaustion or immunotherapy response.
- Retain “provisional contractile stromal program.” ACTA2 rises while COL1A1 and COL3A1 fall, so “more fibrosis” or “global CAF activation” would be too strong.
- Put the numerical ambient-RNA concern in notes: ALB detection is 90.0% in HCC1 T cells, 91.9% in B cells and 94.3% in fibroblasts. Its cause has not been established or corrected.

## Missing HCC content

### Proposed additional slide: network and drug evidence prioritisation

Suggested title: **Drug–gene evidence prioritisation**.

Explain the inputs and output: pooled exploratory DE shortlist, STRING functional associations, DGIdb evidence, and a ranked list for mechanism/indication review. Current counts are 7,121 shortlisted genes, 167,480 STRING associations and 27,452 DGIdb association rows. Distinguish association rows, drug IDs and unique biological drug entities.

Show the current score:

`0.65 × normalized DGIdb score + 0.15 × recorded approval + 0.20 × hub score`.

Publication and clinical-phase weights are zero; clinical phases are unknown. Recorded approval does not mean HCC approval. Network centrality does not establish target dependency. The primary ranking is the original transparent score.

Use TPP1–cerliponase alfa, the top reference association, as a critical-review example. Its high score does not establish an HCC treatment hypothesis. Avoid presenting a top-ten list as efficacious or experimentally validated drugs. Survival evidence was unavailable and must not become a prognostic claim.

### Proposed additional slide: model comparison

Suggested title: **Baseline comparison for evidence-score regression**.

Define the label before reporting accuracy: the model estimates the constructed score for known drug–gene associations. It has no experimental drug-response ground truth.

| Model | Current test MSE | Current test R² |
|---|---:|---:|
| GraphSAGE ensemble | 4.80 × 10⁻⁵ | 0.99113 |
| MLP ensemble | 7.93 × 10⁻⁵ | 0.98533 |
| Architecture without neighbors | 6.83 × 10⁻⁵ | 0.98736 |
| Ridge | 1.18 × 10⁻¹¹ | approximately 1 |
| Training mean | 0.005414 | −0.00109 |

Speaker message: “GraphSAGE performed best among the tested GNN architectures, but Ridge reconstructed the rule-derived score more accurately. I retained the transparent score as the primary ranking.” This complements the interview deck's slide 3 lesson about justifying deep-learning complexity against statistical baselines.

Clarify grouped label splitting, validation-only selection and the known graph retained during evaluation. Do not claim novel-link prediction or performance on unseen drugs. Only one test association has a reference score above 0.5, so accuracy at the top-ranked tail is weakly evaluated. Use scientific notation rather than rounding MSE/MAE to zero.

For a longer deck, insert these two slides after slide 6 and before the spatial section. If the main deck must remain eight slides, retain slides 5–6 and put these two topics in an appendix, with a short mention in the HCC workflow. Do not squeeze all three notebooks into an already dense biological-results slide.

## Old deck: material that must not carry forward

| Old slide | Problem | Current replacement |
|---|---|---|
| 1–2: objective | GNN drug ranking and therapeutic-target wording suggests more predictive/clinical validation than available. | Exploratory single-cell analysis and auditable drug–gene evidence prioritisation. |
| 3: methods | Old 5% mt filter, 2,500 maximum genes, 2,795 cells, 10 PCs, 14 clusters, 1,385 DEGs, curated fallback and five-factor scoring are obsolete. | Current values on interview slide 5; real DGIdb evidence and three active score components. |
| 4: QC and cell populations | Claims the severe filtering was “correct,” and labels macrophages immunosuppressive and T cells functionally active without sufficient evidence. Counts of macrophage clusters do not measure tumour macrophage abundance. | Current recovery fractions and marker-supported broad identities. HCC2 has only 26 macrophage-labelled cells. |
| 5: DE genes | Infers hepatocyte dedifferentiation, invasion and epigenetic dysregulation from pooled DE. XIST is also sensitive to donor/sex context. | Composition-aware interpretation and within-cell-type examples. Only one HCC2 hepatocyte prevents a meaningful hepatocyte contrast. |
| 6: GSEA | “All pathways downregulated” is obsolete. Negative NES is equated with pathway activity suppression. | Current contrast-specific positive and negative enrichment, with adjusted p-values and cautious interpretation. |
| 7: GNN and drugs | Old scores, omitted baselines, errors rounded to zero, and “validated” drug–gene pairs overstate the evidence. | Current baseline table, explicit constructed target and reference-score ranking for review. |
| 8: conclusions | Claims TAM suppression, malignant dedifferentiation, iron-chelation dependency and validation by recovering known HCC drugs. None is established by this analysis. | Recovered composition differences, exploratory immune/stromal patterns, and the baseline lesson. Drug efficacy requires independent experiments. |

## File-opening problem and repair

Both supplied files passed ZIP integrity and XML parsing checks, with no missing internal relationship targets detected. However, the new interview deck contains two structural defects:

1. In `ppt/presentation.xml`, the notes-master list follows the slide list. The PresentationML schema requires the notes-master list before the slide list.
2. Slide 1 contains two shape objects with ID 25. The slide-number placeholder now has a unique ID, 89, in the repaired copy.

Saved a separate copy named `Fatima_Shockor_Interview_Core_Slides_1_8_repaired.pptx`. Only these two XML parts changed. All slide text is identical, all other package parts are byte-identical, and the slide-1 shape properties are identical after accounting for the ID correction. The originals remain untouched.

These are plausible causes of a stricter importer rejecting the file, but opening in Google Drive has not been tested. The repair is structural only; it does not apply the proposed content revisions or verify visual rendering. Upload the repaired file as a new file and open it with Google Slides. If the problem persists, the exact error message is needed to distinguish a remaining import problem from browser/Drive preview issues.

Sources for the file diagnosis: [Microsoft Open XML SDK presentation schema](https://github.com/dotnet/Open-XML-SDK/blob/main/data/schemas/schemas_openxmlformats_org_presentationml_2006_main.json). General opening guidance: [Google Drive viewing/opening files](https://support.google.com/drive/answer/2423485?hl=en-uk), [Google Drive troubleshooting](https://support.google.com/drive/answer/2456903?hl=en).

Current numerical and biological evidence is documented in `INTERPRETATION_AND_PRESENTATION_BRIEF.md` and its supporting CSVs in this directory.
