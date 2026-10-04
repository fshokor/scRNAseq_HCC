# Three gene examples for the HCC presentation

## Suggested slide content

**Title: Selected expression changes in immune and stromal cells**

Comparison: HCC2 relative to HCC1, within the indicated broad cell type. These are selected examples, not the three highest-ranked DE genes. All three meet the current DE criteria and occur in the leading edge of the indicated significant GSEA term.

| Gene | Expression change | Enriched process containing the gene | Biological interpretation |
|---|---|---|---|
| **PDCD1** | Higher in T cells, log2FC **+1.70** | T-cell activation, NES **+1.53** | Higher expression of the PD-1 inhibitory receptor is consistent with altered checkpoint-associated T-cell states or subtype composition. |
| **ACTA2** | Higher in fibroblast-labelled cells, log2FC **+2.38** | Muscle contraction, NES **+2.08** | Higher smooth-muscle actin expression supports a contractile stromal program or a different balance of contractile stromal subtypes. |
| **COL3A1** | Lower in fibroblast-labelled cells, log2FC **−2.32** | Integrin-mediated signaling, NES **−1.79** | Lower type III collagen expression, alongside ACTA2 increasing, suggests differences between matrix-producing and contractile stromal programs. |

**Takeaway:** Immune checkpoint-associated expression and stromal programs differ between the two samples. The contrasting ACTA2/COL3A1 directions argue against describing all stromal changes as increased fibrosis.

**Visible footer:** Exploratory comparison, one sample per group. Expression and enrichment do not establish functional activity or causality.

## Speaker explanation and boundaries

### PDCD1: checkpoint-associated expression in T cells

PDCD1 encodes PD-1, an inhibitory receptor whose ligand-dependent signaling regulates T-cell responses. This biological role is established independently of this dataset. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/5133), [Reactome: co-inhibition by PD-1](https://reactome.org/content/detail/R-HSA-389948)

In this analysis, PDCD1 is higher within broad T cells: log2FC +1.6997, gene FDR 2.24 × 10⁻⁵⁴. Detection is 25.7% in HCC2 T cells versus 9.5% in HCC1 T cells. It is in the leading edge of the positively enriched GO biological process **T cell activation** (GO:0042110), NES +1.5337, pathway FDR 5.39 × 10⁻⁴.

Suggested spoken explanation: “The checkpoint-associated signal persists when I compare T cells rather than pooling all lineages. PD-1 regulates T-cell signaling, so its increase raises a hypothesis about altered T-cell states. However, the broad group mixes subtypes, and this result alone cannot distinguish activation, exhaustion or a change in subtype proportions.”

Do not call T cells functionally exhausted or predict response to checkpoint therapy. The GO term includes regulatory genes, including an inhibitory receptor; its positive enrichment does not mean all genes activate T cells. Reactome PD-1 signaling supplies biological context here; it was not tested as a Reactome enrichment result in this pipeline.

### ACTA2: contractile stromal expression

ACTA2 encodes smooth-muscle alpha-actin, a component of the contractile apparatus. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene?cmd=Retrieve&list_uids=59)

Within fibroblast-labelled cells, ACTA2 rises: log2FC +2.3766, gene FDR 6.87 × 10⁻¹³. Detection is 91.1% in HCC2 versus 65.8% in HCC1. It is in the leading edge of **muscle contraction** (GO:0006936), NES +2.0847, pathway FDR 1.05 × 10⁻⁷. Other leading-edge genes include MYH11, MYL9, CNN1 and MYLK.

Suggested spoken explanation: “ACTA2 and the contraction-related enrichment support a contractile stromal expression pattern. A possible explanation is a different balance of myofibroblast, stellate, pericyte or smooth-muscle-like populations. I would verify those identities before interpreting this as fibroblast activation.”

These are transcript-level hypotheses. They do not establish tissue contraction, cancer-associated fibroblast function or increased fibrosis. The marker combination also warrants checking whether the broad fibroblast label contains vascular smooth-muscle or pericyte-like cells.

### COL3A1: matrix-associated expression decreases

COL3A1 encodes the alpha-1 chain of type III fibrillar collagen, a connective-tissue matrix component. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/1281)

Within fibroblast-labelled cells, COL3A1 falls: log2FC −2.3242, gene FDR 4.94 × 10⁻¹⁷. Detection is 58.4% in HCC2 versus 97.5% in HCC1. It is in the leading edge of the negatively enriched **integrin-mediated signaling pathway** GO biological process (GO:0007229), NES −1.788, pathway FDR 0.003975. It also occurs in the leading edge of the negatively enriched **collagen trimer** cellular-component term (NES −1.742, FDR 0.01180), which describes a structure rather than a signaling pathway.

Suggested spoken explanation: “Not all stromal markers increase. COL3A1 decreases while ACTA2 increases, which suggests a difference in matrix-related and contractile programs or in the stromal subtypes recovered. This is more informative than a blanket claim of increased fibrosis.”

Lower RNA does not establish lower collagen deposition or less fibrosis. Protein abundance, extracellular-matrix persistence and tissue histology were not measured. Nor does the gene alone prove lower integrin signaling activity.

## Why these examples are preferable to ALB/APOA genes for the main biological slide

The old deck used pooled liver-function genes to infer malignant hepatocyte dedifferentiation. The revised data contain only one HCC2 hepatocyte-labelled cell, and widespread ALB expression in HCC1 non-hepatocyte populations remains a technical concern. These three examples use within-cell-type contrasts and explicit links to the actual GSEA results. Subtype composition, sample effects, annotation and contamination remain possible explanations.

The T-cell comparison uses 3,740 HCC2 and 2,469 HCC1 cells; the fibroblast comparison uses 101 HCC2 and 281 HCC1 cells. These cells do not constitute independent donor replicates.

## Interview questions

- **How did you select the genes?** They meet the DE thresholds and illustrate interpretable immune/stromal patterns, with membership verified in significant GSEA leading edges. They are illustrative choices, not an unbiased summary of all results.
- **Does gene-set membership show the gene drives the process?** No. Leading-edge membership links its rank to the enrichment statistic, not causality.
- **Does ACTA2 up and COL3A1 down contradict each other?** No. Contractile and matrix programs differ, and a broad annotation can mix stromal subtypes.
- **Why not claim PD-1 pathway activation?** RNA expression does not measure ligand binding, receptor phosphorylation or downstream inhibitory signaling.
- **What would validate the hypotheses?** Replicated donors, subtype/marker review, PD-1 protein and functional assays, and stromal protein/histology measurements.

Exact gene values and enrichment links are saved in `three_gene_pathway_examples.csv`. No PPTX content has been edited at this stage.
