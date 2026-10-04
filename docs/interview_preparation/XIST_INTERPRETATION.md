# XIST interpretation and mapping warning

The current pooled HCC2-versus-HCC1 contrast reports XIST log2FC +15.0096, with an adjusted p-value stored as zero because of numerical underflow. XIST is detected in 67.48% of retained HCC2 cells and 0.0222% of HCC1 cells. It is significantly higher in every broad cell population with a completed comparison.

| Broad population | HCC2 detection | HCC1 detection |
|---|---:|---:|
| T cells | 64.7% | 0% |
| B cells | 74.0% | 0% |
| Fibroblasts | 97.0% | 0% |
| Endothelial cells | 98.4% | 0% |
| Macrophages | 100% | 0.0143% |
| Monocytes | 70.9% | 0% |

Cell-type log2FC estimates of approximately +31 to +33 in several populations arise with zero reference detection and should not be presented as precise biological fold changes.

XIST is a long noncoding RNA involved in X-chromosome inactivation. The broadly shared sample difference is compatible with donor sex/X-chromosome context or another sample-level effect. It does not establish a cancer-cell-specific mechanism. Donor sex is not verified here. Check authoritative donor metadata and supporting Y-chromosome gene expression before interpreting XIST as tumour-associated dysregulation. Those checks can support a hypothesis but do not replace provenance. [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/7503)

The current GSEA rank mapping includes XIST as Entrez 7503. No saved GO/KEGG result contains XIST in its leading edge, regardless of significance. This is not evidence that it lacks biological function or membership in all gene sets; the exports contain leading-edge memberships rather than full set membership.

No XIST drug association is present in the current DGIdb export.

**Additional network issue:** `results/tables/target_prioritisation/pooled/string_mapping.csv` records the input XIST as STRING ID `9606.ENSP00000491215`, preferred name **HNRNPU**. Consequently, the current XIST-labelled connectivity/hub score cannot be attributed to XIST. It represents a mismatched mapped node and must be corrected/reviewed before using that part of the network. This audit identifies the issue but does not modify the analysis or rerun rankings.

Recommended presentation wording: “XIST is strongly higher throughout HCC2 cell populations. Because XIST regulates X-chromosome inactivation, this sample-wide difference raises a donor-sex or sample-context confounding concern rather than establishing a cancer-specific biomarker.”

Evidence: current DE exports; all saved contrast-specific GSEA exports; pooled rank mapping, STRING mapping and DGIdb evidence. Exact expression results are in `xist_evidence.json`.
