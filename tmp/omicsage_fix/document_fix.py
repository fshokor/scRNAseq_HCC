from pathlib import Path
import json
import h5py
import numpy as np
root=Path('/home/shoko/OmicSage')
output=root/'data/processed/kuppe_heart_verified'
with h5py.File(output/'03_reduced.h5ad') as f:
 c=f['obs/patient_region_id'];labels=c['categories'].asstr()[()][c['codes'][()]]
 g=f['obsp/spatial_connectivities'];idx=g['indices'][()];rows=np.repeat(np.arange(len(labels)),np.diff(g['indptr'][()]))
 counts=f['obs/pct_counts_mt'][()]
 result={'n_spots':len(labels),'cross_section_entries':int(np.sum(labels[rows]!=labels[idx])),
         'stored_adjacency_entries':len(idx),'n_genes':len(f['var/gene_ids']),
         'median_mt_pct':float(np.median(counts)),'mt_filter_applied':False}
 assert result['cross_section_entries']==0
with h5py.File(output/'04_clustered.h5ad') as f:
 result['n_expression_clusters']=int(f['uns/omicsage_spatial_cluster/outputs/n_clusters'][()])
 result['n_svg_fdr05']=int(f['uns/omicsage_spatial_cluster/outputs/n_significant_fdr05'][()])
(root/'docs/spatial_pipeline_fix_verification.json').write_text(json.dumps(result,indent=2))
text='''# Spatial audit fixes and verified rerun

Physical neighbour graphs are now built separately by tissue section and checked for cross-section entries. Global and type-enriched Moran tests reject old unsafe graphs. Neighbourhood permutations respect the library grouping; co-occurrence results are stored per library, not pooled across coordinate systems.

Ingestion preserves mitochondrial features; QC identifies them through feature_name even with Ensembl indices. The original Kuppe source has 13 MT genes. Restoring these and using the former 20% cutoff leaves only 250/11,725 spots. Kuppe configs now explicitly set max_mt_pct: null: percentages are measured, filtering is deferred for tissue-specific review. This is a benchmark setting, not a declaration that every retained spot is healthy. Counts/gene filters still apply. NNLS excludes MT features at its own fitting stage.

Cache sidecars fingerprint input contents, references, code and parameters, recursively checking predecessors and checkpoint contents. Legacy outputs without sidecars are unverified and rerun. The combined report includes only current verified checkpoint reports. Partial reruns require verified predecessors.

Marker ranking selects positive correlations and excludes constant abundance/expression. NNLS now records relative fitting residuals, signature rank and zero-weight types; proportions remain unvalidated until reviewed. Tangram source-cell scores are never assigned to spatial spots; unavailable per-spot scores are reported as N/A. Mean-expression diagnostics no longer renormalize a small sampled gene subset or imply held-out validation. Ligand–receptor q-values use global BH correction and retain only adjacent label groups from the section-specific graph. These are expression hypotheses, not proven signalling. Moran-ranked pathway NES is labelled spatial autocorrelation rather than activation/downregulation.

## Validation

- Affected original tests plus new graph/cache/QC/marker regressions: 279 passed, one optional test skipped. A further ligand–receptor spatial-filter regression is included.
- Full ingestion→QC→reduction→clustering rerun completed into separate data/processed/kuppe_heart_verified and reports/kuppe_heart_verified directories. Original kuppe_heart analysis outputs were preserved.
- Corrected counts/graph metrics are in spatial_pipeline_fix_verification.json. The rerun contains 11,719 spots, six expression clusters and 673 reported HVG spatial-autocorrelation hits. These are exploratory outputs, not independently validated biological findings; donor effects, MT-associated variation and histology still need assessment.
- Deconvolution/downstream/Tangram have not yet been rerun for this corrected chain. In particular, the old NNLS zero-weight results have not been scientifically resolved just by adding diagnostics. Tangram held-out gene/pattern validation remains future analysis.

## Continue the corrected case study

From the OmicSage root in the existing omicsage environment:

```bash
python run_spatial_pipeline.py --config config/runs/kuppe_heart_verified.yaml --from-step deconvolve
```

The verified config protects the original outputs. Review the QC report and MT distribution before selecting a final tissue-specific mitochondrial cutoff. Parameter changes invalidate the appropriate checkpoint chain; rebuilding upstream steps may be required.

Logs: logs/spatial_fix_tests.log and logs/kuppe_heart_verified_core.log. Main report: reports/kuppe_heart_verified/00_spatial_combined_report.html.
'''
(root/'docs/SPATIAL_AUDIT_FIXES.md').write_text(text)
print(json.dumps(result))
