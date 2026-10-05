from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'pipeline/modules/scripts/spatial/spatial_qc.py';s=p.read_text();s=s.replace('max_mt_pct: float = 20.0','max_mt_pct: Optional[float] = 20.0').replace('Maximum mitochondrial gene expression percentage per spot.','Maximum mitochondrial percentage; None records MT QC without filtering.')
s=s.replace('high_mt = int((adata.obs["pct_counts_mt"] > max_mt_pct).sum())','high_mt = int((adata.obs["pct_counts_mt"] > max_mt_pct).sum()) if max_mt_pct is not None and n_mt_genes else 0')
s=s.replace('& (adata.obs["pct_counts_mt"] <= max_mt_pct)','& ((adata.obs["pct_counts_mt"] <= max_mt_pct) if max_mt_pct is not None and n_mt_genes else True)')
s=s.replace('"mt_qc_available": n_mt_genes > 0,','"mt_qc_available": n_mt_genes > 0,\n            "mt_filter_applied": n_mt_genes > 0 and max_mt_pct is not None,');p.write_text(s)
p=root/'reports/templates/spatial/spatial_qc_report.py';s=p.read_text();needle='    stat_rows = ""';s=s.replace(needle,'    if mt_available and params.get("max_mt_pct") is None:\n        mt_note = \'<p class="note">MT percentages measured; mitochondrial filtering intentionally disabled pending tissue-specific threshold review.</p>\'\n'+needle);p.write_text(s)
# Regression tests reflect corrected contracts rather than the former errors.
p=root/'tests/test_spatial_deconvolve.py';s=p.read_text();s=s.replace('def test_h5ad_strips_mt_genes','def test_h5ad_preserves_mt_genes_for_qc').replace('_load_h5ad must strip MT- genes into obsm[\'MT\'].','_load_h5ad retains mitochondrial features and raw counts for QC.')
s=s.replace('        assert "MT" in out.obsm\n        assert out.n_vars == n_other\n        assert out.obsm["MT"].shape == (10, n_mt)','        assert out.n_vars == n_other + n_mt\n        assert sum(out.var_names.str.startswith("MT-")) == n_mt\n        np.testing.assert_array_equal(out.layers["counts"].toarray(), X.toarray())');p.write_text(s)
p=root/'tests/test_spatial_impute.py';s=p.read_text().replace('assert "Imputation Validation" in content or "Spearman" in content','assert "Projection Diagnostic" in content\n        assert "held-out validation" in content');p.write_text(s)
p=root/'tests/test_spatial_audit_regressions.py';s=p.read_text();s+='''
def test_optional_mt_filter_still_records_percentages():
 a=ad.AnnData(sp.csr_matrix(np.array([[10,90]],dtype=float)))
 a.var_names=['GENE','MT-CO1'];a.obsm['spatial']=np.zeros((1,2))
 out,meta=spatial_qc(a,min_counts=0,min_genes=0,max_mt_pct=None)
 assert out.n_obs==1 and out.obs['pct_counts_mt'].iloc[0]==90
 assert not meta['outputs']['mt_filter_applied']
''';p.write_text(s)
for name in ['kuppe_heart.yaml','kuppe_heart_verified.yaml']:
 p=root/'config/runs'/name;s=p.read_text();s=s.replace('max_mt_pct: 20.0','max_mt_pct: null  # Measure MT%; review tissue-specific cutoff before filtering.');p.write_text(s)
print('Optional MT QC and corrected regression contracts applied')
