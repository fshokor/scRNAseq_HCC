from pathlib import Path
root=Path(__file__).parent
def edit(name,old,new):
 p=root/name;s=p.read_text();assert old in s,(name,old[:70]);p.write_text(s.replace(old,new))
edit('run_spatial_pipeline.py','        require_predecessor("impute", cfg, out_path)\n    adata = sc.read_h5ad(input_path)','        adata = sc.read_h5ad(input_path)')
edit('run_spatial_pipeline.py','    impute_cfg   = cfg.get("spatial", {}).get("impute", {})','    require_predecessor("impute", cfg, out_path)\n    impute_cfg   = cfg.get("spatial", {}).get("impute", {})')
edit('run_spatial_pipeline.py','        n_top_genes=reduce_cfg.get("n_top_genes",  3000),','        library_key=reduce_cfg.get("library_key"),\n        n_top_genes=reduce_cfg.get("n_top_genes",  3000),')
# Keep mitochondrial features until QC; model-specific exclusions happen later.
p=root/'spatial/spatial_ingest.py';s=p.read_text();start=s.index('    # 3. Strip mitochondrial genes');end=s.index('    # 4. Strip alpha',start)
s=s[:start]+'''    # 3. Preserve mitochondrial genes for QC. Do not remove them at ingest.

'''+s[end:];s=s.replace('MT genes stripped','MT genes retained for QC');p.write_text(s)
edit('spatial/spatial_qc.py','    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)','    symbols = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else adata.var_names\n    adata.var["mt"] = np.asarray(symbols.str.startswith(mt_prefix))')
edit('spatial/spatial_qc.py','"mt_prefix_zero_match": n_mt_genes == 0,','"mt_prefix_zero_match": n_mt_genes == 0,\n            "mt_qc_available": n_mt_genes > 0,')
edit('templates/spatial_qc_report.py','    stat_rows = ""','    mt_available = outputs.get("mt_qc_available", outputs.get("n_mt_genes_detected", 0) > 0)\n    mt_note = "" if mt_available else \'<p class="note">MT QC unavailable: no mitochondrial features matched. Zero reported percentages do not demonstrate good tissue quality.</p>\'\n    stat_rows = ""')
edit('templates/spatial_qc_report.py','        s = stats.get(col, {})','        if col == "pct_counts_mt" and not mt_available:\n            stat_rows += \'<tr><td>MT gene %</td><td colspan="5">Unavailable</td></tr>\'\n            continue\n        s = stats.get(col, {})')
edit('templates/spatial_qc_report.py','      <div class="stat-grid">{stat_cards}</div>','      <div class="stat-grid">{stat_cards}</div>\n      {mt_note}')
# NNLS memory-safe QC diagnostics; do not substitute another method silently.
edit('spatial/spatial_deconvolve.py','    dom_idx = np.argmax(proportions, axis=1)','    zero_types = [ct for i, ct in enumerate(cell_type_names) if not np.any(proportions[:, i] > 0)]\n    if zero_types:\n        logger.warning("Zero fitted abundance for %s; validate signatures before interpretation", zero_types)\n    dom_idx = np.argmax(proportions, axis=1)')
edit('spatial/spatial_deconvolve.py','            "n_spots": int(adata.n_obs),','            "n_spots": int(adata.n_obs),\n            "zero_weight_cell_types": zero_types,\n            "nnls_diagnostics": adata.uns.get("nnls_diagnostics", {}),\n            "fit_validated": False,')
edit('spatial/spatial_deconvolve.py','"per_sample": per_sample,','"per_sample": per_sample if method == "cell2location" else False,')
edit('spatial/spatial_deconvolve.py','    shared = [g for g in ref_adata.var_names if g in spatial_genes]','    symbols = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else adata.var_names\n    mt_genes = set(adata.var_names[np.asarray(symbols.str.startswith("MT-"))])\n    shared = [g for g in ref_adata.var_names if g in spatial_genes and g not in mt_genes]')
edit('spatial/spatial_deconvolve.py','        p, _ = nnls(W, x, maxiter=200)\n        s = p.sum()\n        return p / s if s > 0 else p','        p, residual = nnls(W, x, maxiter=200)\n        s = p.sum()\n        return (p / s if s > 0 else p), residual / max(float(np.linalg.norm(x)), 1e-12)')
edit('spatial/spatial_deconvolve.py','    proportions = np.asarray(\n        Parallel(n_jobs=n_jobs, prefer="threads")(\n            delayed(_solve)(st_norm[i]) for i in range(st_norm.shape[0])\n        ),\n        dtype=np.float32,\n    )','    fitted = Parallel(n_jobs=n_jobs, prefer="threads")(\n        delayed(_solve)(st_norm[i]) for i in range(st_norm.shape[0]))\n    proportions = np.asarray([result[0] for result in fitted], dtype=np.float32)\n    residuals = np.asarray([result[1] for result in fitted], dtype=np.float32)\n    adata.obs["nnls_relative_residual"] = residuals\n    adata.uns["nnls_diagnostics"] = {"median_relative_residual": float(np.median(residuals)),\n        "max_relative_residual": float(np.max(residuals)),\n        "signature_rank": int(np.linalg.matrix_rank(W)), "n_signatures": len(cell_types),\n        "excluded_mt_genes": len(mt_genes)}')
# Surface diagnostic rather than only logging it.
edit('templates/spatial_deconvolve_report.py','    n_ct        = outputs.get("n_cell_types", 0)','    n_ct        = outputs.get("n_cell_types", 0)\n    zeros = outputs.get("zero_weight_cell_types", [])\n    diagnostic = f"NNLS fit diagnostics: {outputs.get(\'nnls_diagnostics\', {})}; zero-weight types: {\', \'.join(zeros) or \'none\'}. Proportions are fitted weights, not validated cell counts."')
edit('templates/spatial_deconvolve_report.py','      <h2>Run Summary</h2>','      <h2>Run Summary</h2>\n      <p class="note">{diagnostic}</p>')
# Do not project mean-correlation diagnostics as validation of spatial patterns.
p=root/'templates/spatial_impute_report.py';s=p.read_text();s=s.replace('High Spearman correlation indicates that the imputation model has learned','Mean-expression correlation is descriptive, not held-out spatial validation; it compares').replace('biologically realistic gene relationships.','expression means.');p.write_text(s)
# The broad configuration does not need to invalidate ingest for downstream-only changes.
edit('spatial/checkpoint_cache.py',"own = spatial if step == 'ingest' else spatial.get(step, {})","own = ({k:v for k,v in spatial.items() if k in ('source','spatial_type','counts_file','library_id','library_key','load_images','ingest')}\n           if step == 'ingest' else spatial.get(step, {}))")
print('Follow-up fixes staged')
