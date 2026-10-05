from pathlib import Path
root=Path(__file__).parent
def edit(name, old, new):
 p=root/name;s=p.read_text(encoding='utf8');assert old in s,(name,old[:70]);p.write_text(s.replace(old,new),encoding='utf8')
edit('spatial/spatial_reduce.py','import scanpy as sc','import scanpy as sc\nfrom .spatial_graph import build_section_graph')
edit('spatial/spatial_reduce.py','    inplace: bool = False,','    inplace: bool = False,\n    library_key: Optional[str] = None,')
edit('spatial/spatial_reduce.py','    sq.gr.spatial_neighbors(\n        adata,\n        n_neighs=n_neighbors,\n        coord_type=coord_type,\n        key_added="spatial",\n    )','    library_key = build_section_graph(adata, n_neighbors, coord_type, library_key)')
edit('spatial/spatial_reduce.py','"n_neighbors": n_neighbors,','"n_neighbors": n_neighbors,\n            "library_key": library_key or "",')
edit('spatial/spatial_cluster.py','import scanpy as sc','import scanpy as sc\nfrom .spatial_graph import assert_section_graph')
edit('spatial/spatial_cluster.py','    if run_svg and "spatial_connectivities" not in adata.obsp:','    if run_svg and "spatial_connectivities" in adata.obsp:\n        assert_section_graph(adata)\n    if run_svg and "spatial_connectivities" not in adata.obsp:')
edit('spatial/spatial_downstream.py','from __future__ import annotations','from __future__ import annotations\nfrom .spatial_graph import build_section_graph, assert_section_graph, resolve_library_key')
edit('spatial/spatial_downstream.py','            abundance = adata.obs[ct].values.astype(np.float32)','            abundance = adata.obs[ct].values.astype(np.float32)\n            if not np.isfinite(abundance).all() or np.ptp(abundance) == 0:\n                continue')
edit('spatial/spatial_downstream.py','            top_idx = np.argsort(np.abs(corr))[::-1][:n_marker_genes]','            candidates = np.flatnonzero(np.isfinite(corr) & (corr > 0) & (np.ptp(expr_dense, axis=0) > 0))\n            top_idx = candidates[np.argsort(corr[candidates])[::-1]][:n_marker_genes]')
edit('spatial/spatial_downstream.py','            sq.gr.spatial_neighbors(\n                ad_sub, n_neighs=6, coord_type=None, key_added="spatial"\n            )','            build_section_graph(ad_sub, n_neighbors=6)')
edit('spatial/spatial_downstream.py','                sq.gr.spatial_neighbors(\n                    adata, n_neighs=6, coord_type=None, key_added="spatial"\n                )','                build_section_graph(adata, n_neighbors=6)')
edit('spatial/spatial_downstream.py','        sq.gr.nhood_enrichment(\n            adata,','        library_key = assert_section_graph(adata)\n        if library_key:\n            adata.obs[library_key] = adata.obs[library_key].astype("category")\n        sq.gr.nhood_enrichment(\n            adata,\n            library_key=library_key,')
edit('spatial/spatial_downstream.py','        sq.gr.co_occurrence(adata, **kwargs)\n\n        return {"skipped": False}', '''        library_key = resolve_library_key(adata)
        co_key = f"{dominant_celltype_key}_co_occurrence"
        if library_key and adata.obs[library_key].nunique() > 1:
            results = {}
            failures = {}
            adata.uns.pop(co_key, None)  # Never leave a pooled-coordinate result.
            for section, indices in adata.obs.groupby(library_key, observed=True, sort=False).indices.items():
                sub = adata[indices].copy()
                sub.obs[dominant_celltype_key] = sub.obs[dominant_celltype_key].cat.remove_unused_categories()
                try:
                    sq.gr.co_occurrence(sub, **kwargs)
                    results[str(section)] = {**sub.uns[co_key],
                                            "categories": sub.obs[dominant_celltype_key].cat.categories.to_numpy(dtype=str)}
                except Exception as exc:
                    failures[str(section)] = str(exc)
            adata.uns[co_key+"_by_library"] = results
            return {"skipped": not bool(results), "per_library": True, "library_key": library_key,
                    "n_libraries": len(results), "failures": failures}
        sq.gr.co_occurrence(adata, **kwargs)
        adata.uns.pop(co_key+"_by_library", None)
        return {"skipped": False, "per_library": False}''')
# Preserve raw p-values and add global BH q-values, rather than counting raw p as discoveries.
edit('spatial/spatial_downstream.py','        if has_result:\n            _serialize_ligrec_uns(adata, ligrec_key)', '''        if has_result:
            from statsmodels.stats.multitest import multipletests
            result = adata.uns[ligrec_key]
            pvalues = result["pvalues"]
            values = pvalues.to_numpy(dtype=float)
            finite = np.isfinite(values)
            adjusted = np.full(values.shape, np.nan)
            adjusted[finite] = multipletests(values[finite], method="fdr_bh")[1]
            result["qvalues"] = pd.DataFrame(adjusted, index=pvalues.index, columns=pvalues.columns)
            # Expression-based test: graph adjacency is not used by sq.gr.ligrec.
            result["spatially_filtered"] = False
            _serialize_ligrec_uns(adata, ligrec_key)''')
# Cache: all direct runner entry points validate their own dependencies.
p=root/'run_spatial_pipeline.py';s=p.read_text(encoding='utf8')
s=s.replace('import yaml','import yaml\nfrom pipeline.modules.scripts.spatial.checkpoint_cache import valid as cache_valid, record as cache_record, require_predecessor')
import re
for step in ['ingest','qc','reduce','cluster','deconvolve','downstream','impute']:
 pattern=rf'(def run_{step}\(.*?\n)(.*?)(?=\ndef |\nSTEP_RUNNERS)'
 m=re.search(pattern,s,re.S);assert m,step
 body=m.group(2);inp='None' if step=='ingest' else 'input_path'
 body=body.replace('if out_path.exists() and not force:',f'if not force and cache_valid("{step}", {inp}, cfg, out_path):')
 # Cached returns must not restamp; remaining returns record freshly written data.
 body=body.replace('        return out_path\n','        return out_path\n',1)
 lines=body.splitlines();out=[];cached=False
 for line in lines:
  if 'if not force and cache_valid' in line:cached=True
  if line.strip()=='return out_path':
   if cached:cached=False
   else:out.append(' '*(len(line)-len(line.lstrip()))+f'cache_record("{step}", {inp}, cfg, out_path)')
  out.append(line)
 body='\n'.join(out)+'\n'
 if step!='ingest':
  needle='    adata';position=body.index(needle)
  body=body[:position]+f'    require_predecessor("{step}", cfg, out_path)\n'+body[position:]
 s=s[:m.start(2)]+body+s[m.end(2):]
s=s.replace('library_key=spatial_cfg.get("library_key", None),','library_key=spatial_cfg.get("ingest", {}).get("library_key", spatial_cfg.get("library_key")),')
s=s.replace('        n_top_genes=reduce_cfg.get("n_top_genes", 3000),','        library_key=reduce_cfg.get("library_key"),\n        n_top_genes=reduce_cfg.get("n_top_genes", 3000),')
# Full combined report only includes verified current reports.
old='''    generate_spatial_combined_report(
        reports_dir=reports_dir,
        dataset_name=dataset_name,
        output_path=combined_path,
    )'''
assert old in s
s=s.replace(old,'''    import tempfile, shutil
    with tempfile.TemporaryDirectory() as current_reports:
        for report_step, report_name in STEP_REPORT.items():
            output = output_dir / STEP_OUTPUT[report_step]
            source = reports_dir / report_name
            if source.exists() and cache_valid(report_step, resolve_input(report_step, cfg, output_dir), cfg, output):
                shutil.copy2(source, Path(current_reports) / report_name)
        generate_spatial_combined_report(reports_dir=Path(current_reports),
                                         dataset_name=dataset_name, output_path=combined_path)''')
p.write_text(s,encoding='utf8')
# No Tangram source-cell score can be assigned to spots, even when dimensions coincide.
p=root/'spatial/spatial_impute.py';s=p.read_text(encoding='utf8');start=s.index('    # Mapping scores —');end=s.index('    # Store imputed values',start)
s=s[:start]+'''    # Tangram tg_score belongs to source cells/clusters, not spatial spots.
    adata_st.obs.drop(columns=["tangram_mapping_score"], errors="ignore", inplace=True)
    mean_score = float("nan")
    n_poor = -1  # Sentinel: unavailable, never zero successful/failed spots.

'''+s[end:];s=s.replace('"n_poor_spots":       n_poor,','"n_poor_spots":       n_poor,\n            "spot_scores_available": False,')
p.write_text(s,encoding='utf8')
edit('templates/spatial_impute_report.py','    poor_str  = str(n_poor) if method == "tangram" else "N/A"','    poor_str = str(n_poor) if out.get("spot_scores_available", False) and n_poor >= 0 else "N/A"')
edit('templates/spatial_impute_report.py','    if method == "tangram":\n        note = (','    if method == "tangram":\n        note = (') # replaced below with exact block
p=root/'templates/spatial_impute_report.py';s=p.read_text(encoding='utf8');start=s.index('    note = ""');end=s.index('    return (',start)
s=s[:start]+'''    note = '<p class="note">Projected expression is model-derived. Tangram source-cell or cluster scores are not per-spot quality scores. Validate with held-out genes and spatial patterns.</p>'

'''+s[end:];s=s.replace('genes not in the original Visium panel whose spatial expression','projected genes whose spatial expression').replace('High Spearman correlation indicates that the imputation model has learned biologically realistic gene relationships.','Mean-expression correlation is a descriptive diagnostic, not held-out validation or spatial-pattern accuracy.')
p.write_text(s,encoding='utf8')
edit('templates/spatial_downstream_report.py','    if co_key not in adata.uns:\n        return _skip_section','    if co_key+"_by_library" in adata.uns:\n        sections = ", ".join(adata.uns[co_key+"_by_library"])\n        return f\'<section><h2>Spatial Co-occurrence</h2><p>Computed separately by tissue section: {sections}. Coordinate systems were not pooled. Section-specific arrays are stored in the checkpoint.</p></section>\'\n    if co_key not in adata.uns:\n        return _skip_section')
edit('templates/spatial_downstream_report.py','spatially co-localised cell types, using the OmniPath database.','dominant spot-label groups, using the OmniPath database. This expression-based test does not establish spatial contact or signalling.')
edit('templates/spatial_downstream_report.py','Red bars indicate positively enriched pathways; blue = depleted.','Red/blue indicate enrichment toward higher/lower spatial autocorrelation. This ranking does not measure pathway activation or up/down regulation.')
edit('templates/spatial_downstream_report.py','    pvals_df = ligrec_data.get("pvalues")','    pvals_df = ligrec_data.get("qvalues")')
edit('templates/spatial_downstream_report.py','n_sig_001:,} interactions at p&nbsp;&lt;&nbsp;0.001','n_sig_001:,} hypotheses at BH q&nbsp;&lt;&nbsp;0.001')
edit('templates/spatial_downstream_report.py','n_sig_005:,} at p&nbsp;&lt;&nbsp;0.05','n_sig_005:,} at BH q&nbsp;&lt;&nbsp;0.05')
# Summary QC warning is visible even when optional plots fail.
p=root/'templates/spatial_qc_report.py';s=p.read_text(encoding='utf8');loc=s.index('    return',s.index('def _section_summary')) if 'def _section_summary' in s else -1
if loc<0:
 print('QC summary function names require follow-up')
else:
 s=s[:loc]+s[loc:].replace('    return','    return',1)
p.write_text(s,encoding='utf8')
print('Source edits staged')
