from pathlib import Path
root=Path(__file__).parent
p=root/'spatial/spatial_downstream.py';s=p.read_text()
needle='''            result["qvalues"] = pd.DataFrame(adjusted, index=pvalues.index, columns=pvalues.columns)
            # Expression-based test: graph adjacency is not used by sq.gr.ligrec.
            result["spatially_filtered"] = False'''
replacement='''            if "spatial_connectivities" not in adata.obsp:
                build_section_graph(adata)
            assert_section_graph(adata)
            graph = adata.obsp["spatial_connectivities"].tocoo()
            labels = adata.obs[dominant_celltype_key].astype(str).to_numpy()
            contacts = set(zip(labels[graph.row], labels[graph.col]))
            contacts |= {(b, a) for a, b in contacts}
            supported = np.array([tuple(pair) in contacts for pair in pvalues.columns], dtype=bool)
            adjusted[:, ~supported] = np.nan
            result["qvalues"] = pd.DataFrame(adjusted, index=pvalues.index, columns=pvalues.columns)
            result["spatially_filtered"] = True
            result["spatial_filter"] = "label-group adjacency, not proven cell communication"'''
assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
p=root/'templates/spatial_downstream_report.py';s=p.read_text();s=s.replace('This expression-based test does not establish spatial contact or signalling.','BH-adjusted hypotheses are retained only for label groups adjacent in the section-specific graph; this does not prove cell contact or signalling.');p.write_text(s)
p=root/'templates/spatial_impute_report.py';s=p.read_text();start=s.index('    # Log-normalise measured counts');end=s.index('    rho, pval',start)
s=s[:start]+'''    # Use the existing normalized X. Renormalizing only 50 sampled genes
    # changes library sizes and would log-transform an already logged matrix.
    x_arr = adata[:, sample_genes].X
    if sp.issparse(x_arr):
        x_arr = x_arr.toarray()
    measured_mean = np.asarray(x_arr, dtype=np.float64).mean(axis=0)
    imputed_mean = imputed[sample_genes].values.mean(axis=0)

'''+s[end:];start=s.index('    quality_note = ""',start);end=s.index('    return (',start)
s=s[:start]+'''    quality_note = (f'<p class="note">Descriptive mean-expression correlation: rho={rho:.3f}. '
                    'These overlapping genes were not held out; this does not validate spatial patterns.</p>')

'''+s[end:];s=s.replace('4. Imputation Validation','4. Projection Diagnostic').replace('imputed gene set. High Spearman correlation indicates that the ','imputed gene set. This diagnostic compares ').replace('imputation model has learned expression means.','expression means; held-out validation remains necessary.')
p.write_text(s)
# Validate explicitly requested section column at ingest, before any unsafe graph.
p=root/'spatial/spatial_ingest.py';s=p.read_text();needle='    resolved_library_key = library_key or _detect_library_key(adata)';assert needle in s
s=s.replace(needle,needle+'\n    if resolved_library_key and resolved_library_key not in adata.obs:\n        raise ValueError(f"library_key {resolved_library_key!r} is missing from obs")');p.write_text(s)
for p in root.rglob('*.py'):
 if p.name not in ('edit.py','finish.py','last.py'):
  p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
