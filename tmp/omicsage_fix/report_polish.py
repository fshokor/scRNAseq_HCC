from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'reports/templates/spatial/spatial_downstream_report.py';s=p.read_text();s=s.replace('pvals_full = ligrec_data.get("pvalues")','pvals_full = ligrec_data.get("qvalues")').replace(r'$-\log_{10}(p)$',r'$-\log_{10}(q)$').replace('LR pairs  (p < {alpha})','LR pairs  (BH q < {alpha})').replace('ranked by &minus;log&#8321;&#8320;(p)','ranked by &minus;log&#8321;&#8320;(BH q)');p.write_text(s)
p=root/'docs/SPATIAL_AUDIT_FIXES.md';s=p.read_text().replace('279 passed','280 passed').replace('A further ligand–receptor spatial-filter regression is included.','This includes the ligand–receptor spatial-filter and optional-MT regressions.');p.write_text(s)
print('Interaction plots use BH q-values consistently.')
