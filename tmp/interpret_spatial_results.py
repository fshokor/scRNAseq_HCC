import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile,xml.etree.ElementTree as ET
a=ad.read_h5ad('/home/shoko/OmicSage/data/processed/kuppe_heart_verified/06_downstream.h5ad',backed='r')
print('SECTION_COUNTS',a.obs.groupby('patient_region_id',observed=True).size().to_dict())
print('CLUSTER_BY_SECTION\n',pd.crosstab(a.obs.spatial_cluster,a.obs.patient_region_id).to_string())
print('DOMINANT_BY_SECTION\n',pd.crosstab(a.obs.dominant_cell_type,a.obs.patient_region_id).to_string())
types=['Adipocyte','Cardiomyocyte','Cycling cells','Endothelial','Fibroblast','Lymphoid','Mast','Myeloid','Neuronal','Pericyte','vSMCs']
print('MEAN_WEIGHTS',a.obs[types].mean().to_dict())
print('FIT_RESIDUAL',a.obs.nnls_relative_residual.describe().to_dict())
print('MT_BY_SECTION\n',a.obs.groupby('patient_region_id',observed=True).pct_counts_mt.describe().to_string())
print('GRAPH',a.uns['spatial_neighbors'])
t=a.uns['moranI'].copy();t.insert(0,'symbol',a.var.feature_name.reindex(t.index));print('TOP_SVG\n',t.head(25).to_string())
for k,v in a.uns['celltype_marker_genes'].items():print('CORRELATION',k,str(v)[:1400])
print('NH',a.uns['dominant_cell_type_nhood_enrichment'])
lr=a.uns['dominant_cell_type_ligrec'];print('LR_SCHEMA',[(k,np.shape(v)) for k,v in lr.items()]);print('LR_INDEX',str(lr['qvalues_index'][:25]));print('LR_COLUMNS',lr['qvalues_columns']);print('LR_FILTER',lr.get('spatial_filter'))
print('GSEA_COLS',a.uns['svg_gsea'].columns.tolist())
print('REGION_SIZES',a.obs.region_cluster.value_counts().describe().to_dict())
deck=Path('/mnt/c/Users/shoko/OneDrive/Desktop/project/HCC_DD/docs/interview_preparation/presentation/Fatima_Shockor_Interview_Updated_HCC_Spatial_Case_Study.pptx')
with zipfile.ZipFile(deck) as z:
 for n in sorted([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')],key=lambda n:int(n.split('slide')[-1].split('.')[0])):
  text=' | '.join(e.text for e in ET.fromstring(z.read(n)).iter() if e.tag.endswith('}t') and e.text)
  print('SLIDE',n,text)
