from pathlib import Path
import json
import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from pipeline.modules.scripts.spatial.spatial_graph import build_section_graph, assert_section_graph
root=Path('/home/shoko/OmicSage')
config=root/'config/runs/kuppe_heart.yaml'
s=config.read_text().replace('library_key: "sample_name"','library_key: "patient_region_id"');config.write_text(s)
# Verify section graphs at full spot count without loading expression data.
with h5py.File(root/'data/processed/kuppe_heart/02_qc.h5ad') as f:
 obs=f['obs']; col=obs['patient_region_id']; names=col['categories'].asstr()[()]; labels=names[col['codes'][()]]
 coords=f['obsm/spatial'][()]
a=ad.AnnData(X=sp.csr_matrix((len(labels),0)));a.obs['patient_region_id']=labels;a.obsm['spatial']=coords
a.uns['spatial']={'visium':{}}
build_section_graph(a, n_neighbors=6, coord_type='grid');assert_section_graph(a)
graph=a.obsp['spatial_connectivities'].tocoo()
with h5py.File(root/'data/benchmark/kuppe_visium_human_heart_2022_control.h5ad') as f:
 var=f['var']; index=var.attrs['_index']; symbols=var[index].asstr()[()]
 n_mt=sum(str(x).startswith('MT-') for x in symbols)
result={'n_spots':a.n_obs,'n_sections':len(np.unique(labels)),'stored_adjacency_entries':int(graph.nnz),
        'cross_section_entries':int(np.sum(labels[graph.row]!=labels[graph.col])),
        'mt_genes_in_source':int(n_mt)}
out=root/'docs/spatial_pipeline_fix_verification.json';out.write_text(json.dumps(result,indent=2))
cfg=yaml.safe_load(config.read_text());cfg['dataset_id']='kuppe_heart_verified'
cfg['paths']['output_dir']='data/processed/kuppe_heart_verified'
cfg['paths']['reports_dir']='reports/kuppe_heart_verified'
(root/'config/runs/kuppe_heart_verified.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
print(json.dumps(result))
