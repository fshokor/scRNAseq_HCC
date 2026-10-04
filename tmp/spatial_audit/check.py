import h5py, numpy as np, json
from pathlib import Path
from collections import Counter
base=Path('/home/shoko/OmicSage/data/processed/kuppe_heart')
def val(o):
 if isinstance(o,h5py.Dataset):
  a=o[()]
  if isinstance(a,bytes):return a.decode()
  if isinstance(a,np.ndarray):return [x.decode() if isinstance(x,bytes) else x.item() if isinstance(x,np.generic) else x for x in a.flatten()] if a.size<60 else {'shape':list(a.shape)}
  return a.item() if isinstance(a,np.generic) else a
 return {k:val(o[k]) for k in o}
def col(o):
 if isinstance(o,h5py.Dataset):return o.asstr()[()] if o.dtype.kind in 'OS' else o[()]
 return col(o['categories'])[o['codes'][()]]
for filename in ['01_ingested.h5ad','03_reduced.h5ad','04_clustered.h5ad','05_deconvolved.h5ad','06_downstream.h5ad','07_imputed.h5ad']:
 with h5py.File(base/filename) as f:
  print('\nFILE',filename, 'obs',len(f['obs/_index']), 'layers',list(f['layers']))
  for k in f['uns']:
   if k.startswith('omicsage'):print(k,json.dumps(val(f['uns'][k]),default=str))
  if filename=='01_ingested.h5ad':
   print('obs columns',list(f['obs']),'var columns',list(f['var']))
   for k in ['patient','patient_region_id','sample_name','condition']:
    if k in f['obs']:print(k,dict(Counter(col(f['obs'][k]))))
   for k in ['_index','feature_name']:
    if k in f['var']:print('mt matches',k,sum(str(x).startswith('MT-') for x in col(f['var'][k])))
   print('X first values',f['X/data'][:12])
  if filename in ['03_reduced.h5ad','06_downstream.h5ad']:
   ob=f['obs']; key=next((k for k in ['patient_region_id','sample_name'] if k in ob),None)
   if key:
    labels=col(ob[key]);g=f['obsp/spatial_connectivities']; ind=g['indices'][()]; ptr=g['indptr'][()];rows=np.repeat(np.arange(len(labels)),np.diff(ptr)); print('graph_cross_library_entries',int(np.sum(labels[rows]!=labels[ind])),'total',len(ind))
  if filename=='05_deconvolved.h5ad':
   for k in ['Adipocyte','Cardiomyocyte','Cycling cells','Endothelial','Fibroblast','Lymphoid','Mast','Myeloid','Neuronal','Pericyte','vSMCs']:
    if k in f['obs']:
     a=col(f['obs'][k]); print(k,'mean',float(np.mean(a)),'nonzero',int(np.sum(a>0)))
   print('dominant',dict(Counter(col(f['obs/dominant_cell_type']))))
