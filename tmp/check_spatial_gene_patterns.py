from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
a=ad.read_h5ad('/home/shoko/OmicSage/data/processed/kuppe_heart_verified/06_downstream.h5ad',backed='r')
genes=['MYL2','NPPB','ANKRD1','LYVE1','C1QA']
symbols=a.var.feature_name.astype(str)
pos=[np.flatnonzero(symbols.values==g)[0] for g in genes]
order=np.argsort(pos); xx=a.X[:,np.array(pos)[order]]
xx=xx.toarray() if sp.issparse(xx) else np.asarray(xx)
xx=xx[:,np.argsort(order)]
libraries=['control_P1','control_P17','control_P7','control_P8']
w=a.obsp['spatial_connectivities'].tocsr()
def moran(v,ww):
 sums=np.asarray(ww.sum(axis=1)).ravel(); ww=sp.diags(1/np.maximum(sums,1))@ww
 z=v-v.mean()
 return len(v)/ww.sum()*float(z@(ww@z))/float(z@z) if z@z>0 else np.nan
rows=[]
for j,g in enumerate(genes):
 for lib in libraries:
  mask=np.asarray(a.obs.patient_region_id==lib)
  v=xx[mask,j];ww=w[mask][:,mask]
  rows.append({'gene':g,'section':lib,'mean_log_expression':float(v.mean()),'fraction_detected':float((v>0).mean()),'within_section_Moran_I':moran(v,ww)})
print(pd.DataFrame(rows).to_string(index=False))
out=Path('/mnt/c/Users/shoko/OneDrive/Desktop/project/HCC_DD/docs/interview_preparation/spatial_interpretation');out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'selected_genes_by_section.csv',index=False)
fig,axes=plt.subplots(3,4,figsize=(12,9))
for r,g in enumerate(genes[:3]):
 j=genes.index(g);vmax=np.quantile(xx[:,j],.99)
 for c,lib in enumerate(libraries):
  mask=np.asarray(a.obs.patient_region_id==lib); xy=a.obsm['spatial'][mask];ax=axes[r,c]
  im=ax.scatter(xy[:,0],xy[:,1],c=xx[mask,j],s=3,cmap='viridis',vmin=0,vmax=vmax)
  ax.set_aspect('equal');ax.invert_yaxis();ax.set_xticks([]);ax.set_yticks([])
  ax.set_title(f'{g} — {lib.replace("control_", "")}',fontsize=10)
 fig.colorbar(im,ax=axes[r,:],shrink=.7,label='Log-normalized expression')
fig.suptitle('Control-heart spatial expression — shared colour scale within each gene',fontsize=13)
fig.savefig(out/'cardiac_gene_spatial_patterns.png',dpi=180,bbox_inches='tight')
print('FIGURE',out/'cardiac_gene_spatial_patterns.png')
