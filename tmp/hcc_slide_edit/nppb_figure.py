import anndata as ad
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
a=ad.read_h5ad('/home/shoko/OmicSage/data/processed/kuppe_heart_verified/06_downstream.h5ad',backed='r')
pos=np.flatnonzero(a.var.feature_name.astype(str).values=='NPPB')[0]
v=a.X[:,pos:pos+1].toarray().ravel()
fig,axes=plt.subplots(1,4,figsize=(14,3.3),facecolor='#0B1822')
for ax,lib in zip(axes,['control_P1','control_P17','control_P7','control_P8']):
 mask=np.asarray(a.obs.patient_region_id==lib);xy=a.obsm['spatial'][mask]
 ax.set_facecolor('#0B1822')
 im=ax.scatter(xy[:,0],xy[:,1],c=v[mask],s=5,cmap='viridis',vmin=0,vmax=np.quantile(v,.99))
 ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
 ax.set_title(lib.replace('control_','')+f'  |  {(v[mask]>0).mean()*100:.1f}% detected',fontsize=15,color='#F4F8FB',pad=10)
fig.subplots_adjust(left=.01,right=.93,top=.87,bottom=.05,wspace=.12)
cax=fig.add_axes([.95,.18,.012,.57]); cb=fig.colorbar(im,cax=cax)
cb.ax.tick_params(colors='#A9BAC6',labelsize=10);cb.set_label('Log-normalized expression',color='#A9BAC6',fontsize=11)
fig.savefig('/mnt/c/Users/shoko/OneDrive/Desktop/project/HCC_DD/tmp/hcc_slide_edit/nppb_sections.png',dpi=180,facecolor=fig.get_facecolor())
