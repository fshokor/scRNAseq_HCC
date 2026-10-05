import copy
import yaml
from ui.config_builder import build_spatial
from ui.config_io import parse_config_into_state

def test_loaded_spatial_config_keeps_all_parameters_and_nulls():
 original={'dataset_id':'heart','spatial':{
  'source':'input.h5ad','spatial_type':'h5ad','load_images':True,
  'ingest':{'library_key':'patient_region_id'},
  'qc':{'max_mt_pct':None},
  'reduce':{'coord_type':None,'normalize_total':False,'log1p':False,'flavor':'seurat'},
  'cluster':{'run_svg':False,'svg_n_genes':None,'annotation_map':{'0':'Region'}},
  'deconvolve':{'method':'nnls','library_key':None,'per_sample':True,'batch_size_st':None,'max_epochs_st':25},
  'downstream':{'n_perms_nhood':200,'ligrec_n_perms':250,'svg_n_genes':None,'region_resolution':0.7},
  'impute':{'enabled':True,'cell_type_key':'cell_type_original','max_cells_per_type':150}}}
 parsed=parse_config_into_state(original,'config.yaml')
 before=copy.deepcopy(parsed['step_params'])
 result=build_spatial('heart',parsed['data_path'],parsed['rna_path'],'Human',parsed['selected_steps'],parsed['step_params'],dataset_id='heart')
 restored=yaml.safe_load(yaml.safe_dump(result))
 for step,params in before.items():
  for key,value in params.items():assert restored['spatial'][step][key]==value,(step,key)
 assert before==parsed['step_params']

def test_widget_edits_override_imported_values():
 result=build_spatial('heart','input.h5ad','ref.h5ad','Human',['qc','reduce','cluster'],{
  'qc':{'max_mt_pct':None},'reduce':{'coord_type':'generic','n_comps':25},
  'cluster':{'run_svg':False,'svg_n_genes':1000,'annotation_map':{'1':'Region'}}})
 assert result['spatial']['reduce']['n_comps']==25
 assert result['spatial']['reduce']['coord_type']=='generic'
 assert result['spatial']['cluster']['run_svg'] is False
 assert result['spatial']['cluster']['annotation_map']=={'1':'Region'}
