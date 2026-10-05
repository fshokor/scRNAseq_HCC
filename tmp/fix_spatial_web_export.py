from pathlib import Path
root=Path('/home/shoko/OmicSage')
p=root/'ui/config_builder.py'
s=p.read_text()
needle='    # None is meaningful here: omitting it would restore the runner\'s 20% default.\n'
replacement='''    # These widget fields belong at the spatial level, not in ingest.
    for key in ("spatial_type", "load_images"):
        cfg["spatial"]["ingest"].pop(key, None)
    # The spatial runner uses explicit n_comps/n_pcs, not this scRNA widget.
    if "reduce" in cfg["spatial"] and isinstance(cfg["spatial"]["reduce"], dict):
        cfg["spatial"]["reduce"].pop("n_pcs_method", None)

'''+needle
assert needle in s
p.write_text(s.replace(needle,replacement))
p=root/'ui/_pages/p2_configure.py'
s=p.read_text(); old='    elif step == "reduce":\n'
assert s.count(old)==1
p.write_text(s.replace(old,'    elif step == "reduce" and modality != "Spatial":\n'))
p=root/'ui/config_io.py'
s=p.read_text(); needle='        result["selected_steps"] = selected\n'
assert needle in s
s=s.replace(needle,'''        # Ingestion widgets edit these top-level spatial options.
        step_params.setdefault("ingest", {}).update({
            "spatial_type": spatial.get("spatial_type", "h5ad"),
            "load_images": spatial.get("load_images", True),
        })
'''+needle,1)
p.write_text(s)
p=root/'tests/test_spatial_ui_roundtrip.py'
s=p.read_text().replace('  for key,value in params.items():assert restored[\'spatial\'][step][key]==value,(step,key)', '  for key,value in params.items():\n   if step=="ingest" and key in ("spatial_type","load_images"):\n    assert restored["spatial"][key]==value\n   else:assert restored["spatial"][step][key]==value,(step,key)')
s+='''

def test_real_spatial_widgets_preserve_ingestion_and_reduce_config():
 from streamlit.testing.v1 import AppTest
 import json
 original={"dataset_id":"heart","spatial":{
  "source":"input.h5ad","spatial_type":"h5ad","load_images":False,
  "ingest":{"library_key":"patient_region_id"},
  "reduce":{"n_top_genes":3000,"n_comps":50,"n_neighbors":6,
            "coord_type":None,"normalize_total":True,"target_sum":10000,
            "log1p":True,"flavor":"seurat"}}}
 parsed=parse_config_into_state(original,"config.yaml")
 at=AppTest.from_string("""
import streamlit as st
from ui._pages.p2_configure import _render_step
from ui.config_builder import build_spatial
params=st.session_state["params"]
for step in ("ingest","reduce"):
 params[step]=_render_step(step,params[step],"Spatial","Human")
st.json(build_spatial("heart","input.h5ad","","Human",["ingest","reduce"],params,dataset_id="heart"))
""")
 at.session_state["params"]=parsed["step_params"]
 at.run()
 assert not at.exception
 exported=json.loads(at.json[0].value)["spatial"]
 for key in ("spatial_type","load_images","ingest","reduce"):
  assert exported[key]==original["spatial"][key],key

def test_old_session_widget_fields_are_removed_from_export():
 cfg=build_spatial("heart","input.h5ad","","Human",["ingest","reduce"],{
  "ingest":{"library_key":"patient_region_id","spatial_type":"h5ad","load_images":True},
  "reduce":{"n_pcs_method":"elbow"}})
 assert cfg["spatial"]["ingest"]=={"library_key":"patient_region_id"}
 assert "n_pcs_method" not in cfg["spatial"]["reduce"]
'''
p.write_text(s)
