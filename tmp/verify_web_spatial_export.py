from pathlib import Path
import yaml
from ui.config_builder import build_spatial
from ui.config_io import parse_config_into_state
from pipeline.modules.scripts.spatial.checkpoint_cache import require_predecessor, OUTPUTS
root=Path('/home/shoko/OmicSage')
original=yaml.safe_load((root/'config/runs/kuppe_heart_verified.yaml').read_text())
recent=yaml.safe_load(Path('/tmp/omicsage_rni2hfzr.yaml').read_text())
parsed=parse_config_into_state(recent,'web.yaml')
exported=build_spatial(parsed['dataset_name'],parsed['data_path'],parsed['rna_path'],parsed['organism'],parsed['selected_steps'],parsed['step_params'],dataset_id=parsed['dataset_id'])
assert exported['spatial']==original['spatial'], [(k, exported['spatial'].get(k), v) for k,v in original['spatial'].items() if exported['spatial'].get(k)!=v]
require_predecessor('downstream',exported,root/exported['paths']['output_dir']/OUTPUTS['downstream'])
print('Corrected actual web export matches terminal spatial configuration.')
print('Downstream predecessor validation with web configuration: PASSED.')
