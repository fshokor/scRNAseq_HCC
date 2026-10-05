from pathlib import Path
import yaml
import h5py
from pipeline.modules.scripts.spatial.checkpoint_cache import OUTPUTS, predecessor, valid, require_predecessor

root = Path('/home/shoko/OmicSage')
cfg = yaml.safe_load((root / 'config/runs/kuppe_heart_verified.yaml').read_text())
directory = root / cfg['paths']['output_dir']
for step in ('ingest', 'qc', 'reduce', 'cluster', 'deconvolve'):
    output = directory / OUTPUTS[step]
    source = predecessor(step, cfg, output)[1]
    okay = valid(step, source, cfg, output)
    print(f'{step}: chain_valid={okay}', flush=True)
    assert okay, step
with h5py.File(directory / OUTPUTS['ingest'], 'r') as handle:
    print('Ingestion HDF5 readable:', list(handle.keys()), flush=True)
require_predecessor('downstream', cfg, directory / OUTPUTS['downstream'])
print('Downstream predecessor check: PASSED', flush=True)
