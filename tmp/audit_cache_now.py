from pathlib import Path
import json,yaml
from pipeline.modules.scripts.spatial.checkpoint_cache import OUTPUTS,signature,digest,predecessor,valid
root=Path('/home/shoko/OmicSage')
cfg=yaml.safe_load((root/'config/runs/kuppe_heart_verified.yaml').read_text())
for step,name in OUTPUTS.items():
 if step=='impute':continue
 output=root/cfg['paths']['output_dir']/name
 saved=json.loads(Path(str(output)+'.cache.json').read_text())
 source=predecessor(step,cfg,output)[1]
 print(step,{'signature_matches':saved['signature']==signature(step,source,cfg),
             'output_matches':saved['output_sha256']==digest(output),
             'chain_valid':valid(step,source,cfg,output)},flush=True)
 print('output stat', output.stat().st_size,output.stat().st_mtime,flush=True)
