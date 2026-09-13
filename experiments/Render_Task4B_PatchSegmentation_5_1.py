"""Display-only revision: reveal internal hairpin labels in the frozen patches."""
import json
from pathlib import Path
import numpy as np
from experiments.Task4B_PatchSegmentation_5_1 import sha,write,identity
from experiments.Visualize_Task4B_VelocityCurl_3D import render

out=Path('outputs/mainExp_Task4B_PatchSegmentation_5.1')
report=json.loads((out/'training_report.json').read_text())
assert sha(out/'predictions.npz')==report['predictions_sha256']
assert json.loads((out/'audit.json').read_text())['passed']
manifest=json.loads((out/'manifest.json').read_text())
with np.load(out/'predictions.npz') as d:pred={k:d[k] for k in d.files}
folder=out/'figures_3d_r2';folder.mkdir(exist_ok=True)
images=[]
for code,name in enumerate(('channel','tbl')):
    info=manifest['flows'][name];low=np.array(info['low_xyz']);high=np.array(info['high_xyz'])
    resolution=np.array(info['resolution_xyz']);spacing=(high-low)/resolution
    patches=[p for p in info['patches'] if p['split']=='test' and p['kind']=='hairpin_bbox'][:2]
    for p in patches:
        selected=(pred['volume']==code)&(pred['patch']==p['patch_id'])
        indices=np.column_stack(np.unravel_index(pred['source'][selected],tuple(resolution[::-1])))[:,::-1]
        points=low+(indices+.5)*spacing
        plow=low+np.array(p['low'][::-1])*spacing;phigh=low+np.array(p['high'][::-1])*spacing
        for key,title in [('labels','Proxy GT'),('predictions','FMT prediction')]:
            values=pred[key][selected];valid=values>=0
            images.append(render(points[valid],values[valid],spacing,plow,phigh,
                folder/f'{name}_patch{p["patch_id"]}_{key}.png',
                f'{name.upper()} | Test patch {p["patch_id"]} | {title}',
                f'Instance {p["instance_id"]} | ordinary opacity 0.04 | {len(points):,} vortex voxels | invalid primitives {int((~pred["valid"][selected]).sum())}',
                ordinary_opacity=.04))
write(folder/'render_summary.json',dict(predictions_sha256=sha(out/'predictions.npz'),images=images,**identity()))
