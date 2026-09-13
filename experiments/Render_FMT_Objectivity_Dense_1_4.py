"""Re-export complete enlarged 3D projections from saved 1.4 trajectories."""
from pathlib import Path
import os,sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import Visualize_FMT_Objectivity_Translation_1_4 as renderer

root=Path.cwd()/'outputs/Other_FMTObjectivityTranslation_1.4'
names={'cylinder3d':'Re160','halfcylinderRe640':'Re640','halfcylinderRe6400':'Re6400'}
for dataset,title in names.items():
    source=root/dataset;destination=source/'final'
    if destination.exists() and any(destination.iterdir()):raise FileExistsError(destination)
    audit=json.loads((source/'paper.json').read_text())
    audit['render_revision']='2: disable square-axes artist clipping for the enlarged 3D projection'
    audit['render_job_id']=os.environ.get('SLURM_JOB_ID')
    audit['render_source_sha256']=hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest()
    with np.load(source/'observed_pathlines.npz') as data:
        paths,labels,ids=data['pathlines'],data['predictions'],data['original_indices']
    assert len(ids)==2*audit['previous_display_count']
    assert paths.shape[3]==97 and np.isclose(audit['display_duration'],2*audit['classification_duration'])
    for medium in ['paper','slides']:
        print(renderer.render(paths,labels,ids,audit,destination,
              f"{title} 3D | Mean-flow observer | t0 = {audit['source_time']:g}",medium,
              pdf_collision_audit=False),flush=True)
