"""Re-export saved scientific results without rerunning the classifier."""
import sys, os, json, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Visualize_FMT_Objectivity_Translation_1_2 as renderer

root = Path.cwd()
source = root/'outputs/Other_FMTObjectivityTranslation_1.2/cylinder3d'
destination = source/'final'
if destination.exists() and any(destination.iterdir()):
    raise FileExistsError('Final directory must be empty.')
audit = json.loads((source/'paper.json').read_text())
audit['render_job_id'] = os.environ.get('SLURM_JOB_ID')
audit['render_revision'] = '2: larger shared camera zoom; categorical opacity for dense non-vortex paths'
audit['render_source_sha256'] = hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest()
with np.load(source/'observed_pathlines.npz') as data:
    paths, labels, ids = data['pathlines'], data['predictions'], data['original_indices']
for medium in ['paper', 'slides']:
    print(renderer.render(paths,labels,ids,audit,destination,
                         'Cylinder Re160 3D | Mean-flow translating observer',medium,
                         pdf_collision_audit=False),flush=True)
