"""Publish observed fields and manually refined cores, never sampled curves."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.Filter_Task6_CoreLength_1_4 import read_cores
WORK=ROOT/'outputs/mainExp_Task6_CorelineDataset_2.7'
OUT=ROOT/'outputs/Other_Task6_FieldPackage_1.1/package'
NATIVE=ROOT/'outputs/Verify_Task6_CPPParity_1.1/native'
EIGEN=Path('C:/Users/xingdi/sources/optimal-connection/external/eigen-5.0.0')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def copy(src,dst,expected=None):
 if expected is not None:assert sha(src)==expected,str(src)
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 assert sha(dst)==(expected or sha(src)),str(dst)

def export():
 assert not OUT.exists(),'A completed or partial export must be inspected, not overwritten'
 pointer=read(WORK/'current.json');source=WORK/pointer['relative_path']
 assert sha(source/'dataset_frozen.json')==pointer['frozen_sha256']
 frozen=read(source/'dataset_frozen.json');records=[];recipes={};audits=[]
 OUT.mkdir(parents=True)
 for f in frozen['frames']:
  key=f"{f['flow']}:{f['index']}";suffix=Path(f['flow'])/f"frame_{f['index']:03d}"
  old=source/suffix;folder=Path('frames')/suffix;dest=OUT/folder
  reference=read(old/'prepared_input.json');prepared=Path(reference['source'])
  provenance=read(prepared/'preparation_complete.json');info=provenance['input_frame']
  for name in ('relative_velocity.npy','coordinates.npz','ivd.npy','ivd.json'):
   copy(prepared/name,dest/name,reference['files'][name])
  copy(old/'corelines.vtp',dest/'corelines.vtp',f['core_sha256'])
  cores=read_cores(dest/'corelines.vtp');assert len(cores)==f['cores']
  offsets=np.r_[0,np.cumsum([len(c) for c in cores])].astype(np.int64)
  np.save(dest/'core_points.npy',np.concatenate(cores) if cores else np.empty((0,3),np.float64))
  np.save(dest/'core_offsets.npy',offsets)
  field=np.load(dest/'relative_velocity.npy',mmap_mode='r')
  with np.load(dest/'coordinates.npz') as a:axes=[a[c] for c in 'xyz']
  assert field.shape==tuple(map(len,axes[::-1]))+(3,) and np.isfinite(field).all()
  spacing=np.array([(a[-1]-a[0])/(len(a)-1) for a in axes]);h=float(min(spacing))
  assert abs(h-f['h'])<1e-10
  ids=f.get('ui_core_ids',list(range(len(cores))));assert len(ids)==len(cores) and len(set(ids))==len(ids)
  frame=dict(key=key,flow=f['flow'],index=f['index'],time=f['time'],steady=info['steady'],
    folder=folder.as_posix(),shape_xyz=list(map(len,axes)),bounds_xyz=[[float(a[0]),float(a[-1])] for a in axes],
    h=h,spacing_xyz=spacing.tolist(),core_count=len(cores),ui_core_ids=ids,
    velocity_definition='v' if info['steady'] else 'v-u',array_order='z,y,x,component_xyz',
    source_frame_velocity_sha256=reference['files']['relative_velocity.npy'],
    source_corelines_sha256=f['core_sha256'],ivd_definition='norm(curl(v)-full-grid mean curl(v))',
    ivd_candidate_condition='IVD(v) > full-grid arithmetic mean',ivd_threshold=read(dest/'ivd.json')['threshold'])
  clean=dict(provenance,source_preparation_sha256=sha(prepared/'preparation_complete.json'))
  clean['input_frame']=dict(info,source=Path(info['source']).name)
  clean['observer_call']=dict(invariance='Objective',NeighborhoodU=11,prefix_sum=True,middle_slice_only=True)
  write(dest/'observer_provenance.json',clean);write(dest/'frame.json',frame);records.append(frame)
  recipes[f['flow']]=dict(format='amira' if f['flow']=='SquareCylinder' else 'netcdf',steady=info['steady'],
    source_name=Path(info['source']).name,coordinate_variables=info['coordinate_variables'])
  baseline=WORK/'base'/suffix/'corelines.vtp'
  copy(baseline,OUT/'provenance/automatic_corelines'/suffix/'corelines.vtp')
  audits.append(dict(key=key,velocity_sha256=sha(dest/'relative_velocity.npy'),core_sha256=sha(dest/'corelines.vtp'),
    preserved_velocity=True,preserved_manual_cores=True,cores=len(cores),stable_ids=ids))
  print(json.dumps(dict(stage='field_frame',frame=key,cores=len(cores))),flush=True)
 tools=OUT/'tools'
 for name in ('ggt17_observe.py','CMakeLists.txt'):
  destination=tools/'native'/name if name=='CMakeLists.txt' else tools/name
  copy(ROOT/'experiments/task6_field_tools'/name,destination)
 copy(ROOT/'experiments/task6_field_tools/task6_fields.py',OUT/'task6_fields.py')
 copy(ROOT/'FMT_Utils/Task6SquareCylinder_2_5.py',tools/'amira_reader.py')
 sampler=(ROOT/'FMT_Utils/Task6Streamlines_3D.py').read_text();sampler=sampler[:sampler.index('@njit(cache=True)\ndef trace_one')]
 (tools/'vector_sampling.py').write_text(sampler,encoding='utf-8')
 adaptive=(ROOT/'FMT_Utils/Task6AdaptiveStreamlines_3D.py').read_text().replace('from FMT_Utils.Task6Streamlines_3D import sample_vector','from tools.vector_sampling import sample_vector')
 (tools/'adaptive_streamlines.py').write_text(adaptive,encoding='utf-8');(tools/'__init__.py').write_text('',encoding='utf-8')
 for name in ('reference.cpp','original_compute.inc','source_audit.json'):copy(NATIVE/name,tools/'native'/name)
 copy(NATIVE/'build/Release/reference.dll',tools/'bin/reference.dll')
 shutil.copytree(EIGEN/'Eigen',tools/'native/third_party/eigen/Eigen')
 for p in EIGEN.glob('COPYING*'):copy(p,tools/'native/third_party/eigen'/p.name)
 if (EIGEN/'LICENSE').exists():copy(EIGEN/'LICENSE',tools/'native/third_party/eigen/LICENSE')
 write(tools/'flow_recipes.json',recipes)
 write(tools/'native_identity.json',dict(windows_library_sha256=sha(tools/'bin/reference.dll'),
    source_sha256={n:sha(tools/'native'/n) for n in ('reference.cpp','original_compute.inc')},
    source_audit_sha256=sha(tools/'native/source_audit.json')))
 copy(ROOT/'docs/Task6_field_dataset_handoff_2.7.md',OUT/'README.md')
 copy(ROOT/'docs/Task6_ggt17_portable_1.1.md',OUT/'GGT17.md')
 (OUT/'requirements.txt').write_text('numpy>=1.24\nnetCDF4>=1.6\nnumba>=0.59\n',encoding='utf-8')
 edits=read(source/'manual_edits.json');write(OUT/'manual_edits.json',dict(operations=edits['operations'],source_sha256=sha(source/'manual_edits.json')))
 copy(source/'excluded_frames.json',OUT/'excluded_frames.json')
 manifest=dict(version='mainExp_Task6_CorelineDataset_2.7',schema='observed-fields-and-coreline-labels/1',
    packaging_version='Other_Task6_FieldPackage_1.1',source_manual_revision=pointer['relative_path'],
    source_frozen_sha256=pointer['frozen_sha256'],definition='Each flow/time frame consists of an observed v-u field and human-refined coreline labels; steady uses v',
    streamlines_in_dataset=False,frames=records,files_sha256={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
 write(OUT/'dataset.json',manifest)
 write(OUT/'default_split.json',dict(dataset_sha256=sha(OUT/'dataset.json'),policy='unchanged frame roles',
    train=[f"{f['flow']}:{f['index']}" for f in frozen['frames'] if f['role']=='train'],
    test=[f"{f['flow']}:{f['index']}" for f in frozen['frames'] if f['role']=='test'],unused=[]))
 write(OUT/'current.json',dict(manifest='dataset.json',sha256=sha(OUT/'dataset.json'),source_manual_revision=pointer['relative_path']))
 assert read(WORK/'current.json')==pointer
 assert not any(p.name in ('seeds.npy','sample_streamlines.npy','labels.npy','order6.npy','order16.npy') for p in OUT.rglob('*'))
 report=dict(passed=True,frames=len(records),corelines=sum(f['core_count'] for f in records),
   manual_operations=sum(map(len,edits['operations'].values())),no_preintegrated_curves=True,
   manifest_sha256=sha(OUT/'dataset.json'),files=len(manifest['files_sha256']),
   bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),frame_audits=audits)
 write(OUT.parent/'export_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='frame_audits'}),flush=True)
if __name__=='__main__':export()
