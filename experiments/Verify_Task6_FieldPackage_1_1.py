"""Validate portable fields, on-demand integration and the actual observer CLI."""
from pathlib import Path
import hashlib,json,subprocess,sys,urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'outputs/Other_Task6_FieldPackage_1.1/package'
OUT=ROOT/'outputs/Verify_Task6_FieldPackage_1.1'
sys.path.insert(0,str(PACK))
from task6_fields import Dataset,Frame,sha

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def api(path):
 with urllib.request.urlopen('http://127.0.0.1:8872'+path,timeout=180) as response:return json.load(response)
def verify():
 OUT.mkdir(parents=True,exist_ok=True)
 ds=Dataset(PACK);check=ds.verify();report=dict(package_verification=check,frames=[],observer=[],curve_files_written=False)
 assert len(ds.frames('train'))==45 and len(ds.frames('test'))==24
 for f in ds.frames():
  frame=ds.frame(f['key']);cores=frame.corelines()
  assert len(cores)==f['core_count'] and np.array_equal(frame.bounds,np.asarray(f['bounds_xyz']))
  for c in cores:
   assert np.isfinite(c).all() and (c>=frame.bounds[:,0]-1e-5).all() and (c<=frame.bounds[:,1]+1e-5).all()
  assert abs(frame.h-f['h'])<1e-12
  report['frames'].append(dict(key=f['key'],cores=len(cores),shape=list(frame.velocity.shape)))
 # Analytic constant field: direction, length, axis layout and seed changes.
 synthetic=OUT/'analytic';synthetic.mkdir(exist_ok=True)
 v=np.zeros((7,7,7,3),np.float32);v[...,0]=2;np.save(synthetic/'relative_velocity.npy',v)
 np.savez(synthetic/'coordinates.npz',x=np.arange(7.),y=np.arange(7.),z=np.arange(7.))
 f=Frame(OUT,dict(folder='analytic'))
 a=f.integrate(np.array([[3.,3,3]]),1.,points=33)
 b=f.integrate(np.array([[3.,2,1]]),2.,points=65)
 assert a['valid'].all() and b['valid'].all()
 np.testing.assert_allclose(a['curves'][0,[0,-1]],[[2.5,3,3],[3.5,3,3]],atol=1e-6)
 np.testing.assert_allclose(b['curves'][0,[0,-1]],[[2,2,1],[4,2,1]],atol=1e-6)
 report['analytic_direction_seed_and_length']=True
 # Real data, one frame of each flow. Arrays are only kept in memory.
 report['real_integration']=[];seen=set()
 for f in ds.frames():
  if f['flow'] in seen:continue
  seen.add(f['flow']);frame=ds.frame(f['key']);rng=np.random.default_rng(42)
  seeds=rng.uniform(frame.bounds[:,0]+frame.h,frame.bounds[:,1]-frame.h,(12,3))
  result=frame.integrate(seeds,frame.h*8,points=65)
  assert result['valid'].any(),f['key']
  report['real_integration'].append(dict(key=f['key'],valid=int(result['valid'].sum()),
    maximum_error=float(result['error'][result['valid']].max()),limit=frame.h/20))
 before=api('/api/editor/status');responses=[]
 for length,batch in [(1.,901),(2.,901),(1.,902)]:
  r=api(f'/api/domain-streams?key=SquareCylinder%3A65&n=16&length={length}&batch={batch}')
  assert r['source']=='domain' and r['visualization_only'] and r['count']>0
  responses.append(r)
 assert responses[0]['field_sha256']==responses[1]['field_sha256']==responses[2]['field_sha256']
 assert responses[0]['curves']!=responses[1]['curves'] and responses[0]['seeds']!=responses[2]['seeds']
 assert api('/api/editor/status')==before
 report['existing_viewer_on_demand']=[dict(total_length=r['total_length'],batch=r['batch'],count=r['count'],
    field_sha256=r['field_sha256'],curve_sha256=hashlib.sha256(np.asarray(r['curves']).tobytes()).hexdigest()) for r in responses]
 write(OUT/'reader_and_viewer_audit.json',report)
 # Real C++ regeneration, covering original high resolution, odd-plane single thread,
 # Amira input and the steady bypass. Never generate sampled-curve datasets.
 recipes=json.loads((PACK/'tools/flow_recipes.json').read_text())
 raw=Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D')
 for flow,index in [('cylinder3d',81),('deltaWing_resampled',85),('SquareCylinder',65),('tornado3d',0)]:
  output=OUT/'regenerated'/flow
  cmd=[sys.executable,str(PACK/'tools/ggt17_observe.py'),'--flow',flow,'--source',str(raw/recipes[flow]['source_name']),
      '--index',str(index),'--out',str(output)]
  print(json.dumps(dict(stage='observer_regeneration',flow=flow,index=index)),flush=True)
  p=subprocess.run(cmd,cwd=OUT,text=True,capture_output=True,check=True)
  generated=json.loads((output/'generation.json').read_text());original=ds.frame(f'{flow}:{index}')
  actual=np.load(output/'relative_velocity.npy',mmap_mode='r')
  assert np.array_equal(actual,original.velocity),flow
  assert sha(output/'relative_velocity.npy')==original.record['source_frame_velocity_sha256'],flow
  report['observer'].append(dict(flow=flow,index=index,bitwise_equal=True,sha256=sha(output/'relative_velocity.npy'),
    library_sha256=generated['observer']['library_sha256'],threads=generated['observer']['threads'],seconds=generated['seconds']))
  write(OUT/'verification.json',dict(report,complete=False))
 # Confirm CLI rejects boundary frames without creating outputs.
 rejected=OUT/'invalid_boundary'
 p=subprocess.run([sys.executable,str(PACK/'tools/ggt17_observe.py'),'--flow','cylinder3d',
    '--source',str(raw/recipes['cylinder3d']['source_name']),'--index','0','--out',str(rejected)],capture_output=True,text=True)
 assert p.returncode!=0 and not rejected.exists()
 report.update(complete=True,boundary_rejected=True,source_manifest_sha256=sha(PACK/'dataset.json'))
 write(OUT/'verification.json',report)
 print(json.dumps(dict(complete=True,frames=69,observer_checks=report['observer'],on_demand_viewer=True)),flush=True)
if __name__=='__main__':verify()
