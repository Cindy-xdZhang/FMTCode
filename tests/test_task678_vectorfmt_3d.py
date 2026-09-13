"""Directional token reconstruction, input sufficiency and full extension smoke."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from FMT_Utils.VectorFlowMapFMT_3D import vector_fmt, reconstruct_support, features
from FMT_Utils.FlowMapData_3D import cross_offsets, sha256, write_json
from tests.test_task678_hansampling_3d import smoke_spec
from tests.test_task678_flowmap_3d import analytic_field
from FMT_Utils.HanFlowMapData_3D import dense_window
from experiments.Build_Task678_HanSampling_1_1 import save_roles


class VectorFMTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_low_frequency_series_reconstruction(self):
        t=np.arange(31)/31
        rng=np.random.default_rng(77)
        a=rng.normal(size=(3,7,1,3))*.002
        increments=a+np.cos(4*np.pi*t)[None,None,:,None]*.002+np.sin(6*np.pi*t)[None,None,:,None]*.003
        delta=np.concatenate([np.zeros((3,7,1,3)),np.cumsum(increments,axis=2)],axis=2)
        paths=np.concatenate([delta[:,:1],delta[:,:1]+delta[:,1:]+cross_offsets()[None,1:,None]],axis=1)
        paths=paths.astype(np.float32)
        token=vector_fmt(paths,np.ones(3,np.float32))
        self.assertEqual(token.shape,(3,231))
        recovered=reconstruct_support(token,np.zeros((3,3)),np.ones(3))
        np.testing.assert_allclose(recovered,paths,atol=2e-6)

    def test_directional_collision_is_removed(self):
        t=np.arange(32,dtype=np.float32)/32
        displacement=t/8+t*t/16
        d=np.zeros((2,32,3),np.float32);d[0,:,0]=displacement;d[1,:,1]=displacement
        origins=np.array([[0.,0.,-4.],[0.,0.,4.]],np.float32)
        paths=origins[:,None,None]+.25*cross_offsets()[None,:,None]+d[:,None]
        radius=np.full(2,.25,np.float32)
        tokens=vector_fmt(paths,radius)
        self.assertGreater(np.linalg.norm(tokens[0]-tokens[1]),.01)
        p=reconstruct_support(tokens,origins,radius)
        # DC preserves the net displacement even when higher frequencies are truncated.
        np.testing.assert_allclose(p[:,:,-1],paths[:,:,-1],atol=2e-6)

    def test_hidden_targets_do_not_enter_new_tokens(self):
        spec=smoke_spec('unused')
        data,_=dense_window(analytic_field(),np.zeros((2,3)),.1,.5,spec['sampling'],77)
        a=features(data)
        for k in ('target0','target1','target_long'):data[k][:]=12345.
        b=features(data)
        for k in a:np.testing.assert_array_equal(a[k],b[k])


def smoke(output):
    from experiments.Audit_Task678_HanSampling_1_1 import audit_data
    from experiments.Run_Task678_VectorFMT_1_1 import run, DEFAULT_CONFIG
    from experiments.Audit_Task678_VectorFMT_1_1 import audit
    root=Path(output); base=smoke_spec(root/'base'); cfg=root/'base_config.json';write_json(cfg,base)
    records=[]
    for i in range(8):
        data,checks=dense_window(analytic_field(i/8),np.random.default_rng(i).uniform(-1,1,(8,3)),.1,.5,base['sampling'],900+i)
        row=dict(ordinal=i,role=next(k for k,ids in base['splits'].items() if i in ids),start_index=i*9,end_index=i*9+8,time_start=float(i),time_end=float(i+1))
        records.extend(save_roles(root/'base','analytic_unsteady',data,row,base['sampling'],{**row,**checks,'config_sha256':sha256(cfg),'status':'PASS'}))
    write_json(root/'base/build/analytic_unsteady.json',dict(status='PASS',config_sha256=sha256(cfg),records=records))
    audit_data(base,str(cfg))
    spec=json.loads(Path(DEFAULT_CONFIG).read_text())
    spec.update(experiment='Verify_Task678_VectorFMTSmoke_1.1',output_root=str(root/'extension'),datasets=['analytic_unsteady'],families={'analytic_unsteady':'analytic'},seeds=[9110],
                base_config=str(cfg),base_output_root=str(root/'base'),base_config_sha256=sha256(cfg),decoder=base['decoder'],sampling=base['sampling'])
    newcfg=root/'extension_config.json';write_json(newcfg,spec)
    run(spec,str(newcfg),'analytic_unsteady','vector_fmt6',9110,'cpu')
    run(spec,str(newcfg),'analytic_unsteady','vector_affine',0,'cpu')
    audit(spec,str(newcfg))
    report=json.loads((root/'extension/independent_audit.json').read_text())
    assert report['status']=='PASS' and report['metric_records']==51
    print('VECTOR FMT SMOKE PASS: 51 independently replayed records',flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--smoke-output':smoke(sys.argv[2])
    else:unittest.main()
