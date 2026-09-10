"""Same-field analytic non-injectivity check of the frozen FMT, version 1.2."""
import json
from pathlib import Path
import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import cross_offsets, sha256, write_json
from FMT_Utils.HanFlowMapData_3D import support_features
from experiments.Run_Task678_HanSampling_1_1 import arrays_for


def main():
    # One smooth field v(x,y,z,t)=s(t)*(1-g(z),g(z),0), s=1/8+t/8.
    # g is zero for z<=-2, one for z>=2, and a smooth transition between.
    # All first-region supports remain at z<=-3; the second stays at z>=3.
    # Both primitives therefore belong to the SAME field, time and decoder.
    times=np.arange(32,dtype=np.float32)/32
    distance=times/8+times*times/16
    r=np.float32(.25); duration=np.float32(times[-1])
    d=np.zeros((2,32,3),np.float32); d[0,:,0]=distance; d[1,:,1]=distance
    origins=np.array([[0.,0.,-4.],[0.,0.,4.]],np.float32)
    supports=origins[:,None,None,:]+r*cross_offsets().astype(np.float32)[None,:,None,:]+d[:,None]
    context_origins=origins[:,None,:]+np.broadcast_to(3*r*cross_offsets()[None,1:],(2,6,3)).astype(np.float32)
    contexts=context_origins[:,:,None,None,:]+r*cross_offsets()[None,None,:,None,:]+d[:,None,None]
    query_offset=np.array([1/32,1/64,0.],np.float32)
    assert np.linalg.norm(r*cross_offsets()-query_offset,axis=-1).min()>0
    target=d[:,None]+origins[:,None,None,:]+query_offset
    data=dict(support0=supports,support1=supports.copy(),context=contexts.astype(np.float32),
        origin0=origins,origin1=origins.copy(),context_origins=context_origins,
        radius0=np.full(2,r,np.float32),radius1=np.full(2,r,np.float32),duration=np.full(2,duration,np.float32),
        target0=target,target1=target.copy(),target_long=target.copy())
    features=support_features(data)
    comparisons={}
    for task in ('Task6','Task7'):
        arrays=arrays_for({**data,**features},task,'fmt_all')
        equal={k:bool(np.array_equal(arrays[k][0],arrays[k][1])) for k in ('tokens','geometry','query','scales')}
        assert all(equal.values())
        gap=np.linalg.norm(arrays['target'][0,0]-arrays['target'][1,0],axis=-1)
        lower_bound=float(.5*np.sqrt(np.mean(gap[1:].astype(float)**2)))
        assert lower_bound>0
        comparisons[task]=dict(identical_neural_inputs=equal,
            normalized_endpoint_separation=float(gap[-1]),
            minimum_paired_position_nrmse=lower_bound,
            bound='For two equally weighted different targets with the same deterministic input, their midpoint minimizes squared error')
    result=dict(experiment='Verify_Task678_FMTDirectionalAmbiguity_1.2',status='PASS',
        scope='Two regions within one analytic smooth field and one shared decoder; independent of benchmark data; not an objectivity check',
        encoder_sha256=sha256('FMT_Utils/DFT_FMT_3D.py'),token_dimension=161,queries_per_primitive=1,
        number_of_fields=1,number_of_primitives=2,query_offset=query_offset.tolist(),
        physical_times=times.tolist(),radius=float(r),
        velocity='(1/8+t/8)*(1-g(z),g(z),0)',
        g_definition='g=0 for z<=-2, g=1 for z>=2. Between, a=(z+2)/4; g=exp(-1/a)/(exp(-1/a)+exp(-1/(1-a)))',
        primitive_origins=origins.tolist(),comparisons=comparisons,
        inference='Even within one field, these two primitives have identical neural inputs and distinct normalized target trajectories. The current frozen token and metadata do not uniquely specify their coordinate flow maps. This does not predict the nine-field empirical ranking.')
    root=Path('outputs/Verify_Task678_FMTDirectionalAmbiguity_1.2')
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'analytic_inputs.npz',support_paths=supports,context_paths=contexts,
                        fmt_tokens=features['fmt_all__support0'],query_truth=target)
    result['inputs_sha256']=sha256(root/'analytic_inputs.npz')
    write_json(root/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
