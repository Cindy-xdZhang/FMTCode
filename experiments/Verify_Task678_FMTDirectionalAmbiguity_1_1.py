"""Analytic non-injectivity check of the frozen FMT, independent of benchmark data."""
import json
from pathlib import Path
import numpy as np
import torch
from FMT_Utils.FlowMapData_3D import cross_offsets, sha256, write_json
from FMT_Utils.HanFlowMapData_3D import support_features
from experiments.Run_Task678_HanSampling_1_1 import arrays_for


def main():
    # Two different smooth uniform flows: v_a(t)=(1/8+t/8,0,0),
    # v_b(t)=(0,1/8+t/8,0). Neither is an observer transformation test.
    times=np.arange(32,dtype=np.float32)/32
    distance=times/8+times*times/16
    r=np.float32(.25); duration=np.float32(times[-1])
    d=np.zeros((2,32,3),np.float32); d[0,:,0]=distance; d[1,:,1]=distance
    supports=r*cross_offsets().astype(np.float32)[None,:,None,:]+d[:,None]
    context_origins=np.broadcast_to(3*r*cross_offsets()[None,1:],(2,6,3)).astype(np.float32)
    contexts=context_origins[:,:,None,None,:]+r*cross_offsets()[None,None,:,None,:]+d[:,None,None]
    target=d[:,None].copy()
    data=dict(support0=supports,support1=supports.copy(),context=contexts.astype(np.float32),
        origin0=np.zeros((2,3),np.float32),origin1=np.zeros((2,3),np.float32),context_origins=context_origins,
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
    result=dict(experiment='Verify_Task678_FMTDirectionalAmbiguity_1.1',status='PASS',
        scope='Analytic input sufficiency check, independent of all nine benchmark fields; not an objectivity check',
        encoder_sha256=sha256('FMT_Utils/DFT_FMT_3D.py'),token_dimension=161,queries_per_field=1,
        physical_times=times.tolist(),radius=float(r),
        velocity_a='(1/8+t/8,0,0)',velocity_b='(0,1/8+t/8,0)',comparisons=comparisons,
        inference='A universal decoder of absolute-coordinate flow maps cannot be determined by this token and current query metadata alone. This does not predict the nine-field empirical ranking.')
    root=Path('outputs/Verify_Task678_FMTDirectionalAmbiguity_1.1')
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/'analytic_inputs.npz',support_paths=supports,context_paths=contexts,
                        fmt_tokens=features['fmt_all__support0'],query_truth=target)
    result['inputs_sha256']=sha256(root/'analytic_inputs.npz')
    write_json(root/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
