"""Independent Task6 diagnostics on training/validation data only; never load test."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from FMT_Utils.FlowMapData_3D import cross_offsets, integrate, write_json
from FMT_Utils.PrimitiveVAE_3D import resample_time, geometry_metrics, fmt_tokens
from FMT_Utils.Task6Recovery_3D import signed_fmt, inverse_signed_fmt, pca_geometry
from experiments.Task6_PrimitiveVAE_2_1 import strict_window, provenance


def run(config, index):
    spec = json.loads(Path(config).read_text())
    dataset = spec["datasets"][index]
    source_root = Path(spec["source_cache"])
    out = Path(spec["output_root"])/f"{dataset}.json"
    if out.exists(): raise FileExistsError(out)
    manifest = json.loads((source_root/"data"/dataset/"manifest.json").read_text())
    with np.load(source_root/"data"/dataset/"train.npz") as cache:
        train = cache["geometry"].copy()
        old_token = cache["token"].copy()
        chosen = np.linspace(0, len(train)-1, spec["integration_cases"], dtype=int)
        cases = [{key:cache[key][i].copy() for key in ("origin","radius","seed_time","physical_dt","integration_steps","horizon","source_start")} for i in chosen]
    with np.load(source_root/"data"/dataset/"validation.npz") as cache:
        validation = cache["geometry"].copy()
    rng = np.random.default_rng(spec["seed"])
    # Independent zero/constant/shuffled controls in the unchanged physical metric.
    seeds = np.broadcast_to(cross_offsets()[None,:,None], (len(validation),7,32,3))
    static = geometry_metrics(seeds, validation)
    mean_geom = train.mean(0, dtype=np.float64)
    constant = geometry_metrics(np.broadcast_to(mean_geom, validation.shape), validation)
    shuffled = geometry_metrics(train[rng.integers(len(train), size=len(validation))], validation)
    independent_metrics = float(np.sqrt(((train[chosen].astype(float)-mean_geom)[...,1:,:]**2).sum(-1).mean()))
    np.testing.assert_allclose(independent_metrics,geometry_metrics(np.broadcast_to(mean_geom,train[chosen].shape),train[chosen])["position_rmse_r"])
    np.testing.assert_allclose(fmt_tokens(train[chosen]),old_token[chosen],rtol=2e-5,atol=1e-5)
    diagnostics=[]
    for sample_id, case in zip(chosen,cases):
        field, meta = strict_window(manifest["source"]["source"],int(case["source_start"]),13,96)
        radius, horizon = float(case["radius"]), float(case["horizon"])
        origins=np.asarray(case["origin"],float)
        seeds=origins[None,None]+radius*cross_offsets()[None]
        # Re-run the same RK4, then halve its step. Compare in the original units.
        raw, flags=integrate(field,seeds,0,horizon,int(case["integration_steps"]))
        refined, flags2=integrate(field,seeds,0,horizon,2*int(case["integration_steps"]))
        assert flags.all() and flags2.all()
        local=(resample_time(raw)-origins)/radius
        finer=(resample_time(refined)-origins)/radius
        cache_rmse=geometry_metrics(local,train[sample_id:sample_id+1])["position_rmse_r"]
        step_rmse=geometry_metrics(local,finer)["position_rmse_r"]
        # SciPy supplies both independent interpolation and independent DOP853.
        spatial=[np.linspace(field.domainMinBoundary[i],field.domainMaxBoundary[i],field.field.shape[3-i]) for i in range(3)]
        interp=RegularGridInterpolator((np.linspace(0,field.tmax,13),spatial[2],spatial[1],spatial[0]),field.field,bounds_error=True)
        def velocity(t,x):
            xyz=x.reshape(7,3)
            query=np.column_stack([np.full(7,t),xyz[:,2],xyz[:,1],xyz[:,0]])
            return interp(query).reshape(-1)
        sample_times=np.linspace(0,horizon,int(case["integration_steps"])+1)
        solution=solve_ivp(velocity,(0,horizon),seeds.reshape(-1),method="DOP853",t_eval=sample_times,rtol=2e-8,atol=radius*1e-8,max_step=float(case["physical_dt"])/2)
        assert solution.success and np.isfinite(solution.y).all()
        paths=solution.y.reshape(7,3,-1).transpose(0,2,1)[None]
        scipy_local=(resample_time(paths)-origins)/radius
        diagnostics.append(dict(sample_id=int(sample_id),source_start=int(case["source_start"]),
            cache_replay_rmse_r=cache_rmse,rk4_step_halving_rmse_r=step_rmse,
            scipy_dop853_rmse_r=geometry_metrics(local,scipy_local)["position_rmse_r"],function_evaluations=solution.nfev))
    spectral=[]
    for k in spec["frequencies"]:
        for role,x in (("train",train),("validation",validation)):
            recon=inverse_signed_fmt(signed_fmt(x,k),k)
            spectral.append(dict(frequencies=k,role=role,**geometry_metrics(recon,x)))
    mean,basis,variance=pca_geometry(train)
    pca=[]
    for dim in spec["latent_dimensions"]:
        for role,x in (("train",train),("validation",validation)):
            flat=x.reshape(len(x),672)
            recon=((flat-mean)@basis[:,:dim])@basis[:,:dim].T+mean
            pca.append(dict(dimensions=dim,role=role,**geometry_metrics(recon.reshape(x.shape),x)))
    report=dict(experiment=spec["experiment"],dataset=dataset,provenance=provenance(config),
        roles_read=["train","validation"],integration=diagnostics,
        controls_validation=dict(static_seed=static,constant_training_mean=constant,random_training_primitive=shuffled),
        signed_fourier_inverse=spectral,pca_geometry=pca,training_shape=list(train.shape),validation_shape=list(validation.shape))
    write_json(out,report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="config/Verify_Task6_ReconstructionAudit_2.2.json"); parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args(); run(args.config,args.index)
