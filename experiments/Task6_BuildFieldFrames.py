"""Reusable field/core builder; a JSON plan selects frames, never saved streamlines."""
from pathlib import Path
import argparse, importlib.util, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from FMT_Utils.Task6CPPExtraction_2_1 import extract_fixed,spacing_h
from FMT_Utils.Task6CPPExtraction_2_5 import filter_merged_corelines
from FMT_Utils.Task6VortexCore_3D import instantaneous_vorticity_deviation
from experiments.Extract_Task6_VortexCore_1_2 import write_lines
from experiments.Filter_Task6_CoreLength_1_4 import read_cores
from experiments.Export_Task6_FieldPackage_1_1 import read,write,sha

def build(plan,row):
    package=Path(plan['dataset_root']);tools=package/'tools'
    sys.path.insert(0,str(tools))
    spec=importlib.util.spec_from_file_location('task6_observer',tools/'ggt17_observe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    flow,index=row['flow'],row['index'];dest=Path(plan['staging'])/flow/f'frame_{index:03d}'
    if (dest/'complete.json').exists():
        complete=read(dest/'complete.json')
        for n,h in complete['files'].items():assert sha(dest/n)==h
        return
    dest.mkdir(parents=True,exist_ok=True);recipes=read(tools/'flow_recipes.json');recipe=recipes[flow]
    started=time.monotonic()
    if not (dest/'observer_provenance.json').exists():
        series,axes,timing,physical_time,inputs=module.load_source(recipe,Path(row['source']),index)
        lib=tools/'bin/reference.dll';assert sha(lib)==read(tools/'native_identity.json')['windows_library_sha256']
        field,threads=module.compute(series,axes,*timing,lib)
        np.save(dest/'relative_velocity.npy',field);np.savez(dest/'coordinates.npz',**dict(zip('zyx',axes)))
        ivd,meta=instantaneous_vorticity_deviation(axes,series[1]);ivd=ivd.astype(np.float32)
        threshold=float(ivd.mean(dtype=np.float64));np.save(dest/'ivd.npy',ivd)
        meta.update(field='original_velocity_v',threshold=threshold,threshold_definition='full-grid arithmetic mean',candidate_nodes=int((ivd>threshold).sum()))
        write(dest/'ivd.json',meta)
        write(dest/'observer_provenance.json',dict(input_frame=row,observer_call=dict(invariance='Objective',NeighborhoodU=11,prefix_sum=True,middle_slice_only=True),input_files=inputs,library_sha256=sha(lib),threads=threads,seconds=time.monotonic()-started))
        del series,ivd,field
    with np.load(dest/'coordinates.npz') as z:axes=[z[a] for a in 'zyx']
    spacing,h=spacing_h(axes)
    if not (dest/'extraction.json').exists():extract_fixed(np.load(dest/'relative_velocity.npy',mmap_mode='r'),axes,dest,4.,10)
    merged=read_cores(dest/'vtk_merged.vtp');cores,report=filter_merged_corelines(merged,h)
    write_lines(dest/'corelines.vtp',cores)
    loaded=read_cores(dest/'corelines.vtp');assert len(cores)==len(loaded)
    for a,b in zip(cores,loaded):np.testing.assert_array_equal(a,b)
    lengths=[float(np.linalg.norm(np.diff(c.astype(np.float64),axis=0),axis=1).sum()) for c in cores]
    assert all(v>=16*h for v in lengths)
    np.save(dest/'core_points.npy',np.concatenate(loaded) if loaded else np.empty((0,3),np.float64))
    np.save(dest/'core_offsets.npy',np.r_[0,np.cumsum([len(c) for c in loaded])].astype(np.int64))
    write(dest/'length_filter.json',report)
    frame=dict(key=f'{flow}:{index}',flow=flow,index=index,time=row['time'],steady=False,folder=f'frames/{flow}/frame_{index:03d}',shape_xyz=list(map(len,axes[::-1])),bounds_xyz=[[float(a[0]),float(a[-1])] for a in axes[::-1]],h=h,spacing_xyz=spacing.tolist(),core_count=len(cores),ui_core_ids=list(range(len(cores))),velocity_definition='v-u',array_order='z,y,x,component_xyz',source_frame_velocity_sha256=sha(dest/'relative_velocity.npy'),source_corelines_sha256=sha(dest/'corelines.vtp'),ivd_definition='norm(curl(v)-full-grid mean curl(v))',ivd_candidate_condition='IVD(v) > full-grid arithmetic mean',ivd_threshold=read(dest/'ivd.json')['threshold'])
    write(dest/'frame.json',frame)
    write(dest/'complete.json',dict(frame=frame,seconds=time.monotonic()-started,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'},streamlines_saved=False))
    print(json.dumps(dict(frame=frame['key'],cores=len(cores),seconds=time.monotonic()-started)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--row',type=int);p.add_argument('--workers',type=int,default=1);a=p.parse_args();plan=read(a.plan)
    if a.row is not None:return build(plan,plan['frames'][a.row])
    def run(i):subprocess.run([sys.executable,'-u',__file__,str(a.plan),'--row',str(i)],check=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(run,i):i for i in range(len(plan['frames']))};completed=0
        for job in as_completed(jobs):
            job.result();completed+=1
            write(Path(plan['staging'])/'status.json',dict(completed=completed,total=len(plan['frames']),last=plan['frames'][jobs[job]]))
if __name__=='__main__':main()
