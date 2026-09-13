"""Export frozen 5.2 test predictions in original local 3D flow patches."""
import json
from pathlib import Path
import tarfile
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from experiments.Task4B_GeometrySearch_5_2 import sha,write,identity,load_npz
from experiments.Visualize_Task4B_VelocityCurl_3D import render


def main():
    out=Path('outputs/Other_Task4B_GeometrySearch_5.2')
    baseline=Path('outputs/mainExp_Task4B_PatchSegmentation_5.1')
    assert json.loads((out/'audit.json').read_text())['passed']
    report=json.loads((out/'final_report.json').read_text())
    assert sha(out/'final_test_predictions.npz')==report['prediction_sha256']
    pred=load_npz(out/'final_test_predictions.npz')
    manifest=json.loads((baseline/'manifest.json').read_text())
    old_export=json.loads((baseline/'export_report.json').read_text())
    folder=out/'test_patch_volumes';folder.mkdir(exist_ok=True)
    figures=out/'figures_3d';figures.mkdir(exist_ok=True)
    if (out/'export_report.json').exists():raise RuntimeError('Frozen export exists')
    result=dict(flows={},images=[],**identity())
    for code,name in enumerate(('channel','tbl')):
        info=manifest['flows'][name];low=np.array(info['low_xyz']);high=np.array(info['high_xyz'])
        resolution=np.array(info['resolution_xyz']);spacing=(high-low)/resolution
        records=[];old_hash={p['patch_id']:p['sha256'] for p in old_export['flows'][name]['files']}
        patches=[p for p in info['patches'] if p['split']=='test']
        display={p['patch_id'] for p in patches if p['kind']=='hairpin_bbox'}
        display=set(sorted(display)[:2])
        destination=folder/name;destination.mkdir(exist_ok=True)
        for p in patches:
            pid=p['patch_id'];path=baseline/'patch_volumes'/name/'test'/f'{name}_patch{pid:04d}.vtk'
            assert sha(path)==old_hash[pid]
            reader=vtk.vtkDataSetReader();reader.SetFileName(str(path));reader.ReadAllScalarsOn();reader.ReadAllVectorsOn();reader.ReadAllFieldsOn();reader.Update()
            volume=reader.GetOutput();cell=volume.GetCellData()
            truth=vtk_to_numpy(cell.GetArray('proxy_label'))
            values=np.where(truth>=0,-2,-1).astype(np.int32);confidence=np.zeros(len(truth),np.float32)
            use=(pred['volume']==code)&(pred['patch']==pid)
            index=np.column_stack(np.unravel_index(pred['source'][use],tuple(resolution[::-1])))
            plow=np.array(p['low']);shape=np.array(p['high'])-plow
            local=np.ravel_multi_index((index-plow).T,tuple(shape))
            np.testing.assert_array_equal(truth[local],pred['labels'][use])
            values[local]=pred['predictions'][use];confidence[local]=pred['probabilities'][use].max(1)
            for key,array in [('predicted_label',values),('prediction_confidence',confidence)]:
                cell.RemoveArray(key);v=numpy_to_vtk(array,deep=True);v.SetName(key);cell.AddArray(v)
            cell.SetActiveScalars('predicted_label')
            output=destination/path.name
            writer=vtk.vtkDataSetWriter();writer.SetFileName(str(output));writer.SetInputData(volume);writer.SetFileTypeToBinary();assert writer.Write()==1
            check=vtk.vtkDataSetReader();check.SetFileName(str(output));check.ReadAllScalarsOn();check.ReadAllVectorsOn();check.ReadAllFieldsOn();check.Update()
            restored=check.GetOutput()
            for i in range(cell.GetNumberOfArrays()):
                array=cell.GetArray(i)
                np.testing.assert_array_equal(vtk_to_numpy(restored.GetCellData().GetArray(array.GetName())),vtk_to_numpy(array))
            records.append(dict(patch_id=pid,file=str(output.relative_to(folder)),sha256=sha(output)))
            if pid in display:
                points=low+(index[:,::-1]+.5)*spacing
                boxlow=low+plow[::-1]*spacing;boxhigh=low+np.array(p['high'][::-1])*spacing
                for key,title in [('labels','Proxy GT'),('predictions','5.2 prediction')]:
                    labels=pred[key][use];valid=labels>=0
                    result['images'].append(render(points[valid],labels[valid],spacing,boxlow,boxhigh,
                        figures/f'{name}_patch{pid}_{key}.png',f'{name.upper()} | Test patch {pid} | {title}',
                        f'Instance {p["instance_id"]} | ordinary opacity 0.04 | {len(points):,} vortex voxels',ordinary_opacity=.04))
        archive=out/f'{name}_100_test_patch_volumes.tar.gz'
        with tarfile.open(archive,'w:gz') as handle:handle.add(destination,arcname=name)
        result['flows'][name]=dict(files=records,count=len(records),roundtrip_checked=len(records),archive=archive.name,archive_sha256=sha(archive))
        print(name,'exported',len(records),'test volumes',flush=True)
    write(out/'export_report.json',result)


if __name__=='__main__':main()
