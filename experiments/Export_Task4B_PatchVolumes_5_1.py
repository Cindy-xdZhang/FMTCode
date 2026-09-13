"""Export all 2000 local flow volumes and final test segmentations to VTK."""
import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import yaml
from experiments.Task4B_PatchSegmentation_5_1 import sha,write,identity
from experiments.Verify_Task4B_VelocityCurlMemorization import load_field


def main(config,input_root):
    spec=json.loads(Path(config).read_text());out=Path(spec['output'])
    assert json.loads((out/'audit.json').read_text())['passed']
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['config_sha256']==sha(config)
    field_spec=yaml.safe_load(Path(spec['field_source_config']).read_text())
    build=json.loads((out/'build_summary.json').read_text())
    with np.load(out/'predictions.npz') as d: predictions={k:d[k] for k in d.files}
    destination=out/'patch_volumes';destination.mkdir(exist_ok=True)
    if (out/'export_report.json').exists():raise RuntimeError('Frozen export report exists')
    report=dict(version=spec['version'],config_sha256=sha(config),flows={},**identity())
    instructions='''Local flow patches for mainExp_Task4B_PatchSegmentation_5.1
All arrays are CELL_DATA. Physical x/y/z aspect and coordinates are retained.
velocity and vorticity are interpolated from the original full flow.
IVD and proxy_label use frozen 4.4 labels; recomputing derivatives from these
coarser cropped velocity grids need not reproduce the original-field IVD.
Labels: -1 non-vortex/ignored, 0 ordinary_streamwise, 1 ordinary_spanwise,
2 hairpin_head, 3 hairpin_leg. Predicted -2 means an invalid primitive.
Only test files contain predicted_label and prediction_confidence.
Test non-vortex cells remain -1 through the fixed IVD mask, not model inference.
Training files contain velocity, vorticity, IVD, VortexIds and proxy_label.
The original native fields, not these downsampled exports, supplied training
primitive interpolation. See the JSON manifest for exact splits and boxes.
'''
    (destination/'README.txt').write_text(instructions)
    for code,name in enumerate(('channel','tbl')):
        path=Path(input_root)/field_spec['flows'][name]['flow']
        assert sha(path)==build['flows'][name]['flow_sha256']
        field,_,mean,_=load_field(path,name)
        info=manifest['flows'][name];low=np.array(info['low_xyz']);high=np.array(info['high_xyz'])
        resolution=np.array(info['resolution_xyz']);spacing=(high-low)/resolution
        with np.load(Path(spec['labels_source'])/f'{name}_groundtruth.npz') as d:
            labels=d['labels'];ids=d['vortex_ids']
        with np.load(Path('outputs/Verify_Task4B_VelocityCurlMemorization_4.1')/f'{name}_label_volume.npz') as d:
            ivd=d['ivd']
        records=[];checked=set();max_ivd_difference=0.
        for p in info['patches']:
            plow=np.array(p['low']);phigh=np.array(p['high']);shape=phigh-plow
            slices=tuple(slice(a,b) for a,b in zip(plow,phigh))
            index=np.indices(shape).reshape(3,-1).T+plow
            points=low+(index[:,::-1]+.5)*spacing
            velocity=field.velocity(points).astype(np.float32)
            omega=field.vorticity(points)
            recalculated=np.linalg.norm(omega.astype(np.float64)-mean,axis=1)
            difference=float(np.max(np.abs(recalculated-ivd[slices].ravel())))
            max_ivd_difference=max(max_ivd_difference,difference)
            np.testing.assert_allclose(recalculated,ivd[slices].ravel(),rtol=1e-6,atol=1e-7)
            dataset=vtk.vtkImageData();dataset.SetDimensions(*(shape[::-1]+1))
            dataset.SetOrigin(*(low+plow[::-1]*spacing));dataset.SetSpacing(*spacing)
            values=dict(proxy_label=labels[slices].ravel().astype(np.int32),
                VortexIds=ids[slices].ravel().astype(np.int32),IVD=ivd[slices].ravel().astype(np.float32),
                velocity=velocity,vorticity=omega.astype(np.float32))
            if p['split']=='test':
                pred=np.where(labels[slices].ravel()>=0,-2,-1).astype(np.int32)
                confidence=np.zeros(len(pred),np.float32)
                selected=(predictions['volume']==code)&(predictions['patch']==p['patch_id'])
                original=np.column_stack(np.unravel_index(predictions['source'][selected],labels.shape))
                local=np.ravel_multi_index((original-plow).T,tuple(shape))
                pred[local]=predictions['predictions'][selected]
                confidence[local]=predictions['probabilities'][selected].max(1)
                values.update(predicted_label=pred,prediction_confidence=confidence)
            for key,array in values.items():
                vtk_array=numpy_to_vtk(np.ascontiguousarray(array),deep=True);vtk_array.SetName(key)
                dataset.GetCellData().AddArray(vtk_array)
            dataset.GetCellData().SetActiveScalars('proxy_label');dataset.GetCellData().SetActiveVectors('velocity')
            folder=destination/name/p['split'];folder.mkdir(parents=True,exist_ok=True)
            output=folder/f'{name}_patch{p["patch_id"]:04d}.vtk'
            writer=vtk.vtkDataSetWriter();writer.SetInputData(dataset);writer.SetFileName(str(output));writer.SetFileTypeToBinary()
            assert writer.Write()==1
            if p['split'] not in checked:
                reader=vtk.vtkDataSetReader();reader.SetFileName(str(output));reader.ReadAllScalarsOn();reader.ReadAllVectorsOn();reader.ReadAllFieldsOn();reader.Update()
                restored=reader.GetOutput();assert restored.GetNumberOfCells()==len(points)
                for key,array in values.items():
                    np.testing.assert_allclose(vtk_to_numpy(restored.GetCellData().GetArray(key)),array,rtol=0,atol=0)
                np.testing.assert_allclose(restored.GetBounds(),dataset.GetBounds(),rtol=1e-5,atol=1e-5)
                checked.add(p['split'])
            records.append(dict(patch_id=p['patch_id'],split=p['split'],cell_count=len(points),sha256=sha(output),file=str(output.relative_to(destination))))
            if p['patch_id']%100==0:print(name,'exported',p['patch_id']+1,'/',len(info['patches']),flush=True)
        archive=out/f'{name}_1000_patch_volumes.tar.gz'
        write(destination/name/'patch_manifest.json',info)
        with tarfile.open(archive,'w:gz') as handle:
            handle.add(destination/name,arcname=name)
            handle.add(destination/'README.txt',arcname='README.txt')
        report['flows'][name]=dict(files=records,count=len(records),max_ivd_reconstruction_difference=max_ivd_difference,
            vtk_roundtrip_checked_splits=sorted(checked),archive=archive.name,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
        del field
    write(out/'export_report.json',report)
    print('Export complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/mainExp_Task4B_PatchSegmentation_5.1.json')
    parser.add_argument('--input-root',default='inputs')
    args=parser.parse_args();main(args.config,args.input_root)
