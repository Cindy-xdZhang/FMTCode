"""Export all new paired training heads and the complete expanded test split."""
from pathlib import Path
import json
import numpy as np
from FMT_Utils.Task4C_GTHeadCoverage_1_1 import sha
from experiments.Task4C_GTHeadCoverage_1_1 import metadata,write


def export(spec,config,identity):
    from experiments.Task4C_FPSAugmentSearch_1_1 import metrics
    root=Path(spec['output']);run=root/'final/c156/seed96721'
    result=json.loads((run/'result.json').read_text())
    assert result['complete'] and result['candidate']==spec['candidate'] and result['seed']==96721
    assert result['identity']['config_sha256']==sha(config)
    package=root/'viewer_package';package.mkdir(exist_ok=False)
    splits={};scores={};head_reports={}
    for role in ('train','test'):
        file=run/(role+'_predictions.npz');assert sha(file)==result['predictions'][role]
        with np.load(file) as z:prediction={k:z[k] for k in z.files}
        assert len(prediction['labels'])==spec['expected_counts'][role]
        scores[role]=dict(pooled=metrics(prediction['labels'],prediction['probability']),per_flow={})
        for fi,flow in enumerate(spec['flows']):
            name=flow['name'];folder=root/'physical'/name/role;m=metadata(folder)
            mask=prediction['flow_index']==fi;ids=prediction['row_in_split'][mask]
            assert np.array_equal(ids,np.arange(len(m['labels'])))
            assert np.array_equal(prediction['labels'][mask],m['labels'])
            assert np.array_equal(prediction['instance'][mask],m['instance'])
            probability=prediction['probability'][mask];assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
            scores[role]['per_flow'][name]=metrics(m['labels'],probability)
            old=metadata(Path(spec['source_output'])/'physical'/name/role);n=len(old['labels'])
            selected=np.arange(n,len(m['labels'])) if role=='train' else np.arange(len(m['labels']))
            g=np.load(folder/'geometry.npy',mmap_mode='r')
            world=np.asarray(g[selected],dtype=float)*m['radius'][selected,None,None,None]+m['centroid'][selected,None,None,:]
            valid=np.arange(27)[None]<m['counts'][selected,None];world[~valid]=0
            pack={k:m[k][selected] for k in ('counts','labels','center','head_component','instance','scale_id','radius','neighbor_distance')}
            pack.update(geometry=world.astype(np.float32),row_ids=selected,p_c156=probability[selected],
                mandatory_head=(selected>=n),owner_instance=m['instance'][selected])
            path=package/(name+'_'+role+'.npz');np.savez_compressed(path,**pack)
            splits[name+'/'+role]=dict(file=path.name,sha256=sha(path),population=len(m['labels']),selected=len(selected),
                metadata_sha256=sha(folder/'metadata.npz'),geometry_sha256=sha(folder/'geometry.npy'),
                selection='all_added_training_heads' if role=='train' else 'all_test_rows',
                class_counts=dict(c156=dict(hairpin=int(np.sum(pack['p_c156']>=.5)),non_hairpin=int(np.sum(pack['p_c156']<.5)))))
            if role=='test':
                head_reports[name]=dict(original_test=metrics(m['labels'][:n],probability[:n]),
                    added_head_counts={str(i):dict(bundles=int(np.sum(m['instance'][n:]==i)),
                        correctly_predicted=int(np.sum((m['instance'][n:]==i)&(probability[n:]>=.5)))) for i in np.unique(m['instance'][n:])},
                    added_head_recall=float(np.mean(probability[n:]>=.5)),
                    added_test_positive_only=True)
    for role in ('validation','test'):
        r=result['validation'] if role=='validation' else result['test']['combined']
        if role=='test':
            for k,v in r.items():assert abs(v-scores['test']['pooled'][k])<1e-12,k
    manifest=dict(version=spec['version'],identity=identity(config),flows=spec['flows'],splits=splits,
        models=[dict(id='c156',label='FMT c156 · GT头部补样1.1',seed=96721,parameters=992386,version=spec['version'],
            metrics=scores,result_sha256=sha(run/'result.json'),selected_epoch=result['selected_epoch'])],pending=[],
        config=dict(splits=['test','train'],geometry_chunk_size=128,display_bundles=200,default_hairpin_count='all',
            default_nonhairpin_count=100,default_view_mode='heads'),
        evidence_note='固定c156，seed96721，阈值0.5。测试视图包含全部11,320束；训练视图仅展示新增的3,960束GT头部邻近训练样本（完整训练196,960束）。F1按所选流场完整集合计算。新增样本按人工GT头部归属标正类；旧标签不变。共享GT实例的局部评估，不是未见实例测试。',
        head_coverage=head_reports,training_seconds=result['training_seconds'],fixed_model_single_seed=True)
    write(package/'manifest.json',manifest)
    write(root/'completion.json',dict(complete=True,identity=identity(config),test=result['test'],head_coverage=head_reports,
        exported_bundles=sum(x['selected'] for x in splits.values()),package_manifest_sha256=sha(package/'manifest.json'),
        no_weights_saved=True))
    print(json.dumps(dict(complete=True,package=str(package),head_recall={k:v['added_head_recall'] for k,v in head_reports.items()})),flush=True)
