"""Export the first primary-test primitive and eight fixed particles per flow."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--vector',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();base,vector,out=Path(a.base).resolve(),Path(a.vector).resolve(),Path(a.output).resolve()
    specs=[];audits=[]
    for root,expected in [(base,2376),(vector,999)]:
        audit=json.loads((root/'independent_audit.json').read_text())
        assert audit['status']=='PASS' and audit['metric_records']==expected
        assert audit['metrics_sha256']==sha(root/'metrics.csv')
        assert audit['config_sha256']==sha(root/'runtime_config.json')
        audits.append(audit);specs.append(json.loads((root/'runtime_config.json').read_text()))
    assert specs[1]['base_config_sha256']==audits[0]['config_sha256']
    assert specs[0]['datasets']==specs[1]['datasets'] and specs[0]['seeds']==specs[1]['seeds']
    out.mkdir(parents=True,exist_ok=False)
    role='primitive_test';seed=specs[0]['seeds'][0];manifest=[]
    for dataset in specs[0]['datasets']:
        build=json.loads((base/'build'/f'{dataset}.json').read_text())
        records=[r for r in build['records'] if r['role']==role]
        first=records[0];path=base/'cache'/dataset/Path(first['cache_file']).name
        assert sha(path)==first['cache_sha256']
        keys=['origin0','radius0','support0','support1','context','target0','target_long',
              'query_ids0','window_ordinal','material_bundle_ids','duration']
        with np.load(path,allow_pickle=False) as archive:
            data={k:archive[k][0] for k in keys}
        arrays={};sources=[]
        for task in ('Task6','Task7','Task8'):
            truth=data['target_long'] if task=='Task8' else data['target0']
            arrays[task+'__truth']=truth[:8]
            arrays[task+'__support']=(data['context'].reshape(42,32,3) if task=='Task7' else
                np.concatenate([data['support0'],data['support1']]) if task=='Task8' else data['support0'])
            for arm,root in [('fmt_all',base),('vector_fmt6',vector),('raw_positions',base)]:
                directory=root/'runs'/dataset/arm/f'seed{seed}'
                record=json.loads((directory/f'{task}_{role}_normal.json').read_text())
                assert record['source_records'][0]['cache_sha256']==first['cache_sha256']
                prediction=directory/record['prediction_file']
                assert sha(prediction)==record['prediction_sha256']
                with np.load(prediction,allow_pickle=False) as archive:
                    arrays[task+'__'+arm]=archive['prediction'][0,:8]
                assert arrays[task+'__'+arm].shape==arrays[task+'__truth'].shape
                sources.append(dict(task=task,arm=arm,prediction_sha256=record['prediction_sha256']))
        arrays['origin']=data['origin0'];arrays['radius']=data['radius0']
        output=out/(dataset+'.npz');np.savez_compressed(output,**arrays)
        entry=dict(dataset=dataset,role=role,seed=seed,source_cache_sha256=first['cache_sha256'],
            region_index=0,window_ordinal=int(data['window_ordinal']),material_bundle_id=int(data['material_bundle_ids']),
            query_indices=list(range(8)),query_ids=data['query_ids0'][:8].tolist(),duration=float(data['duration']),
            file=output.name,sha256=sha(output),sources=sources)
        manifest.append(entry)
        print('EXPORTED',dataset,flush=True)
    evidence=dict(status='PASS',selection='First primitive in primary-test cache order, first eight saved query particles, first registered seed. No selection by error or appearance.',
        task6_scope='Only the first-stage query is shown; quantitative Task6 evaluation includes both stages.',
        task7_scope='Visible support contains all six external seven-line crosses.',
        task8_scope='Predicted first-stage endpoints were passed to the second stage.',
        base_audit_sha256=sha(base/'independent_audit.json'),vector_audit_sha256=sha(vector/'independent_audit.json'),
        exporter_sha256=sha(Path(__file__)),datasets=manifest)
    (out/'fixed_examples.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':main()
