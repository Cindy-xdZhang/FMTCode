"""Build an offline, chunked Plotly viewer from verified attribution arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def get_plotly(path):
    if path:
        text=Path(path).read_text(encoding='utf-8')
        candidates=re.findall(r'<script[^>]*>([\s\S]*?)</script>',text)
        for candidate in candidates:
            if 'plotly.js' in candidate[:2000] and len(candidate)>1000000:return candidate
        raise ValueError('Existing viewer has no embedded Plotly runtime')
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def build(package,output,plotly_source=None):
    package=Path(package);output=Path(output);(output/'data').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((package/'manifest.json').read_text())
    assert manifest['complete'] and len(manifest['records'])==400 and not manifest['weights_saved']
    records=[];all_values={k:[] for k in ('gradient','smooth_gradient','local_shape_delta')}
    files={};valid_points=0;max_error=0
    for original in manifest['records']:
        r=dict(original);file=package/r['file'];assert sha(file)==r['sha256']
        with np.load(file) as z:values={k:z[k] for k in z.files}
        n=r['count'];assert 10<=n<=27
        assert values['geometry'].shape==(27,32,3) and int(values['count'])==n
        assert r['probability']>=.5 and abs(float(values['probability'])-r['probability'])<1e-4
        max_error=max(max_error,abs(float(values['probability'])-r['probability']))
        r['color_limits']={}
        for key in all_values:
            v=values[key][:n];assert np.isfinite(v).all() and np.all(values[key][n:]==0)
            all_values[key].append(v.ravel());r['color_limits'][key]=max(float(np.quantile(np.abs(v),.98)),1e-12)
        patches=[dict(indices=i.tolist(),margin_delta=float(m),probability_delta=float(p),probability=float(q))
                 for i,m,p,q in zip(values['patch_indices'],values['patch_margin_delta'],
                                   values['patch_probability_delta'],values['patch_probability'])]
        assert len(patches)==n*7
        assert np.isfinite(values['patch_margin_delta']).all()
        assert np.isfinite(values['patch_probability']).all()
        assert np.all((values['patch_probability']>=0)&(values['patch_probability']<=1))
        assert np.allclose(float(values['probability'])-values['patch_probability'],
                           values['patch_probability_delta'],rtol=0,atol=1e-12)
        changed_margin=float(values['margin'])-values['patch_margin_delta']
        exponential=np.exp(-np.abs(changed_margin))
        reconstructed=np.where(changed_margin>=0,1/(1+exponential),exponential/(1+exponential))
        assert np.allclose(reconstructed,values['patch_probability'],rtol=0,atol=2e-7)
        # Independently reconstruct the point-display averaging from patch values.
        displayed=np.zeros((27,32));multiplicity=np.zeros((27,32))
        for patch in patches:
            line,a,b=patch['indices'];displayed[line,a+1:b]+=patch['margin_delta'];multiplicity[line,a+1:b]+=1
        assert np.array_equal(displayed/np.maximum(multiplicity,1),values['local_shape_delta'])
        data={k:values[k][:n].tolist() for k in ('geometry','normalized_geometry',*all_values)}
        data['patches']=patches
        target=output/'data'/f"{r['id']}.js"
        target.write_text('window.registerSaliency('+json.dumps(r['id'])+','+
            json.dumps(data,separators=(',',':'),allow_nan=False)+');\n',encoding='utf-8')
        files[str(target.relative_to(output))]=sha(target);records.append(r);valid_points+=n*32
    payload=dict(records=records,model=manifest['model'],
        color_limits={k:max(float(np.quantile(np.abs(np.concatenate(v)),.98)),1e-12) for k,v in all_values.items()},
        methods=dict(gradient='norm(d(hairpin_logit-nonhairpin_logit)/d(normalized_geometry))',
                     smooth_gradient='norm(mean_of_16_gradient_vectors_sigma_0.002)',
                     local_shape_delta='original_minus_chord_replacement_logit_margin_overlap_averaged_for_display',
                     point_indices='zero_based',neighbors='fixed_original_seed_neighbors',
                     shape_caveat='straightening also changes length, tangents and geometry normalization'))
    template=Path(__file__).parent/'templates/task4c_saliency.html'
    html=template.read_text(encoding='utf-8').replace('/*__PLOTLY__*/',get_plotly(plotly_source)).replace(
        '/*__PAYLOAD__*/',json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/'))
    (output/'index.html').write_text(html,encoding='utf-8')
    audit=dict(complete=True,version=manifest['version'],source_manifest_sha256=sha(package/'manifest.json'),
        model=manifest['model'],bundles=len(records),valid_points=valid_points,
        flows={f:dict(samples=sum(r['flow']==f for r in records),
                      true_positive=sum(r['flow']==f and r['label']==1 for r in records),
                      false_positive=sum(r['flow']==f and r['label']==0 for r in records)) for f in ('channel','tbl')},
        source_files_checked=len(records),all_display_averages_independently_recomputed=True,
        max_probability_error=max_error,html_sha256=sha(output/'index.html'),template_sha256=sha(template),
        builder_sha256=sha(__file__),files=files,methods=payload['methods'],color_limits=payload['color_limits'],
        randomization_diagnostics=manifest['randomization_diagnostics'],
        browser_interaction_verified=False,weights_present=False)
    write_json(output/'viewer_manifest.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('files','randomization_diagnostics','model')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--plotly-source')
    args=parser.parse_args();build(args.package,args.output,args.plotly_source)
