"""Non-destructive 1:2 class-balanced, 4:1 train/test load indices for Couette."""
import json
import numpy as np
from experiments.Build_Task4C_CouetteDataset_2_1 import OUT, write, sha


def pick(meta, available, positive, negative, seed):
    priority=np.random.default_rng(seed).permutation(available)
    mandatory=[]
    # Preserve a strict candidate/GT intersection seed for every represented instance.
    for iid in np.unique(meta['instance'][available]):
        eligible=priority[(meta['gt_owner'][priority]==iid)&(meta['instance'][priority]==iid)]
        if not len(eligible):raise ValueError(f'No strict GT head seed for instance {iid}')
        mandatory.append(int(eligible[0]))
    selected=list(mandatory)
    for label,count in [(1,positive),(0,negative)]:
        have=sum(meta['label'][i]==label for i in mandatory)
        assert have<=count
        allowed=priority[(meta['label'][priority]==label)&~np.isin(priority,mandatory)]
        assert len(allowed)>=count-have
        selected.extend(allowed[:count-have].tolist())
    selected=np.sort(selected).astype(np.int64)
    assert int(meta['label'][selected].sum())==positive and len(selected)==positive+negative
    return selected


def main():
    audit=json.loads((OUT/'data_audit.json').read_text());assert audit['complete']
    with np.load(OUT/'physical/couette/metadata.npz') as z:meta=dict(z)
    report=dict(version='Verify_Task4C_CouetteLoadIndices_2.1',complete=True,seed=96611,
        source_audit_sha256=sha(OUT/'data_audit.json'),geometry_and_labels_unchanged=True,
        positive_to_negative='1:2',train_to_test='4:1',max_train_seeds=12000,folds={})
    for f in range(5):
        train=np.flatnonzero(meta['fold']!=f);test=np.flatnonzero(meta['fold']==f)
        ptrain=int(meta['label'][train].sum());ptest=int(meta['label'][test].sum())
        unit=min(1000,ptrain//4,ptest,(len(train)-ptrain)//8,(len(test)-ptest)//2)
        if unit:
            tr=pick(meta,train,4*unit,8*unit,[96611,2,f,0]);te=pick(meta,test,unit,2*unit,[96611,2,f,1])
            assert not set(meta['instance'][tr])&set(meta['instance'][te])
            assert len(tr)==4*len(te)
        else:
            # Couette has four independent groups. No fabricated fifth held-out group.
            tr=te=np.empty(0,np.int64)
        dest=OUT/f'load_indices/fold{f}';dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'train_seed_indices.npy',tr);np.save(dest/'test_seed_indices.npy',te)
        report['folds'][str(f)]=dict(train=len(tr),test=len(te),train_positive=int(meta['label'][tr].sum()),test_positive=int(meta['label'][te].sum()),
            train_instances=np.unique(meta['instance'][tr]).tolist(),test_instances=np.unique(meta['instance'][te]).tolist(),
            unavailable_reason=None if unit else 'No Couette test group in this fold; retain the existing Channel/TBL experiment unchanged',
            files={p.name:sha(p) for p in dest.glob('*.npy')})
    write(OUT/'load_indices/audit.json',report)
    dataset=json.loads((OUT/'dataset.json').read_text());dataset['couette_load_indices']=dict(path='load_indices/audit.json',sha256=sha(OUT/'load_indices/audit.json'),
        default_fold=0,description='Balanced supervised views; full geometry and original instance-fold memberships remain intact')
    write(OUT/'dataset.json',dataset)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
