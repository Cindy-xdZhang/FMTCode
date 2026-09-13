"""Select an exact, label-independent count of complete long pathline primitives."""
import numpy as np
from scipy.stats import qmc
from FMT_Utils.TranslationObserver_3D import integrate_pathlines


def dense_candidates(initial, offset, seed=7080):
    initial=np.asarray(initial,dtype=float)
    low=initial[:,0].min(axis=0);high=initial[:,0].max(axis=0)
    points=qmc.scale(qmc.Sobol(3,scramble=True,seed=seed).random_base2(15),low,high)
    offsets=np.array([[0,0,0],[offset,0,0],[-offset,0,0],[0,offset,0],
                      [0,-offset,0],[0,0,offset],[0,0,-offset]])
    return np.concatenate([initial,points[:,None]+offsets]),[low.tolist(),high.tolist()]


def select_complete(lab_velocity,candidates,times,max_step,target,batch_size=1024):
    selected=[];selected_ids=[];invalid=[];unused_valid=[];count=0;attempted=0
    for start in range(0,len(candidates),batch_size):
        batch=candidates[start:start+batch_size]
        paths=integrate_pathlines(lab_velocity,batch.reshape(-1,3),times,max_step)
        valid=np.isfinite(paths.reshape(len(batch),7,len(times),3)).all(axis=(1,2,3))
        ids=np.flatnonzero(valid);take=ids[:target-count]
        selected.append(batch[take]);selected_ids.extend((start+take).tolist())
        invalid.extend((start+np.flatnonzero(~valid)).tolist())
        unused_valid.extend((start+ids[len(take):]).tolist())
        count+=len(take);attempted+=len(batch)
        print(f'Complete long primitives selected: {count}/{target}; attempted: {attempted}',flush=True)
        if count==target:break
    if count!=target:raise ValueError(f'Only {count} complete primitives for requested {target}.')
    audit={'candidate_pool_count':len(candidates),'candidate_attempted_count':attempted,
           'candidate_invalid_count':len(invalid),'candidate_invalid_indices':invalid,
           'candidate_valid_not_needed_indices':unused_valid,'selected_candidate_indices':selected_ids,
           'selection_rule':'Original seeds first, then scrambled Sobol seed7080 within their spatial extent; first complete 7-line primitives; no labels or features used'}
    return np.concatenate(selected),audit
