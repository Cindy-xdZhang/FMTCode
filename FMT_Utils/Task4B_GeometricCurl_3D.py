"""Geometry-only derivatives and train-internal spatial splits for Task4-b 5.2."""
import numpy as np
from FMT_Utils.Task4B_PatchSplit_3D import components,select_test_components,overlaps


def internal_split(manifest,seed,validation_patch_count=90):
    output={}
    for code,name in enumerate(('channel','tbl')):
        patches=[p for p in manifest['flows'][name]['patches'] if p['split']=='train']
        anchors=[p for p in patches if p['kind']=='hairpin_bbox']
        groups=components(anchors,1);rng=np.random.default_rng(seed+code)
        chosen=select_test_components(groups,int(np.floor(len(anchors)*.1+.5)),rng)
        validation=[p for i,p in enumerate(anchors) if i in chosen]
        fit_anchors=[p for i,p in enumerate(anchors) if i not in chosen]
        lows=np.array([p['low'] for p in fit_anchors]);highs=np.array([p['high'] for p in fit_anchors])
        windows=[p for p in patches if p['kind']=='random_window' and not overlaps(p['low'],p['high'],lows,highs,1).any()]
        required=validation_patch_count-len(validation)
        if len(windows)<required:raise ValueError('Insufficient train-only validation windows')
        validation.extend([windows[i] for i in rng.permutation(len(windows))[:required]])
        val_ids={p['patch_id'] for p in validation}
        vl=np.array([p['low'] for p in validation]);vh=np.array([p['high'] for p in validation])
        roles={p['patch_id']:(1 if p['patch_id'] in val_ids else
               -1 if overlaps(p['low'],p['high'],vl,vh,1).any() else 0) for p in patches}
        assert all(roles[p['patch_id']]==0 for p in fit_anchors)
        output[name]=dict(patch_roles=roles,counts={str(k):sum(v==k for v in roles.values()) for k in (-1,0,1)},
            fit_instance_ids=[p['instance_id'] for p in fit_anchors],
            validation_instance_ids=[p['instance_id'] for i,p in enumerate(anchors) if i in chosen],
            original_test_used=False,minimum_gap_voxels=1)
    return output


def integrate_geometry(velocity,seeds,h,low,high,speed_bound,*,mode,steps=16,offset=1.):
    offsets=np.r_[np.zeros((1,3)),np.eye(3)*offset,-np.eye(3)*offset]
    initial=np.broadcast_to(offsets,(len(seeds),7,3)).copy()
    qlow=np.full((len(seeds),3),np.inf);qhigh=np.full((len(seeds),3),-np.inf)
    def derivative(q):
        points=seeds[:,None,:]+q*h[:,None,None]
        assert np.isfinite(points).all()
        assert np.all(points>=low[:,None,:]-1e-11) and np.all(points<=high[:,None,:]+1e-11)
        v=velocity(points.reshape(-1,3)).reshape(points.shape)
        speed=np.linalg.norm(v,axis=-1,keepdims=True)
        assert np.isfinite(v).all() and (speed>0).all()
        if mode=='unit':result=v/speed
        elif mode=='time':
            result=v/speed_bound[:,None,None]
            assert np.max(np.linalg.norm(result,axis=-1))<=1+1e-8
        else:raise ValueError(mode)
        np.minimum(qlow,points.min(1),out=qlow);np.maximum(qhigh,points.max(1),out=qhigh)
        return result
    paths=[]
    for sign in (-1.,1.):
        q=initial.copy();half=[q]
        for _ in range(steps):
            k1=derivative(q);k2=derivative(q+sign*.5*k1)
            k3=derivative(q+sign*.5*k2);k4=derivative(q+sign*k3)
            q=q+sign*(k1+2*k2+2*k3+k4)/6;half.append(q)
        paths.append(half)
    return np.stack(paths[0][:0:-1]+paths[1],axis=2),qlow,qhigh


def geometry_derivatives(q):
    q=np.asarray(q,np.float64)
    tangent=np.gradient(q,axis=2,edge_order=2)
    separation=(q[:,1:4]-q[:,4:7]).transpose(0,2,3,1)
    difference=(tangent[:,1:4]-tangent[:,4:7]).transpose(0,2,3,1)
    jacobian=difference@np.linalg.pinv(separation,rcond=1e-8)
    curl=np.stack((jacobian[:,:,2,1]-jacobian[:,:,1,2],
                   jacobian[:,:,0,2]-jacobian[:,:,2,0],jacobian[:,:,1,0]-jacobian[:,:,0,1]),axis=-1)
    center=q.shape[2]//2;t=tangent[:,0,center];j=jacobian[:,center];omega=curl[:,center]
    speed=np.linalg.norm(t,axis=-1,keepdims=True);norm=np.linalg.norm(omega,axis=-1,keepdims=True)
    dot=np.sum(t*omega,axis=-1,keepdims=True)
    cosine=np.abs(dot)/np.maximum(speed*norm,1e-15)
    strain=(j+j.transpose(0,2,1))*.5
    descriptors=np.concatenate((t/np.maximum(speed,1e-15),tangent[:,:,center].reshape(len(q),-1),
        j.reshape(len(q),-1),(j/np.maximum(speed[:,None],1e-15)).reshape(len(q),-1),
        omega,omega/np.maximum(norm,1e-15),speed,norm,cosine,dot,
        np.trace(j,axis1=1,axis2=2)[:,None],np.linalg.norm(strain,axis=(1,2))[:,None]),axis=1)
    assert np.isfinite(descriptors).all() and np.isfinite(curl).all()
    return tangent,curl,descriptors,cosine.ravel()


def encoded_geometry(q):
    import torch
    from FMT_Utils.DFT_FMT_3D import pathline_dft_features_3d,dft_rotation_invariants_3d
    tangent,curl,descriptors,cosine=geometry_derivatives(q)
    options=dict(num_freq=6,neighbor_weight=1.,neighbor_scale=100.,neighbor_pool='sort',mode='gram',include_chirality=True)
    base=pathline_dft_features_3d(torch.from_numpy(q.astype(np.float32)),**options)
    differential=pathline_dft_features_3d(torch.from_numpy(tangent.astype(np.float32)),**options)
    curl_fmt=dft_rotation_invariants_3d(torch.from_numpy(curl.astype(np.float32)),6,'gram',True).numpy()
    return np.concatenate((base,differential,curl_fmt,descriptors),axis=1).astype(np.float32),cosine
