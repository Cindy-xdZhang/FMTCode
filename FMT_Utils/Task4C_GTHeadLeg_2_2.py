"""Additional binary rule: positive omega_y and nearer perpendicular to velocity."""
import numpy as np


def classify(velocity, omega, inside=None):
    velocity=np.asarray(velocity,np.float64);omega=np.asarray(omega,np.float64)
    if velocity.shape!=omega.shape or omega.ndim!=2 or omega.shape[1]!=3:
        raise ValueError('velocity and omega must both have shape (N, 3)')
    inside=np.ones(len(omega),bool) if inside is None else np.asarray(inside,bool)
    if inside.shape!=(len(omega),):raise ValueError('inside must have shape (N,)')
    finite=np.isfinite(velocity).all(1)&np.isfinite(omega).all(1)
    speed=np.linalg.norm(np.where(finite[:,None],velocity,0),axis=1)
    strength=np.linalg.norm(np.where(finite[:,None],omega,0),axis=1)
    valid=inside&finite&(speed>0)&(strength>0)
    v=velocity[valid]/speed[valid,None];w=omega[valid]/strength[valid,None]
    dot=np.abs(np.einsum('ij,ij->i',v,w));cross=np.linalg.norm(np.cross(v,w),axis=1)
    nearer=np.zeros(len(omega),bool);nearer[valid]=cross>dot
    angle=np.full(len(omega),np.nan);angle[valid]=np.degrees(np.arctan2(cross,dot))
    head=valid&(omega[:,1]>0)&nearer
    # User requests all remaining cells as leg; undefined angles cannot be head.
    return dict(label=head.astype(np.uint8),head=head,valid_angle=valid,
                nearer_perpendicular=nearer,acute_angle_deg=angle)


def summary(labels,volumes):
    counts=np.bincount(labels,minlength=2);volume=np.bincount(labels,weights=volumes,minlength=2)
    return dict(cells=int(counts.sum()),volume=float(volume.sum()),counts=counts.tolist(),volumes=volume.tolist(),
                volume_fractions=(volume/volume.sum()).tolist() if volume.sum()>0 else [0.,0.])
