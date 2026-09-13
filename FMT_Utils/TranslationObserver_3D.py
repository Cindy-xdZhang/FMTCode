"""Pure translating Killing observer, in lab-velocity convention.

u_alpha(t) = alpha*b(t); d_alpha(t) = integral(u_alpha, t0, t).
The lab-to-observer map is y=x-d_alpha(t), hence Q=I, c=-d_alpha.
The observed field reads the original field at x=y+d_alpha(t).
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


class TranslationObserver:
    def __init__(self, times, velocities, t0):
        self.times = np.asarray(times, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.t0 = float(t0)
        if (self.times.ndim != 1 or len(self.times) < 2
                or self.velocities.shape != (len(self.times), 3)
                or not np.isfinite(self.times).all() or not np.isfinite(self.velocities).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Require increasing finite times [T] and velocities [T,3].')
        self._check_time(self.t0)
        # Interpolate velocity, then integrate this very same spline exactly.
        self._velocity = CubicSpline(self.times, self.velocities, axis=0,
                                     bc_type='natural', extrapolate=False)
        self._integral = self._velocity.antiderivative()

    def _check_time(self, t):
        t = np.asarray(t, dtype=np.float64)
        if not np.isfinite(t).all() or np.any(t < self.times[0]) or np.any(t > self.times[-1]):
            raise ValueError('Observer time outside supplied velocity schedule; no extrapolation.')

    def velocity(self, t, alpha=1.0):
        self._check_time(t)
        return float(alpha)*self._velocity(t)

    def displacement(self, t, alpha=1.0):
        self._check_time(t)
        return float(alpha)*(self._integral(t)-self._integral(self.t0))

    def observed_field(self, lab_velocity, alpha):
        def evaluate(y, t):
            original_position = np.asarray(y, dtype=np.float64) + self.displacement(t, alpha)
            return np.asarray(lab_velocity(original_position, t)) - self.velocity(t, alpha)
        return evaluate

    def transform_paths(self, lab_paths, times, alpha):
        """Exact coordinate pushforward at matching physical sample times."""
        return np.asarray(lab_paths, dtype=np.float64)-self.displacement(times, alpha)


def integrate_pathlines(velocity, initial, times, max_step):
    """RK4 at exact requested times; invalidate a line at its first invalid stage.

velocity(points[N,3], physical_time) returns [N,3]. An out-of-domain
query must return NaN, rather than clipping/extrapolating the original field.
"""
    times = np.asarray(times, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if (times.ndim != 1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or not np.isfinite(max_step) or max_step<=0
            or initial.ndim!=2 or initial.shape[1]!=3 or not np.isfinite(initial).all()):
        raise ValueError('Invalid initial points, sample times, or max_step.')
    paths = np.full((len(initial),len(times),3),np.nan)
    paths[:,0]=initial
    current=initial.copy()
    active=np.ones(len(initial),bool)
    for j,(left,right) in enumerate(zip(times[:-1],times[1:]),1):
        steps=max(1,int(np.ceil((right-left)/max_step)))
        h=(right-left)/steps
        for k in range(steps):
            ids=np.flatnonzero(active)
            if not len(ids):break
            x=current[ids];t=left+k*h
            k1=velocity(x,t)
            k2=velocity(x+.5*h*k1,t+.5*h)
            k3=velocity(x+.5*h*k2,t+.5*h)
            k4=velocity(x+h*k3,min(right,t+h))
            next_x=x+h*(k1+2*k2+2*k3+k4)/6
            # Validate the endpoint too; RK intermediate validity is insufficient.
            endpoint=velocity(next_x,min(right,t+h))
            valid=np.isfinite(next_x).all(axis=1)&np.isfinite(endpoint).all(axis=1)
            active[ids[~valid]]=False
            current[ids[valid]]=next_x[valid]
        paths[active,j]=current[active]
    return paths


def evaluate_observers(lab_velocity, initial_primitives, times, classifier, observer,
                       max_step, correspondence_tolerance=1e-5):
    """Reintegrate and reclassify all four frames with one fixed classifier.

classifier(observed_primitives[N,7,L,3]) returns binary labels [N] or
(labels, feature_matrix). Never refit inside this callback. Line zero is the
classified center path; the six neighbours supply its FMT representation.
"""
    initial=np.asarray(initial_primitives,dtype=np.float64)
    times=np.asarray(times,dtype=np.float64)
    if initial.ndim!=3 or initial.shape[1:]!=(7,3):
        raise ValueError('Expected center and six neighbours [N,7,3].')
    if not np.isclose(observer.t0,times[0],rtol=0,atol=1e-12):
        raise ValueError('All four observers must have identity transformation at the first sample.')
    levels=(0.,.25,.5,1.)
    paths=[]
    for alpha in levels:
        transformed_initial=initial-observer.displacement(times[0],alpha)
        paths.append(integrate_pathlines(observer.observed_field(lab_velocity,alpha),
                     transformed_initial.reshape(-1,3),times,max_step).reshape(len(initial),7,len(times),3))
    # Shared material primitives only; never choose survivors by their predictions.
    common=np.logical_and.reduce([np.isfinite(p).all(axis=(1,2,3)) for p in paths])
    if not common.any():raise ValueError('No complete material primitive survives all four integrations.')
    paths=np.stack([p[common] for p in paths])
    errors=[]
    for alpha,p in zip(levels,paths):
        expected=observer.transform_paths(paths[0],times,alpha)
        errors.append(float(np.max(np.linalg.norm(p-expected,axis=-1))))
    if max(errors)>correspondence_tolerance:
        raise ValueError(f'Observed integration / coordinate pushforward disagree: {errors}; reduce max_step.')
    labels=[];features=[]
    for p in paths:
        prediction=classifier(p.copy())
        y,f=prediction if isinstance(prediction,tuple) else (prediction,None)
        y=np.asarray(y)
        if y.shape!=(common.sum(),) or not np.isin(y,[0,1]).all():
            raise ValueError('Classifier must return one binary vortex label per center path.')
        labels.append(y.astype(bool))
        if f is not None:
            f=np.asarray(f)
            if f.ndim!=2 or f.shape[0]!=common.sum() or not np.isfinite(f).all():
                raise ValueError('Invalid feature matrix returned by classifier.')
        features.append(f)
    labels=np.stack(labels)
    audit={'levels':list(levels),'input_primitives':len(initial),'common_primitives':int(common.sum()),
           'excluded_incomplete_primitives':int((~common).sum()),'excluded_original_indices':np.flatnonzero(~common).tolist(),
           'exclusion_rule':'incomplete original-domain integration in at least one frame; shared mask before classification',
           'trajectory_correspondence_max_error':errors,'correspondence_tolerance':correspondence_tolerance,
           'changed_labels_vs_original':[int(np.count_nonzero(y!=labels[0])) for y in labels],
           'changed_label_fraction':[float(np.mean(y!=labels[0])) for y in labels],
           'feature_max_absolute_change':[None if f is None or features[0] is None else float(np.max(np.abs(f-features[0]))) for f in features],
           'classification':'Same fixed classifier, independently recomputed features at every observer level',
           'line_semantics':'One classified center path per seven-line primitive; all seven lines transformed and encoded',
           'frame_convention':'u=alpha*b(t); d=integral(u); y=x-d; observed(y,t)=v(y+d,t)-u(t)',
           'velocity_interpolation':'natural cubic spline in physical time; antiderivative of the same spline',
           'observer_velocity_times':observer.times.tolist(),'observer_velocity_samples':observer.velocities.tolist(),
           'sample_times':times.tolist(),'max_step':float(max_step)}
    return paths,labels,np.flatnonzero(common),audit
