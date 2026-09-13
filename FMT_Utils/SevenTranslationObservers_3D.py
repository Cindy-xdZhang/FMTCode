"""Seven-level verification; frozen four-level implementation remains unchanged."""
import numpy as np
from FMT_Utils.TranslationObserver_3D import integrate_pathlines

def evaluate_observers(lab_velocity, initial_primitives, times, classifier, observer,
                       max_step, correspondence_tolerance=1e-5):
    """Reintegrate and reclassify all seven frames with one fixed classifier.

classifier(observed_primitives[N,7,L,3]) returns binary labels [N] or
(labels, feature_matrix). Never refit inside this callback. Line zero is the
classified center path; the six neighbours supply its FMT representation.
"""
    initial=np.asarray(initial_primitives,dtype=np.float64)
    times=np.asarray(times,dtype=np.float64)
    if initial.ndim!=3 or initial.shape[1:]!=(7,3):
        raise ValueError('Expected center and six neighbours [N,7,3].')
    if not np.isclose(observer.t0,times[0],rtol=0,atol=1e-12):
        raise ValueError('All seven observers must have identity transformation at the first sample.')
    levels=tuple(np.linspace(0.,1.,7))
    paths=[]
    for alpha in levels:
        transformed_initial=initial-observer.displacement(times[0],alpha)
        paths.append(integrate_pathlines(observer.observed_field(lab_velocity,alpha),
                     transformed_initial.reshape(-1,3),times,max_step).reshape(len(initial),7,len(times),3))
    # Shared material primitives only; never choose survivors by their predictions.
    common=np.logical_and.reduce([np.isfinite(p).all(axis=(1,2,3)) for p in paths])
    if not common.any():raise ValueError('No complete material primitive survives all seven integrations.')
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
