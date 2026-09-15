"""Independent float64 reference and float32 forward-error bounds for statistics."""
import numpy as np


def audit_training_statistics(data,feature,reported,scalar_expected):
    for key,value in scalar_expected.items():
        np.testing.assert_allclose(reported[key],value,rtol=1e-8,atol=1e-9)
    if feature=='none':return {'feature':feature,'status':'PASS'}
    raw=np.concatenate([d[feature][:,d['valid_low']] for d in data],1).astype(np.float64)
    values=np.sign(raw)*np.log1p(np.abs(raw))
    mean=values.mean(1);deviation=values.std(1)
    rms=np.sqrt(np.mean(values*values,axis=1))
    # Production applies log1p in float32 before float64 reductions. Four
    # float32 epsilons cover the elementary transform's rounding budget;
    # mean and standard deviation propagate that pointwise budget differently.
    epsilon=4*np.finfo(np.float32).eps
    mean_bound=epsilon*np.abs(values).mean(1)+1e-12
    std_bound=epsilon*rms+1e-12
    mean_actual=np.asarray(reported['feature_mean'],dtype=np.float64)
    std_actual=np.asarray(reported['feature_std'],dtype=np.float64)
    assert mean_actual.shape==mean.shape and std_actual.shape==deviation.shape
    assert np.isfinite(mean_actual).all() and np.isfinite(std_actual).all() and (std_actual>0).all()
    np.testing.assert_array_less(np.abs(mean_actual-mean),np.nextafter(mean_bound,np.inf))
    small=deviation<1e-8
    constant=np.ptp(raw,axis=1)==0
    assert not np.any((np.abs(deviation-1e-8)<=std_bound)&~constant),'Ambiguous standard-deviation cutoff'
    np.testing.assert_array_equal(std_actual[small],np.ones(small.sum()))
    np.testing.assert_array_less(np.abs(std_actual[~small]-deviation[~small]),np.nextafter(std_bound[~small],np.inf))
    reference_std=np.where(small,1.,deviation)
    actual=np.clip((values-mean_actual[:,None])/std_actual[:,None],-8,8)
    reference=np.clip((values-mean[:,None])/reference_std[:,None],-8,8)
    normalized_difference=float(np.max(np.abs(actual-reference)))
    assert normalized_difference<1e-4,normalized_difference
    return {'feature':feature,'status':'PASS','float32_epsilon_multiplier':4,
            'max_mean_absolute_error':float(np.max(np.abs(mean_actual-mean))),
            'max_std_absolute_error':float(np.max(np.abs(std_actual-reference_std))),
            'max_mean_bound_ratio':float(np.max(np.abs(mean_actual-mean)/mean_bound)),
            'max_std_bound_ratio':float(np.max(np.abs(std_actual-reference_std)/std_bound)),
            'max_normalized_difference':normalized_difference}
