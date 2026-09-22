import numpy as np
from FMT_Utils.Task6Sampling_2_1 import CandidateSampler,SegmentQuery
from FMT_Utils.Task6CorelineDataset_1_1 import coreline_labels


def test_independent_labels_match_brute_segments_including_strict_boundary():
    core=np.array([[0,0,0],[1,0,0],[1,1,0]],float);h=.25
    points=np.r_[np.random.default_rng(96611).uniform(-.5,1.5,(500,3)),[[.5,h,0],[.5,np.nextafter(h,0),0]]]
    exact=np.full(len(points),np.inf)
    for a,b in zip(core[:-1],core[1:]):
        d=b-a;t=np.clip((points-a)@d/(d@d),0,1)
        exact=np.minimum(exact,np.linalg.norm(points-a-t[:,None]*d,axis=1))
    query=SegmentQuery([core]).distance(points,h)
    np.testing.assert_array_equal(query<h,exact<h)
    np.testing.assert_array_equal(query<h,coreline_labels(points,[core],h)['label'])
    assert not (query[-2]<h) and query[-1]<h


def test_candidate_support_preserves_unseedable_label_geometry():
    axes=[np.linspace(-1,1,21)]*3
    ivd=np.broadcast_to((axes[2]<0)[None,None,:],(21,21,21)).astype(float)
    cores=[np.array([[-.4,0,-.8],[-.4,0,.8]]),np.array([[.9,0,-.8],[.9,0,.8]])]
    sampler=CandidateSampler(ivd,axes,cores,.1,.5)
    assert sampler.eligible==[0] and len(sampler.cores)==2
    targets=sampler.targets(400);seeds,_=sampler.propose(targets)
    assert (sampler.interpolate(seeds[:,::-1])>.5).all()
    assert (SegmentQuery([cores[0]]).distance(seeds[targets==0],.2)<.2).all()
    assert np.sum(targets==0)==200 and np.sum(targets==-1)==200
