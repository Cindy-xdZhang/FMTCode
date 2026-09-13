"""Verify the annotation-only revision against the preceding exported figures."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--original',type=Path,default=Path('outputs/Verify_AIVDTranslationObservers_1.2/cylinder3d'))
    parser.add_argument('--revised',type=Path,default=Path('outputs/Verify_AIVDTranslationObservers_1.3/cylinder3d'))
    args=parser.parse_args();result={}
    for medium in ('paper','slides'):
        a=np.asarray(Image.open(args.original/'final'/f'{medium}.png').convert('RGB'))
        b=np.asarray(Image.open(args.revised/'final'/f'{medium}.png').convert('RGB'))
        assert a.shape==b.shape
        diff=np.any(a!=b,axis=2);height,width=diff.shape
        allowed=np.zeros(diff.shape,dtype=bool)
        def box(x0,y0,x1,y1):
            allowed[int((1-y1)*height):int((1-y0)*height)+1,
                    int(x0*width):int(x1*width)+1]=True
        # Only the top subtitle and right-hand annotations may change.
        box(.12,.963,.88,.981)
        for i in range(7):
            baseline=(.735-i*.225+3*.225+.191)/1.675
            box(.68,baseline-.023,.97,baseline+.014)
        changed_outside=int(np.sum(diff&~allowed))
        assert changed_outside==0,changed_outside
        meta=json.loads((args.revised/'final'/f'{medium}.json').read_text())
        old=json.loads((args.original/'final'/f'{medium}.json').read_text())
        for name in ('shared_bounds','camera','levels','changed_labels_vs_original','vortex_counts'):
            assert meta[name]==old[name],name
        annotations=meta['camera_annotations'];assert len(annotations)==7
        length=60 if medium=='paper' else 90
        direction=np.array(annotations[0]['arrow_direction_screen'])
        full_velocity=np.array(annotations[6]['velocity_vector_at_arrow_time'])
        for i,item in enumerate(annotations):
            assert item['arrow_time']==10.5
            assert np.isclose(item['arrow_length_pt'],length*i/6)
            np.testing.assert_allclose(item['arrow_direction_screen'],direction,atol=1e-12)
            np.testing.assert_allclose(item['velocity_vector_at_arrow_time'],i/6*full_velocity,atol=1e-12)
        ids={el.attrib.get('id') for el in ET.parse(args.revised/'final'/f'{medium}.svg').iter()}
        for i in range(7):
            assert f'camera_{i}' in ids and f'camera_velocity_{i}' in ids
            assert (f'camera_velocity_arrow_{i}' if i else 'camera_velocity_zero_0') in ids
        result[medium]={'status':'PASS','pixels_changed':int(diff.sum()),
                        'pixels_changed_outside_annotations':changed_outside,
                        'velocity_arrow_groups':6,'zero_velocity_markers':1,
                        'arrow_lengths_pt':[item['arrow_length_pt'] for item in annotations],
                        'full_velocity_at_t0':full_velocity.tolist(),
                        'projected_direction':direction.tolist(),
                        'preserved':'trajectory pixels, bounds, camera, original labels and counts'}
    qa=args.revised/'qa';qa.mkdir(exist_ok=True)
    (qa/'pixel_preservation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
