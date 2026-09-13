"""Audit the downloaded image files and package the user-requested figures.

Run after all rendering jobs finish. Overview images are browsing previews;
the individually exported figures remain the scientific submission artifacts.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/task123_plotdeps'),str(Path.home()/'.codex/skills/nature-figure/scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import audit_figure_collisions as collision
import audit_pdf_text as pdf_text

out=ROOT/'outputs/Other_Task123_PaperTriptychs_1.2'
datasets=['cylinder3d','halfcylinderRe640','halfcylinderRe6400','tangaroa','boeing747','deltaWing_LBM']
images=sorted(p for medium in ['paper','slides'] for p in (out/medium).iterdir()
              if p.suffix.lower() in ['.png','.pdf','.svg','.tiff'])
assert len(images)==126,f'Expected 126 images, got {len(images)}'
records=[]
for p in images:
    row={'file':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,
         'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    if p.suffix in ['.png','.tiff']:
        with Image.open(p) as im:
            row['pixels']=list(im.size);im.verify()
    if p.suffix=='.pdf':
        a=collision.audit_pdf(p);b=pdf_text.audit_pdf(p.read_bytes(),5)
        p.with_suffix('.collision-audit.json').write_text(json.dumps(a,indent=2))
        p.with_suffix('.font-audit.json').write_text(json.dumps(b,indent=2))
        assert a['summary']['fail']==0,(p.name,a)
        assert b['below_minimum_count']==0,(p.name,b)
        assert all(f['severity']=='WARN' and f['kind']=='text-fill-edge' and f['text']=='z' for f in a['findings']),(p.name,a)
        row.update(minimum_font_pt=b['minimum_found_pt'],collision_summary=a['summary'],
                   warning_review='Pending final visual review: z direction label crosses white background edge.')
    records.append(row)
for task in ['task1','task2','task3']:
    fig=plt.figure(figsize=(14.4,8.7),facecolor='white')
    for i,dataset in enumerate(datasets):
        ax=fig.add_axes([(i%2)*.5,1-(i//2+1)/3,.5,1/3])
        ax.imshow(plt.imread(out/'paper'/f'{dataset}_{task}_triptych.png'))
        ax.set_axis_off()
    fig.savefig(out/f'overview_{task}_mode_{"a" if task=="task1" else "b"}.png',dpi=240)
    plt.close(fig)
archive=out/'Task123_SixFlows_ModeAB_figures.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in images+sorted(out.glob('overview_*.png')):
        z.write(p,p.relative_to(out).as_posix())
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==129
    assert z.testzip() is None
manifest={'experiment':out.name,'scientific_triptychs':18,'layout_variants':2,'image_files':126,
          'overview_png':3,'images':records,'visual_review':'pending',
          'archive':{'file':archive.name,'bytes':archive.stat().st_size,
                     'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},
          'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [ROOT/'experiments/Visualize_Task123_PaperTriptychs_1_2.py',
                                     ROOT/'experiments/Build_Task123_DisplayAssets_1_2.py',
                                     ROOT/'config/Other_Task123_PaperTriptychs_1.2.json']}}
(out/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'images':126,'pdf_audits':36,'collision_fail':0,
                  'minimum_font_pt':sorted({r['minimum_font_pt'] for r in records if 'minimum_font_pt' in r}),
                  'archive_MB':round(archive.stat().st_size/1e6,2)},indent=2))
