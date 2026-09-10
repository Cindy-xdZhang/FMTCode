"""Audit actual exported figures and create QA contact sheets of every example."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pymupdf as fitz
from PIL import Image


def audit(folder,scripts):
    reports=[]
    pdfs=[p for p in sorted(folder.glob('*.pdf')) if not p.name.endswith('.collisions.pdf')]
    if not pdfs: raise ValueError('No actual exported figures found')
    for pdf in pdfs:
        text=subprocess.run([sys.executable,str(scripts/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
        (folder/(pdf.stem+'.text_audit.json')).write_text(text.stdout,encoding='utf-8')
        collision=subprocess.run([sys.executable,str(scripts/'audit_figure_collisions.py'),str(pdf),
            '--json-out',str(folder/(pdf.stem+'.collisions.json')),
            '--overlay-pdf',str(folder/(pdf.stem+'.collisions.pdf'))],capture_output=True,text=True)
        item=dict(figure=pdf.name,text_exit=text.returncode,collision_exit=collision.returncode,
            collision_report=collision.stdout)
        reports.append(item)
        print(json.dumps(item))
        if text.returncode or collision.returncode:
            raise RuntimeError(text.stdout+text.stderr+collision.stdout+collision.stderr)
    for dataset in sorted({p.stem.split('_primitive_')[0] for p in pdfs if '_primitive_' in p.stem}):
        rendered=[]
        for pdf in [p for p in pdfs if p.stem.startswith(dataset+'_primitive_')]:
            with fitz.open(pdf) as doc:
                pix=doc[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False)
                rendered.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
        contact=Image.new('RGB',(max(p.width for p in rendered),sum(p.height for p in rendered)),color='white')
        y=0
        for panel in rendered:
            contact.paste(panel,(0,y));y+=panel.height
        contact.save(folder/(dataset+'.qa_contact.png'))
    (folder/'export_audit.json').write_text(json.dumps(dict(reports=reports,
        visual_review='Contact sheets require explicit human or agent visual inspection; not inferred from automated tests'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--audit-scripts',type=Path,required=True)
    args=parser.parse_args();audit(args.folder,args.audit_scripts)
