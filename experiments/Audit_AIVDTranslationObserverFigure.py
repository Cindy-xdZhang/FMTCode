"""Inspect the final exported PDFs and render every panel for human review."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--skill-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_root/'scripts'))
    import fitz
    import audit_pdf_text
    import audit_figure_collisions
    result = {}
    qa = args.directory/'qa'
    qa.mkdir(exist_ok=True)
    for medium in ['paper', 'slides']:
        stem = args.directory/'final'/medium
        pdf = stem.with_suffix('.pdf')
        fonts = audit_pdf_text.audit_pdf(pdf.read_bytes(), 5)
        collisions = audit_figure_collisions.audit_pdf(pdf)
        Path(str(stem)+'.font-audit.json').write_text(json.dumps(fonts, indent=2))
        Path(str(stem)+'.collision-audit.json').write_text(json.dumps(collisions, indent=2))
        assert not fonts['below_minimum_count']
        assert not collisions['summary']['fail']
        with fitz.open(pdf) as document:
            page = document[0]
            width, height = page.rect.width, page.rect.height
            page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(qa/f'{medium}_whole.png')
            for i in range(7):
                bottom = (.735-i*.225+3*.225)/1.675
                rect = fitz.Rect(.025*width, (1-bottom-.213/1.675)*height,
                                 .985*width, (1-bottom+.023/1.675)*height)
                page.get_pixmap(matrix=fitz.Matrix(1.6 if medium=='paper' else 1, 1.6 if medium=='paper' else 1),
                               clip=rect, alpha=False).save(qa/f'{medium}_panel_{"abcdefg"[i]}.png')
        result[medium] = {'fonts': fonts, 'collision_summary': collisions['summary'],
                          'panel_alignment': json.loads(Path(str(stem)+'.alignment.json').read_text()),
                          'human_review': 'pending'}
    (qa/'automatic_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
