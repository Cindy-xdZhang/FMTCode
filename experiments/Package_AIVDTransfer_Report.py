"""Package audited metrics, figures, explanation and reproducible source links."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/Verify_AIVDTransfer_1.1')
    args = parser.parse_args()
    root = Path(args.root)
    audit = json.loads((root/'local_aggregate_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['underlying_metric_rows'] == 350
    shutil.copy2('docs/aivd1w3_dft_explained_zh.md', root/'aivd1w3_dft_explained_zh.md')
    shutil.copy2('docs/Verify_AIVDTransfer_1.1.md', root/'protocol.md')
    shutil.copy2('config/Verify_AIVDTransfer_1.1.json', root/'config.json')
    code = root/'code'
    code.mkdir(exist_ok=True)
    names = ['Verify_AIVDTransfer_3D.py', 'Preflight_AIVDTransfer_3D.py',
             'Audit_AIVDTransfer_3D.py', 'Summarize_AIVDTransfer_3D.py',
             'Plot_AIVDTransfer_Comparison.py', 'Package_AIVDTransfer_Report.py']
    for name in names:
        shutil.copy2(Path('experiments')/name, code/name)
    files = [p for p in root.glob('*') if p.is_file() and p.suffix in ('.md', '.csv', '.json')
             and p.name != 'delivery_manifest.json']
    files += [p for p in (root/'figures').glob('*') if p.is_file() and
              p.suffix in ('.pdf', '.svg', '.png', '.json', '.csv', '.md') and
              not any(s in p.name for s in ('.alignment.svg', '.collision-audit.pdf'))]
    files += list(code.glob('*.py'))
    files += [p for p in (root/'reproducibility').glob('*') if p.is_file()]
    assert all(p.suffix.lower() not in ('.pt', '.pth', '.ckpt') for p in files)
    archive = root/'figures_and_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(files):
            z.write(path, path.relative_to(root).as_posix())
    manifest = {
        'experiment': 'Verify_AIVDTransfer_1.1', 'archive': archive.name,
        'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
        'files': {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)},
        'models_included': False,
        'predictions': 'Complete scientific predictions, per-run metrics and logs remain on Ibex; this bundle contains aggregate performance data and source code.',
    }
    (root/'delivery_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
