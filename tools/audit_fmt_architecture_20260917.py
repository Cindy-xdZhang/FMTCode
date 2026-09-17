"""Read-only inventory of current, archived and Git-history research sources.

Static matches are evidence locations, NOT proof that a branch executed.
The accompanying human-reviewed report follows the actual task adapters.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/Verify_FMTArchitectureAudit_1.1'
SUFFIXES = {'.py', '.json', '.yaml', '.yml', '.sh', '.js', '.ts', '.tsx', '.html', '.ipynb', '.ps1', '.bat', '.cpp', '.h', '.cu'}
CONV = re.compile(r'^(?:Conv(?:Transpose)?[123]d|conv(?:_transpose)?[123]d|convolve|convolution|fftconvolve)$')
CENTER = re.compile(r'(?:central|center|anchor)\s*=\s*torch\.arange|ids\s*=\s*torch\.cat\(\((?:central|center)|for\s+\w+\s+in\s+range\((?:7|n|line_count)\)|\[j for j in range\(7\) if j != i\]|(?:bundle_fmt|line_fmt|fourier_tokens|fmt_bundle_features)\(')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def csv_write(name, rows, columns):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, contents = [], {}

    def add(origin, path, blob, commit, data):
        digest = hashlib.sha256(data).hexdigest()
        contents.setdefault(digest, data)
        items.append(dict(origin=origin, path=path, blob=blob, commit=commit, sha256=digest, bytes=len(data)))

    current = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    for name in sorted(set(current)):
        path = ROOT/name
        if path.is_file() and path.suffix in SUFFIXES:
            add('working_tree', name, '', '', path.read_bytes())
    for archive in sorted((ROOT/'.research_archive').glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if Path(name).suffix in SUFFIXES:
                    add('archive:'+archive.name, name, '', '', z.read(name))

    # Every added/changed blob reachable from every local Git ref, not just HEAD.
    raw = git('log', '--all', '--raw', '--no-renames', '--no-abbrev', '--format=COMMIT %H', '--', *['*'+suffix for suffix in sorted(SUFFIXES)]).decode('utf-8')
    references = {}
    commit = ''
    for line in raw.splitlines():
        if line.startswith('COMMIT '):
            commit = line.split()[1]
        elif line.startswith(':'):
            meta, path = line.split('\t', 1)
            fields = meta.split(); blob = fields[3]
            if set(blob) == {'0'} or Path(path).suffix not in SUFFIXES:
                continue
            references.setdefault((path, blob), commit)
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for (path, blob), commit in references.items():
        process.stdin.write((blob+'\n').encode()); process.stdin.flush()
        header = process.stdout.readline().decode().strip().split()
        if len(header) != 3 or header[1] != 'blob':
            raise RuntimeError(header)
        data = process.stdout.read(int(header[2])); assert process.stdout.read(1) == b'\n'
        add('git_history', path, blob, commit, data)
    process.stdin.close(); process.wait()

    parsed, errors = {}, []
    for digest, data in contents.items():
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            parsed[digest] = dict(imports=[], calls=[], center_hits=[], parse_status='non_utf8')
            continue
        result = dict(imports=[], calls=[], center_hits=[], parse_status='not_python')
        result['center_hits'] = [dict(line=i, text=line.strip()) for i, line in enumerate(text.splitlines(), 1) if CENTER.search(line)]
        paths = [x['path'] for x in items if x['sha256'] == digest]
        if any(p.endswith('.py') for p in paths):
            try:
                tree = ast.parse(text); result['parse_status'] = 'ok'
                stack = []

                class Visitor(ast.NodeVisitor):
                    def visit_ClassDef(self, node):
                        stack.append(node.name); self.generic_visit(node); stack.pop()
                    def visit_FunctionDef(self, node):
                        stack.append(node.name); self.generic_visit(node); stack.pop()
                    visit_AsyncFunctionDef = visit_FunctionDef
                    def visit_ImportFrom(self, node):
                        result['imports'].append(dict(module=node.module, names=[n.name for n in node.names], line=node.lineno))
                    def visit_Import(self, node):
                        result['imports'].extend(dict(module=n.name, names=[], line=node.lineno) for n in node.names)
                    def visit_Call(self, node):
                        name = ast.unparse(node.func)
                        tail = name.split('.')[-1]
                        if CONV.match(tail) or tail in ('rfft', 'fft', 'dct'):
                            result['calls'].append(dict(kind='convolution' if CONV.match(tail) else 'Fourier_or_cosine_transform',
                                 call=name, owner='.'.join(stack), line=node.lineno, expression=ast.unparse(node)[:1200]))
                        self.generic_visit(node)
                Visitor().visit(tree)
            except (SyntaxError, ValueError) as exc:
                result['parse_status'] = str(exc)
                errors.append(dict(sha256=digest, paths=paths, error=str(exc)))
        parsed[digest] = result

    conv, centers, configs = [], [], []
    import yaml
    for item in items:
        info = parsed[item['sha256']]
        item.update(parse_status=info['parse_status'], convolution_calls=sum(c['kind'] == 'convolution' for c in info['calls']),
                    transform_calls=sum(c['kind'] != 'convolution' for c in info['calls']), center_candidates=len(info['center_hits']))
        for call in info['calls']:
            if call['kind'] == 'convolution':
                conv.append({**item, **call})
        for hit in info['center_hits']:
            centers.append({**item, **hit, 'status': 'candidate_requires_call_chain_review'})
        if Path(item['path']).suffix in {'.json', '.yaml', '.yml'} and item['path'].startswith(('config/', 'pnn/config')):
            text = contents[item['sha256']].decode('utf-8-sig')
            try:
                data = json.loads(text) if item['path'].endswith('.json') else yaml.safe_load(text)
                experiment = data.get('version', data.get('experiment', data.get('name', ''))) if isinstance(data, dict) else ''
                configs.append({**item, 'experiment':experiment, 'configuration':data})
            except Exception as exc:
                configs.append({**item, 'experiment':'', 'configuration':None, 'error':str(exc)})
    csv_write('source_inventory.csv', items, list(items[0]))
    csv_write('convolution_calls.csv', conv, list(items[0])+['owner', 'line', 'call', 'expression'])
    csv_write('center_candidates.csv', centers, list(items[0])+['line', 'text', 'status'])
    (OUT/'configurations.json').write_text(json.dumps(configs, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    csv_write('configuration_inventory.csv', [dict(c, configuration=json.dumps(c['configuration'],ensure_ascii=False,default=str)) for c in configs],
              list(items[0])+['experiment','configuration','error'])
    (OUT/'parsed_sources.json').write_text(json.dumps(parsed, ensure_ascii=False),encoding='utf-8')
    (OUT/'parse_errors.json').write_text(json.dumps(errors, ensure_ascii=False,indent=2),encoding='utf-8')
    directories = [dict(directory=p.name, status='presence_is_not_proof_of_completed_training') for p in sorted((ROOT/'outputs').iterdir()) if p.is_dir()]
    csv_write('output_directory_inventory.csv', directories, ['directory','status'])
    summary = dict(head=git('rev-parse','HEAD').decode().strip(), sources=len(items), unique_contents=len(contents),
        source_origins={origin:sum(x['origin']==origin for x in items) for origin in sorted({x['origin'] for x in items})},
        configurations=len(configs), distinct_config_paths=len({c['path'] for c in configs}), convolution_occurrences=len(conv),
        unique_conv_source_contents=len({c['sha256'] for c in conv}), parse_errors=len(errors), output_directories=len(directories),
        limitation='AST hits do not establish executed branches. All local reachable Git source versions are inventoried; remote-only/uncommitted historical copies cannot be certified from this inventory.')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
