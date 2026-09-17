"""Compare read-only Ibex snapshots with local/Git/archive sources."""
import ast
import base64
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/Verify_FMTArchitectureAudit_1.1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    rows = list(csv.DictReader((OUT/'source_inventory.csv').open(encoding='utf-8-sig')))
    exact, normalized, syntax = {}, {}, {}
    by_path = {}
    proc = subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,cwd=ROOT)
    archives = {}
    def add(row,data):
        exact.setdefault(digest(data),row)
        normalized.setdefault(digest(data.replace(b'\r\n',b'\n')),row)
        by_path.setdefault(row['path'],[]).append((row,data))
        if row['path'].endswith('.py'):
            try:
                sig=ast.dump(ast.parse(data.decode('utf-8-sig')),include_attributes=False)
                syntax.setdefault(digest(sig.encode()),row)
            except (ValueError,SyntaxError,UnicodeError):
                pass
    for row in rows:
        if row['origin']=='working_tree':
            p=ROOT/row['path']
            if p.exists(): add(row,p.read_bytes())
        elif row['origin']=='git_history':
            proc.stdin.write((row['blob']+'\n').encode());proc.stdin.flush()
            head=proc.stdout.readline().decode().split()
            data=proc.stdout.read(int(head[2]));proc.stdout.read(1)
            add(row,data)
        else:
            name=row['origin'].split(':',1)[1]
            if name not in archives: archives[name]=zipfile.ZipFile(ROOT/'.research_archive'/name)
            add(row,archives[name].read(row['path']))
    proc.stdin.close();proc.wait()
    remote=json.loads((OUT/'ibex_unmatched_contents.json').read_text(encoding='utf-8-sig'))
    results=[]
    for row in remote:
        data=base64.b64decode(row['content_base64'])
        assert digest(data)==row['sha256'],row['path']
        match=exact.get(digest(data));status='exact_match'
        if match is None:
            match=normalized.get(digest(data.replace(b'\r\n',b'\n')));status='line_endings_only'
        if match is None and row['path'].endswith('.py'):
            try:
                sig=ast.dump(ast.parse(data.decode('utf-8-sig')),include_attributes=False)
                match=syntax.get(digest(sig.encode()));status='same_python_AST'
            except (ValueError,SyntaxError,UnicodeError):pass
        if match is None:status='different_content_requires_review'
        result={k:v for k,v in row.items() if k!='content_base64'}
        result.update(status=status,local_match=match)
        if match is None:
            target=OUT/'remote_only'/row['sha256']/row['path']
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            result['local_snapshot']=str(target.relative_to(ROOT))
        results.append(result)
    (OUT/'ibex_reconciliation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    from collections import Counter
    summary=dict(remote_roots=198,remote_file_records=71744,initial_unmatched_contents=len(remote),statuses=dict(Counter(r['status'] for r in results)))
    (OUT/'ibex_reconciliation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(json.dumps([dict(path=r['path'],sha256=r['sha256'],root=r['roots'][0]) for r in results if r['status']=='different_content_requires_review'],indent=2))


if __name__=='__main__':main()
