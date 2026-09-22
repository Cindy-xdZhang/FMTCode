"""Publish the verified field package at the user-selected local data address."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,datetime
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'outputs/Other_Task6_FieldPackage_1.1/package'
QA=ROOT/'outputs/Verify_Task6_FieldPackage_1.1'
BASE=Path('C:/Users/xingdi/OneDrive - KAUST/WorkingInProcess/FLowVisAssets/flowData3D')
TARGET=BASE/'Task6_CorelineDataset_2.7_share'
STAGE=BASE/'Task6_CorelineDataset_2.7_share_fields_staging_20260922'
BACKUP=BASE/'Task6_CorelineDataset_2.7_share_legacy_streamlines_20260922'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def finish():
 import numpy as np
 report=read(QA/'verification.json');assert report['complete']
 rebuilt=QA/'recompiled_re640/relative_velocity.npy'
 expected=PACK/'frames/halfcylinderRe640/frame_067/relative_velocity.npy'
 assert sha(rebuilt)==sha(expected)
 report['windows_source_build']=dict(passed=True,compiler='MSVC 19.42.34436.0 x64',
   library_sha256=sha(QA/'native_build/Release/reference.dll'),
   frame='halfcylinderRe640:67',bitwise_equal=True,field_sha256=sha(rebuilt))
 write(QA/'verification.json',report)
 shutil.copyfile(ROOT/'docs/Task6_ggt17_portable_1.1.md',PACK/'GGT17.md')
 write(PACK/'provenance/verification.json',report)
 manifest=read(PACK/'dataset.json');old_sha=sha(PACK/'dataset.json')
 manifest['files_sha256']['GGT17.md']=sha(PACK/'GGT17.md')
 manifest['files_sha256']['provenance/verification.json']=sha(PACK/'provenance/verification.json')
 write(PACK/'dataset.json',manifest);digest=sha(PACK/'dataset.json')
 split=read(PACK/'default_split.json');split['dataset_sha256']=digest;write(PACK/'default_split.json',split)
 pointer=read(PACK/'current.json');pointer['sha256']=digest;write(PACK/'current.json',pointer)
 write(PACK.parent/'final_package.json',dict(complete=True,manifest_sha256=digest,
   prior_manifest_sha256=old_sha,update='Correct Re640 filename; attach completed portability and numerical verification',
   field_and_label_payloads_unchanged=True,files=len(manifest['files_sha256']),
   bytes=sum((PACK/p).stat().st_size for p in manifest['files_sha256'])))
 print(json.dumps(read(PACK.parent/'final_package.json')),flush=True)
def publish():
 final=read(PACK.parent/'final_package.json');assert final['complete'] and final['manifest_sha256']==sha(PACK/'dataset.json')
 for p in (TARGET,STAGE,BACKUP):assert p.resolve().parent==BASE.resolve()
 assert not STAGE.exists() and not BACKUP.exists(),'Inspect previous publication before retrying'
 old=read(TARGET/'dataset.json');assert 'total_samples' in old and 'observed-fields' not in old.get('schema','')
 old_sha=sha(TARGET/'dataset.json')
 expected=read(ROOT/'outputs/mainExp_Task6_CorelineDataset_2.7/current.json')
 assert expected['frozen_sha256']==read(PACK/'dataset.json')['source_frozen_sha256']
 print('Copying verified field package to destination staging',flush=True)
 shutil.copytree(PACK,STAGE,ignore=shutil.ignore_patterns('__pycache__'))
 manifest=read(STAGE/'dataset.json')
 for i,(name,digest) in enumerate(manifest['files_sha256'].items()):
  assert sha(STAGE/name)==digest,name
  if (i+1)%200==0:print(json.dumps(dict(stage='destination_hashes',files=i+1)),flush=True)
 assert sha(STAGE/'dataset.json')==final['manifest_sha256']
 assert read(STAGE/'current.json')['sha256']==final['manifest_sha256']
 assert not any(p.name in ('sample_streamlines.npy','seeds.npy','labels.npy','order6.npy','order16.npy') for p in STAGE.rglob('*'))
 # Same-parent directory renames; preserve the old dataset and roll back on failure.
 TARGET.rename(BACKUP)
 try:STAGE.rename(TARGET)
 except BaseException:
  BACKUP.rename(TARGET)
  raise
 assert sha(BACKUP/'dataset.json')==old_sha
 report=dict(complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   dataset_root=str(TARGET),manifest_sha256=sha(TARGET/'dataset.json'),files=len(manifest['files_sha256']),
   frames=len(manifest['frames']),corelines=sum(f['core_count'] for f in manifest['frames']),
   primary_data='observed v-u field frames plus human-refined coreline labels',streamlines_stored=False,
   historical_streamline_package=str(BACKUP),historical_manifest_sha256=old_sha,
   windows_observer_cli='tools/ggt17_observe.py',manual_source_revision=expected)
 write(PACK.parent/'publication.json',report)
 write(ROOT/'config/Task6_current_dataset.json',dict(dataset_root=str(TARGET),
   manifest_sha256=report['manifest_sha256'],schema=manifest['schema'],
   editor_revision_workspace='outputs/mainExp_Task6_CorelineDataset_2.7',
   editor_current_at_publication=expected,streamlines_are_dataset=False))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['finish','publish']);a=p.parse_args()
 finish() if a.phase=='finish' else publish()
