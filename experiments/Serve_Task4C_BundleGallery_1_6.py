"""Serve frozen Couette data in the original gallery without changing old assets."""
import json
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote
from experiments.Serve_Task4C_BundleGallery_1_3 import GalleryServer

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/Verify_Task4C_BundleGallery_1.1/viewer'
DATA=ROOT/'outputs/mainExp_Task4C_CouetteDataset_2.1'
GALLERY=DATA/'gallery'
VERSION='Other_Task4C_BundleGallery_1.6'


def load(name,root=GALLERY):return json.loads((root/name).read_text(encoding='utf8'))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(BASE),**kwargs)

    def translate_path(self,path):
        request=unquote(urlparse(path).path)
        if request.startswith('/couette/'):
            resolved=(GALLERY/request.removeprefix('/couette/')).resolve()
            if resolved.is_relative_to(GALLERY.resolve()):return str(resolved)
            return str(GALLERY/'nonexistent')
        return super().translate_path(path)

    def send_data(self,value,kind='application/json; charset=utf-8'):
        body=(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)).encode('utf8')
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)

    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/status':return self.send_data(dict(version=VERSION,dataset_complete=True,preview_only=False,dataset=load('data_audit.json',DATA)))
        if path=='/manifest.json':
            m=load('manifest.json',BASE);m['flows']['couette']=load('flow.json');m['couette_extension']=load('complete.json')
            return self.send_data(m)
        if path=='/instances_manifest_1_2.json':
            m=load('instances_manifest_1_2.json',BASE);m['flows']['couette']=load('instances.json');return self.send_data(m)
        if path in ('/','/index.html'):
            html=(BASE/'index.html').read_text(encoding='utf8')
            html=html.replace("fetch('manifest.json')","fetch('manifest.json?v=couette-dataset-2p1')")
            html=html.replace('src="bundle_gallery_instances_1_2.js"','src="bundle_gallery_instances_1_2.js?v=couette-dataset-2p1"')
            html=html.replace('<option value="tbl">TBL</option>','<option value="tbl">TBL</option><option value="couette">Couette</option>')
            html=html.replace("value('flow')==='channel'?'Channel':'TBL'","({channel:'Channel',tbl:'TBL',couette:'Couette'})[value('flow')]")
            html=html.replace('</body>','<script src="/couette_dataset_gallery.js"></script></body>')
            return self.send_data(html,'text/html; charset=utf-8')
        if path=='/bundle_gallery_instances_1_2.js':
            script=(BASE/'bundle_gallery_instances_1_2.js').read_text(encoding='utf8')
            script=script.replace("fetch('instances_manifest_1_2.json')","fetch('instances_manifest_1_2.json?v=couette-dataset-2p1')")
            return self.send_data(script,'text/javascript; charset=utf-8')
        if path=='/couette_dataset_gallery.js':return self.send_data((ROOT/'experiments/templates/task4c_bundle_gallery_1_6.js').read_text(encoding='utf8'),'text/javascript; charset=utf-8')
        return super().do_GET()


if __name__=='__main__':
    assert load('complete.json')['complete'] and load('data_audit.json',DATA)['complete']
    print(VERSION+' http://127.0.0.1:8768/index.html?flow=couette',flush=True)
    GalleryServer(('127.0.0.1',8768),Handler).serve_forever()
