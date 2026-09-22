"""Strict reader for the supplied uniform 3D Amira file series."""
from pathlib import Path
import re
import numpy as np


def read_amira(path):
    raw=Path(path).read_bytes()
    marker=re.search(br'(?m)^#\s*Data section follows\r?\n@1\r?\n',raw)
    if not marker:raise ValueError('Missing Amira binary block marker')
    header=raw[:marker.start()].decode('ascii')
    if not header.startswith('# AmiraMesh BINARY-LITTLE-ENDIAN 2.1'):raise ValueError('Unsupported Amira encoding')
    if not re.search(r'Lattice\s*\{\s*float\[3\]\s+\w+\s*\}\s*@1',header):raise ValueError('Expected three interleaved components')
    if 'CoordType "uniform"' not in header:raise ValueError('Expected uniform coordinates')
    shape=tuple(map(int,re.search(r'define Lattice (\d+) (\d+) (\d+)',header).groups()))
    bounds=np.array(list(map(float,re.search(r'BoundingBox ([^\r\n]+)',header).group(1).strip(' ,').split()))).reshape(3,2)
    if np.any(bounds[:,1]<=bounds[:,0]):raise ValueError('Invalid physical bounds')
    count=int(np.prod(shape))*3;end=marker.end()+4*count
    if len(raw)<end or raw[end:].strip():raise ValueError('Unexpected binary payload length')
    field=np.frombuffer(raw,dtype='<f4',count=count,offset=marker.end()).reshape(*shape[::-1],3).copy()
    if not np.isfinite(field).all():raise ValueError('Non-finite source velocity')
    axes=[np.linspace(*b,n,dtype=np.float64) for b,n in zip(bounds,shape)][::-1]
    return field,axes,dict(shape_xyz=list(shape),bounds_xyz=bounds.tolist(),payload_offset=marker.end(),byte_count=count*4)


def read_series(folder):
    folder=Path(folder);text=(folder/'SquareCylinder.fileseries').read_text()
    bounds=list(map(float,re.search(r'AdditionalBounds ([^\r\n]+)',text).group(1).split()))
    names=re.findall(r'(?m)^flow_t\d+\.am\s*$',text);names=[n.strip() for n in names]
    count=int(re.search(r'NumOfFileSteps (\d+)',text).group(1))
    if len(names)!=count or len(set(names))!=count or any(not (folder/n).is_file() for n in names):raise ValueError('Incomplete file series')
    return [folder/n for n in names],np.linspace(bounds[0],bounds[1],count,dtype=np.float64)
