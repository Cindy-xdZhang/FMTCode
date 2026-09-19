"""3h adaptive ball query with exact center-started FPS on frozen v2 samples."""
from types import FunctionType
from pathlib import Path
import numpy as np
from numba import njit
from FMT_Utils import Task4C_BallQuery_2_2 as frozen


@njit(cache=True)
def _fps(points, center, ids, k):
    minimum = np.empty(len(ids), np.float64)
    for i in range(len(ids)):
        d = points[ids[i]]-center
        minimum[i] = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
    result = np.empty(k, np.int64)
    for j in range(k):
        pick = np.argmax(minimum); result[j] = ids[pick]
        for i in range(len(ids)):
            d = points[ids[i]]-points[ids[pick]]
            distance = d[0]*d[0]+d[1]*d[1]+d[2]*d[2]
            minimum[i] = min(minimum[i], distance)
        minimum[pick] = -np.inf
    return result


def select_fps(points, center, ids, k):
    return _fps(points, center, np.sort(np.asarray(ids, np.int64)), k)


ball_tables = FunctionType(frozen.ball_tables.__code__, dict(frozen.__dict__, select_fps=select_fps),
                           argdefs=frozen.ball_tables.__defaults__)
_build = FunctionType(frozen.build_neighbor_tables.__code__, dict(frozen.__dict__, ball_tables=ball_tables))
Dataset = frozen.Dataset
load_neighbors = frozen.load_neighbors


def build_neighbor_tables(spec, sha):
    assert spec['neighbors']['radius_h'] == 3 and spec['neighbors']['h_mode'] == 'minimum_axis_mean_spacing'
    for flow in spec['flows']:
        h = np.load(Path(spec['grid_scale_output'])/f"h_{flow['name']}.npy")
        assert np.all(h == flow['h_value'])
    return _build(spec, sha)
