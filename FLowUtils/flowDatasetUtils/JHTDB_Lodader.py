"""JHTDB velocity sampling, adapted from PyflowVis JHTDB_Lodader.py.

getData queries physical coordinates in the stationary-wall frame. Output arrays
use FMT order (T,Z,Y,X,3) or (T,B,A,2). No credentials are stored in this module.
"""
from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np



dataset_meta={
    "channel":{
        "range":{
            "x":(0,8*np.pi),
            "y":(-1,1),
            "z":(0,3*np.pi)
        },
        "resolution":{
            "x":2048 ,
            "y":512 ,
            "z":1536 
        }
    },
    "channel5200":{
        "range":{
            "x":(0,8*np.pi),
            "y":(-1,1),
            "z":(0,3*np.pi)
        }
}
}

class JHTDB_Lodader:

    def __init__(self, auth_token=None, output_path='./outputs/jhtdb',
                 max_query_points=65536, max_retries=3):
        self.auth_token = auth_token or os.environ.get('JHTDB_AUTH_TOKEN')
        if not self.auth_token:
            raise ValueError('Set JHTDB_AUTH_TOKEN or pass auth_token explicitly')
        self.output_path = str(output_path)
        self.max_query_points = self._positive_int(max_query_points, 'max_query_points')
        self.max_retries = self._positive_int(max_retries, 'max_retries')
        self._dataset_cache = {}

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
        return int(value)

    @classmethod
    def _axis(cls, resolution, bounds):
        n = cls._positive_int(resolution, 'resolution')
        if len(bounds) != 2 or not np.isfinite(bounds).all():
            raise ValueError('Axis bounds must contain two finite numbers')
        if bounds[1] < bounds[0] or (n > 1 and bounds[1] == bounds[0]):
            raise ValueError('Multi-point axis bounds must strictly increase')
        return np.linspace(*bounds, num=n, dtype=np.float64)

    def _get_or_create_dataset(self, dataset_name: str):
        """Instantiate or fetch cached givernylocal dataset object."""
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        from givernylocal.turbulence_dataset import turb_dataset
        dataset = turb_dataset(dataset_title=dataset_name, output_path=self.output_path, auth_token=self.auth_token)
        self._dataset_cache[dataset_name] = dataset
        return dataset



    # ---------- internal query helpers ----------
    def _query_getData(self, dataset_name, variable_name, times, temporal_method,
                       spatial_method, spatial_operator, points, option=None,
                       maxQueryOnce=None):
        """Query one time at a time; option is deliberately not a time-series request.

        Passing option=[end,dt] inside this time loop would request overlapping
        series. The original loader computed that option but did not use it.
        """
        from givernylocal.turbulence_toolkit import getData
        import time
        if variable_name != 'velocity' or spatial_operator != 'field':
            raise ValueError('Vector-field loaders require velocity / field')
        if option is not None:
            raise ValueError('Pass explicit times, not a getData time-series option')
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('points must be finite (N,3) coordinates')
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
            raise ValueError('times must be a nonempty finite vector')
        if dataset_name == 'channel':
            if np.any((times < 0) | (times > 25.9935)):
                raise ValueError('channel time must lie in [0,25.9935]')
            if np.any(np.abs(points[:, 1]) > 1):
                raise ValueError('channel y must lie in [-1,1]')
            if temporal_method == 'none' and not np.allclose(times / 0.0065, np.round(times / 0.0065), rtol=0, atol=1e-8):
                raise ValueError('Use pchip for channel times off the stored 0.0065 grid')
        batch = self._positive_int(maxQueryOnce or self.max_query_points, 'maxQueryOnce')
        dataset = self._get_or_create_dataset(dataset_name)
        result = np.empty((len(times), len(points), 3), dtype=np.float32)
        for ti, query_time in enumerate(times):
            for start in range(0, len(points), batch):
                end = min(start + batch, len(points))
                for attempt in range(self.max_retries):
                    try:
                        response = getData(dataset, variable_name, float(query_time),
                                           temporal_method, spatial_method, spatial_operator,
                                           points[start:end], verbose=False)
                        if len(response) != 1:
                            raise ValueError('Expected exactly one returned time slice')
                        values = np.asarray(response[0], dtype=np.float32)
                        if values.shape != (end-start, 3):
                            raise ValueError(f'Wrong velocity response shape: {values.shape}')
                        if not np.isfinite(values).all() or np.any(values == np.float32(-999.9)):
                            raise ValueError('Invalid/missing velocity values returned')
                        result[ti, start:end] = values
                        break
                    except Exception as exc:
                        # Giverny/requests errors may embed a token-bearing URL.
                        message = str(exc).replace(self.auth_token, '<redacted>')
                        if attempt + 1 == self.max_retries:
                            raise RuntimeError(f'JHTDB t={query_time}, points {start}:{end}: {message}') from None
                        time.sleep(min(2 ** attempt, 8))
        return result

    @staticmethod
    def _generate_times(temporal_method, time_start, time_res, time_end, integerTime=False):
        """Return exactly time_res uniform samples; integerTime means snapshot IDs.

        Snapshot IDs apply only to datasets whose API uses snapshot numbering,
        not channel physical times. Temporal 'none' also supports multiple times.
        """
        n = JHTDB_Lodader._positive_int(time_res, 'time_res')
        if temporal_method not in ('none', 'pchip'):
            raise ValueError('temporal_method must be none or pchip')
        if not np.isfinite(time_start):
            raise ValueError('time_start must be finite')
        if n == 1:
            if time_end is not None and time_end != time_start:
                raise ValueError('A single sample requires time_end == time_start or None')
            times = np.array([time_start], dtype=np.float64)
        else:
            if time_end is None or not np.isfinite(time_end) or time_end <= time_start:
                raise ValueError('Multiple samples require finite time_end > time_start')
            times = np.linspace(time_start, time_end, n, dtype=np.float64)
        if integerTime:
            if not np.allclose(times, np.round(times), rtol=0, atol=1e-9):
                raise ValueError('integerTime requires integral snapshot IDs')
            times = np.round(times).astype(np.int64)
        return times, None

    @staticmethod
    def _generate_grid_3d(x_res: int, y_res: int, z_res: int,
                          x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D grid and flattened points array (N,3)."""
        x_vals = JHTDB_Lodader._axis(x_res, x_range)
        y_vals = JHTDB_Lodader._axis(y_res, y_range)
        z_vals = JHTDB_Lodader._axis(z_res, z_range)
        gx, gy, gz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # (X,Y,Z)
        points = np.stack([gx.ravel(order='C'), gy.ravel(order='C'), gz.ravel(order='C')], axis=1)
        return x_vals, y_vals, z_vals, points

    @staticmethod
    def _generate_grid_2d(plane: str, a_res: int, b_res: int, fixed_value: float,
                          a_range: Tuple[float, float], b_range: Tuple[float, float]) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Generate 2D plane grid points given plane id: 'xy','xz','yz'.

        Returns:
            plane: normalized plane string
            a_vals, b_vals: coordinate arrays along the plane
            points: flattened points array (N,3)
        """
        p = plane.lower()
        a_vals = JHTDB_Lodader._axis(a_res, a_range)
        b_vals = JHTDB_Lodader._axis(b_res, b_range)
        ga, gb = np.meshgrid(a_vals, b_vals, indexing='ij')  # (A,B)
        if p == 'xy':
            x_flat = ga.ravel(order='C')
            y_flat = gb.ravel(order='C')
            z_flat = np.full_like(x_flat, float(fixed_value))
        elif p == 'xz':
            x_flat = ga.ravel(order='C')
            z_flat = gb.ravel(order='C')
            y_flat = np.full_like(x_flat, float(fixed_value))
        elif p == 'yz':
            y_flat = ga.ravel(order='C')
            z_flat = gb.ravel(order='C')
            x_flat = np.full_like(y_flat, float(fixed_value))
        else:
            raise ValueError("plane must be one of 'xy','xz','yz'")
        points = np.stack([x_flat, y_flat, z_flat], axis=1)
        return p, a_vals, b_vals, points

    @staticmethod
    def _pack_structured_grid_3d(per_time_values: List[np.ndarray], x_res: int, y_res: int, z_res: int) -> np.ndarray:
        """Pack (N,3) over time into (T, Z, Y, X, 3) using reshape + transpose (no index inference)."""
        T = len(per_time_values)
        data_5d = np.zeros((T, int(z_res), int(y_res), int(x_res), 3), dtype=np.float32)
        for t, vals in enumerate(per_time_values):
            expected = int(x_res) * int(y_res) * int(z_res)
            if vals.shape[0] != expected or vals.shape[1] != 3:
                raise ValueError("Each time slice must be (X*Y*Z, 3)")
            data_5d[t] = vals.reshape(int(x_res), int(y_res), int(z_res), 3).transpose(2, 1, 0, 3)
        return data_5d

    @staticmethod
    def _pack_structured_grid_2d(per_time_values: List[np.ndarray], a_res: int, b_res: int, plane: str) -> Tuple[np.ndarray, Tuple[str, str]]:
        """Pack plane data (N,3) into (T, H, W, 2) by plane selection.

        Plane mapping:
            'xy' → keep (u,v) → axes names ('ydim','xdim') [H=Y, W=X]
            'xz' → keep (u,w) → axes names ('zdim','xdim') [H=Z, W=X]
            'yz' → keep (v,w) → axes names ('zdim','ydim') [H=Z, W=Y]
        """
        T = len(per_time_values)
        H, W = int(b_res), int(a_res)  # map to (H=Wrt second axis, W=Wrt first axis)
        data_4d = np.zeros((T, H, W, 2), dtype=np.float32)
        if plane == 'xy':
            axis_names = ('ydim', 'xdim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (X, Y, 3)
                # transpose to (Y, X, comp)
                arr = arr.transpose(1, 0, 2)
                data_4d[t, :, :, 0] = arr[:, :, 0]
                data_4d[t, :, :, 1] = arr[:, :, 1]
        elif plane == 'xz':
            axis_names = ('zdim', 'xdim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (X, Z, 3)
                arr = arr.transpose(1, 0, 2)  # (Z, X, 3)
                data_4d[t, :, :, 0] = arr[:, :, 0]
                data_4d[t, :, :, 1] = arr[:, :, 2]
        elif plane == 'yz':
            axis_names = ('zdim', 'ydim')
            for t, vals in enumerate(per_time_values):
                arr = vals.reshape(int(a_res), int(b_res), 3)  # (Y, Z, 3) if we feed (A=Y, B=Z)
                arr = arr.transpose(1, 0, 2)  # (Z, Y, 3)
                data_4d[t, :, :, 0] = arr[:, :, 1]
                data_4d[t, :, :, 1] = arr[:, :, 2]
        else:
            raise ValueError("plane must be 'xy','xz','yz'")
        return data_4d, axis_names


    def load_2d_unsteadyFlow(self, dataset_name: str,    
                              plane: str, a_res: int, b_res: int, fixed_value: float,
                              a_range: Tuple[float, float], b_range: Tuple[float, float],
                              time_start: float,  time_end: float,time_res: int,
                              spatial_method: str="lag8",
                              spatial_operator: str="field",
                              variable_name: str="velocity",
                              integerTime:bool=False, temporal_method:Optional[str]=None, return_array:bool=False
                              ) -> UnsteadyVectorField2D:
        """Generate plane grid from resolution, query, and return UnsteadyVectorField2D."""
        p, a_vals, b_vals, points = self._generate_grid_2d(plane, a_res, b_res, fixed_value, a_range, b_range)
        if integerTime and dataset_name == 'channel':
            raise ValueError('channel uses physical time, not integer snapshot IDs')
        if temporal_method is None:
            temporal_method = "none" if time_res == 1 or integerTime else "pchip"
            if dataset_name == 'channel' and time_res == 1 and not np.isclose(
                    time_start / 0.0065, round(time_start / 0.0065), rtol=0, atol=1e-8):
                temporal_method = 'pchip'
        times, option = self._generate_times(temporal_method, time_start, time_res, time_end,integerTime)
        per_time_values = self._query_getData(dataset_name, variable_name, times, temporal_method,
                                                 spatial_method, spatial_operator, points, option)
        data_4d, axis_names = self._pack_structured_grid_2d(per_time_values, a_res, b_res, p)
        if return_array:
            return data_4d
        from FLowUtils.VectorField2d import UnsteadyVectorField2D
        Ydim, Xdim = data_4d.shape[1], data_4d.shape[2]
        xmin, xmax = float(a_vals.min()), float(a_vals.max())
        ymin, ymax = float(b_vals.min()), float(b_vals.max())
        tmin = float(times[0]) if times.size else 0.0
        tmax = float(times[-1]) if times.size else 0.0
        vf2d = UnsteadyVectorField2D(Xdim=Xdim, Ydim=Ydim, time_steps=int(times.size if times.size else 1),
                                     domainMinBoundary=[xmin, ymin, tmin], domainMaxBoundary=[xmax, ymax, tmax])
        vf2d.jhtdb_coordinates = {p[0]: a_vals, p[1]: b_vals}
        vf2d.jhtdb_times = times
        vf2d.jhtdb_plane = p
        vf2d.jhtdb_fixed_value = float(fixed_value)
        vf2d.field = data_4d
        return vf2d













    def load_3d_unsteadyFlow(self, dataset_name: str, variable_name: str,
                              temporal_method: str, spatial_method: str, spatial_operator: str,
                              x_res: int, y_res: int, z_res: int,
                              time_start: float, time_res: int, time_end: Optional[float],
                              x_range: Tuple[float, float], y_range: Tuple[float, float], z_range: Tuple[float, float], return_array:bool=False) -> UnsteadyVectorField3D:
        """Generate 3D grid from resolution, query, and return UnsteadyVectorField3D."""
        times, option = self._generate_times(temporal_method, time_start, time_res, time_end)
        x_vals, y_vals, z_vals, points = self._generate_grid_3d(x_res, y_res, z_res, x_range, y_range, z_range)
        per_time_values = self._query_getData(dataset_name, variable_name, times, temporal_method,
                                              spatial_method, spatial_operator, points, option)
        data_5d = self._pack_structured_grid_3d(per_time_values, x_res, y_res, z_res)
        if return_array:
            return data_5d
        from FLowUtils.VectorField3d import UnsteadyVectorField3D
        tmin = float(times[0])
        tmax = float(times[-1])
        vf3d = UnsteadyVectorField3D(Xdim=int(x_res), Ydim=int(y_res), Zdim=int(z_res),
                                     time_steps=int(times.size if times.size else 1),
                                     domainMinBoundary=[float(x_vals.min()), float(y_vals.min()), float(z_vals.min())],
                                     domainMaxBoundary=[float(x_vals.max()), float(y_vals.max()), float(z_vals.max())],
                                     tmin=tmin, tmax=tmax)
        vf3d.jhtdb_coordinates = dict(x=x_vals, y=y_vals, z=z_vals)
        vf3d.jhtdb_times = times
        vf3d.field = data_5d
        return vf3d


# Correctly spelled alias; keep the original import path/API compatible.
JHTDBLoader = JHTDB_Lodader
