"""VTK extraction and strict IVD primitives for Task6 vortex-core datasets.

These functions accept an explicitly supplied instantaneous vector field.
They do not choose or approximate the reference-frame transformation.
"""
from __future__ import annotations

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from FMT_Utils.GlobalMeanObserver_3D import volume_mean


def eligible_time_indices(times, original_interval, cylinder=False):
    """Select existing frames using physical time, never cropped-file indices."""
    times = np.asarray(times, dtype=np.float64)
    start, end = map(float, original_interval)
    if end <= start or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid original interval or non-increasing time coordinates.")
    lower = start + (0.5 if cylinder else 0.05) * (end - start)
    upper = start + 0.95 * (end - start)
    return np.flatnonzero((times >= lower - 1e-6) & (times <= upper + 1e-6))


def rectilinear_field(coordinates_zyx, velocity):
    """Preserve physical coordinates and VTK's x-fastest point ordering."""
    coordinates = [np.asarray(c, dtype=np.float64) for c in coordinates_zyx]
    values = np.asarray(velocity, dtype=np.float64)
    if values.shape != tuple(len(c) for c in coordinates) + (3,):
        raise ValueError("Expected velocity shape (Z,Y,X,3).")
    if not np.isfinite(values).all():
        raise ValueError("Missing velocities require an explicit valid-domain policy.")
    if any(len(c) < 3 or not np.isfinite(c).all() or np.any(np.diff(c) <= 0)
           for c in coordinates):
        raise ValueError("Coordinates must be finite, increasing, and have >=3 nodes.")
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(*[len(c) for c in coordinates[::-1]])
    for setter, coordinate in zip(
        (grid.SetZCoordinates, grid.SetYCoordinates, grid.SetXCoordinates), coordinates
    ):
        setter(numpy_to_vtk(coordinate, deep=True))
    vectors = numpy_to_vtk(np.ascontiguousarray(values.reshape(-1, 3)), deep=True)
    vectors.SetName("task6_velocity")
    grid.GetPointData().SetVectors(vectors)
    return grid


def instantaneous_vorticity_deviation(coordinates_zyx, velocity):
    """IVD = |curl(v) - volume_mean(curl(v))| on the supplied field.

    Derivatives use physical axis coordinates, with second-order edge formulas.
    The mean uses tensor-product trapezoidal volume weights on the whole domain.
    """
    values = np.asarray(velocity, dtype=np.float64)
    coordinates = [np.asarray(c, dtype=np.float64) for c in coordinates_zyx]
    if values.shape != tuple(len(c) for c in coordinates) + (3,):
        raise ValueError("Expected velocity shape (Z,Y,X,3).")
    if not np.isfinite(values).all():
        raise ValueError("Missing velocities require an explicit valid-domain policy.")
    z, y, x = coordinates
    wx = np.gradient(values[..., 2], y, axis=1, edge_order=2)
    wx -= np.gradient(values[..., 1], z, axis=0, edge_order=2)
    wy = np.gradient(values[..., 0], z, axis=0, edge_order=2)
    wy -= np.gradient(values[..., 2], x, axis=2, edge_order=2)
    wz = np.gradient(values[..., 1], x, axis=2, edge_order=2)
    wz -= np.gradient(values[..., 0], y, axis=1, edge_order=2)
    vorticity = np.stack((wx, wy, wz), axis=-1)
    mean, missing = volume_mean(vorticity, coordinates)
    if missing:
        raise ValueError("Unexpected missing vorticity.")
    ivd = np.linalg.norm(vorticity - mean, axis=-1)
    return ivd, {"mean_vorticity": mean.tolist(), "maximum": float(ivd.max()),
                 "threshold": float(0.5 * ivd.max()), "comparison": "strictly_greater"}


def candidate_mask(ivd):
    """Half of the maximum, not the median or any percentile."""
    ivd = np.asarray(ivd, dtype=np.float64)
    if not np.isfinite(ivd).all() or np.any(ivd < 0):
        raise ValueError("IVD must be finite and nonnegative.")
    return ivd > 0.5 * ivd.max()


def extract_vortex_core(grid, higher_order=False, faster_approximation=False):
    """Call vtkVortexCore on tetrahedra, retaining native criteria arrays.

    On VTK 9.5.0 the analytic axial-vortex test returns no lines for the
    image/rectilinear voxel representation, but succeeds after VTK's standard
    tetrahedral conversion. This conversion adds no velocity interpolation
    parameters and does not alter the supplied grid-point coordinates.
    """
    errors = []
    tetrahedra = vtk.vtkDataSetTriangleFilter()
    tetrahedra.SetInputData(grid)
    tetrahedra.TetrahedraOnlyOn()
    tetrahedra.Update()
    extractor = vtk.vtkVortexCore()
    extractor.AddObserver(vtk.vtkCommand.ErrorEvent,
                          lambda _obj, event: errors.append(str(event)))
    extractor.SetInputConnection(tetrahedra.GetOutputPort())
    extractor.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "task6_velocity"
    )
    extractor.SetHigherOrderMethod(bool(higher_order))
    extractor.SetFasterApproximation(bool(faster_approximation))
    extractor.Update()
    if errors or extractor.GetErrorCode():
        raise RuntimeError(f"vtkVortexCore failed: {errors}, code={extractor.GetErrorCode()}")
    result = vtk.vtkPolyData()
    result.DeepCopy(extractor.GetOutput())
    return result


def polylines(polydata):
    """Return each VTK line without connecting unrelated endpoints."""
    if polydata.GetNumberOfPoints() == 0:
        return []
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    lines = []
    ids = vtk.vtkIdList()
    cells = polydata.GetLines()
    cells.InitTraversal()
    while cells.GetNextCell(ids):
        index = np.fromiter((ids.GetId(i) for i in range(ids.GetNumberOfIds())), dtype=int)
        if len(index) >= 2:
            lines.append(points[index].copy())
    return lines


def signed_winding_about_axis(curve, center, tangent):
    """Signed turns around a specified local straight axis, for diagnostics.

    A curved core requires transported normal frames and nearest-segment
    association; this local-axis helper alone is not that acceptance test.
    An open curve returns accumulated turns, not an integer topological invariant.
    """
    curve = np.asarray(curve, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 3:
        raise ValueError("A winding diagnostic needs at least 3 three-dimensional points.")
    if not np.isfinite(curve).all() or not np.isfinite(center).all():
        raise ValueError("Non-finite curve or center.")
    norm = np.linalg.norm(tangent)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid tangent.")
    tangent = tangent / norm
    reference = np.eye(3)[np.argmin(np.abs(tangent))]
    first = np.cross(tangent, reference)
    first /= np.linalg.norm(first)
    second = np.cross(tangent, first)
    relative = curve - center
    a, b = relative @ first, relative @ second
    if np.min(np.hypot(a, b)) <= 1e-12:
        raise ValueError("Curve intersects the reference axis; winding is undefined.")
    delta = np.arctan2(a[:-1] * b[1:] - b[:-1] * a[1:],
                       a[:-1] * a[1:] + b[:-1] * b[1:])
    if np.any(np.abs(delta) >= np.pi * 0.95):
        raise ValueError("Angular sampling is too coarse to resolve winding.")
    return float(delta.sum() / (2 * np.pi))
