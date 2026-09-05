"""Discrete 3D voxel segmentations loaded from VTK label fields.

The regular voxel grid is deliberately independent of the source flow-field grid.
Labels use NumPy order ``(Z, Y, X)``.  Positive integers identify segmented
regions; zero and negative source values are background.

This module ports the data and rendering behavior of optimal-connection's
``ObjectVoxelAnnotation3D`` without depending on the PyflowVis GUI engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VoxelGrid3D:
    """A regular cube grid over a physical 3D bounding box."""

    resolution_xyz: tuple[int, int, int]
    domain_min_xyz: np.ndarray
    domain_max_xyz: np.ndarray

    def __post_init__(self):
        resolution = tuple(int(value) for value in self.resolution_xyz)
        if len(resolution) != 3 or any(value < 1 for value in resolution):
            raise ValueError("voxel resolution must contain three positive integers")
        domain_min = np.asarray(self.domain_min_xyz, dtype=np.float64).reshape(3)
        domain_max = np.asarray(self.domain_max_xyz, dtype=np.float64).reshape(3)
        if not (np.isfinite(domain_min).all() and np.isfinite(domain_max).all()):
            raise ValueError("voxel-grid bounds must be finite")
        if np.any(domain_max <= domain_min):
            raise ValueError("voxel-grid maximum must exceed minimum on every axis")
        object.__setattr__(self, "resolution_xyz", resolution)
        object.__setattr__(self, "domain_min_xyz", domain_min)
        object.__setattr__(self, "domain_max_xyz", domain_max)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        nx, ny, nz = self.resolution_xyz
        return nz, ny, nx

    @property
    def voxel_size_xyz(self) -> np.ndarray:
        return (self.domain_max_xyz - self.domain_min_xyz) / np.asarray(
            self.resolution_xyz, dtype=np.float64
        )

    @property
    def voxel_count(self) -> int:
        return int(np.prod(self.resolution_xyz, dtype=np.int64))

    @property
    def bounds_vtk(self) -> tuple[float, float, float, float, float, float]:
        return tuple(
            value
            for axis in range(3)
            for value in (self.domain_min_xyz[axis], self.domain_max_xyz[axis])
        )


@dataclass
class SegmentationField3D:
    """An integer region identifier at every voxel."""

    grid: VoxelGrid3D
    labels_zyx: np.ndarray
    name: str = "segmentation"

    def __post_init__(self):
        labels = np.asarray(self.labels_zyx)
        if labels.shape != self.grid.shape_zyx:
            raise ValueError(
                f"labels shape {labels.shape} does not match voxel grid "
                f"{self.grid.shape_zyx}"
            )
        finite = np.isfinite(labels)
        rounded = np.rint(np.where(finite, labels, 0.0))
        labels = np.where((finite) & (rounded > 0), rounded, 0)
        self.labels_zyx = np.asarray(labels, dtype=np.int32, order="C")

    @property
    def positive_labels(self) -> np.ndarray:
        labels = np.unique(self.labels_zyx)
        return labels[labels > 0]

    @property
    def positive_voxel_count(self) -> int:
        return int(np.count_nonzero(self.labels_zyx > 0))


def _read_vtk_dataset(path: str | Path):
    import vtk

    path = Path(path)
    readers = {
        ".vtk": vtk.vtkDataSetReader,
        ".vti": vtk.vtkXMLImageDataReader,
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
        ".vts": vtk.vtkXMLStructuredGridReader,
        ".vtr": vtk.vtkXMLRectilinearGridReader,
        ".vtp": vtk.vtkXMLPolyDataReader,
    }
    reader_type = readers.get(path.suffix.lower())
    if reader_type is None:
        raise ValueError(
            f"unsupported VTK extension '{path.suffix}'; expected one of "
            f"{sorted(readers)}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = reader_type()
    reader.SetFileName(str(path))
    if hasattr(reader, "ReadAllScalarsOn"):
        reader.ReadAllScalarsOn()
    if hasattr(reader, "ReadAllVectorsOn"):
        reader.ReadAllVectorsOn()
    if hasattr(reader, "ReadAllFieldsOn"):
        reader.ReadAllFieldsOn()
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"VTK file contains no dataset points: {path}")
    return dataset


def _safe_bounds(dataset) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(dataset.GetBounds(), dtype=np.float64).reshape(3, 2)
    domain_min = bounds[:, 0].copy()
    domain_max = bounds[:, 1].copy()
    for axis in range(3):
        if not np.isfinite([domain_min[axis], domain_max[axis]]).all():
            raise ValueError("VTK dataset has non-finite bounds")
        if domain_max[axis] <= domain_min[axis]:
            domain_max[axis] = domain_min[axis] + 1.0
    return domain_min, domain_max


def _label_array_descriptions(dataset) -> list[dict]:
    arrays: list[dict] = []
    for association, attributes in (
        ("point", dataset.GetPointData()),
        ("cell", dataset.GetCellData()),
    ):
        for index in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(index)
            name = array.GetName() if array is not None else None
            if array is None or not name or array.GetNumberOfComponents() != 1:
                continue
            value_range = array.GetRange()
            arrays.append(
                {
                    "name": str(name),
                    "association": association,
                    "dtype": str(array.GetDataTypeAsString()),
                    "range": [float(value_range[0]), float(value_range[1])],
                }
            )
    return arrays


def choose_label_array(arrays: Sequence[dict]) -> dict:
    """Choose a likely categorical label array by its name."""

    if not arrays:
        raise ValueError("VTK dataset has no named single-component point/cell arrays")

    def rank(item: dict) -> tuple[int, int]:
        name = item["name"].lower()
        if "vortex" in name:
            score = 5
        elif "segment" in name or "seg" in name:
            score = 4
        elif "label" in name or "class" in name:
            score = 3
        elif "id" in name and "pointid" not in name:
            score = 2
        else:
            score = 1
        # Cell-associated identifiers are the normal convention for segmented meshes.
        return score, int(item["association"] == "cell")

    return dict(max(arrays, key=rank))


def _select_label_array(
    arrays: Sequence[dict], array_name: str | None, association: str | None
) -> dict:
    if association not in (None, "point", "cell"):
        raise ValueError("association must be 'point', 'cell', or None")
    if array_name in (None, "", "auto", "(auto)"):
        candidates = [
            item for item in arrays if association is None or item["association"] == association
        ]
        return choose_label_array(candidates)
    matches = [
        item
        for item in arrays
        if item["name"] == array_name
        and (association is None or item["association"] == association)
    ]
    if not matches:
        available = [f"{item['name']} ({item['association']})" for item in arrays]
        raise ValueError(
            f"label array '{array_name}' with association={association!r} was not found; "
            f"available={available}"
        )
    if len(matches) > 1 and association is None:
        raise ValueError(
            f"label array '{array_name}' exists in point and cell data; specify association"
        )
    return dict(matches[0])


def suggest_native_resolution(
    dataset, association: str, max_axis_resolution: int = 1024
) -> tuple[int, int, int]:
    """Estimate the source mesh's per-axis cell fidelity.

    Structured datasets use their recorded dimensions.  Unstructured meshes use
    physical extent divided by the median sampled cell bound on each axis, matching
    the robust estimate used by the C++ implementation.
    """

    max_axis_resolution = max(1, int(max_axis_resolution))
    domain_min, domain_max = _safe_bounds(dataset)
    extents = domain_max - domain_min

    if hasattr(dataset, "GetDimensions"):
        dimensions = [0, 0, 0]
        try:
            dataset.GetDimensions(dimensions)
        except TypeError:
            dimensions = list(dataset.GetDimensions())
        if len(dimensions) == 3 and any(int(value) > 1 for value in dimensions):
            if association == "cell":
                dimensions = [max(1, int(value) - 1) for value in dimensions]
            return tuple(
                min(max_axis_resolution, max(1, int(value))) for value in dimensions
            )

    number_of_cells = int(dataset.GetNumberOfCells())
    volume = float(np.prod(np.maximum(extents, 1e-12)))
    density_scale = (max(8, int(dataset.GetNumberOfPoints())) / volume) ** (1.0 / 3.0)
    density_resolution = np.maximum(1, np.rint(extents * density_scale)).astype(int)
    if number_of_cells <= 0:
        return tuple(
            min(max_axis_resolution, int(value)) for value in density_resolution
        )

    step = max(1, number_of_cells // 4000)
    cell_sizes: list[list[float]] = []
    bounds = [0.0] * 6
    for cell_id in range(0, number_of_cells, step):
        cell = dataset.GetCell(cell_id)
        if cell is None:
            continue
        cell.GetBounds(bounds)
        cell_sizes.append(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        )
    if not cell_sizes:
        return tuple(
            min(max_axis_resolution, int(value)) for value in density_resolution
        )
    median_size = np.median(np.asarray(cell_sizes, dtype=np.float64), axis=0)
    resolution = []
    for axis in range(3):
        if median_size[axis] > 1e-12 and extents[axis] > 0:
            value = int(np.rint(extents[axis] / median_size[axis]))
        else:
            value = int(density_resolution[axis])
        resolution.append(min(max_axis_resolution, max(1, value)))
    return tuple(resolution)


def cap_resolution_by_voxels(
    resolution_xyz: Sequence[int], max_voxels: int | None
) -> tuple[int, int, int]:
    """Reduce all three axes by one common factor so the voxel budget is respected."""

    resolution = np.asarray(resolution_xyz, dtype=np.int64).reshape(3)
    if np.any(resolution < 1):
        raise ValueError("resolution values must be positive")
    if max_voxels is None:
        return tuple(int(value) for value in resolution)
    max_voxels = int(max_voxels)
    if max_voxels < 1:
        raise ValueError("max_voxels must be positive or None")
    count = int(np.prod(resolution, dtype=np.int64))
    if count <= max_voxels:
        return tuple(int(value) for value in resolution)
    scale = (max_voxels / count) ** (1.0 / 3.0)
    reduced = np.maximum(1, np.floor(resolution * scale).astype(np.int64))
    while int(np.prod(reduced, dtype=np.int64)) > max_voxels:
        axis = int(np.argmax(reduced))
        reduced[axis] = max(1, reduced[axis] - 1)
    return tuple(int(value) for value in reduced)


def inspect_vtk_label_file(path: str | Path) -> dict:
    """Return topology, bounds, arrays, and a label-safe resolution suggestion."""

    dataset = _read_vtk_dataset(path)
    arrays = _label_array_descriptions(dataset)
    selected = choose_label_array(arrays)
    domain_min, domain_max = _safe_bounds(dataset)
    suggested = suggest_native_resolution(dataset, selected["association"])
    return {
        "path": str(Path(path).resolve()),
        "dataset_type": str(dataset.GetClassName()),
        "num_points": int(dataset.GetNumberOfPoints()),
        "num_cells": int(dataset.GetNumberOfCells()),
        "domain_min_xyz": domain_min.tolist(),
        "domain_max_xyz": domain_max.tolist(),
        "arrays": arrays,
        "auto_selected_array": selected,
        "suggested_native_resolution_xyz": list(suggested),
    }


def _copy_selected_array(dataset, selected: dict):
    """Add the selected array under an unambiguous temporary name."""

    import vtk

    working = vtk.vtkDataSet.SafeDownCast(dataset.NewInstance())
    working.ShallowCopy(dataset)
    attributes = (
        dataset.GetPointData()
        if selected["association"] == "point"
        else dataset.GetCellData()
    )
    source = attributes.GetArray(selected["name"])
    if source is None:
        raise ValueError(
            f"selected {selected['association']} array '{selected['name']}' disappeared"
        )
    copied = source.NewInstance()
    copied.DeepCopy(source)
    copied.SetName("__fmt_voxel_label__")
    target_attributes = (
        working.GetPointData()
        if selected["association"] == "point"
        else working.GetCellData()
    )
    target_attributes.AddArray(copied)
    return working, source


def _center_sampling_bounds(grid: VoxelGrid3D) -> list[float]:
    bounds = []
    voxel_size = grid.voxel_size_xyz
    for axis, count in enumerate(grid.resolution_xyz):
        if count == 1:
            center = 0.5 * (
                grid.domain_min_xyz[axis] + grid.domain_max_xyz[axis]
            )
            bounds.extend([float(center), float(center)])
        else:
            bounds.extend(
                [
                    float(grid.domain_min_xyz[axis] + 0.5 * voxel_size[axis]),
                    float(grid.domain_max_xyz[axis] - 0.5 * voxel_size[axis]),
                ]
            )
    return bounds


def _source_positive_labels(source_array) -> list[int]:
    from vtk.util.numpy_support import vtk_to_numpy

    source = np.asarray(vtk_to_numpy(source_array)).reshape(-1)
    source = source[np.isfinite(source)]
    return [
        int(value)
        for value in np.unique(np.rint(source[source > 0]).astype(np.int64))
    ]


def _sample_cell_labels(working, grid: VoxelGrid3D) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    nx, ny, nz = grid.resolution_xyz
    resampler = vtk.vtkResampleToImage()
    resampler.SetInputDataObject(working)
    resampler.UseInputBoundsOff()
    resampler.SetSamplingBounds(*_center_sampling_bounds(grid))
    resampler.SetSamplingDimensions(nx, ny, nz)
    resampler.Update()
    image = resampler.GetOutput()
    if image is None:
        raise RuntimeError("VTK label resampling produced no image")
    point_data = image.GetPointData()
    sampled = point_data.GetArray("__fmt_voxel_label__")
    if sampled is None:
        raise RuntimeError("selected cell label array was not produced by VTK resampling")
    labels = np.asarray(vtk_to_numpy(sampled)).reshape((nz, ny, nx), order="C")
    valid_array = point_data.GetArray("vtkValidPointMask")
    if valid_array is None:
        valid = np.ones((nz, ny, nx), dtype=bool)
    else:
        valid = np.asarray(vtk_to_numpy(valid_array), dtype=bool).reshape(
            (nz, ny, nx), order="C"
        )
    return labels, valid


def _sample_point_labels(working, grid: VoxelGrid3D) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point sampling, masked to the source geometry's valid support."""

    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    nx, ny, nz = grid.resolution_xyz
    center_bounds = _center_sampling_bounds(grid)
    target = vtk.vtkImageData()
    target.SetDimensions(nx, ny, nz)
    target.SetOrigin(center_bounds[0], center_bounds[2], center_bounds[4])
    spacing = []
    for axis, count in enumerate(grid.resolution_xyz):
        spacing.append(float(grid.voxel_size_xyz[axis] if count > 1 else 1.0))
    target.SetSpacing(*spacing)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(target)
    interpolator.SetSourceData(working)
    interpolator.SetKernel(vtk.vtkVoronoiKernel())
    interpolator.SetNullPointsStrategyToNullValue()
    interpolator.SetNullValue(0.0)
    interpolator.Update()
    sampled_array = interpolator.GetOutput().GetPointData().GetArray(
        "__fmt_voxel_label__"
    )
    if sampled_array is None:
        raise RuntimeError("nearest-point VTK label sampling produced no label array")
    labels = np.asarray(vtk_to_numpy(sampled_array)).reshape(
        (nz, ny, nx), order="C"
    )

    # vtkPointInterpolator fills the full bounding box.  Probe the actual dataset
    # geometry separately so points outside a sparse/unstructured mesh remain background.
    validity_probe = vtk.vtkResampleToImage()
    validity_probe.SetInputDataObject(working)
    validity_probe.UseInputBoundsOff()
    validity_probe.SetSamplingBounds(*center_bounds)
    validity_probe.SetSamplingDimensions(nx, ny, nz)
    validity_probe.Update()
    valid_array = validity_probe.GetOutput().GetPointData().GetArray(
        "vtkValidPointMask"
    )
    if valid_array is None:
        valid = np.ones((nz, ny, nx), dtype=bool)
    else:
        valid = np.asarray(vtk_to_numpy(valid_array), dtype=bool).reshape(
            (nz, ny, nx), order="C"
        )
    return labels, valid


def load_vtk_segmentation(
    path: str | Path,
    resolution_xyz: Sequence[int] | None = None,
    array_name: str | None = None,
    association: str | None = None,
    max_voxels: int | None = 2_000_000,
    max_axis_resolution: int = 1024,
) -> tuple[SegmentationField3D, dict, dict]:
    """Load a VTK segmentation onto an independent regular voxel grid.

    Cell labels are sampled piecewise-constantly at voxel centers.  Point labels
    use nearest-point sampling.  No label identifier is linearly interpolated.
    Voxels outside the source geometry and all source values ``<=0`` become zero.
    """

    dataset = _read_vtk_dataset(path)
    arrays = _label_array_descriptions(dataset)
    selected = _select_label_array(arrays, array_name, association)
    domain_min, domain_max = _safe_bounds(dataset)
    suggested = suggest_native_resolution(
        dataset, selected["association"], max_axis_resolution=max_axis_resolution
    )
    if resolution_xyz is None:
        used_resolution = cap_resolution_by_voxels(suggested, max_voxels)
        resolution_origin = (
            "suggested_native"
            if used_resolution == suggested
            else f"suggested_native_capped_to_{int(max_voxels)}_voxels"
        )
    else:
        used_resolution = tuple(int(value) for value in resolution_xyz)
        if len(used_resolution) != 3 or any(value < 1 for value in used_resolution):
            raise ValueError("resolution_xyz must contain three positive integers")
        resolution_origin = "explicit"
    grid = VoxelGrid3D(used_resolution, domain_min, domain_max)
    working, source_array = _copy_selected_array(dataset, selected)
    source_labels = _source_positive_labels(source_array)

    # Exact native ImageData arrays do not need a geometric probe and preserve every id.
    dimensions = tuple(int(value) for value in dataset.GetDimensions()) if dataset.IsA(
        "vtkImageData"
    ) else None
    exact_shape = None
    if dimensions is not None:
        exact_resolution = (
            dimensions
            if selected["association"] == "point"
            else tuple(max(1, value - 1) for value in dimensions)
        )
        if used_resolution == exact_resolution and int(source_array.GetNumberOfTuples()) == int(
            np.prod(exact_resolution)
        ):
            from vtk.util.numpy_support import vtk_to_numpy

            exact_shape = grid.shape_zyx
            labels = np.asarray(vtk_to_numpy(source_array)).reshape(
                exact_shape, order="C"
            )
            valid = np.ones(exact_shape, dtype=bool)
    if exact_shape is None:
        if selected["association"] == "cell":
            labels, valid = _sample_cell_labels(working, grid)
        else:
            labels, valid = _sample_point_labels(working, grid)

    labels = np.where(valid, labels, 0)
    segmentation = SegmentationField3D(grid, labels, selected["name"])
    voxel_labels = [int(value) for value in segmentation.positive_labels]
    metadata = {
        "path": str(Path(path).resolve()),
        "dataset_type": str(dataset.GetClassName()),
        "num_points": int(dataset.GetNumberOfPoints()),
        "num_cells": int(dataset.GetNumberOfCells()),
        "domain_min_xyz": domain_min.tolist(),
        "domain_max_xyz": domain_max.tolist(),
        "arrays": arrays,
        "suggested_native_resolution_xyz": list(suggested),
        "used_resolution_xyz": list(used_resolution),
        "resolution_origin": resolution_origin,
        "sampling": "voxel centers; cell labels piecewise-constant; point labels nearest",
        "source_positive_labels": source_labels,
        "voxelized_positive_labels": voxel_labels,
        "missing_positive_labels_after_voxelization": sorted(
            set(source_labels) - set(voxel_labels)
        ),
        "positive_voxel_count": segmentation.positive_voxel_count,
        "valid_sample_count": int(np.count_nonzero(valid)),
    }
    return segmentation, metadata, selected


def visible_voxel_mask(
    labels_zyx: np.ndarray,
    surface_mode: str = "outer",
    label_filter: Iterable[int] | None = None,
) -> np.ndarray:
    """Select positive cubes, optionally omitting fully hidden interiors.

    ``outer`` reproduces the C++ six-neighbor exposed-voxel rule. ``interfaces``
    also keeps cubes next to a different positive label. ``all`` keeps every
    selected positive cube.
    """

    labels = np.asarray(labels_zyx)
    if labels.ndim != 3:
        raise ValueError("labels_zyx must be a 3D array")
    visible = labels > 0
    if label_filter is not None:
        wanted = np.asarray([int(value) for value in label_filter], dtype=np.int64)
        if wanted.size == 0:
            return np.zeros(labels.shape, dtype=bool)
        visible &= np.isin(labels, wanted)
    if surface_mode == "all":
        return visible
    if surface_mode not in ("outer", "interfaces"):
        raise ValueError("surface_mode must be 'outer', 'interfaces', or 'all'")

    padded_visible = np.pad(visible, 1, constant_values=False)
    neighbors_visible = (
        padded_visible[1:-1, 1:-1, :-2]
        & padded_visible[1:-1, 1:-1, 2:]
        & padded_visible[1:-1, :-2, 1:-1]
        & padded_visible[1:-1, 2:, 1:-1]
        & padded_visible[:-2, 1:-1, 1:-1]
        & padded_visible[2:, 1:-1, 1:-1]
    )
    if surface_mode == "outer":
        return visible & ~neighbors_visible

    padded_labels = np.pad(labels, 1, constant_values=0)
    same_neighbors = (
        (padded_labels[1:-1, 1:-1, :-2] == labels)
        & (padded_labels[1:-1, 1:-1, 2:] == labels)
        & (padded_labels[1:-1, :-2, 1:-1] == labels)
        & (padded_labels[1:-1, 2:, 1:-1] == labels)
        & (padded_labels[:-2, 1:-1, 1:-1] == labels)
        & (padded_labels[2:, 1:-1, 1:-1] == labels)
    )
    return visible & ~(neighbors_visible & same_neighbors)


def visible_voxel_centers(
    segmentation: SegmentationField3D,
    surface_mode: str = "outer",
    label_filter: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical centers, labels, and the Boolean mask used for rendering."""

    mask = visible_voxel_mask(
        segmentation.labels_zyx, surface_mode=surface_mode, label_filter=label_filter
    )
    iz, iy, ix = np.nonzero(mask)
    indices_xyz = np.column_stack((ix, iy, iz)).astype(np.float64, copy=False)
    centers = segmentation.grid.domain_min_xyz + (
        indices_xyz + 0.5
    ) * segmentation.grid.voxel_size_xyz
    labels = segmentation.labels_zyx[iz, iy, ix]
    return (
        np.asarray(centers, dtype=np.float32, order="C"),
        np.asarray(labels, dtype=np.int32, order="C"),
        mask,
    )


def _golden_angle_rgb(label: int) -> tuple[float, float, float]:
    hue = (0.6180339887498949 * max(1, int(label))) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.68, 0.95)


def _make_lookup_table(max_label: int):
    import vtk

    max_label = max(1, int(max_label))
    table = vtk.vtkLookupTable()
    table.SetNumberOfTableValues(max_label + 1)
    table.SetTableRange(0.0, float(max_label))
    table.Build()
    table.SetTableValue(0, 0.65, 0.65, 0.65, 0.0)
    for label in range(1, max_label + 1):
        red, green, blue = _golden_angle_rgb(label)
        table.SetTableValue(label, red, green, blue, 1.0)
    return table


def _camera_definition(view: str) -> tuple[np.ndarray, np.ndarray]:
    definitions = {
        "isometric": (np.asarray([1.4, -1.2, 0.9]), np.asarray([0.0, 0.0, 1.0])),
        "+x": (np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        "-x": (np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        "+y": (np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        "-y": (np.asarray([0.0, -1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
        "+z": (np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0])),
        "-z": (np.asarray([0.0, 0.0, -1.0]), np.asarray([0.0, 1.0, 0.0])),
    }
    if view not in definitions:
        raise ValueError(f"unknown camera view '{view}'; choices={sorted(definitions)}")
    return definitions[view]


def _set_camera(renderer, grid: VoxelGrid3D, view: str):
    direction, view_up = _camera_definition(view)
    direction = direction / np.linalg.norm(direction)
    center = 0.5 * (grid.domain_min_xyz + grid.domain_max_xyz)
    diagonal = float(np.linalg.norm(grid.domain_max_xyz - grid.domain_min_xyz))
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + 2.5 * diagonal * direction))
    camera.SetViewUp(*view_up)
    camera.SetParallelProjection(view != "isometric")
    renderer.ResetCamera()
    if view == "isometric":
        camera.Zoom(1.08)
    else:
        # vtkRenderer.ResetCamera fits a 3D bounding sphere, which wastes most of
        # an orthographic image when looking down the long channel axis. Fit the
        # projected rectangle instead.
        right = np.cross(direction, view_up)
        right /= np.linalg.norm(right)
        corners = np.asarray(
            [
                [x, y, z]
                for x in (grid.domain_min_xyz[0], grid.domain_max_xyz[0])
                for y in (grid.domain_min_xyz[1], grid.domain_max_xyz[1])
                for z in (grid.domain_min_xyz[2], grid.domain_max_xyz[2])
            ],
            dtype=np.float64,
        )
        span_up = float(np.ptp(corners @ view_up))
        span_right = float(np.ptp(corners @ right))
        render_window = renderer.GetRenderWindow()
        width, height = render_window.GetSize() if render_window is not None else (4, 3)
        aspect = max(float(width) / max(float(height), 1.0), 1e-6)
        camera.SetParallelScale(0.55 * max(span_up, span_right / aspect, 1e-12))
    renderer.ResetCameraClippingRange()


def render_voxel_segmentation(
    segmentation: SegmentationField3D,
    output_dir: str | Path | None = None,
    views: Sequence[str] = ("isometric", "+x", "+y", "+z"),
    surface_mode: str = "outer",
    label_filter: Iterable[int] | None = None,
    shrink: float = 0.92,
    opacity: float = 1.0,
    image_size: Sequence[int] = (1200, 900),
    interactive: bool = False,
) -> dict:
    """Render shrunken cube glyphs and optionally open a trackball-camera window."""

    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    shrink = float(np.clip(shrink, 0.05, 1.0))
    opacity = float(np.clip(opacity, 0.02, 1.0))
    width, height = (int(image_size[0]), int(image_size[1]))
    if width < 64 or height < 64:
        raise ValueError("image_size must be at least 64 x 64")

    centers, render_labels, render_mask = visible_voxel_centers(
        segmentation, surface_mode=surface_mode, label_filter=label_filter
    )
    if centers.size == 0:
        raise ValueError("no positive voxels remain after label/surface filtering")

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(centers, deep=True))
    point_data = vtk.vtkPolyData()
    point_data.SetPoints(points)
    vtk_labels = numpy_to_vtk(render_labels, deep=True)
    vtk_labels.SetName("segmentation_label")
    point_data.GetPointData().AddArray(vtk_labels)
    point_data.GetPointData().SetActiveScalars("segmentation_label")

    cube = vtk.vtkCubeSource()
    cube.SetXLength(shrink * segmentation.grid.voxel_size_xyz[0])
    cube.SetYLength(shrink * segmentation.grid.voxel_size_xyz[1])
    cube.SetZLength(shrink * segmentation.grid.voxel_size_xyz[2])
    cube.Update()

    maximum_label = int(render_labels.max())
    lookup_table = _make_lookup_table(maximum_label)
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(point_data)
    mapper.SetSourceConnection(cube.GetOutputPort())
    mapper.ScalingOff()
    mapper.OrientOff()
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("segmentation_label")
    mapper.SetColorModeToMapScalars()
    mapper.SetLookupTable(lookup_table)
    mapper.SetScalarRange(0.0, float(maximum_label))

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetInterpolationToFlat()

    outline = vtk.vtkOutlineSource()
    outline.SetBounds(*segmentation.grid.bounds_vtk)
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline.GetOutputPort())
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(0.22, 0.22, 0.22)
    outline_actor.GetProperty().SetLineWidth(1.0)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.97, 0.97, 0.97)
    renderer.AddActor(actor)
    renderer.AddActor(outline_actor)
    renderer.UseDepthPeelingOn()
    renderer.SetMaximumNumberOfPeels(80)
    renderer.SetOcclusionRatio(0.1)

    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(
        f"{segmentation.name}: {len(segmentation.positive_labels)} labels | "
        f"{len(render_labels):,} rendered cubes"
    )
    text_actor.SetPosition(16, 16)
    text_actor.GetTextProperty().SetFontSize(17)
    text_actor.GetTextProperty().SetColor(0.08, 0.08, 0.08)
    renderer.AddActor2D(text_actor)

    window = vtk.vtkRenderWindow()
    window.SetSize(width, height)
    window.SetWindowName(f"Voxel segmentation: {segmentation.name}")
    window.AddRenderer(renderer)
    if not interactive:
        window.SetOffScreenRendering(1)

    view_paths: list[str] = []
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for view in views:
            _set_camera(renderer, segmentation.grid, view)
            window.Render()
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff()
            capture.Update()
            safe_name = {
                "+x": "plus-x",
                "-x": "minus-x",
                "+y": "plus-y",
                "-y": "minus-y",
                "+z": "plus-z",
                "-z": "minus-z",
            }.get(view, view)
            output_path = output_dir / f"{segmentation.name}_{safe_name}.png"
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(output_path))
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"failed to render {output_path}")
            view_paths.append(str(output_path.resolve()))

    if interactive:
        _set_camera(renderer, segmentation.grid, "isometric")
        interactor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        interactor.SetRenderWindow(window)
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)

        def report_picked_label(caller, _event):
            x, y = caller.GetEventPosition()
            if not picker.Pick(x, y, 0.0, renderer):
                return
            position = np.asarray(picker.GetPickPosition(), dtype=np.float64)
            index_xyz = np.floor(
                (position - segmentation.grid.domain_min_xyz)
                / segmentation.grid.voxel_size_xyz
            ).astype(int)
            index_xyz = np.clip(
                index_xyz,
                0,
                np.asarray(segmentation.grid.resolution_xyz, dtype=int) - 1,
            )
            ix, iy, iz = (int(value) for value in index_xyz)
            label = int(segmentation.labels_zyx[iz, iy, ix])
            message = f"label={label} | voxel=({ix}, {iy}, {iz})"
            text_actor.SetInput(message)
            print(message, flush=True)
            window.Render()

        interactor.AddObserver("LeftButtonPressEvent", report_picked_label, 1.0)
        window.Render()
        interactor.Initialize()
        interactor.Start()

    return {
        "rendered_cube_count": int(len(render_labels)),
        "surface_mode": surface_mode,
        "label_filter": None
        if label_filter is None
        else [int(value) for value in label_filter],
        "shrink": shrink,
        "opacity": opacity,
        "image_size": [width, height],
        "view_paths": view_paths,
        "render_mask_positive_count": int(np.count_nonzero(render_mask)),
    }
