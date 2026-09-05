from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FMT_Utils.VoxelSegmentation_3D import (
    VoxelGrid3D,
    SegmentationField3D,
    cap_resolution_by_voxels,
    inspect_vtk_label_file,
    load_vtk_segmentation,
    visible_voxel_mask,
)


try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError:
    vtk = None
    numpy_to_vtk = None


def _write_image(path: Path, association: str) -> None:
    image = vtk.vtkImageData()
    image.SetDimensions(3, 3, 3)
    image.SetOrigin(-1.0, -2.0, -3.0)
    image.SetSpacing(1.0, 2.0, 3.0)
    if association == "point":
        values = np.zeros(27, dtype=np.int32)
        values[1] = -4
        values[13] = 3
        values[26] = 7
        array = numpy_to_vtk(values, deep=True)
        array.SetName("VortexIds")
        image.GetPointData().AddArray(array)
    else:
        values = np.arange(1, 9, dtype=np.int32)
        array = numpy_to_vtk(values, deep=True)
        array.SetName("SegmentLabels")
        image.GetCellData().AddArray(array)
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    assert writer.Write() == 1


@unittest.skipUnless(vtk is not None, "VTK is required")
class TestVoxelSegmentation3D(unittest.TestCase):
    def test_point_labels_preserve_ids_and_convert_nonpositive_to_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "channel_GTs.vtk"
            _write_image(path, "point")
            metadata = inspect_vtk_label_file(path)
            self.assertEqual(
                metadata["suggested_native_resolution_xyz"], [3, 3, 3]
            )
            segmentation, _, selected = load_vtk_segmentation(
                path, max_voxels=None
            )
            self.assertEqual(
                selected,
                {
                    "name": "VortexIds",
                    "association": "point",
                    "dtype": "int",
                    "range": [-4.0, 7.0],
                },
            )
            self.assertEqual(segmentation.labels_zyx.shape, (3, 3, 3))
            self.assertEqual(segmentation.positive_labels.tolist(), [3, 7])
            self.assertEqual(int(segmentation.labels_zyx.reshape(-1)[1]), 0)

    def test_cell_labels_use_cell_resolution_and_keep_integer_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell_labels.vtk"
            _write_image(path, "cell")
            segmentation, metadata, selected = load_vtk_segmentation(
                path,
                array_name="SegmentLabels",
                association="cell",
                max_voxels=None,
            )
            self.assertEqual(selected["association"], "cell")
            self.assertEqual(metadata["used_resolution_xyz"], [2, 2, 2])
            self.assertEqual(segmentation.labels_zyx.shape, (2, 2, 2))
            self.assertEqual(
                segmentation.positive_labels.tolist(), list(range(1, 9))
            )

    def test_arbitrary_cell_resolution_uses_only_discrete_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell_labels.vtk"
            _write_image(path, "cell")
            segmentation, _, _ = load_vtk_segmentation(
                path,
                resolution_xyz=(5, 4, 3),
                array_name="SegmentLabels",
                association="cell",
            )
            self.assertEqual(segmentation.labels_zyx.shape, (3, 4, 5))
            self.assertTrue(
                set(np.unique(segmentation.labels_zyx)).issubset(set(range(9)))
            )

    def test_outer_surface_omits_only_fully_hidden_voxel(self):
        labels = np.ones((3, 3, 3), dtype=np.int32)
        exposed = visible_voxel_mask(labels, surface_mode="outer")
        self.assertEqual(int(exposed.sum()), 26)
        self.assertFalse(bool(exposed[1, 1, 1]))

    def test_interfaces_keep_voxels_next_to_a_different_positive_label(self):
        labels = np.ones((5, 5, 5), dtype=np.int32)
        labels[:, :, 3:] = 2
        outer = visible_voxel_mask(labels, surface_mode="outer")
        interfaces = visible_voxel_mask(labels, surface_mode="interfaces")
        self.assertFalse(bool(outer[2, 2, 2]))
        self.assertTrue(bool(interfaces[2, 2, 2]))
        self.assertTrue(bool(interfaces[2, 2, 3]))

    def test_label_filter_and_resolution_budget_are_explicit(self):
        grid = VoxelGrid3D((4, 3, 2), np.zeros(3), np.ones(3))
        labels = np.zeros(grid.shape_zyx, dtype=np.int32)
        labels[:, :, :2] = 4
        labels[:, :, 2:] = 9
        segmentation = SegmentationField3D(grid, labels)
        mask = visible_voxel_mask(
            segmentation.labels_zyx, surface_mode="all", label_filter=[9]
        )
        self.assertTrue(np.all(segmentation.labels_zyx[mask] == 9))
        capped = cap_resolution_by_voxels((255, 191, 302), 2_000_000)
        self.assertLessEqual(int(np.prod(capped)), 2_000_000)
        ratios = np.asarray(capped) / np.asarray((255, 191, 302))
        self.assertLess(float(ratios.max() - ratios.min()), 0.01)


if __name__ == "__main__":
    unittest.main()
