import numpy as np
import unittest

from Visualize_Task23_3D_Horizontal import (
    _closeup_bounds,
    _confusion_masks,
    _count_blue_like_pixels,
    _validate_render_payload,
    _visible_mask,
)


def _payload():
    seeds = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    reference = np.asarray([False, False, True, True])
    return {
        "seeds": seeds,
        "reference": reference,
        "predictions": [
            np.asarray([False, True, False, True]),
            np.asarray([True, False, True, False]),
        ],
        "ivd_mesh": (
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
        ),
        "bounds": np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
    }


class VisualizationContractTests(unittest.TestCase):
    def test_confusion_masks_are_an_exact_partition_of_the_same_seeds(self):
        reference = np.asarray([False, False, True, True])
        prediction = np.asarray([False, True, False, True])
        masks = _confusion_masks(reference, prediction)

        self.assertEqual(
            {name: int(mask.sum()) for name, mask in masks.items()},
            {
                "true_negative": 1,
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 1,
            },
        )
        np.testing.assert_array_equal(
            np.stack(list(masks.values())).sum(axis=0), np.ones(4, dtype=int)
        )
        self.assertFalse(np.any(masks["false_positive"] & reference))
        self.assertFalse(np.any(masks["false_negative"] & ~reference))

    def test_render_payload_preserves_seed_count_and_rejects_densification(self):
        payload = _payload()
        seeds, reference, predictions = _validate_render_payload(payload, 1.0)

        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(reference), 4)
        self.assertEqual(len(predictions[0]), 4)
        np.testing.assert_array_equal(seeds, payload["seeds"])
        with self.assertRaisesRegex(ValueError, "densification is forbidden"):
            _validate_render_payload(payload, 1.01)

    def test_render_payload_rejects_prediction_on_a_different_seed_set(self):
        payload = _payload()
        payload["predictions"][1] = payload["predictions"][1][:-1]
        with self.assertRaisesRegex(ValueError, "one label per seed"):
            _validate_render_payload(payload, 1.0)

    def test_closeup_is_ivd_only_and_shared_by_both_prediction_arms(self):
        coordinates = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        seeds = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        reference = np.linalg.norm(seeds - 0.5, axis=1) < 0.45
        bounds = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64)

        first = _closeup_bounds(bounds, seeds, reference)
        second = _closeup_bounds(bounds, seeds, reference)

        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        visible = _visible_mask(seeds, first)
        self.assertTrue(np.any(reference & visible))
        self.assertTrue(np.all(first[0] >= bounds[0]))
        self.assertTrue(np.all(first[1] <= bounds[1]))

    def test_blue_pixel_audit_rejects_background_blue(self):
        rgb = np.ones((2, 2, 3), dtype=np.float64)
        rgb[0, 0] = [0.10, 0.35, 0.80]
        rgb[0, 1] = [0.85, 0.10, 0.10]
        self.assertEqual(_count_blue_like_pixels(rgb), 1)


if __name__ == "__main__":
    unittest.main()
