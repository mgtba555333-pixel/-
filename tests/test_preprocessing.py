from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.model_service import ModelService, decode_scores


class DummyModel:
    # Non-square input makes the resize test catch accidental aspect-ratio cropping.
    input_shape = (None, 2, 4, 3)


class PreprocessingTests(unittest.TestCase):
    def test_default_matches_model_internal_rescaling(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            service = ModelService()
        self.assertEqual(service.preprocessing, "auto")
        self.assertEqual(service.color_order, "rgb")

    def test_binary_sigmoid_score_maps_to_normal_and_pneumonia(self) -> None:
        labels, scores = decode_scores([0.75], "binary_sigmoid", ["NORMAL", "PNEUMONIA"])
        self.assertEqual(labels, ["NORMAL", "PNEUMONIA"])
        np.testing.assert_allclose(scores, [0.25, 0.75], atol=1e-8)

    def test_multiclass_softmax_is_not_applied_twice(self) -> None:
        labels, scores = decode_scores([0.2, 0.8], "multiclass", ["NORMAL", "PNEUMONIA"])
        self.assertEqual(labels, ["NORMAL", "PNEUMONIA"])
        np.testing.assert_allclose(scores, [0.2, 0.8], atol=1e-8)

    def make_service(self, preprocessing: str = "unit", color_order: str = "rgb") -> ModelService:
        service = ModelService()
        service.model = DummyModel()
        service.labels = ["NORMAL", "PNEUMONIA"]
        service.load_error = None
        service.preprocessing = preprocessing
        service.color_order = color_order
        return service

    def test_unit_scaling_and_expected_shape(self) -> None:
        service = self.make_service()
        image = Image.new("RGB", (8, 1), (255, 128, 0))
        batch = service._prepare_input(image)

        self.assertEqual(batch.shape, (1, 2, 4, 3))
        np.testing.assert_allclose(batch[0, 0, 0], [1.0, 128 / 255.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(batch.max(), 1.0, atol=1e-5)
        self.assertGreaterEqual(float(batch.min()), 0.0)

    def test_bgr_channel_order_is_explicit(self) -> None:
        service = self.make_service(color_order="bgr")
        image = Image.new("RGB", (8, 1), (255, 128, 0))
        batch = service._prepare_input(image)
        np.testing.assert_allclose(batch[0, 0, 0], [0.0, 128 / 255.0, 1.0], atol=1e-5)

    def test_raw_mode_does_not_rescale(self) -> None:
        service = self.make_service(preprocessing="raw")
        image = Image.new("RGB", (8, 1), (240, 128, 12))
        batch = service._prepare_input(image)
        np.testing.assert_allclose(batch[0, 0, 0], [240.0, 128.0, 12.0], atol=1e-5)

    def test_resize_stretches_to_model_shape(self) -> None:
        service = self.make_service()
        image = Image.new("RGB", (8, 1), (30, 70, 90))
        batch = service._prepare_input(image)
        self.assertEqual(batch.shape[1:3], (2, 4))
        np.testing.assert_allclose(batch[0, :, :, 0], 30 / 255.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
