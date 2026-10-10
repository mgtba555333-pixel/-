from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_model import (
    bootstrap_confidence_intervals,
    compute_specificity,
    expected_calibration_error,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_specificity_is_one_vs_rest_per_class(self) -> None:
        confusion = np.asarray([[8, 2], [1, 9]], dtype=np.int64)
        actual = compute_specificity(confusion, ["NORMAL", "PNEUMONIA"])
        self.assertAlmostEqual(actual["NORMAL"], 0.9)
        self.assertAlmostEqual(actual["PNEUMONIA"], 0.8)

    def test_expected_calibration_error(self) -> None:
        ece = expected_calibration_error([0, 1], [0.1, 0.9], bins=10)
        self.assertAlmostEqual(ece, 0.1, places=6)

    def test_bootstrap_intervals_are_reproducible_and_include_auc(self) -> None:
        y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 1]
        binary_truth = [0, 0, 1, 1, 0, 1, 0, 1]
        positive_scores = [0.08, 0.2, 0.85, 0.62, 0.42, 0.93, 0.55, 0.79]
        first = bootstrap_confidence_intervals(
            y_true, y_pred, binary_truth, positive_scores, num_classes=2, resamples=100, seed=7
        )
        second = bootstrap_confidence_intervals(
            y_true, y_pred, binary_truth, positive_scores, num_classes=2, resamples=100, seed=7
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(first["accuracy"]["lower_95"])
        self.assertGreater(first["roc_auc_pneumonia_vs_normal"]["valid_resamples"], 25)


if __name__ == "__main__":
    unittest.main()
