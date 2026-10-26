import numpy as np
import pytest

from scripts.evaluation.evaluate_research import transform, detected_box_calibration, object_size_analysis


def test_stress_preserves_coordinates_shape_and_grayscale():
    image = np.repeat(np.arange(100, dtype=np.uint8).reshape(10, 10, 1), 3, axis=2)
    for scenario in ["blur_sigma1.2", "blur_sigma2.0", "noise_sd5", "noise_sd12", "jpeg_quality40", "brightness0.7", "brightness1.3", "downsample0.5"]:
        result = transform(image, scenario, np.random.default_rng(42))
        assert result.shape == image.shape
        assert result.dtype == np.uint8
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result, transform(image, scenario, np.random.default_rng(42)))


def test_calibration_is_detection_conditioned_and_size_uses_ground_truth_denominator():
    record = {"sample_id": "a", "width": 100, "height": 100, "gt_boxes": [[0, 0, 1, 1], [20, 20, 40, 40]],
              "gt_classes": [0, 0], "boxes": [[20, 20, 40, 40]], "classes": [0], "scores": [.8]}
    calibration = detected_box_calibration([record])
    assert calibration["n_predictions"] == 1
    assert calibration["brier_detected_boxes_only"] == pytest.approx(.04)
    assert calibration["patient_disease_probability"] is False
    size = object_size_analysis([record], .25)
    assert size["area_lt_0.001"]["recall"] == 0
    assert size["area_ge_0.01"]["recall"] == 1
