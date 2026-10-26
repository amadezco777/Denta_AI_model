import copy
import numpy as np
import pytest

from src.evaluation.detection import (bootstrap, compute_ap, match_predictions,
                                      record_statistics, select_threshold, summarize)


def sample():
    return {"sample_id": "a", "boxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
            "classes": [0, 1], "scores": [.9, .7],
            "gt_boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "gt_classes": [0, 1]}


def test_wrong_class_and_duplicate_do_not_inflate_recall():
    r = sample()
    r["boxes"] += [[0, 0, 10, 10]]
    r["scores"] += [.4]
    r["classes"] += [0]
    result = summarize([r], ["a", "b"])
    assert result["by_class"][0]["tp"] == 1
    assert result["by_class"][0]["fp"] == 1
    r["classes"] = [1, 0, 1]
    assert summarize([r], ["a", "b"])["by_class"][0]["tp"] == 0


def test_threshold_rematches_and_missing_classes_are_explicit():
    r = sample()
    r["scores"] = [.01, .9]
    result = summarize([r], ["a", "b", "c"], .25)
    assert result["by_class"][0]["fn"] == 1
    assert result["by_class"][0]["ap50"] > .9
    assert result["by_class"][2]["recall"] is None
    assert result["classes_without_reference"] == ["c"]


def test_empty_predictions_and_empty_references_are_valid():
    r = sample()
    r.update(boxes=[], scores=[], classes=[])
    assert summarize([r], ["a", "b"])["map50"] == 0
    r.update(gt_boxes=[], gt_classes=[])
    assert summarize([r], ["a", "b"])["map50"] is None


def test_paired_bootstrap_requires_same_images_and_truth():
    a = sample()
    b = copy.deepcopy(a)
    result = bootstrap([a], ["a", "b"], .25, n=15, paired=[b])
    assert result["intervals"]["map50"] == {"low": 0, "high": 0, "valid_replicates": 15}
    b["sample_id"] = "different"
    with pytest.raises(ValueError, match="aligned"):
        bootstrap([a], ["a", "b"], .25, n=15, paired=[b])


def test_nonfinite_and_invalid_geometry_rejected():
    r = sample()
    r["scores"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        record_statistics(r)
    r = sample()
    r["boxes"][0] = [10, 0, 0, 10]
    with pytest.raises(ValueError, match="Inverted"):
        record_statistics(r)


@pytest.mark.integration
def test_ap_and_matching_match_installed_ultralytics():
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import compute_ap as native_ap
    from src.evaluation.detection import iou_matrix
    rng = np.random.default_rng(42)
    for _ in range(20):
        recall = np.sort(rng.uniform(size=30))
        precision = rng.uniform(size=30)
        assert compute_ap(recall, precision) == pytest.approx(native_ap(recall, precision)[0], abs=1e-12)
    gt = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    boxes = np.array([[1, 1, 10, 10], [0, 0, 10, 10], [6, 6, 15, 15]])
    gc, pc = np.array([0, 1]), np.array([0, 1, 1])
    validator = BaseValidator()
    validator.iouv = torch.linspace(.5, .95, 10)
    expected = validator.match_predictions(torch.tensor(pc), torch.tensor(gc), torch.tensor(iou_matrix(gt, boxes))).numpy()
    assert np.array_equal(match_predictions(gt, gc, boxes, pc), expected)


@pytest.mark.integration
def test_application_predictor_preserves_validation_multilabel_policy():
    import torch
    from types import SimpleNamespace
    from src.inference.reference_predictor import ReferenceDetectionPredictor
    predictor = ReferenceDetectionPredictor(overrides={"conf": .25, "iou": .7, "task": "detect", "max_det": 300})
    predictor.model = SimpleNamespace(names={0: "a", 1: "b"}, end2end=False)
    predictor.batch = (["toy.png"],)
    raw = torch.tensor([[[5.], [5.], [10.], [10.], [.8], [.7]]])
    results = predictor.postprocess(raw, torch.zeros(1, 3, 20, 20), [np.zeros((20, 20, 3), dtype=np.uint8)])
    assert len(results[0].boxes) == 2
    assert set(results[0].boxes.cls.tolist()) == {0, 1}


@pytest.mark.integration
def test_predictor_uses_native_ceil_geometry_and_inverts_both_axes():
    import cv2
    import torch
    from types import SimpleNamespace
    from src.inference.reference_predictor import ReferenceDetectionPredictor
    predictor = ReferenceDetectionPredictor(overrides={"task": "detect"})
    predictor.imgsz = [32, 32]
    predictor.model = SimpleNamespace(names={0: "a"})
    image = np.full((18, 40, 3), 200, dtype=np.uint8)
    prepared = predictor.pre_transform([image])[0]
    assert np.array_equal(prepared[8:23], cv2.resize(image, (32, 15), interpolation=cv2.INTER_LINEAR))
    native = torch.tensor([[8., 8 + 4 * 15 / 18, 16., 8 + 10 * 15 / 18, .9, 0.]])
    result = predictor.construct_result(native, torch.zeros(1, 3, 32, 32), image, "test.png")
    assert np.allclose(result.boxes.xyxy.numpy(), [[10, 4, 20, 10]], atol=1e-5)
