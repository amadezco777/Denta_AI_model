"""Object-detection metrics, paired image bootstrap, and threshold selection.

AP integration and IoU matching follow Ultralytics 8.4.173. Scores are object
detection scores, never patient-level diagnostic probabilities.
"""
import numpy as np

IOU_THRESHOLDS = np.linspace(.5, .95, 10)


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite boxes required")
    if np.any(a[:, 2:] < a[:, :2]) or np.any(b[:, 2:] < b[:, :2]):
        raise ValueError("Inverted box coordinates")
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.maximum(hi - lo, 0).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return overlap / np.maximum(area_a[:, None] + area_b[None, :] - overlap, 1e-12)


def match_predictions(gt_boxes, gt_classes, boxes, classes, thresholds=IOU_THRESHOLDS):
    classes, gt_classes = np.asarray(classes), np.asarray(gt_classes)
    overlap = iou_matrix(gt_boxes, boxes)
    overlap *= gt_classes[:, None] == classes[None, :]
    correct = np.zeros((len(classes), len(thresholds)), dtype=bool)
    for k, threshold in enumerate(thresholds):
        matches = np.array(np.nonzero(overlap >= threshold)).T
        if len(matches) > 1:
            matches = matches[overlap[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        if len(matches):
            correct[matches[:, 1], k] = True
    return correct


def compute_ap(recall, precision):
    if not len(recall):
        return 0.0
    recall, precision = np.asarray(recall), np.asarray(precision)
    mr = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mp = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mp = np.maximum.accumulate(mp[::-1])[::-1]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(np.linspace(0, 1, 101), mr, mp), np.linspace(0, 1, 101)))


def record_statistics(record, threshold=.001):
    scores = np.asarray(record["scores"], dtype=float)
    raw_classes = np.asarray(record["classes"], dtype=float)
    raw_gt = np.asarray(record["gt_classes"], dtype=float)
    if any(not np.isfinite(a).all() or np.any(a < 0) or np.any(a != np.floor(a)) for a in [raw_classes, raw_gt]):
        raise ValueError("Nonnegative integer class IDs required")
    classes = raw_classes.astype(int)
    boxes = np.asarray(record["boxes"], dtype=float).reshape(-1, 4)
    if (scores.ndim != 1 or classes.ndim != 1 or len(scores) != len(classes) or len(boxes) != len(scores)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("Aligned finite detection scores, classes and boxes required")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Score threshold must be in [0, 1]")
    keep = scores >= threshold
    classes, scores, boxes = classes[keep], scores[keep], boxes[keep]
    gt_classes = raw_gt.astype(int)
    if gt_classes.ndim != 1 or len(record["gt_boxes"]) != len(gt_classes):
        raise ValueError("Aligned reference boxes and classes required")
    correct = match_predictions(record["gt_boxes"], gt_classes, boxes, classes)
    return correct, scores, classes, gt_classes


def summarize(records, names, threshold=.25):
    if any(c < 0 or c >= len(names) for r in records for values in (r["classes"], r["gt_classes"]) for c in values):
        raise ValueError("Unknown class ID")
    stats = [record_statistics(r) for r in records]
    operating = [record_statistics(r, threshold) for r in records]
    return _summarize_statistics(stats, operating, names, threshold)


def _summarize_statistics(stats, operating, names, threshold):
    if not stats:
        raise ValueError("At least one evaluated image is required")
    correct = np.concatenate([s[0] for s in stats])
    scores = np.concatenate([s[1] for s in stats])
    classes = np.concatenate([s[2] for s in stats])
    gt = np.concatenate([s[3] for s in stats])
    order = np.argsort(-scores)
    correct, scores, classes = correct[order], scores[order], classes[order]
    rows = []
    for c, name in enumerate(names):
        support = int(np.count_nonzero(gt == c))
        selected = classes == c
        flags = correct[selected]
        values = []
        for k in range(len(IOU_THRESHOLDS)):
            tp = np.cumsum(flags[:, k])
            fp = np.cumsum(~flags[:, k])
            values.append(compute_ap(tp / max(support, 1), tp / np.maximum(tp + fp, 1)) if support else None)
        # Recompute matching at the frozen display threshold. A low-confidence
        # box must not steal a match from an accepted high-confidence detection.
        tp = sum(int(s[0][s[2] == c, 0].sum()) for s in operating)
        n_pred = sum(int((s[2] == c).sum()) for s in operating)
        fp, fn = n_pred - tp, support - tp
        p, r = tp / max(n_pred, 1), tp / max(support, 1)
        rows.append({"class_id": c, "class": name, "support": support, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r if support else None,
                     "f1": 2 * p * r / max(p + r, 1e-12) if support else None,
                     "ap50": values[0], "ap50_95": float(np.mean(values)) if support else None})
    present = [r for r in rows if r["support"]]
    return {"images": len(stats), "boxes": len(gt), "threshold": threshold,
            "map50": float(np.mean([r["ap50"] for r in present])) if present else None,
            "map50_95": float(np.mean([r["ap50_95"] for r in present])) if present else None,
            "false_positives_per_image": sum(r["fp"] for r in rows) / len(stats),
            "classes_without_reference": [r["class"] for r in rows if not r["support"]], "by_class": rows}


def select_threshold(records, names):
    candidates = []
    for threshold in np.arange(.05, .81, .05):
        result = summarize(records, names, float(threshold))
        score = np.mean([result["by_class"][c]["f1"] or 0 for c in (0, 1)])
        candidates.append({"threshold": float(threshold), "pathology_macro_f1": float(score),
                           "false_positives_per_image": result["false_positives_per_image"]})
    selected = max(candidates, key=lambda r: (r["pathology_macro_f1"], r["threshold"]))
    return selected, candidates


def bootstrap(records, names, threshold, n=1000, seed=42, paired=None):
    """Resample whole images; image identity does not certify patient identity."""
    if paired is not None and [r["sample_id"] for r in records] != [r["sample_id"] for r in paired]:
        raise ValueError("Paired bootstrap requires exactly aligned image IDs")
    if not records or type(n) is not int or n < 1:
        raise ValueError("Nonempty image cohort and positive bootstrap count required")
    if paired is not None and any(a["gt_classes"] != b["gt_classes"] or a["gt_boxes"] != b["gt_boxes"] for a, b in zip(records, paired)):
        raise ValueError("Paired reference annotations differ")
    stats = [record_statistics(r) for r in records]
    operating = [record_statistics(r, threshold) for r in records]
    other_stats = [record_statistics(r) for r in paired] if paired is not None else None
    other_operating = [record_statistics(r, threshold) for r in paired] if paired is not None else None
    rng = np.random.default_rng(seed)
    metrics = {"map50": [], "map50_95": []}
    for name in names:
        metrics[name + "/recall"] = []
        metrics[name + "/ap50"] = []
    for _ in range(n):
        indexes = rng.integers(0, len(records), len(records))
        result = _summarize_statistics([stats[i] for i in indexes], [operating[i] for i in indexes], names, threshold)
        other = _summarize_statistics([other_stats[i] for i in indexes], [other_operating[i] for i in indexes], names, threshold) if paired is not None else None
        for key in ["map50", "map50_95"]:
            value = result[key]
            if result["classes_without_reference"]:
                value = None
            if other is not None:
                value = value - other[key] if value is not None and other[key] is not None else None
            metrics[key].append(value)
        for c, name in enumerate(names):
            for metric in ["recall", "ap50"]:
                value = result["by_class"][c][metric]
                if other is not None:
                    v2 = other["by_class"][c][metric]
                    value = value - v2 if value is not None and v2 is not None else None
                metrics[name + "/" + metric].append(value)
    intervals = {}
    for key, values in metrics.items():
        valid = [v for v in values if v is not None and np.isfinite(v)]
        intervals[key] = {"low": float(np.percentile(valid, 2.5)) if valid else None,
                          "high": float(np.percentile(valid, 97.5)) if valid else None,
                          "valid_replicates": len(valid)}
    return {"method": "paired_image_percentile" if paired is not None else "image_percentile",
            "patient_independence": "unknown", "replicates": n, "seed": seed, "intervals": intervals}
