"""Frozen validation decisions, independent holdouts, stress, and uncertainty."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "4"
import cv2
import numpy as np
import yaml

from src.data.annotations import read_yolo_boxes
from src.data.contract import sha256_file
from src.data.research import NAMES, PREPROCESSING, REFERENCE, verify_research_contract
from src.evaluation.detection import bootstrap, iou_matrix, record_statistics, select_threshold, summarize
from scripts.training.complete_research import write_json


def transform(image, scenario, rng):
    if scenario == "clean":
        return image
    if scenario.startswith("blur_sigma"):
        return cv2.GaussianBlur(image, (0, 0), float(scenario.removeprefix("blur_sigma")))
    if scenario.startswith("noise_sd"):
        noise = rng.normal(0, float(scenario.removeprefix("noise_sd")), image.shape[:2])
        # Same noise in all channels, retaining radiographic grayscale.
        return np.clip(image.astype(float) + noise[:, :, None], 0, 255).astype(np.uint8)
    if scenario.startswith("jpeg_quality"):
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(scenario.removeprefix("jpeg_quality"))])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if scenario.startswith("brightness"):
        return np.clip(image.astype(float) * float(scenario.removeprefix("brightness")), 0, 255).astype(np.uint8)
    if scenario == "downsample0.5":
        h, w = image.shape[:2]
        low = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown stress scenario {scenario}")


def prediction_cache(weights, dataset, rows, imgsz, output, scenario="clean", device=0):
    spec = {"weights_sha256": sha256_file(weights), "imgsz": imgsz, "conf": .001, "iou": .7,
            "rect": False, "max_det": 300, "scenario": scenario, "multi_label": True, "device": device,
            "geometry": "native_validation_ceil_resize_inverse_affine_unclipped_v1",
            "sample_ids": [r["sample_id"] for r in rows], "reference_status": REFERENCE}
    if output.exists():
        saved = json.loads(output.read_text())
        if saved["spec"] != spec:
            raise ValueError("Prediction cache specification changed")
        return saved["records"]
    import torch
    from ultralytics import YOLO
    from src.inference.reference_predictor import ReferenceDetectionPredictor
    torch.set_num_threads(4)
    model = YOLO(str(weights))
    records, timings = [], []
    for start in range(0, len(rows), 8):
        batch, rng = rows[start:start + 8], None
        images = []
        for row in batch:
            image = cv2.imread(str(dataset / row["image"]))
            # Stable per-image corruption across compared models.
            import hashlib
            seed = int.from_bytes(hashlib.sha256(row["sample_id"].encode()).digest()[:4], "little")
            images.append(transform(image, scenario, np.random.default_rng(seed)))
        tick = time.perf_counter()
        predictions = model.predict(images, imgsz=imgsz, conf=.001, iou=.7, max_det=300,
                                    rect=False, device=device, verbose=False, save=False, half=False,
                                    predictor=ReferenceDetectionPredictor)
        timings.append((time.perf_counter() - tick) / len(batch))
        for row, image, result in zip(batch, images, predictions):
            h, w = image.shape[:2]
            targets, _ = read_yolo_boxes(dataset / row["label"], range(len(NAMES)))
            gt = [[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                  for c, x, y, bw, bh in targets]
            records.append({"sample_id": row["sample_id"], "width": w, "height": h,
                            "original_shape": row["original_shape"], "group_id": row["group_id"],
                            "source_split": row["source_split"], "gt_boxes": gt,
                            "gt_classes": [b[0] for b in targets], "boxes": result.boxes.xyxy.cpu().tolist(),
                            "scores": result.boxes.conf.cpu().tolist(), "classes": result.boxes.cls.int().cpu().tolist()})
        if start % 80 == 0:
            print(f"{output.stem}: {min(start + 8, len(rows))}/{len(rows)}", flush=True)
    write_json(output, {"spec": spec, "records": records, "batch_seconds_per_image": timings})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def detected_box_calibration(records, min_score=.05):
    flags, scores = [], []
    for r in records:
        matched, confidence, _, _ = record_statistics(r, min_score)
        flags.extend(matched[:, 0].astype(int).tolist())
        scores.extend(confidence.tolist())
    flags, scores = np.asarray(flags), np.asarray(scores)
    bins, ece = [], 0.
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        keep = (scores >= a) & ((scores < b) if b < 1 else (scores <= b))
        n = int(keep.sum())
        acc, mean = float(flags[keep].mean()) if n else None, float(scores[keep].mean()) if n else None
        if n:
            ece += n / len(scores) * abs(acc - mean)
        bins.append({"low": float(a), "high": float(b), "n": n, "tp_fraction": acc, "mean_score": mean})
    return {"target": "matched_detection_at_IoU0.5_among_predictions_above_0.05", "n_predictions": len(scores),
            "ece_10_equal_width_bins": ece if len(scores) else None,
            "brier_detected_boxes_only": float(np.mean((scores - flags) ** 2)) if len(scores) else None,
            "patient_disease_probability": False, "missed_objects_not_in_calibration_denominator": True, "bins": bins}


def object_size_analysis(records, threshold):
    counts = {k: {"reference_boxes": 0, "tp": 0} for k in ["area_lt_0.001", "area_0.001_to_0.01", "area_ge_0.01"]}
    for r in records:
        keep = np.asarray(r["scores"]) >= threshold
        boxes = np.asarray(r["boxes"]).reshape(-1, 4)[keep]
        classes = np.asarray(r["classes"])[keep]
        gt = np.asarray(r["gt_boxes"]).reshape(-1, 4)
        overlap = iou_matrix(gt, boxes)
        overlap *= np.asarray(r["gt_classes"])[:, None] == classes[None, :]
        # Same one-to-one assignment as the primary detector metrics.
        matches = np.array(np.nonzero(overlap >= .5)).T
        if len(matches) > 1:
            matches = matches[overlap[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        detected = set(matches[:, 0]) if len(matches) else set()
        for i, box in enumerate(gt):
            area = float(np.prod(box[2:] - box[:2]) / (r["width"] * r["height"]))
            key = "area_lt_0.001" if area < .001 else "area_0.001_to_0.01" if area < .01 else "area_ge_0.01"
            counts[key]["reference_boxes"] += 1
            counts[key]["tp"] += i in detected
    for row in counts.values():
        row["recall"] = row["tp"] / row["reference_boxes"] if row["reference_boxes"] else None
    return counts


def confusion_and_errors(records, threshold):
    n = len(NAMES)
    matrix = np.zeros((n + 1, n + 1), dtype=int)  # row truth, col predicted
    errors = []
    for r in records:
        keep = np.asarray(r["scores"]) >= threshold
        boxes = np.asarray(r["boxes"]).reshape(-1, 4)[keep]
        pred = np.asarray(r["classes"], dtype=int)[keep]
        gt = np.asarray(r["gt_boxes"]).reshape(-1, 4)
        labels = np.asarray(r["gt_classes"], dtype=int)
        overlap = iou_matrix(gt, boxes)
        pairs = np.array(np.nonzero(overlap >= .5)).T
        if len(pairs) > 1:
            pairs = pairs[overlap[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        gt_used, pred_used = set(), set()
        for a, b in pairs:
            matrix[labels[a], pred[b]] += 1
            gt_used.add(a)
            pred_used.add(b)
        for i, c in enumerate(labels):
            if i not in gt_used:
                matrix[c, n] += 1
        for i, c in enumerate(pred):
            if i not in pred_used:
                matrix[n, c] += 1
        stats = summarize([r], NAMES, threshold)
        errors.append({"sample_id": r["sample_id"], "source_split": r["source_split"],
                       "fn": sum(v["fn"] for v in stats["by_class"]), "fp": sum(v["fp"] for v in stats["by_class"]),
                       "by_class": [{k: v[k] for k in ["class", "support", "tp", "fp", "fn"]} for v in stats["by_class"]]})
    return {"labels": NAMES + ["background"], "row": "reference", "column": "prediction", "iou": .5,
            "matching": "class_agnostic_one_to_one", "matrix": matrix.tolist(),
            "error_ranking": sorted(errors, key=lambda r: r["fn"] + r["fp"], reverse=True)}


def evaluate(local, reports, bootstrap_n=1000):
    protocol_path = ROOT / "configs/training/research_protocol.yaml"
    p = yaml.safe_load(protocol_path.read_text())
    frozen = json.loads((reports / "protocol_frozen.json").read_text())
    if sha256_file(protocol_path) != frozen["sha256"]:
        raise ValueError("Protocol changed")
    contract, all_rows = verify_research_contract(ROOT / p["contract"])
    dataset = Path(contract["root"])
    selected = json.loads((reports / "selected_configuration.json").read_text())
    confirmations = json.loads((reports / "confirmation.json").read_text())
    if len(confirmations) != 6:
        raise ValueError("All six confirmation runs must finish before holdout evaluation")
    contract_hash=sha256_file(ROOT / p["contract"])
    for item in confirmations:
        directory=Path(item["run"])
        record=json.loads((directory / "run.json").read_text(encoding="utf-8"))
        digest=sha256_file(directory / "weights/best.pt")
        if (record["status"] != "completed" or record["best_sha256"] != digest
                or item["validation"]["weights_sha256"] != digest
                or record["source_contract_sha256"] != contract_hash):
            raise ValueError("Confirmation provenance or checkpoint integrity changed")
    runs = {r["name"]: Path(r["run"]) for r in confirmations}
    final_name = "selected_n_s42"
    final_weights = runs[final_name] / "weights/best.pt"
    imgsz = selected["params"]["imgsz"]
    cache = local / "evaluation/research_v2"
    cache.mkdir(parents=True, exist_ok=True)
    val_rows = [r for r in all_rows if r["split"] == "val"]
    validation = prediction_cache(final_weights, dataset, val_rows, imgsz, cache / "selected_val.json")
    # Independent implementation check against the installed official validator
    # on the same development set. A material discrepancy blocks reporting.
    from scripts.training.complete_research import native_validation
    native = native_validation(runs[final_name], local, imgsz)
    check = summarize(validation, NAMES, .25)
    native_by_id = {int(k): v for k, v in native["by_class_ap"].items()}
    per_class_differences = {r["class"]: abs(r["ap50"] - native_by_id[r["class_id"]][0])
                             for r in check["by_class"]}
    parity = {"native_map50": native["map50"], "cached_map50": check["map50"],
              "native_map50_95": native["map50_95"], "cached_map50_95": check["map50_95"],
              "absolute_tolerance": .0001, "per_class_ap50_absolute_differences": per_class_differences}
    write_json(reports / "metric_parity.json", parity)
    if (abs(check["map50"] - native["map50"]) > .0001 or abs(check["map50_95"] - native["map50_95"]) > .0001
            or max(per_class_differences.values()) > .0001):
        raise ValueError("Prediction/metric parity failed; investigate geometry before reporting")
    operating, grid = select_threshold(validation, NAMES)
    threshold = operating["threshold"]
    decision = {"final_run": final_name, "weights_sha256": sha256_file(final_weights),
                "imgsz": imgsz, "threshold": threshold, "objective": "validation_pathology_macro_F1",
                "grid": grid, "protocol_sha256": frozen["sha256"], "decided_before_holdout_predictions": True}
    decision_path = reports / "evaluation_decision.json"
    if decision_path.exists() and json.loads(decision_path.read_text()) != decision:
        raise ValueError("Frozen evaluation decision changed")
    write_json(decision_path, decision)
    write_json(ROOT / "models/research_v2/model.json", {"version": "2.0.0-source-reference", "weights": str(final_weights),
               "weights_sha256": sha256_file(final_weights), "names": NAMES, "preprocessing": PREPROCESSING,
               "reference_status": REFERENCE, "clinical_approval": False, "imgsz": imgsz, "threshold": threshold,
               "nms_multi_label": True,
               "box_coordinate_policy": "inverse_native_affine_unclipped",
               "protocol_sha256": frozen["sha256"]})
    # Verify every architecture's validation path before any holdout prediction.
    # The final model and threshold are already frozen and are not reselected.
    validation_results = []
    all_models_parity = {}
    for name, directory in runs.items():
        size = 640 if name.startswith("baseline") else imgsz
        records = prediction_cache(directory / "weights/best.pt", dataset, val_rows, size, cache / f"{name}_val.json")
        item = {"run": name, "metrics": summarize(records, NAMES, threshold)}
        native_comparison = next(r["validation"] for r in confirmations if r["name"] == name)
        by_id = {int(k): v for k, v in native_comparison["by_class_ap"].items()}
        class_differences = {r["class"]: {"ap50": abs(r["ap50"]-by_id[r["class_id"]][0]),
                                         "ap50_95": abs(r["ap50_95"]-float(np.mean(by_id[r["class_id"]])))}
                             for r in item["metrics"]["by_class"]}
        differences = {key: abs(item["metrics"][key]-native_comparison[key]) for key in ["map50","map50_95"]}
        passed = max([*differences.values(), *[v for row in class_differences.values() for v in row.values()]]) <= .0001
        all_models_parity[name] = {"passed": passed, "absolute_tolerance": .0001,
                                  "global_absolute_differences": differences, "per_class_absolute_differences": class_differences}
        write_json(reports / "metric_parity_all_models.json", all_models_parity)
        if not passed:
            raise ValueError(f"Native validation parity failed for {name}; holdouts remain unopened")
        item["native_parity"] = all_models_parity[name]
        if name == "baseline_n640_s42":
            item["paired_delta_selected_minus_baseline"] = bootstrap(validation, NAMES, threshold, bootstrap_n, paired=records)
        validation_results.append(item)
        write_json(reports / "validation_comparisons.json", validation_results)
    summary = {"reference_status": REFERENCE, "patient_independence": "unknown", "decision": decision, "partitions": {}}
    for split in ["train", "val", "test", "external"]:
        rows = [r for r in all_rows if r["split"] == split]
        records = validation if split == "val" else prediction_cache(final_weights, dataset, rows, imgsz, cache / f"selected_{split}.json")
        result = summarize(records, NAMES, threshold)
        if split != "train":
            result["bootstrap95"] = bootstrap(records, NAMES, threshold, bootstrap_n)
        result["detected_box_calibration"] = detected_box_calibration(records)
        result["object_size"] = object_size_analysis(records, threshold)
        write_json(reports / f"errors_{split}.json", confusion_and_errors(records, threshold))
        summary["partitions"][split] = result
        write_json(reports / "evaluation.json", summary)
    stress = []
    clean = summarize(validation, NAMES, threshold)
    for scenario in p["evaluation"]["stress_scenarios"]:
        records = prediction_cache(final_weights, dataset, val_rows, imgsz, cache / f"stress_{scenario}.json", scenario)
        metrics = summarize(records, NAMES, threshold)
        stress.append({"scenario": scenario, "metrics": metrics,
                       "delta_map50": metrics["map50"] - clean["map50"],
                       "delta_pathology_recall": sum(metrics["by_class"][c]["recall"] - clean["by_class"][c]["recall"] for c in (0, 1)) / 2,
                       "paired_delta_clean_minus_stress": bootstrap(validation, NAMES, threshold, bootstrap_n, paired=records)})
        write_json(reports / "stress.json", stress)
    print("FROZEN EVALUATION COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    evaluate(Path(os.environ.get("DENTAL_LOCAL_ROOT", "C:/Users/crisc/AppData/Local/DentalAI")), ROOT / "reports/completion", args.bootstrap)
