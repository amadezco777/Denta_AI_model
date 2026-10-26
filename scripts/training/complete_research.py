"""Execute the frozen local source-reference experiment; no holdout tuning."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "4"
os.environ["YOLO_AUTOINSTALL"] = "false"
import yaml

from src.data.contract import sha256_file
from src.training.research import train_source_reference


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")
    temp.replace(path)


def load_protocol():
    path = ROOT / "configs/training/research_protocol.yaml"
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = Path(p["local_root"])
    reports = ROOT / "reports/completion"
    reports.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_file(path)
    frozen = reports / "protocol_frozen.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["sha256"] != fingerprint:
            raise ValueError("Frozen protocol changed; version the experiment instead")
    else:
        write_json(frozen, {"sha256": fingerprint, "frozen_at_unix": time.time(), "protocol": p})
    return p, local, reports


def release():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def run(p, local, name, model="yolo11n.pt", seed=42, **overrides):
    import torch
    torch.set_num_threads(4)
    target = local / "runs/research_v2" / name
    if target.exists():
        record = json.loads((target / "run.json").read_text())
        if record["status"] != "completed":
            raise ValueError(f"Previous run {name} did not complete; inspect its record")
        if record["best_sha256"] != sha256_file(target / "weights/best.pt"):
            raise ValueError("Completed weights changed")
        return target
    pretrained = local / "models/pretrained" / model
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    if not pretrained.is_file():
        # The official model name allows Ultralytics to fetch its release asset.
        from ultralytics import YOLO
        previous = Path.cwd()
        try:
            os.chdir(pretrained.parent)
            downloaded = YOLO(model)
            del downloaded
        finally:
            os.chdir(previous)
    params = p["train"] | overrides
    directory = train_source_reference(pretrained, ROOT / p["contract"], target.parent, name, seed, **params)
    release()
    return directory


def native_validation(run_dir, local, imgsz):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    output = run_dir / "selection_validation.json"
    if output.exists():
        return json.loads(output.read_text())
    model = YOLO(str(run_dir / "weights/best.pt"))
    m = model.val(data=str(local / "data/research_v2/dataset.yaml"), split="val", imgsz=imgsz,
                  batch=8, workers=0, device=0, conf=.001, iou=.7, max_det=300,
                  rect=False, plots=False, verbose=False, project=str(run_dir), name="selection", exist_ok=True)
    by_id = {int(c): list(map(float, ap)) for c, ap in zip(m.box.ap_class_index, m.box.all_ap)}
    result = {"pathology_ap50": sum(by_id.get(c, [0])[0] for c in (0, 1)) / 2,
              "map50": float(m.box.map50), "map50_95": float(m.box.map),
              "by_class_ap": by_id, "metrics": dict(m.results_dict), "imgsz": imgsz,
              "weights_sha256": sha256_file(run_dir / "weights/best.pt")}
    write_json(output, result)
    del model
    release()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["pilot", "experiments"])
    args = parser.parse_args()
    p, local, reports = load_protocol()
    if args.phase == "pilot":
        directory = run(p, local, "pilot_n640", epochs=p["pilot_epochs"], patience=0)
        write_json(reports / "pilot.json", json.loads((directory / "run.json").read_text()))
        return
    import optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=6))
    # Each ask/tell is replayed in order after interruption; immutable run dirs
    # prevent accidentally fitting a new model over a completed result.
    trials = []
    for i in range(p["search"]["trials"]):
        trial = study.ask()
        params = {"lr0": trial.suggest_float("lr0", .0002, .003, log=True),
                  "weight_decay": trial.suggest_float("weight_decay", .00001, .002, log=True),
                  "scale": trial.suggest_float("scale", .05, .4),
                  "imgsz": trial.suggest_categorical("imgsz", [640, 960])}
        directory = run(p, local, f"search_{i:02d}", epochs=p["search"]["epochs"], patience=0, **params)
        actual = json.loads((directory / "run.json").read_text())["config"]["train"]
        if any(actual[k] != v for k, v in params.items()):
            raise ValueError("Replayed TPE parameters differ from stored trial")
        metrics = native_validation(directory, local, params["imgsz"])
        study.tell(trial, metrics["pathology_ap50"])
        trials.append({"trial": i, "params": params, "value": metrics["pathology_ap50"],
                       "map50": metrics["map50"], "map50_95": metrics["map50_95"], "run": str(directory)})
        write_json(reports / "search_trials.json", trials)
    selected = trials[study.best_trial.number]
    write_json(reports / "selected_configuration.json", selected | {"selection_partition": "val", "budget_epochs": p["search"]["epochs"]})
    epochs, patience = p["confirmation"]["epochs"], p["confirmation"]["patience"]
    variants = [("baseline_n640_s42", "yolo11n.pt", 42, {}),
                *[(f"selected_n_s{s}", "yolo11n.pt", s, selected["params"]) for s in p["confirmation"]["seeds"]],
                ("architecture_26n_s42", "yolo26n.pt", 42, selected["params"]),
                ("ablation_mosaic_s42", "yolo11n.pt", 42, selected["params"] | {"mosaic": .5})]
    results = []
    for name, model, seed, params in variants:
        directory = run(p, local, name, model=model, seed=seed, epochs=epochs, patience=patience, **params)
        metrics = native_validation(directory, local, params.get("imgsz", p["train"]["imgsz"]))
        results.append({"name": name, "seed": seed, "model": model, "params": params, "run": str(directory), "validation": metrics})
        write_json(reports / "confirmation.json", results)
    print("ALL FROZEN EXPERIMENTS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
