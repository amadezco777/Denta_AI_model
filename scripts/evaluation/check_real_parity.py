"""Preflight metric parity using only the pilot and development images."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "4"


def main():
    import torch
    from ultralytics import YOLO
    from src.data.research import NAMES, verify_research_contract
    from src.evaluation.detection import summarize, bootstrap
    from scripts.evaluation.evaluate_research import prediction_cache
    from scripts.training.complete_research import write_json
    torch.set_num_threads(4)
    contract, rows = verify_research_contract(ROOT / "configs/data/contracts/research_v2.json")
    local = Path.home() / "AppData/Local/DentalAI"
    weights = local / "runs/research_v2/pilot_n640/weights/best.pt"
    selected = [r for r in rows if r["split"] == "val"]
    records = prediction_cache(weights, Path(contract["root"]), selected, 640,
                               local / "evaluation/research_v2/pilot_preflight_v2_val.json", device="cpu")
    computed = summarize(records, NAMES, .25)
    model = YOLO(str(weights))
    native = model.val(data=str(Path(contract["root"]) / "dataset.yaml"), imgsz=640, batch=8, workers=0,
                       device="cpu", rect=False, conf=.001, iou=.7, max_det=300, plots=False, verbose=False,
                       project=str(local / "evaluation/research_v2"), name="pilot_parity_native", exist_ok=True)
    tick = time.perf_counter()
    intervals = bootstrap(records, NAMES, .25, n=1000)
    report = {"scope": "pilot_and_validation_only", "native_map50": float(native.box.map50),
              "cached_map50": computed["map50"], "native_map50_95": float(native.box.map),
              "cached_map50_95": computed["map50_95"], "multi_label": True, "bootstrap": intervals,
              "bootstrap_seconds": time.perf_counter() - tick}
    report["passed"] = abs(report["native_map50"] - report["cached_map50"]) <= .003 and abs(report["native_map50_95"] - report["cached_map50_95"]) <= .003
    write_json(ROOT / "reports/completion/pilot_metric_parity.json", report)
    if not report["passed"]:
        raise ValueError("Pilot metric parity failed")
    print(json.dumps({k: v for k, v in report.items() if k != "bootstrap"}), flush=True)


if __name__ == "__main__":
    main()
