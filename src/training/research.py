"""Traceable training against published source annotations without fake reviews.

The clinician-approved training path in train_baseline (v1) remains unchanged.
This runner explicitly records an exploratory, unadjudicated reference level.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from src.data.contract import sha256_file
from src.data.research import verify_research_contract, REFERENCE
from src.training.train_research import run_ultralytics, training_environment, validate_config
from src.data.provenance import git_state


def train_source_reference(model, contract_path, project, name, seed=42, **parameters):
    contract, _ = verify_research_contract(contract_path, for_training=True)
    root = Path(contract["root"])
    # Reuse argument validation, but never call the approved-contract runner or
    # construct a CSV with a fictional reviewer/coverage assertion.
    cfg = {"model": str(model), "data": str(root / "dataset.yaml"),
           "dataset_lock": str(contract_path), "seeds": [seed], "project": str(project),
           "name": name, "deterministic": True, "train": parameters}
    validate_config(cfg)
    project = Path(project).resolve()
    save_dir = project / name
    if save_dir.exists():
        raise FileExistsError(f"Run already exists: {save_dir}")
    save_dir.mkdir(parents=True)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "reference_status": REFERENCE, "clinical_approval": False,
              "config": copy.deepcopy(cfg), "environment": training_environment(parameters.get("device")),
              "code": git_state(Path(__file__).resolve().parents[2]),
              "source_contract_sha256": sha256_file(contract_path),
              "dataset_inventory_digest": hashlib.sha256(json.dumps(contract["files"], sort_keys=True).encode()).hexdigest()}
    (save_dir / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    started = time.perf_counter()
    try:
        directory, seconds, memory = run_ultralytics(Path(model), root / "dataset.yaml", seed,
                                                    project, name, parameters | {"deterministic": True})
        verify_research_contract(contract_path, for_training=True)
        record = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
        record.update({"status": "completed", "elapsed_seconds": time.perf_counter() - started,
                       "best_sha256": sha256_file(directory / "weights/best.pt")})
        return directory
    except BaseException as exc:
        record = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
        record.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (save_dir / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
