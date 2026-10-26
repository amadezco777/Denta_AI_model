"""Entrena YOLO con contrato externo y registros de corridas incluso fallidas.

Uso: python -m src.training.train_research --config configs/training/baseline.yaml
Las pruebas rapidas usan un runner inyectado sin importar PyTorch/Ultralytics.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.contract import sha256_file, verify_contract
from src.data.provenance import environment, git_state
from src.utils.config import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVED = {"data", "model", "seed", "project", "name", "deterministic", "exist_ok", "resume",
            "pretrained", "cfg", "task", "mode", "save_dir", "classes", "single_cls", "fraction", "split"}
SUPPORTED_TRAIN_ARGS = {
    "epochs", "batch", "imgsz", "device", "workers", "optimizer", "lr0", "lrf", "momentum",
    "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "patience", "amp",
    "cache", "rect", "cos_lr", "close_mosaic", "nbs", "dropout", "val", "plots", "save",
    "save_period", "verbose", "time", "freeze", "multi_scale", "profile", "overlap_mask",
    "mask_ratio", "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
    "scale", "shear", "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup",
    "cutmix", "copy_paste", "copy_paste_mode", "auto_augment", "erasing", "iou", "max_det",
    "conf", "agnostic_nms",
}


class DatasetMismatch(ValueError):
    """Datos o semantica distintos del contrato aprobado."""


def parse_overrides(pairs):
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key or not value.strip() or key in out:
            raise ValueError(f"--set espera clave=valor unica y no vacia, no {pair!r}")
        out[key] = yaml.safe_load(value)
    return out


def validate_config(cfg):
    required = {"model", "data", "dataset_lock", "seeds", "project", "name", "train"}
    if not isinstance(cfg, dict) or not required.issubset(cfg):
        raise ValueError(f"Configuracion requiere {sorted(required)}")
    extra = set(cfg) - required - {"deterministic"}
    if extra:
        raise ValueError(f"Claves de configuracion desconocidas: {sorted(extra)}")
    if not isinstance(cfg["train"], dict):
        raise ValueError("train debe ser un mapa")
    reserved = sorted(set(cfg["train"]) & RESERVED)
    if reserved:
        raise ValueError(f"{reserved} van fuera de 'train:' o estan reservados por el ejecutor")
    unknown = sorted(set(cfg["train"]) - SUPPORTED_TRAIN_ARGS)
    if unknown:
        raise ValueError(f"El ejecutor no conoce estos parametros: {unknown}")
    for key in ("model", "data", "dataset_lock", "project", "name"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} requiere una cadena no vacia")
    if not re.fullmatch(r"[\w.-]+", cfg["name"]) or cfg["name"] in {".", ".."}:
        raise ValueError("name no puede contener rutas")
    seeds = cfg["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("seeds requiere enteros unicos en [0, 2**32)")
    if type(cfg.get("deterministic", True)) is not bool:
        raise ValueError("deterministic requiere true/false fuera de train")
    if cfg["train"].get("val", True) is not True or cfg["train"].get("save", True) is not True:
        raise ValueError("El baseline requiere val=true y save=true")
    cache = cfg["train"].get("cache", False)
    if type(cache) is not bool and cache != "ram":
        raise ValueError("cache admite false/true/ram; disk alteraria los archivos bloqueados")
    device = cfg["train"].get("device")
    if device is not None and (type(device) not in (str, int) or (isinstance(device, str) and "," in device)):
        raise ValueError("El registro de corridas admite un solo dispositivo, no entrenamiento distribuido")
    for key in ("epochs", "imgsz", "workers", "patience", "close_mosaic", "nbs"):
        if key in cfg["train"]:
            minimum = 0 if key in {"workers", "patience", "close_mosaic"} else 1
            if type(cfg["train"][key]) is not int or cfg["train"][key] < minimum:
                raise ValueError(f"{key} requiere un entero >= {minimum}")
    for key, value in cfg["train"].items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} debe ser finito")
    return cfg


def load_config(path, overrides=()):
    cfg = load_yaml(path)
    if not isinstance(cfg.get("train", {}), dict):
        raise ValueError("train debe ser un mapa")
    cfg["train"] = {**cfg.get("train", {}), **parse_overrides(overrides)}
    return validate_config(cfg)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def check_dataset(data_yaml, lock_path=None):
    if lock_path is None:
        raise DatasetMismatch("No registra contrato externo dataset_lock; la huella historica no basta")
    try:
        ok, expected, actual = verify_contract(data_yaml, lock_path, for_training=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DatasetMismatch(f"Contrato no valido: {exc}") from exc
    if not ok:
        raise DatasetMismatch(f"{data_yaml}: contrato no coincide\n  registrado: {expected}\n  actual: {actual}")
    return actual


def training_environment(device=None):
    env = environment()
    for pkg in ("torch", "torchvision", "ultralytics"):
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = None
    env.update({"requested_device": device, "cuda": None, "gpu": None})
    return env


def _write_record(save_dir, record):
    temporary = save_dir / "run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(save_dir / "run.json")


def run_ultralytics(model_path, data_yaml, seed, project, name, train_args):
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    unknown = sorted(set(train_args) - set(DEFAULT_CFG_DICT))
    if unknown:
        raise ValueError(f"Ultralytics instalado no acepta: {unknown}")
    save_dir = project / name
    record = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError("El contrato de este ejecutor es YOLO detect; no admite segmentacion/clasificacion")
    loaded = getattr(model, "ckpt_path", None)
    if str(model_path).endswith(".pt"):
        if not loaded or not Path(loaded).is_file():
            raise ValueError("No se puede identificar el checkpoint realmente cargado")
        record["pretrained"] = {"path": str(Path(loaded).resolve()), "sha256": sha256_file(loaded),
                                "kind": "checkpoint"}
    else:
        record["pretrained"] = {"path": str(model_path), "kind": "architecture_random_initialization",
                                "sha256": sha256_file(model_path) if model_path.is_file() else None,
                                "architecture": getattr(model.model, "yaml", {})}
    _write_record(save_dir, record)
    kwargs = {**train_args, "data": str(data_yaml), "seed": seed, "project": str(project),
              "name": name, "exist_ok": True}
    actual_device = None

    def started(trainer):
        nonlocal actual_device
        actual_device = trainer.device
        record["effective_config"] = dict(vars(trainer.args))
        record["environment"]["device"] = str(actual_device)
        record["environment"]["cuda"] = torch.version.cuda
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            record["optimizer"] = {"class": type(optimizer).__name__,
                                   "parameter_groups_at_start": [{k: group[k] for k in ("lr", "weight_decay", "momentum", "betas")
                                                                  if k in group} for group in optimizer.param_groups]}
        if actual_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actual_device)
            record["environment"]["gpu"] = torch.cuda.get_device_name(actual_device)
        _write_record(save_dir, record)

    model.add_callback("on_train_start", started)
    start = time.perf_counter()
    try:
        model.train(**kwargs)
    finally:
        record["resources"] = {"train_seconds": round(time.perf_counter() - start, 3),
                               "gpu_mem_peak_gb": (round(torch.cuda.max_memory_reserved(actual_device) / 1e9, 3)
                                                   if actual_device is not None and actual_device.type == "cuda" else None)}
        if getattr(model, "trainer", None) is not None:
            record["effective_config"] = dict(vars(model.trainer.args))
        _write_record(save_dir, record)
    return Path(model.trainer.save_dir), record["resources"]["train_seconds"], record["resources"]["gpu_mem_peak_gb"]


def train(cfg, runner=run_ultralytics):
    cfg = copy.deepcopy(validate_config(cfg))
    data_yaml, lock_path = resolve(cfg["data"]), resolve(cfg["dataset_lock"])
    dataset_digest = check_dataset(data_yaml, lock_path)
    model_path = resolve(cfg["model"])
    if not model_path.is_file():
        if Path(cfg["model"]).is_absolute() or len(Path(cfg["model"]).parts) > 1:
            raise ValueError(f"No existen los pesos locales: {model_path}")
        model_path = Path(cfg["model"])
    project = resolve(cfg["project"])
    names = [f"{cfg['name']}_s{seed}" for seed in cfg["seeds"]]
    if any((project / name).exists() for name in names):
        raise FileExistsError("Ya existe una corrida solicitada; usa otro name o project")
    meta = load_yaml(data_yaml).get("meta", {})
    run_dirs = []
    for seed, name in zip(cfg["seeds"], names):
        check_dataset(data_yaml, lock_path)
        save_dir = project / name
        save_dir.mkdir(parents=True, exist_ok=False)
        record = {"schema_version": 1, "status": "running",
                  "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
                  "dataset": {"yaml": str(data_yaml), "contract": str(lock_path), "contract_hash": dataset_digest,
                              "taxonomy_version": meta.get("taxonomy_version")},
                  "code": git_state(REPO_ROOT), "seed": seed,
                  "requested_config": copy.deepcopy(cfg), "effective_config": None,
                  "environment": training_environment(cfg["train"].get("device")),
                  "pretrained": {"path": str(model_path),
                                 "sha256": sha256_file(model_path) if model_path.is_file() else None},
                  "resources": {}, "error": None}
        _write_record(save_dir, record)
        start = time.perf_counter()
        try:
            kwargs = copy.deepcopy(cfg["train"])
            kwargs["deterministic"] = cfg.get("deterministic", True)
            result_dir, seconds, peak = runner(model_path, data_yaml, seed, project, name, kwargs)
            if Path(result_dir).resolve() != save_dir:
                raise ValueError("El entrenador cambio la carpeta reservada para la corrida")
            check_dataset(data_yaml, lock_path)
            record = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
            record["resources"] = {"train_seconds": seconds, "gpu_mem_peak_gb": peak}
            record["status"] = "completed"
        except BaseException as exc:
            record = json.loads((save_dir / "run.json").read_text(encoding="utf-8"))
            record["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["resources"].setdefault("train_seconds", round(time.perf_counter() - start, 3))
            record["resources"].setdefault("gpu_mem_peak_gb", None)
            _write_record(save_dir, record)
            provenance = {"started_at": record["started_at"], "status": record["status"],
                          "dataset": {**record["dataset"], "fingerprint": dataset_digest},
                          "code": record["code"], "pretrained": record["pretrained"], "seed": seed,
                          "config": {k: v for k, v in cfg.items() if k != "seeds"},
                          "environment": record["environment"], "resources": record["resources"],
                          "error": record["error"]}
            (save_dir / "provenance.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True),
                                                     encoding="utf-8")
        print(f"Semilla {seed}: {save_dir} ({record['status']})")
        run_dirs.append(save_dir)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/training/baseline.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--name")
    parser.add_argument("--model")
    parser.add_argument("--check-only", action="store_true", help="valida config y contrato sin entrenar")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config, args.set)
        for key in ("seeds", "name", "model"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        validate_config(cfg)
        if args.check_only:
            print(check_dataset(resolve(cfg["data"]), resolve(cfg["dataset_lock"])))
        else:
            train(cfg)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"NO SE ENTRENA. {exc}") from exc


if __name__ == "__main__":
    main()
