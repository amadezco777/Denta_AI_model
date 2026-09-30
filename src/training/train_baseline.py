"""Entrena el baseline YOLO sobre el dataset maestro, con procedencia.

Antes de entrenar recalcula la huella del dataset y la compara con la que
registra su dataset.yaml (meta.provenance.output_fingerprint). Si no coincide,
no entrena: el modelo saldria de datos distintos a los registrados.

Por cada semilla crea una corrida en <project>/<name>_s<semilla>/ con lo que
guarda Ultralytics (weights/best.pt, results.csv, args.yaml, graficas) y un
provenance.yaml propio:
  - dataset: yaml usado, huella, version de la taxonomia
  - code: commit de git y si habia cambios sin commitear (dirty)
  - pretrained: pesos de partida y su SHA-256
  - seed y config: la semilla y la configuracion tal como se pidio
  - environment: Python, torch, CUDA, Ultralytics, GPU
  - resources: tiempo de entrenamiento y pico de VRAM (para la prueba de humo)

La GPU no es determinista: dos corridas con la misma semilla dan metricas
parecidas pero no identicas. Por eso se entrenan varias semillas y se
reporta la media y la variacion.

Uso (desde la raiz del repositorio):
    python -m src.training.train_baseline --config configs/training/baseline.yaml
    # Prueba de humo: cambiar valores sin tocar el yaml
    python -m src.training.train_baseline --config configs/training/baseline.yaml \\
        --name humo_s960 --model models/pretrained/yolo11s.pt --set epochs=1 imgsz=960
"""

import argparse
import hashlib
import importlib.metadata
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.data.provenance import environment, fingerprint, git_state, verify_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


class DatasetMismatch(Exception):
    """El dataset en disco no es el que registra su yaml."""


def parse_overrides(pairs):
    """["epochs=1", "device=cpu"] -> {"epochs": 1, "device": "cpu"} (tipos via YAML)."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--set espera clave=valor, no {pair!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_config(path, overrides=()):
    """Lee el yaml de entrenamiento, aplica --set y valida los nombres de train."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["train"] = {**cfg.get("train", {}), **parse_overrides(overrides)}
    from ultralytics.cfg import DEFAULT_CFG_DICT

    unknown = sorted(set(cfg["train"]) - set(DEFAULT_CFG_DICT))
    if unknown:
        raise ValueError(f"Ultralytics no conoce estos parametros: {unknown}")
    reserved = sorted(set(cfg["train"]) & {"data", "model", "seed", "project", "name"})
    if reserved:
        raise ValueError(f"{reserved} van fuera de 'train:' (los fija este script)")
    return cfg


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(data_yaml):
    """Devuelve la huella del dataset o lanza DatasetMismatch."""
    ok, expected, actual = verify_dataset(data_yaml)
    if expected is None:
        raise DatasetMismatch(f"{data_yaml} no registra huella (meta.provenance.output_fingerprint)")
    if not ok:
        raise DatasetMismatch(f"{data_yaml}: la huella no coincide\n"
                              f"  registrada: {expected}\n  actual:     {actual}")
    return actual


def training_environment():
    env = environment()
    for pkg in ("torch", "torchvision", "ultralytics"):
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = None
    import torch

    env["cuda"] = torch.version.cuda
    env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return env


def run_ultralytics(model_path, data_yaml, seed, project, name, train_args):
    """Entrena una corrida. Devuelve (carpeta de salida, segundos, pico de VRAM en GB)."""
    import torch
    from ultralytics import YOLO

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(model_path))
    start = time.perf_counter()
    model.train(data=str(data_yaml), seed=seed, project=str(project), name=name,
                deterministic=True, **train_args)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None
    return Path(model.trainer.save_dir), seconds, peak


def train(cfg, runner=run_ultralytics):
    """Verifica el dataset y entrena una corrida por semilla. Devuelve las carpetas."""
    data_yaml = resolve(cfg["data"])
    model_path = resolve(cfg["model"])
    if not model_path.is_file():
        # Un nombre suelto ("yolo11n.pt", "yolo11n.yaml") lo resuelve Ultralytics.
        model_path = Path(cfg["model"])
    project = resolve(cfg["project"])

    print(f"Verificando la huella de {data_yaml} ...")
    dataset_fp = check_dataset(data_yaml)
    print(f"OK: {dataset_fp}")
    meta = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("meta", {})

    git = git_state(REPO_ROOT)
    if git is None or git["dirty"]:
        print("AVISO: hay cambios sin commitear (o no hay git). El modelo quedara marcado como\n"
              "       no reproducible desde un commit (provenance.code.dirty).\n")
    env = training_environment()
    pretrained = {"path": cfg["model"],
                  "sha256": sha256_file(model_path) if model_path.is_file() else None}

    run_dirs = []
    for seed in cfg["seeds"]:
        started = datetime.now().isoformat(timespec="seconds")
        save_dir, seconds, peak = runner(model_path, data_yaml, seed, project,
                                         f"{cfg['name']}_s{seed}", dict(cfg["train"]))
        provenance = {
            "started_at": started,
            "dataset": {"yaml": cfg["data"], "fingerprint": dataset_fp,
                        "taxonomy_version": meta.get("taxonomy_version")},
            "code": git,
            "pretrained": pretrained,
            "seed": seed,
            "config": {k: v for k, v in cfg.items() if k != "seeds"},
            "environment": env,
            "resources": {"train_seconds": round(seconds, 1),
                          "gpu_mem_peak_gb": round(peak, 2) if peak is not None else None},
        }
        (save_dir / "provenance.yaml").write_text(
            yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Semilla {seed}: {save_dir}  ({seconds / 60:.1f} min, VRAM pico {peak} GB)")
        run_dirs.append(save_dir)
    return run_dirs


def main():
    parser = argparse.ArgumentParser(description="Entrena el baseline YOLO con procedencia")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/training/baseline.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR",
                        help="cambia parametros de 'train:' sin editar el yaml")
    parser.add_argument("--seeds", type=int, nargs="+", help="sustituye 'seeds' del yaml")
    parser.add_argument("--name", help="sustituye 'name' del yaml")
    parser.add_argument("--model", help="sustituye 'model' del yaml (p. ej. models/pretrained/yolo11s.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.name:
        cfg["name"] = args.name
    if args.model:
        cfg["model"] = args.model
    try:
        train(cfg)
    except DatasetMismatch as e:
        raise SystemExit(f"NO SE ENTRENA. {e}")


if __name__ == "__main__":
    main()
