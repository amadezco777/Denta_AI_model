"""Transporte de un dataset aprobado sin confundir traslado con nueva revision.

El paquete preserva bytes, revisiones y contrato original. Al restaurar solo
cambia las raices de los YAML y crea otro contrato con la misma revision.
"""

import copy
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from src.data.contract import _canonical, create_contract, sha256_file, verify_contract
from src.training.train_research import resolve, validate_config
from src.utils.config import load_yaml


def package_training(config_path, archive_path):
    cfg = validate_config(load_yaml(config_path))
    data_yaml, lock_path = resolve(cfg["data"]), resolve(cfg["dataset_lock"])
    ok, _, _ = verify_contract(data_yaml, lock_path, for_training=True)
    if not ok:
        raise ValueError("Datos distintos del contrato; no se empaquetan")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    snapshot = lock["snapshot"]
    root = Path(snapshot["root"])
    model = resolve(cfg["model"])
    if not model.is_file() and (Path(cfg["model"]).is_absolute() or len(Path(cfg["model"]).parts) > 1):
        raise ValueError("No existe el checkpoint local solicitado")
    archive_path = Path(archive_path).resolve()
    if archive_path.exists() or archive_path.is_relative_to(root):
        raise ValueError("Archivo de salida nuevo y fuera del dataset requerido")
    files = {f"dataset/{rel}": root / rel for rel in snapshot["files"]}
    files["review.csv"] = Path(snapshot["review_path"])
    if model.is_file():
        files[f"model/{model.name}"] = model
        cfg["model"] = f"model/{model.name}"
    descriptor = {"schema_version": 1, "original_lock": lock, "training_config": cfg,
                  "dataset_yaml": data_yaml.relative_to(root).as_posix(),
                  "files": {name: sha256_file(p) for name, p in files.items()}}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".training-package-", suffix=".zip", dir=archive_path.parent, delete=False) as f:
        temporary = Path(f.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("package.json", json.dumps(descriptor, ensure_ascii=False, indent=2, allow_nan=False))
            for name, path in files.items():
                z.write(path, name)
        # Revalidar antes de publicar: el dataset no puede cambiar mientras se empaqueta.
        if not verify_contract(data_yaml, lock_path, for_training=True)[0]:
            raise ValueError("El dataset cambio durante el empaquetado")
        # rename en Windows falla si otro proceso ya creo el destino.
        os.rename(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(archive_path)


def restore_training(archive_path, out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Destino debe ser nuevo; no se sobrescribe un entrenamiento")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as z:
        members = z.infolist()
        member_names = [m.filename for m in members]
        if len(member_names) != len(set(member_names)):
            raise ValueError("Miembros ZIP repetidos")
        for m in members:
            path = PurePosixPath(m.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in m.filename or ":" in m.filename
                    or (m.external_attr >> 16) & 0o170000 == 0o120000 or m.is_dir()):
                raise ValueError("Ruta ZIP no permitida")
        descriptor = json.loads(z.read("package.json"))
        if descriptor.get("schema_version") != 1:
            raise ValueError("Version de paquete no admitida")
        expected = descriptor["files"]
        if set(member_names) != set(expected) | {"package.json"}:
            raise ValueError("Inventario ZIP distinto del declarado")
        original = descriptor["original_lock"]
        snapshot = original["snapshot"]
        if hashlib.sha256(_canonical(snapshot)).hexdigest() != original["digest"]:
            raise ValueError("Contrato original alterado")
        if {k[8:]: v for k, v in expected.items() if k.startswith("dataset/")} != snapshot["files"]:
            raise ValueError("Archivos del paquete distintos del contrato original")
        if expected.get("review.csv") != snapshot["review_sha256"]:
            raise ValueError("Revision distinta de la aprobada")
        dataset_yaml = PurePosixPath(descriptor["dataset_yaml"])
        if dataset_yaml.is_absolute() or ".." in dataset_yaml.parts or "\\" in str(dataset_yaml):
            raise ValueError("dataset_yaml fuera de la raiz")
        with tempfile.TemporaryDirectory(prefix=".training-restore-", dir=out.parent) as temporary:
            stage = Path(temporary) / "restored"
            stage.mkdir()
            for name, digest in expected.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, path.open("xb") as target:
                    while chunk := source.read(1 << 20):
                        target.write(chunk)
                if sha256_file(path) != digest:
                    raise ValueError(f"Hash no coincide: {name}")
            # Todas las vistas conservan sus claves; solo se sustituye path.
            for path in (stage / "dataset").rglob("*.yaml"):
                view = load_yaml(path)
                if "path" in view:
                    if view["path"] != snapshot["root"]:
                        raise ValueError("Una vista apunta a otra raiz; no se traslada implicitamente")
                    view["path"] = str(out / "dataset")
                    path.write_text(yaml.safe_dump(view, sort_keys=False, allow_unicode=True), encoding="utf-8")
            write_path = stage / "original.lock.json"
            write_path.write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")
            cfg = copy.deepcopy(descriptor["training_config"])
            cfg.update(data=str(out / "dataset" / str(dataset_yaml)), dataset_lock=str(out / "dataset.lock.json"),
                       project=str(out / "runs"))
            if cfg["model"].startswith("model/"):
                cfg["model"] = str(out / cfg["model"])
            validate_config(cfg)
            (stage / "training.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            os.rename(stage, out)
    # El contrato depende de rutas finales; el registro distingue restauracion incompleta.
    state = out / "restore.json"
    state.write_text(json.dumps({"status": "creating_contract", "source_contract": original["digest"]}), encoding="utf-8")
    new_digest = create_contract(out / "dataset" / str(dataset_yaml), out / "dataset.lock.json", out / "review.csv")
    state.write_text(json.dumps({"status": "completed", "source_contract": original["digest"],
                                 "new_contract": new_digest, "archive_sha256": sha256_file(archive_path)}, indent=2), encoding="utf-8")
    return out / "training.yaml"
