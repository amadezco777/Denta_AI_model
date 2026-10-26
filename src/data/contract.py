"""Contrato externo de datos, semantica YOLO y declaraciones de revision.

La firma detecta diferencias respecto a un artefacto aprobado. No autentica
al revisor ni convierte una declaracion de cobertura en evidencia clinica.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from src.data.annotations import read_yolo_boxes
from src.data.preprocessing import boxes_under_mask
from src.utils.config import load_yaml

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REVIEW_FIELDS = ["archivo", "image_sha256", "label_sha256", "label_status", "coverage",
                 "reviewer", "group_id", "group_basis", "group_verified"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value):
    # JSON canonico rechaza valores NaN/inf y tipos que no forman parte del contrato.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _inside(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Ruta fuera de la raiz del dataset: {path}")
    return resolved


def class_names(value):
    if isinstance(value, list):
        names = dict(enumerate(value))
    elif isinstance(value, dict):
        names = {}
        for key, name in value.items():
            if isinstance(key, bool) or not re.fullmatch(r"[0-9]+", str(key)):
                raise ValueError("Los ids de names deben ser enteros no negativos")
            cls = int(key)
            if cls in names:
                raise ValueError("Ids repetidos en names")
            names[cls] = name
    else:
        raise ValueError("names debe ser una lista o un mapa de clases")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Los ids de names deben ser consecutivos desde cero")
    if any(not isinstance(n, str) or not n.strip() for n in names.values()):
        raise ValueError("Nombres de clase no vacios requeridos")
    if len(set(names.values())) != len(names):
        raise ValueError("Nombres de clase repetidos")
    return dict(sorted(names.items()))


def inspect_dataset(yaml_path):
    """Valida directorios YOLO simples, cajas, imagenes y pares imagen/etiqueta."""
    yaml_path = Path(yaml_path).resolve()
    cfg = load_yaml(yaml_path)
    root_value = cfg.get("path")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("path debe ser una raiz absoluta explicita")
    root = Path(root_value).resolve()
    if not root.is_dir():
        raise ValueError(f"Raiz de dataset inexistente: {root}")
    _inside(yaml_path, root)
    if Path(root_value).is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("El dataset no admite enlaces simbolicos")
    if cfg.get("download"):
        raise ValueError("Un contrato local no permite download automatico")
    names = class_names(cfg.get("names"))
    if "nc" in cfg and (type(cfg["nc"]) is not int or cfg["nc"] != len(names)):
        raise ValueError("nc no coincide con names")
    meta = cfg.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("meta debe ser un mapa con el rol del dataset")
    role = meta.get("role")
    if role not in {"development", "external_evaluation"}:
        raise ValueError("meta.role debe declarar development o external_evaluation")
    required = ("train", "val") if role == "development" else ("val", "test")
    if any(not cfg.get(k) for k in required):
        raise ValueError(f"Faltan particiones requeridas: {required}")
    if role == "external_evaluation" and "train" in cfg:
        raise ValueError("La configuracion externa no puede contener train")
    samples, directories, pixels = [], {}, {}
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        rel = cfg[split]
        if (not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts
                or len(Path(rel).parts) != 2 or Path(rel).parts[0] != "images"):
            raise ValueError(f"{split}: se requiere un directorio relativo images/<particion>")
        image_dir = _inside(root / rel, root)
        label_dir = _inside(root / "labels" / Path(rel).name, root)
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Faltan directorios de imagenes o etiquetas en {split}")
        if image_dir in directories:
            if role == "external_evaluation" and {split, directories[image_dir]} == {"val", "test"}:
                continue
            raise ValueError("Dos particiones reutilizan el mismo directorio")
        directories[image_dir] = split
        images = sorted(image_dir.iterdir())
        if not images:
            raise ValueError(f"Particion vacia: {split}")
        stems = set()
        for img in images:
            if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Archivo de imagen no admitido: {img}")
            if img.stem.casefold() in stems:
                raise ValueError(f"Dos imagenes comparten etiqueta: {img.stem}")
            stems.add(img.stem.casefold())
            label = label_dir / f"{img.stem}.txt"
            boxes, _ = read_yolo_boxes(label, names)
            mask = meta.get("mask")
            if mask is not None and boxes_under_mask(boxes, mask):
                raise ValueError(f"Caja bajo la mascara: {label}")
            image = cv2.imdecode(np.frombuffer(img.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Imagen corrupta o no decodificable: {img}")
            pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if pixel_hash in pixels and pixels[pixel_hash] != split:
                raise ValueError("Imagen con pixeles identicos en distintas particiones")
            pixels[pixel_hash] = split
            samples.append({"archivo": img.relative_to(root).as_posix(), "split": split,
                            "image_sha256": sha256_file(img), "label_sha256": sha256_file(label)})
        for label in label_dir.iterdir():
            if not label.is_file() or label.suffix != ".txt" or label.stem.casefold() not in stems:
                raise ValueError(f"Etiqueta huerfana o archivo inesperado: {label}")
    # names se normaliza para que claves JSON numericas no dependan del volcado YAML.
    cfg["names"] = {str(k): v for k, v in names.items()}
    return root, cfg, sorted(samples, key=lambda row: row["archivo"])


def write_review_template(yaml_path, out_path):
    _, cfg, samples = inspect_dataset(yaml_path)
    with Path(out_path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow({**{k: sample[k] for k in ("archivo", "image_sha256", "label_sha256")},
                             "label_status": "pending", "reviewer": "", "group_id": "",
                             "group_basis": "", "group_verified": "false",
                             "coverage": json.dumps({n: "unknown" for n in cfg["names"].values()})})


def _review(path, samples, names):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or set(reader.fieldnames) != set(REVIEW_FIELDS):
            raise ValueError(f"Revision CSV requiere las columnas {REVIEW_FIELDS}")
        rows = list(reader)
    by_file = {}
    groups = {}
    sample_map = {s["archivo"]: s for s in samples}
    for row in rows:
        key = row["archivo"]
        if key in by_file or key not in sample_map or None in row:
            raise ValueError("Revision duplicada, huerfana o fila CSV malformada")
        if row["label_status"] != "approved" or not row["reviewer"].strip():
            raise ValueError(f"{key}: falta revision aprobada y revisor")
        for kind in ("image_sha256", "label_sha256"):
            if row[kind] != sample_map[key][kind]:
                raise ValueError(f"{key}: la revision corresponde a otros datos ({kind})")
        try:
            coverage = json.loads(row["coverage"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{key}: coverage requiere un objeto JSON") from exc
        if (not isinstance(coverage, dict) or set(coverage) != set(names.values())
                or any(v != "complete" for v in coverage.values())):
            raise ValueError(f"{key}: cobertura completa de todas las clases requerida")
        if (row["group_verified"] != "true" or not row["group_id"].strip()
                or row["group_basis"] not in {"patient_metadata", "verified_acquisition", "image_only"}):
            raise ValueError(f"{key}: grupo confirmado y fundamento requeridos")
        groups.setdefault(row["group_id"], []).append((sample_map[key]["split"], row["group_basis"]))
        row["coverage"] = coverage
        by_file[key] = row
    if set(by_file) != set(sample_map):
        raise ValueError("La revision no cubre todas las imagenes del dataset")
    for group, entries in groups.items():
        if len({split for split, _ in entries}) > 1:
            raise ValueError(f"Grupo confirmado compartido entre particiones: {group}")
        if len(entries) > 1 and any(basis == "image_only" for _, basis in entries):
            raise ValueError("image_only solo identifica una imagen, no varias visitas")
    return [by_file[key] for key in sorted(by_file)]


def _inventory(root):
    # Solo las caches directas de etiquetas de Ultralytics son regenerables.
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            if len(rel.parts) == 2 and rel.parts[0] == "labels" and p.suffix == ".cache":
                continue
            files[rel.as_posix()] = sha256_file(p)
    return files


def _snapshot(yaml_path, review_path):
    root, cfg, samples = inspect_dataset(yaml_path)
    reviews = _review(review_path, samples, cfg["names"])
    _check_similarity_groups(root, reviews)
    return {"schema_version": SCHEMA_VERSION, "root": str(root),
            "yaml": str(Path(yaml_path).resolve()), "config": cfg, "files": _inventory(root),
            "samples": samples, "review_path": str(Path(review_path).resolve()),
            "review_sha256": sha256_file(review_path), "reviews": reviews}


def _check_similarity_groups(root, reviews):
    """Los grupos propuestos por build_master requieren confirmacion conjunta."""
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "grupo_similitud" not in reader.fieldnames:
            raise ValueError("Manifiesto historico sin grupo_similitud: regenerar o adaptar con revision")
        rows = list(reader)
    by_file = {row["archivo"]: row for row in reviews}
    groups = {}
    seen = set()
    for row in rows:
        key = f"images/{row['particion']}/{row['archivo']}"
        if key in by_file:
            if key in seen or not row["grupo_similitud"]:
                raise ValueError("Manifiesto con imagen repetida o grupo de similitud vacio")
            seen.add(key)
            groups.setdefault(row["grupo_similitud"], []).append(by_file[key])
    if seen != set(by_file):
        raise ValueError("El manifiesto no cubre todas las imagenes de la vista")
    for entries in groups.values():
        if len(entries) > 1 and (len({e["group_id"] for e in entries}) != 1
                                or any(e["group_basis"] == "image_only" for e in entries)):
            raise ValueError("Grupo de similitud no confirmado conjuntamente en la revision")


def create_contract(yaml_path, lock_path, review_path):
    root, _, _ = inspect_dataset(yaml_path)
    lock_path = Path(lock_path).resolve()
    if lock_path.is_relative_to(root):
        raise ValueError("El contrato debe guardarse fuera del dataset")
    snapshot = _snapshot(yaml_path, review_path)
    digest = hashlib.sha256(_canonical(snapshot)).hexdigest()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as f:
        json.dump({"digest": digest, "snapshot": snapshot}, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return digest


def verify_contract(yaml_path, lock_path, *, for_training=False):
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or set(lock) != {"digest", "snapshot"}:
        raise ValueError("Contrato invalido")
    expected = lock["digest"]
    recorded = lock["snapshot"]
    if recorded.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Version de contrato no admitida")
    if hashlib.sha256(_canonical(recorded)).hexdigest() != expected:
        raise ValueError("Contrato alterado: la firma interna no coincide")
    root = Path(recorded["root"])
    if lock_path.is_relative_to(root):
        raise ValueError("El contrato debe estar fuera del dataset")
    if Path(yaml_path).resolve() != Path(recorded["yaml"]):
        raise ValueError("El contrato pertenece a otro YAML")
    current = _snapshot(yaml_path, recorded["review_path"])
    if for_training and current["config"]["meta"]["role"] != "development":
        raise ValueError("La prueba externa esta reservada para evaluacion; no se entrena con ella")
    actual = hashlib.sha256(_canonical(current)).hexdigest()
    return expected == actual, expected, actual
