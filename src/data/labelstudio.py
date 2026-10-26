"""Paquetes locales Label Studio y conversion estricta de revisiones a YOLO detect.

Las predicciones son sugerencias. Solo una anotacion humana no cancelada,
aprobada y con cobertura explicita puede generar etiquetas de entrenamiento.
"""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import yaml
from PIL import Image

from src.data.annotations import read_yolo_boxes
from src.data.contract import IMAGE_SUFFIXES, REVIEW_FIELDS, class_names, inspect_dataset, sha256_file
from src.data.similarity import SIMILARITY_THRESHOLD, connected_groups, similar_pairs, thumbnail
from src.data.splitting import three_way_split
from src.utils.config import load_yaml

SCHEMA_VERSION = 1
COLORS = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c", "#f1c40f", "#2ecc71"]
GROUP_FIELDS = ["sample_id", "group_id", "group_basis", "group_verified"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def labeling_config(names):
    names = class_names(names)
    view = ET.Element("View")
    ET.SubElement(view, "Header", value="Hallazgos dentales: revisar toda la imagen y todas las clases")
    ET.SubElement(view, "Image", name="image", value="$image", zoom="true", zoomControl="true", rotateControl="false")
    labels = ET.SubElement(view, "RectangleLabels", name="findings", toName="image", canRotate="false")
    for cls, name in names.items():
        ET.SubElement(labels, "Label", value=name, background=COLORS[cls % len(COLORS)], hotkey=str(cls + 1))
    ET.SubElement(view, "Header", value="Cobertura: complete significa que revise presencia y ausencia de esta clase")
    for cls, name in names.items():
        ET.SubElement(view, "Header", value=name)
        choices = ET.SubElement(view, "Choices", name=f"coverage_{cls}", toName="image", choice="single-radio", required="true")
        ET.SubElement(choices, "Choice", value="complete")
        ET.SubElement(choices, "Choice", value="uncertain")
    ET.SubElement(view, "Header", value="Estado de revision; approved solo tras terminar la revision")
    status = ET.SubElement(view, "Choices", name="review_status", toName="image", choice="single-radio", required="true")
    for value in ("approved", "needs_review", "unusable"):
        ET.SubElement(status, "Choice", value=value)
    ET.SubElement(view, "TextArea", name="reviewer", toName="image", required="true", maxSubmissions="1",
                  placeholder="Identificador del revisor odontologico")
    ET.SubElement(view, "TextArea", name="notes", toName="image", maxSubmissions="1", placeholder="Dudas o adjudicacion")
    ET.indent(view)
    return ET.tostring(view, encoding="unicode") + "\n"


def _decode(path):
    # Orientacion EXIF no resuelta puede desalinear cajas al mostrarlas en un navegador.
    with Image.open(path) as im:
        if im.getexif().get(274, 1) != 1:
            raise ValueError(f"Orientacion EXIF pendiente de normalizar: {path}")
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError(f"Imagen multiframe no admitida: {path}")
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError(f"Se requiere PNG/JPEG decodificable de 8 bits: {path}")
    if image.ndim == 3 and image.shape[2] not in (3,):
        raise ValueError(f"Imagen con canal alfa no admitida: {path}")
    return image


def prepare_bundle(out, taxonomy_path, *, images=None, dataset=None, document_root=None):
    """Publica un paquete completo sin modificar origen ni sobrescribir otra salida."""
    if (images is None) == (dataset is None):
        raise ValueError("Elige exactamente images o dataset")
    out = Path(out).resolve()
    document_root = Path(document_root).resolve() if document_root else out.parent
    if out.exists() or not out.is_relative_to(document_root):
        raise ValueError("Salida debe ser nueva y estar dentro del document root")
    spec = load_yaml(taxonomy_path)
    names = class_names(spec["names"])
    source_meta = {}
    if dataset:
        root, cfg, samples = inspect_dataset(dataset)
        if class_names(cfg["names"]) != names or cfg["meta"].get("taxonomy_version") != spec["version"]:
            raise ValueError("El dataset no coincide con la taxonomia seleccionada")
        source_meta = cfg["meta"]
        inputs = [(root / s["archivo"], s["archivo"], s["split"]) for s in samples]
    else:
        root = Path(images).resolve()
        if not root.is_dir():
            raise ValueError(f"No existe la carpeta de imagenes: {root}")
        inputs = [(p, p.relative_to(root).as_posix(), "pool") for p in sorted(root.rglob("*"))
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("No se admiten enlaces simbolicos de entrada")
    if not inputs:
        raise ValueError("No hay radiografias PNG/JPEG para importar")
    if out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Entrada y paquete de salida no pueden contenerse")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ls-prepare-", dir=out.parent) as temporary:
        stage = Path(temporary) / "bundle"
        (stage / "images").mkdir(parents=True)
        rows, tasks, pixel_hashes = [], [], set()
        for path, source, split in inputs:
            image = _decode(path)
            h, w = image.shape[:2]
            pixels = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if pixels in pixel_hashes and not dataset:
                raise ValueError("Radiografias con pixeles identicos: deduplicar antes de etiquetar")
            pixel_hashes.add(pixels)
            sample_id = hashlib.sha256(source.encode()).hexdigest()[:24]
            relative = f"images/{sample_id}.png"
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError(f"No se pudo codificar: {path}")
            (stage / relative).write_bytes(encoded.tobytes())
            uri = "/data/local-files/?" + urlencode({"d": (out / relative).relative_to(document_root).as_posix()})
            row = {"sample_id": sample_id, "image": relative, "uri": uri, "source": source, "split": split,
                   "width": w, "height": h, "image_sha256": sha256_file(stage / relative),
                   "source_image_sha256": sha256_file(path)}
            task = {"data": {"image": uri, "sample_id": sample_id}, "meta": {"taxonomy_version": spec["version"]}}
            if dataset:
                label = root / "labels" / Path(source).parent.name / f"{path.stem}.txt"
                boxes, _ = read_yolo_boxes(label, names)
                row["source_label_sha256"] = sha256_file(label)
                task["predictions"] = [{"model_version": "source-labels-unreviewed", "result": [
                    {"id": f"box-{i}", "type": "rectanglelabels", "from_name": "findings", "to_name": "image",
                     "original_width": w, "original_height": h, "image_rotation": 0,
                     "value": {"x": 100 * (x - bw / 2), "y": 100 * (y - bh / 2),
                               "width": 100 * bw, "height": 100 * bh, "rotation": 0,
                               "rectanglelabels": [names[c]]}}
                    for i, (c, x, y, bw, bh) in enumerate(boxes)]}]
            rows.append(row)
            tasks.append(task)
        manifest = {"schema_version": SCHEMA_VERSION, "taxonomy_version": spec["version"],
                    "names": {str(k): v for k, v in names.items()}, "source_meta": source_meta, "samples": rows}
        write_json(stage / "manifest.json", manifest)
        write_json(stage / "tasks.json", tasks)
        (stage / "label_config.xml").write_text(labeling_config(names), encoding="utf-8")
        with (stage / "groups.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GROUP_FIELDS)
            writer.writeheader()
            writer.writerows({"sample_id": row["sample_id"], "group_verified": "false"} for row in rows)
        os.rename(stage, out)
    return len(rows)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Geometria numerica finita requerida")
    return value


def parse_annotation(task, sample, names, annotation_id=None):
    annotations = task.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("annotations debe ser una lista")
    active = [a for a in annotations if isinstance(a, dict) and not a.get("was_cancelled", False)]
    if annotation_id is not None:
        active = [a for a in active if str(a.get("id")) == str(annotation_id)]
    if len(active) != 1:
        raise ValueError(f"{sample['sample_id']}: requiere una anotacion activa o seleccion explicita para adjudicar")
    annotation = active[0]
    results = annotation.get("result")
    if not isinstance(results, list):
        raise ValueError("La anotacion humana debe tener una lista result")
    boxes, controls, region_ids = [], {}, set()
    by_name = {n: c for c, n in names.items()}
    for result in results:
        if not isinstance(result, dict) or result.get("to_name") != "image":
            raise ValueError("Resultado sin destino image")
        kind, control, value = result.get("type"), result.get("from_name"), result.get("value")
        if not isinstance(value, dict):
            raise ValueError("value debe ser un objeto")
        if kind == "rectanglelabels" and control == "findings":
            region_id = result.get("id")
            if not isinstance(region_id, str) or not region_id or region_id in region_ids:
                raise ValueError("Region sin id unico")
            region_ids.add(region_id)
            if (result.get("original_width") != sample["width"] or result.get("original_height") != sample["height"]
                    or _number(result.get("image_rotation", 0)) != 0 or _number(value.get("rotation", 0)) != 0):
                raise ValueError("Dimensiones distintas o caja/imagen rotada: no se convierte implicitamente")
            labels = value.get("rectanglelabels")
            if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in by_name:
                raise ValueError("Una clase conocida por caja requerida")
            x, y, w, h = (_number(value.get(k)) for k in ("x", "y", "width", "height"))
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > 100 + 1e-6 or y + h > 100 + 1e-6:
                raise ValueError("Caja fuera de imagen o degenerada")
            boxes.append((by_name[labels[0]], (x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100))
        elif ((kind == "choices" and (control == "review_status" or control in {f"coverage_{c}" for c in names}))
              or (kind == "textarea" and control in {"reviewer", "notes"})):
            if control in controls:
                raise ValueError("Control de revision repetido")
            values = value.get("choices" if kind == "choices" else "text")
            if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
                raise ValueError("Una respuesta textual por control requerida")
            controls[control] = values[0].strip()
        else:
            raise ValueError(f"Resultado no soportado: {kind}/{control}; solo YOLO detect")
    if controls.get("review_status") != "approved" or not controls.get("reviewer"):
        raise ValueError("Falta aprobacion explicita o revisor")
    coverage = {name: controls.get(f"coverage_{c}", "unknown") for c, name in names.items()}
    if any(v != "complete" for v in coverage.values()):
        raise ValueError("Falta cobertura completa de todas las clases; no convertir omisiones en negativos")
    if len(boxes) != len(set(boxes)):
        raise ValueError("Cajas exactamente duplicadas; adjudicar antes de exportar")
    return boxes, {"reviewer": controls["reviewer"], "coverage": coverage,
                   "annotation_id": annotation.get("id"), "completed_by": annotation.get("completed_by"),
                   "notes": controls.get("notes", "")}


def _read_groups(path, samples):
    groups = {s["sample_id"]: {"group_id": "", "group_basis": "", "group_verified": "false"} for s in samples}
    if path is None:
        return groups
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != GROUP_FIELDS:
            raise ValueError(f"groups CSV requiere {GROUP_FIELDS}")
        rows = list(reader)
    if len(rows) != len(groups) or {r.get("sample_id") for r in rows} != set(groups):
        raise ValueError("groups CSV debe cubrir cada sample_id exactamente una vez")
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Fila groups CSV malformada")
        if row["group_verified"] not in {"true", "false"}:
            raise ValueError("group_verified requiere true/false")
        if row["group_verified"] == "true" and (not row["group_id"].strip()
                or row["group_basis"] not in {"patient_metadata", "verified_acquisition", "image_only"}):
            raise ValueError("Un grupo confirmado requiere id y fundamento")
        groups[row["sample_id"]] = {k: row[k] for k in GROUP_FIELDS[1:]}
    return groups


def export_bundle(bundle, export_path, out, *, groups_path=None, annotation_ids=None,
                  val_fraction=0.15, test_fraction=0.15, seed=42):
    """Genera dataset + review.csv; los grupos pendientes siguen bloqueando el contrato."""
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(bundle) or bundle.is_relative_to(out):
        raise ValueError("Salida nueva y separada del paquete requerida")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Version de paquete no admitida")
    if bundle.is_symlink() or any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("No se admiten enlaces simbolicos en el paquete")
    names = class_names(manifest["names"])
    samples = manifest["samples"]
    if not isinstance(samples, list) or not samples or any(not isinstance(s, dict) for s in samples):
        raise ValueError("Manifiesto requiere una lista de muestras")
    for sample in samples:
        key = sample.get("sample_id")
        if (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{24}", key)
                or sample.get("image") != f"images/{key}.png"
                or any(type(sample.get(k)) is not int or sample[k] <= 0 for k in ("width", "height"))):
            raise ValueError("Identidad, ruta o dimensiones no validas en el manifiesto")
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Manifiesto vacio o con sample_id repetido")
    for sample in samples:
        image_path = (bundle / sample["image"]).resolve()
        if not image_path.is_relative_to(bundle) or image_path.is_symlink():
            raise ValueError("Imagen fuera del paquete")
        if sha256_file(image_path) != sample["image_sha256"]:
            raise ValueError("Imagen de anotacion modificada")
        image = _decode(image_path)
        if image.shape[:2] != (sample["height"], sample["width"]):
            raise ValueError("Dimensiones del manifiesto no coinciden")
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8-sig"))
    if not isinstance(tasks, list):
        raise ValueError("Usa exportacion JSON completa de Label Studio")
    by_id = {}
    for task in tasks:
        key = task.get("data", {}).get("sample_id") if isinstance(task, dict) else None
        if key in by_id or key not in {s["sample_id"] for s in samples}:
            raise ValueError("Tarea duplicada o no pertenece al paquete")
        by_id[key] = task
    if set(by_id) != {s["sample_id"] for s in samples}:
        raise ValueError("Faltan tareas del paquete; exporta todo el lote")
    selections = annotation_ids or {}
    if not isinstance(selections, dict) or set(selections) - set(by_id):
        raise ValueError("Seleccion de anotaciones contiene tareas desconocidas")
    groups_review = _read_groups(groups_path, samples)
    for s in samples:
        task = by_id[s["sample_id"]]
        if task["data"].get("image") != s["uri"]:
            raise ValueError("La tarea apunta a otra imagen; importar tasks.json sin editar")
        s["boxes"], s["review"] = parse_annotation(task, s, names, selections.get(s["sample_id"]))
    pairs = similar_pairs([thumbnail(bundle / s["image"]) for s in samples], SIMILARITY_THRESHOLD)
    # Metadatos confirmados y similitud visual son restricciones de particionado.
    by_group = {}
    for i, s in enumerate(samples):
        g = groups_review[s["sample_id"]]
        if g["group_id"]:
            if g["group_id"] in by_group:
                pairs.append((by_group[g["group_id"]], i))
            by_group[g["group_id"]] = i
    components = connected_groups(len(samples), pairs)
    pool = {s["split"] for s in samples} == {"pool"}
    if pool:
        summary = [(len(g), Counter(b[0] for i in g for b in samples[i]["boxes"])) for g in components]
        assignments = three_way_split(summary, len(names), val_fraction, test_fraction, seed)
    else:
        if any(s["split"] not in {"train", "val", "test"} for s in samples):
            raise ValueError("No mezclar pool ni holdout externo con dataset de desarrollo")
        assignments = []
        for g in components:
            splits = {samples[i]["split"] for i in g}
            if len(splits) != 1:
                raise ValueError("Grupo de similitud/adquisicion cruza particiones; resolver antes de exportar")
            assignments.append(splits.pop())
    for gi, (component, split) in enumerate(zip(components, assignments)):
        for i in component:
            samples[i].update(split=split, similarity_group=str(gi))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ls-export-", dir=out.parent) as temporary:
        stage = Path(temporary) / "output"
        dataset_root = stage / "dataset"
        review_rows, manifest_rows, audit_rows = [], [], []
        for s in samples:
            stem, split = s["sample_id"], s["split"]
            rel = f"images/{split}/{stem}.png"
            image, label = dataset_root / rel, dataset_root / "labels" / split / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / s["image"], image)
            label.write_text("".join(f"{c} {x!r} {y!r} {w!r} {h!r}\n" for c, x, y, w, h in s["boxes"]), encoding="utf-8")
            read_yolo_boxes(label, names)
            group = groups_review[stem]
            review_rows.append({"archivo": rel, "image_sha256": sha256_file(image), "label_sha256": sha256_file(label),
                                "label_status": "approved", "coverage": json.dumps(s["review"]["coverage"]),
                                "reviewer": s["review"]["reviewer"], **group})
            manifest_rows.append({"archivo": image.name, "particion": split, "grupo_similitud": s["similarity_group"],
                                  "origen": s["source"], "source_image_sha256": s["source_image_sha256"], "cajas": len(s["boxes"])})
            audit_rows.append({"sample_id": stem, **s["review"]})
        source_meta = manifest.get("source_meta", {})
        role = source_meta.get("role", "development")
        config = {"path": str(out / "dataset"), "names": names,
                  "meta": {"role": role, "taxonomy_version": manifest["taxonomy_version"],
                           "reference_status": "requires_group_verification_and_contract",
                           "annotation_source": "label_studio", "labelstudio_export_sha256": sha256_file(export_path),
                           "labelstudio_manifest_sha256": sha256_file(bundle / "manifest.json")},
                  **{split: f"images/{split}" for split in ("train", "val", "test") if any(s["split"] == split for s in samples)}}
        if role == "external_evaluation":
            if set(s["split"] for s in samples) != {"val"}:
                raise ValueError("La vista externa debe conservar su unico holdout")
            config["test"] = config["val"]
        for key in ("mask", "preprocessing_version"):
            if key in source_meta:
                config["meta"][key] = source_meta[key]
        (dataset_root / "dataset.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        for path, rows, fields in ((stage / "review.csv", review_rows, REVIEW_FIELDS),
                                   (dataset_root / "manifest.csv", manifest_rows, list(manifest_rows[0]))):
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        write_json(stage / "annotation_audit.json", audit_rows)
        counts = {split: Counter(b[0] for s in samples if s["split"] == split for b in s["boxes"])
                  for split in sorted({s["split"] for s in samples})}
        write_json(stage / "export_report.json", {
            "images": dict(Counter(s["split"] for s in samples)),
            "boxes": {split: {name: counts[split][c] for c, name in names.items()} for split in counts},
            "classes_without_examples": {split: [name for c, name in names.items() if not counts[split][c]] for split in counts},
            "groups_pending": sum(g["group_verified"] != "true" for g in groups_review.values()),
            "role": role,
        })
        # Validacion semantica antes de publicar, con raiz temporal solo en memoria del YAML.
        config["path"] = str(dataset_root.resolve())
        check_yaml = dataset_root / "dataset.yaml"
        final_yaml = check_yaml.read_text(encoding="utf-8")
        check_yaml.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        inspect_dataset(check_yaml)
        check_yaml.write_text(final_yaml, encoding="utf-8")
        os.rename(stage, out)
    return len(samples)
