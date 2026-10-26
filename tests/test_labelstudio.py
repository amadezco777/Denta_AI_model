"""Regresiones de conversion Label Studio y barreras contra falsos negativos."""

import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.data.contract import create_contract, verify_contract
from src.data.labelstudio import export_bundle, labeling_config, parse_annotation, prepare_bundle
from tests.helpers import synthetic_dataset

NAMES = {0: "caries", 1: "implante"}


def make_bundle(tmp_path, *, fresh=False):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(yaml.safe_dump({"names": NAMES, "version": "0.2.0"}), encoding="utf-8")
    bundle = tmp_path / "media" / "batch"
    if fresh:
        images = tmp_path / "images"
        images.mkdir()
        for i in range(12):
            cv2.imwrite(str(images / f"{i}.png"), np.random.default_rng(i).integers(0, 255, (64, 128), dtype=np.uint8))
        prepare_bundle(bundle, taxonomy, images=images)
    else:
        data, _, _ = synthetic_dataset(tmp_path / "master")
        # El fixture historico lleva el mismo rectangulo en todos los splits;
        # aqui necesitamos adquisiciones visualmente independientes.
        for i, path in enumerate(sorted((data.parent / "images").rglob("*.png"))):
            cv2.imwrite(str(path), np.random.default_rng(i).integers(0, 255, (64, 128, 3), dtype=np.uint8))
        prepare_bundle(bundle, taxonomy, dataset=data)
    return bundle


def annotation(sample, boxes=True):
    result = []
    if boxes:
        result.append({"id": "a", "type": "rectanglelabels", "from_name": "findings", "to_name": "image",
                       "original_width": sample["width"], "original_height": sample["height"], "image_rotation": 0,
                       "value": {"x": 20, "y": 30, "width": 40, "height": 20, "rotation": 0,
                                 "rectanglelabels": ["caries"]}})
    for name, value in [("review_status", "approved"), ("coverage_0", "complete"), ("coverage_1", "complete")]:
        result.append({"type": "choices", "from_name": name, "to_name": "image", "value": {"choices": [value]}})
    result.append({"type": "textarea", "from_name": "reviewer", "to_name": "image", "value": {"text": ["synthetic-only"]}})
    return {"id": 10, "was_cancelled": False, "completed_by": 1, "result": result}


def reviewed_export(bundle, *, boxes=True):
    tasks = json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))
    samples = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["samples"]
    for task, sample in zip(tasks, samples):
        task["annotations"] = [annotation(sample, boxes)]
    path = bundle.parent / "export.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def verified_groups(bundle):
    path = bundle / "groups.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(group_id=row["sample_id"], group_basis="image_only", group_verified="true")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_preparacion_no_aprueba_sugerencias(tmp_path):
    bundle = make_bundle(tmp_path)
    tasks = json.loads((bundle / "tasks.json").read_text())
    assert all("annotations" not in t for t in tasks)
    assert all(t["predictions"][0]["model_version"] == "source-labels-unreviewed" for t in tasks)
    assert all("/data/local-files/?d=batch%2Fimages%2F" in t["data"]["image"] for t in tasks)
    assert labeling_config(NAMES).count('name="coverage_') == 2


def test_roundtrip_contrato_y_geometria(tmp_path):
    bundle = make_bundle(tmp_path)
    out = tmp_path / "reviewed"
    export_bundle(bundle, reviewed_export(bundle), out, groups_path=verified_groups(bundle))
    labels = list((out / "dataset/labels/train").glob("*.txt"))
    assert labels[0].read_text().split() == ["0", "0.4", "0.4", "0.4", "0.2"]
    lock = tmp_path / "reviewed.lock.json"
    create_contract(out / "dataset/dataset.yaml", lock, out / "review.csv")
    assert verify_contract(out / "dataset/dataset.yaml", lock)[0]


def test_negativo_revisado_y_grupos_pendientes(tmp_path):
    bundle = make_bundle(tmp_path)
    out = tmp_path / "reviewed"
    export_bundle(bundle, reviewed_export(bundle, boxes=False), out)
    assert all(p.read_text() == "" for p in (out / "dataset/labels").rglob("*.txt"))
    with pytest.raises(ValueError, match="grupo"):
        create_contract(out / "dataset/dataset.yaml", tmp_path / "lock.json", out / "review.csv")


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(was_cancelled=True),
    lambda a: a["result"].pop(),
    lambda a: a["result"][1]["value"].update(choices=["needs_review"]),
    lambda a: a["result"][2]["value"].update(choices=["uncertain"]),
    lambda a: a["result"][0].update(original_width=999),
    lambda a: a["result"][0].update(image_rotation=90),
    lambda a: a["result"][0]["value"].update(rotation=2),
    lambda a: a["result"][0]["value"].update(x=-1),
    lambda a: a["result"][0]["value"].update(width=0),
    lambda a: a["result"][0]["value"].update(width=float("nan")),
    lambda a: a["result"][0]["value"].update(rectanglelabels=["unknown"]),
    lambda a: a["result"].append(copy.deepcopy(a["result"][0])),
    lambda a: a["result"][0].update(type="polygonlabels"),
])
def test_anotacion_invalida_se_bloquea(mutate):
    sample = {"sample_id": "s", "width": 128, "height": 64}
    ann = annotation(sample)
    mutate(ann)
    with pytest.raises(ValueError):
        parse_annotation({"annotations": [ann]}, sample, NAMES)


def test_predicciones_no_sustituyen_anotacion():
    sample = {"sample_id": "s", "width": 128, "height": 64}
    with pytest.raises(ValueError):
        parse_annotation({"predictions": [annotation(sample)]}, sample, NAMES)


def test_multiples_anotaciones_exigen_adjudicacion():
    sample = {"sample_id": "s", "width": 128, "height": 64}
    a, b = annotation(sample), annotation(sample)
    b["id"] = 11
    task = {"annotations": [a, b]}
    with pytest.raises(ValueError):
        parse_annotation(task, sample, NAMES)
    assert parse_annotation(task, sample, NAMES, 11)[1]["annotation_id"] == 11


@pytest.mark.parametrize("change", ["image", "missing_task", "uri", "duplicate_task"])
def test_exportacion_alterada_no_deja_salida(tmp_path, change):
    bundle = make_bundle(tmp_path)
    path = reviewed_export(bundle)
    tasks = json.loads(path.read_text())
    if change == "image":
        next((bundle / "images").iterdir()).write_bytes(b"corrupt")
    elif change == "missing_task":
        tasks.pop()
    elif change == "uri":
        tasks[0]["data"]["image"] = "another.png"
    else:
        tasks.append(tasks[0])
    path.write_text(json.dumps(tasks))
    out = tmp_path / "failed"
    with pytest.raises(ValueError):
        export_bundle(bundle, path, out)
    assert not out.exists()


def test_imagenes_nuevas_particionan_y_pueden_bloquearse(tmp_path):
    bundle = make_bundle(tmp_path, fresh=True)
    out = tmp_path / "reviewed"
    export_bundle(bundle, reviewed_export(bundle), out, groups_path=verified_groups(bundle))
    cfg = yaml.safe_load((out / "dataset/dataset.yaml").read_text())
    assert all(list((out / "dataset" / cfg[s]).iterdir()) for s in ("train", "val", "test"))
    create_contract(out / "dataset/dataset.yaml", tmp_path / "lock.json", out / "review.csv")


def test_dos_imagenes_nuevas_no_inventan_tres_splits(tmp_path):
    bundle = make_bundle(tmp_path, fresh=True)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["samples"] = manifest["samples"][:2]
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "tasks.json").write_text(json.dumps(json.loads((bundle / "tasks.json").read_text())[:2]))
    with pytest.raises(ValueError, match="tres grupos"):
        export_bundle(bundle, reviewed_export(bundle), tmp_path / "out")


def test_grupo_confirmado_no_cruza_holdout(tmp_path):
    bundle = make_bundle(tmp_path)
    groups = verified_groups(bundle)
    rows = list(csv.DictReader(groups.open()))
    for row in rows:
        row.update(group_id="same-patient", group_basis="patient_metadata")
    with groups.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="cruza"):
        export_bundle(bundle, reviewed_export(bundle), tmp_path / "out", groups_path=groups)


def test_manifest_path_traversal_is_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    path = reviewed_export(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["samples"][0]["sample_id"] = "../../escape"
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Identidad"):
        export_bundle(bundle, path, tmp_path / "out")


def test_external_view_stays_reserved_for_evaluation(tmp_path):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(yaml.safe_dump({"names": NAMES, "version": "0.2.0"}))
    data, _, _ = synthetic_dataset(tmp_path / "external", splits=("holdout",), external=True)
    for i, path in enumerate(sorted((data.parent / "images").rglob("*.png"))):
        cv2.imwrite(str(path), np.random.default_rng(i).integers(0, 255, (64, 128, 3), dtype=np.uint8))
    bundle = tmp_path / "media/batch"
    prepare_bundle(bundle, taxonomy, dataset=data)
    out = tmp_path / "reviewed"
    export_bundle(bundle, reviewed_export(bundle), out, groups_path=verified_groups(bundle))
    lock = tmp_path / "reviewed_external.lock.json"
    create_contract(out / "dataset/dataset.yaml", lock, out / "review.csv")
    assert verify_contract(out / "dataset/dataset.yaml", lock)[0]
    with pytest.raises(ValueError, match="externa"):
        verify_contract(out / "dataset/dataset.yaml", lock, for_training=True)
