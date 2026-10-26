"""El traslado conserva revisiones y detecta bytes alterados o rutas ZIP hostiles."""

import json
import zipfile

import pytest
import yaml

from scripts.training.estimate_cost import estimate
from src.data.contract import verify_contract
from src.training.portable import package_training, restore_training
from src.training.train_research import load_config
from tests.helpers import synthetic_dataset


def archive(tmp_path):
    data, lock, _ = synthetic_dataset(tmp_path / "master")
    cfg = {"model": "yolo26n.pt", "data": str(data), "dataset_lock": str(lock),
           "seeds": [42], "project": str(tmp_path / "runs"), "name": "pilot", "train": {"epochs": 3}}
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    package = tmp_path / "package.zip"
    package_training(path, package)
    return package


def test_restore_relocates_contract_preserving_reviews(tmp_path):
    package = archive(tmp_path)
    restored = tmp_path / "another" / "remote"
    cfg_path = restore_training(package, restored)
    cfg = load_config(cfg_path)
    assert cfg["data"] == str(restored / "dataset/dataset.yaml")
    assert cfg["project"] == str(restored / "runs")
    assert verify_contract(cfg["data"], cfg["dataset_lock"], for_training=True)[0]
    state = json.loads((restored / "restore.json").read_text())
    assert state["status"] == "completed"
    assert state["source_contract"] != state["new_contract"]


@pytest.mark.parametrize("name,content", [("../escape.txt", b"bad"), ("dataset/images/train/0.png", b"bad")])
def test_hostile_or_corrupt_zip_fails_before_publish(tmp_path, name, content):
    package = archive(tmp_path)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(changed, "w") as target:
        for entry in source.infolist():
            target.writestr(entry, content if entry.filename == name else source.read(entry.filename))
        if name not in source.namelist():
            target.writestr(name, content)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        restore_training(changed, out)
    assert not out.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_modified_approved_dataset_cannot_be_packaged(tmp_path):
    archive(tmp_path)
    (tmp_path / "master/labels/train/0.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="otros datos|distintos"):
        package_training(tmp_path / "training.yaml", tmp_path / "new.zip")


def test_cost_uses_measured_epoch_deltas(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text(" epoch, time\n1, 100\n2, 160\n3, 220\n", encoding="utf-8")
    result = estimate(csv, 60, 0.5, runs=2, setup_minutes=0)
    assert result["seconds_per_epoch"] == 60
    assert result["estimated_hours"] == 2
    assert result["estimated_total_usd"] == 1


@pytest.mark.parametrize("times", ["100\n90\n120", "NaN\n160\n220", "100\n160"])
def test_cost_rejects_invalid_pilot(tmp_path, times):
    path = tmp_path / "results.csv"
    path.write_text("time\n" + times)
    with pytest.raises(ValueError):
        estimate(path, 50, 0.5)


def test_yolo26_and_yolo11_comparison_change_only_model():
    a = load_config("configs/training/yolo26_baseline.yaml")
    b = load_config("configs/training/yolo11_comparison.yaml")
    assert a["model"] != b["model"]
    for key in ("data", "dataset_lock", "train", "seeds", "deterministic"):
        assert a[key] == b[key]
