"""Pruebas del entrenamiento con procedencia (src/training/train_baseline.py).

Las pruebas rapidas sustituyen a Ultralytics por un entrenador falso; la
ultima entrena de verdad (CPU, 1 epoca, imagenes de 64 px) para comprobar que
Ultralytics acepta un dataset.yaml con el bloque meta.
"""

import cv2
import numpy as np
import pytest
import yaml

from src.data.provenance import DATASET_EXCLUDE, fingerprint
from src.training import train_baseline as tb

NAMES = {0: "caries", 1: "implante"}


def _dataset(root, n=2):
    """Dataset YOLO sintetico con train/val y su huella registrada en el yaml."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((64, 64, 3), 40, np.uint8)
            cv2.rectangle(img, (20, 20), (40, 40), (220, 220, 220), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            (root / "labels" / split / f"{i}.txt").write_text(f"{i % 2} 0.47 0.47 0.33 0.33\n")
    meta = {"taxonomy_version": "0.2.0",
            "provenance": {"output_fingerprint": fingerprint(root, DATASET_EXCLUDE)}}
    data = {"path": str(root), "names": NAMES, "meta": meta,
            "train": "images/train", "val": "images/val"}
    (root / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root / "dataset.yaml"


def _cfg(tmp_path, data_yaml, **train):
    return {"model": "yolo11n.yaml", "data": str(data_yaml), "seeds": [1, 2],
            "project": str(tmp_path / "runs"), "name": "prueba", "train": train}


class FakeRunner:
    """Registra las llamadas en lugar de entrenar."""

    def __init__(self):
        self.calls = []

    def __call__(self, model_path, data_yaml, seed, project, name, train_args):
        self.calls.append({"seed": seed, "name": name, "train": train_args})
        save_dir = project / name
        save_dir.mkdir(parents=True)
        return save_dir, 12.3, 1.5


# --- configuracion -------------------------------------------------------------

def test_set_convierte_los_tipos():
    assert tb.parse_overrides(["epochs=1", "fliplr=0.0", "device=cpu", "cache=false"]) == \
        {"epochs": 1, "fliplr": 0.0, "device": "cpu", "cache": False}


def test_set_sin_igual_es_un_error():
    with pytest.raises(ValueError):
        tb.parse_overrides(["epochs"])


def _write_cfg(tmp_path, train):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"model": "m.pt", "data": "d.yaml", "seeds": [42],
                                    "project": "runs", "name": "x", "train": train}))
    return path


def test_set_sustituye_al_yaml(tmp_path):
    cfg = tb.load_config(_write_cfg(tmp_path, {"epochs": 50, "imgsz": 640}), ["epochs=1"])
    assert cfg["train"] == {"epochs": 1, "imgsz": 640}


@pytest.mark.parametrize("train, overrides", [
    ({"epoch": 50}, []),          # errata en el yaml
    ({"epochs": 50}, ["imgsize=960"]),  # errata en --set
])
def test_parametro_desconocido_detiene(tmp_path, train, overrides):
    with pytest.raises(ValueError, match="no conoce"):
        tb.load_config(_write_cfg(tmp_path, train), overrides)


def test_seed_dentro_de_train_detiene(tmp_path):
    # La semilla la fija el script por corrida; en train: se pisaria en silencio.
    with pytest.raises(ValueError, match="fuera de 'train:'"):
        tb.load_config(_write_cfg(tmp_path, {"seed": 7}))


# --- verificacion del dataset ----------------------------------------------------

def test_dataset_modificado_no_entrena(tmp_path):
    data_yaml = _dataset(tmp_path / "master")
    (tmp_path / "master/labels/train/0.txt").write_text("1 0.47 0.47 0.33 0.33\n")  # clase cambiada a mano
    runner = FakeRunner()
    with pytest.raises(tb.DatasetMismatch):
        tb.train(_cfg(tmp_path, data_yaml), runner=runner)
    assert runner.calls == []


def test_dataset_sin_huella_no_entrena(tmp_path):
    data_yaml = _dataset(tmp_path / "master")
    data = yaml.safe_load(data_yaml.read_text())
    del data["meta"]["provenance"]
    data_yaml.write_text(yaml.safe_dump(data))
    with pytest.raises(tb.DatasetMismatch, match="no registra"):
        tb.train(_cfg(tmp_path, data_yaml), runner=FakeRunner())


def test_cache_de_ultralytics_no_rompe_la_verificacion(tmp_path):
    data_yaml = _dataset(tmp_path / "master")
    (tmp_path / "master/labels/train.cache").write_bytes(b"creado por ultralytics")
    tb.train(_cfg(tmp_path, data_yaml), runner=FakeRunner())


# --- corridas y procedencia ---------------------------------------------------------

def test_una_corrida_por_semilla_con_su_procedencia(tmp_path):
    data_yaml = _dataset(tmp_path / "master")
    runner = FakeRunner()
    dirs = tb.train(_cfg(tmp_path, data_yaml, epochs=3), runner=runner)

    assert [c["seed"] for c in runner.calls] == [1, 2]
    assert [c["name"] for c in runner.calls] == ["prueba_s1", "prueba_s2"]
    assert all(c["train"] == {"epochs": 3} for c in runner.calls)

    for seed, d in zip([1, 2], dirs):
        prov = yaml.safe_load((d / "provenance.yaml").read_text(encoding="utf-8"))
        assert prov["seed"] == seed
        assert prov["dataset"]["fingerprint"] == fingerprint(tmp_path / "master", DATASET_EXCLUDE)
        assert prov["dataset"]["taxonomy_version"] == "0.2.0"
        assert prov["config"]["train"] == {"epochs": 3}
        assert prov["resources"] == {"train_seconds": 12.3, "gpu_mem_peak_gb": 1.5}
        assert "commit" in prov["code"] and "torch" in prov["environment"]


def test_el_runner_no_puede_alterar_la_config_de_la_siguiente_semilla(tmp_path):
    data_yaml = _dataset(tmp_path / "master")

    class Mutador(FakeRunner):
        def __call__(self, *args):
            args[-1]["epochs"] = 999
            return super().__call__(*args)

    runner = Mutador()
    tb.train(_cfg(tmp_path, data_yaml, epochs=3), runner=runner)
    assert runner.calls[1]["train"]["epochs"] == 999   # el que muto fue su propia copia
    prov = yaml.safe_load((tmp_path / "runs/prueba_s2/provenance.yaml").read_text())
    assert prov["config"]["train"]["epochs"] == 3


# --- integracion real con Ultralytics -------------------------------------------------

def test_entrenamiento_real_minimo(tmp_path, monkeypatch):
    # project relativo: debe quedar bajo la raiz del repo (aqui tmp_path), no en
    # la carpeta global de Ultralytics, que es a donde manda las rutas relativas.
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    data_yaml = _dataset(tmp_path / "master")
    cfg = _cfg(tmp_path, data_yaml, epochs=1, imgsz=64, batch=2, device="cpu",
               workers=0, plots=False, val=True, amp=False)
    cfg["seeds"] = [3]
    cfg["project"] = "runs"
    (run_dir,) = tb.train(cfg)
    assert (run_dir / "weights/best.pt").is_file()
    assert run_dir.parent == tmp_path / "runs"
    prov = yaml.safe_load((run_dir / "provenance.yaml").read_text(encoding="utf-8"))
    assert prov["seed"] == 3
    assert yaml.safe_load((run_dir / "args.yaml").read_text())["seed"] == 3
