"""Pruebas del entrenamiento con procedencia (src/training/train_research.py).

Las pruebas rapidas sustituyen a Ultralytics por un entrenador falso; la
ultima entrena de verdad (CPU, 1 epoca, imagenes de 64 px) para comprobar que
Ultralytics acepta un dataset.yaml con el bloque meta.
"""

import os

import pytest
import yaml

from tests.helpers import synthetic_dataset
from src.training import train_research as tb

NAMES = {0: "caries", 1: "implante"}


def _dataset(root, n=2):
    return synthetic_dataset(root, n)[0]


def _cfg(tmp_path, data_yaml, **train):
    return {"model": "yolo11n.yaml", "data": str(data_yaml), "seeds": [1, 2],
            "project": str(tmp_path / "runs"), "name": "prueba", "train": train,
            "dataset_lock": str(data_yaml.parent.parent / f"{data_yaml.parent.name}.lock.json")}


class FakeRunner:
    """Registra las llamadas en lugar de entrenar."""

    def __init__(self):
        self.calls = []

    def __call__(self, model_path, data_yaml, seed, project, name, train_args):
        self.calls.append({"seed": seed, "name": name, "train": train_args})
        save_dir = project / name
        save_dir.mkdir(parents=True, exist_ok=True)
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
                                    "project": "runs", "name": "x", "train": train,
                                    "dataset_lock": "d.lock.json"}))
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
    cfg = _cfg(tmp_path, data_yaml)
    cfg["dataset_lock"] = str(tmp_path / "missing.lock.json")
    with pytest.raises(tb.DatasetMismatch, match="Contrato no valido"):
        tb.train(cfg, runner=FakeRunner())


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
    assert all(c["train"] == {"epochs": 3, "deterministic": True} for c in runner.calls)

    for seed, d in zip([1, 2], dirs):
        prov = yaml.safe_load((d / "provenance.yaml").read_text(encoding="utf-8"))
        assert prov["seed"] == seed
        assert prov["dataset"]["fingerprint"] == tb.check_dataset(data_yaml, tmp_path / "master.lock.json")
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

@pytest.mark.integration
@pytest.mark.parametrize("architecture", ["yolo11n.yaml", "yolo26n.yaml"])
def test_entrenamiento_real_minimo(tmp_path, monkeypatch, architecture):
    # project relativo: debe quedar bajo la raiz del repo (aqui tmp_path), no en
    # la carpeta global de Ultralytics, que es a donde manda las rutas relativas.
    monkeypatch.setattr(tb, "REPO_ROOT", tmp_path)
    data_yaml = _dataset(tmp_path / "master")
    device = os.environ.get("DENTA_TEST_DEVICE", "cpu")
    cfg = _cfg(tmp_path, data_yaml, epochs=1, imgsz=64, batch=2, device=device,
               workers=0, plots=False, val=True, amp=False)
    cfg["seeds"] = [3]
    cfg["model"] = architecture
    cfg["project"] = "runs"
    (run_dir,) = tb.train(cfg)
    assert (run_dir / "weights/best.pt").is_file()
    assert run_dir.parent == tmp_path / "runs"
    prov = yaml.safe_load((run_dir / "provenance.yaml").read_text(encoding="utf-8"))
    assert prov["seed"] == 3
    assert yaml.safe_load((run_dir / "args.yaml").read_text())["seed"] == 3
    if device != "cpu":
        assert prov["resources"]["gpu_mem_peak_gb"] > 0
