"""Pruebas de la procedencia de datasets (src/data/provenance.py)."""

import subprocess

import pytest
import yaml

from src.data.provenance import fingerprint, git_state, verify_dataset


def _tree(root, files):
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return root


BASE = {"images/a.jpg": b"\xff\xd8abc", "labels/a.txt": b"0 0.5 0.5 0.1 0.1\n"}


def test_misma_carpeta_misma_huella(tmp_path):
    a = _tree(tmp_path / "a", BASE)
    b = _tree(tmp_path / "b", BASE)
    assert fingerprint(a) == fingerprint(b)


@pytest.mark.parametrize("cambio", [
    {"labels/a.txt": b"0 0.5 0.5 0.1 0.2\n"},        # un byte distinto
    {"labels/b.txt": b""},                            # archivo nuevo, aunque vacio
])
def test_cualquier_cambio_cambia_la_huella(tmp_path, cambio):
    a = _tree(tmp_path / "a", BASE)
    b = _tree(tmp_path / "b", {**BASE, **cambio})
    assert fingerprint(a) != fingerprint(b)


def test_renombrar_cambia_la_huella(tmp_path):
    a = _tree(tmp_path / "a", BASE)
    b = _tree(tmp_path / "b", {"images/z.jpg": BASE["images/a.jpg"], "labels/a.txt": BASE["labels/a.txt"]})
    assert fingerprint(a) != fingerprint(b)


def test_mover_bytes_entre_nombre_y_contenido_no_colisiona(tmp_path):
    # Sin separar ruta y contenido, "ab"+"c" y "a"+"bc" darian el mismo flujo de bytes.
    a = _tree(tmp_path / "a", {"ab": b"c"})
    b = _tree(tmp_path / "b", {"a": b"bc"})
    assert fingerprint(a) != fingerprint(b)


def test_excluye_cache_y_yaml(tmp_path):
    a = _tree(tmp_path / "a", BASE)
    b = _tree(tmp_path / "b", {**BASE, "labels/train.cache": b"x", "dataset.yaml": b"path: otra"})
    assert fingerprint(a, (".yaml", ".cache")) == fingerprint(b, (".yaml", ".cache"))


def test_verify_dataset_detecta_una_etiqueta_modificada(tmp_path):
    root = _tree(tmp_path / "master", BASE)
    meta = {"provenance": {"output_fingerprint": fingerprint(root, (".yaml", ".cache"))}}
    (root / "dataset.yaml").write_text(yaml.safe_dump({"meta": meta}), encoding="utf-8")
    assert verify_dataset(root / "dataset.yaml")[0]
    (root / "labels/a.txt").write_bytes(b"1 0.5 0.5 0.1 0.1\n")   # alguien cambia una clase a mano
    assert not verify_dataset(root / "dataset.yaml")[0]


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_state_commit_y_cambios_sin_commitear(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "script.py").write_text("x = 1\n")
    (tmp_path / ".gitignore").write_text("data/\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    state = git_state(tmp_path)
    assert len(state["commit"]) == 40 and not state["dirty"]

    (tmp_path / "data").mkdir()
    (tmp_path / "data/x.jpg").write_bytes(b"1")          # datos ignorados: no ensucian
    (tmp_path / "notas.txt").write_text("borrador")      # sin versionar: no ensucia
    assert not git_state(tmp_path)["dirty"]

    (tmp_path / "script.py").write_text("x = 2\n")       # codigo modificado: si
    assert git_state(tmp_path)["dirty"]


def test_git_state_fuera_de_un_repo_devuelve_none(tmp_path):
    assert git_state(tmp_path) is None
