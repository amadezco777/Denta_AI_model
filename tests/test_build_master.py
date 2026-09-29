"""Pruebas de la construccion del dataset maestro (decisiones del 2026-09-28)."""

import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dataset_conversion/build_master.py"

spec = importlib.util.spec_from_file_location("build_master", SCRIPT)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

MAPPING, TAXONOMY = bm.DEFAULT_MAPPING, bm.DEFAULT_TAXONOMY


# --- mapeo de clases -------------------------------------------------------------

def test_mapeo_traduce_y_descarta(tmp_path):
    mapping, names, _ = bm.load_mapping(MAPPING, TAXONOMY)
    label = tmp_path / "a.txt"
    # 4 Carious lesion -> caries (0); 0 Implant -> implante (6); 5 Bone resorbtion -> descartada
    label.write_text("4 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1\n5 0.2 0.2 0.1 0.1\n")
    boxes, dropped = bm.map_boxes(label, mapping)
    assert [b[0] for b in boxes] == [0, 6]
    assert names[0] == "caries" and names[6] == "implante"
    assert dropped == Counter({5: 1})


def test_id_sin_mapeo_detiene(tmp_path):
    mapping, _, _ = bm.load_mapping(MAPPING, TAXONOMY)
    label = tmp_path / "a.txt"
    label.write_text("14 0.5 0.5 0.1 0.1\n")
    with pytest.raises(SystemExit):
        bm.map_boxes(label, mapping)


def test_mapeo_de_otra_version_de_taxonomia_detiene(tmp_path):
    spec_ = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    spec_["taxonomy_version"] = "0.1.3-draft"
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(spec_), encoding="utf-8")
    with pytest.raises(SystemExit):
        bm.load_mapping(old, TAXONOMY)


def test_mapeo_real_cubre_los_14_ids_y_las_7_clases():
    mapping, names, _ = bm.load_mapping(MAPPING, TAXONOMY)
    assert sorted(mapping) == list(range(14))
    assert {t for t in mapping.values() if t is not None} == set(names)


# --- mascara ---------------------------------------------------------------------

def test_mascara_tapa_bordes_y_respeta_el_centro():
    img = np.full((100, 200), 255, np.uint8)
    out = bm.apply_mask(img)
    assert out[:, :6].max() == 0 and out[:, -6:].max() == 0 and out[-4:, :].max() == 0
    assert out[:96, 6:194].min() == 255
    assert out[0, 100] == 255  # el borde superior no se tapa


def test_cajas_bajo_mascara():
    centro = (0, 0.5, 0.5, 0.1, 0.1)
    izquierda = (0, 0.03, 0.5, 0.04, 0.1)
    abajo = (0, 0.5, 0.97, 0.1, 0.04)
    assert bm.boxes_under_mask([centro, izquierda, abajo]) == [izquierda, abajo]


# --- reparto estratificado por grupos ----------------------------------------------

def test_reparto_respeta_fraccion_y_clase_escasa():
    rng = np.random.default_rng(0)
    groups = [(1, Counter({0: int(rng.integers(1, 5)), 3: int(rng.integers(0, 4))})) for _ in range(180)]
    groups += [(2, Counter({0: 2, 6: 1})) for _ in range(20)]  # clase escasa, en grupos de 2 imagenes
    val = bm.stratified_group_split(groups, n_classes=7, val_fraction=0.15, seed=1)
    n_val = sum(groups[i][0] for i in val)
    assert 0.12 <= n_val / 220 <= 0.18
    rare_val = sum(groups[i][1][6] for i in val)
    assert 2 <= rare_val <= 4  # ~15 % de 20


def test_reparto_no_acapara_las_imagenes_cargadas():
    # Caso del dataset real: unas pocas imagenes con muchas cajas y la clase escasa.
    # El reparto voraz anterior las mandaba a val y val se llevaba ~20-28 % de las cajas.
    groups = [(1, Counter({0: 1})) for _ in range(180)]
    groups += [(1, Counter({0: 10, 6: 1})) for _ in range(20)]
    val = bm.stratified_group_split(groups, n_classes=7, val_fraction=0.15, seed=1)
    n_val = sum(groups[i][0] for i in val)
    assert n_val == 30  # 15 % de 200
    for c, total in ((0, 380), (6, 20)):
        frac = sum(groups[i][1][c] for i in val) / total
        assert abs(frac - 0.15) <= 0.03, (c, frac)


def test_reparto_es_reproducible():
    groups = [(1, Counter({i % 7: 1})) for i in range(100)]
    assert bm.stratified_group_split(groups, 7, seed=5) == bm.stratified_group_split(groups, 7, seed=5)


# --- de principio a fin ------------------------------------------------------------

def _textured(path, seed, quality=95):
    small = np.random.default_rng(seed).integers(0, 255, (12, 24), dtype=np.uint8)
    Image.fromarray(small).resize((240, 120), Image.BILINEAR).save(path, quality=quality)


def _make_interim(root, n_train=40):
    for split in ("train", "valid", "test", "externa"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    def add(split, name, seed, labels="4 0.5 0.5 0.1 0.1\n", quality=95):
        _textured(root / split / "images" / name, seed, quality)
        (root / split / "labels" / f"{Path(name).stem}.txt").write_text(labels)

    for k in range(n_train):
        add("train", f"{300 + k}.jpg", seed=100 + k)
    add("valid", "600.jpg", seed=500)
    add("test", "v1.jpg", seed=600)
    add("externa", "60.jpg", seed=700)  # mismo nombre que no choca con train gracias al prefijo
    return add


def _run(src, dst, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst), *extra],
                          capture_output=True, text=True)


def test_extremo_a_extremo(tmp_path):
    src, dst = tmp_path / "interim", tmp_path / "master"
    add = _make_interim(src)
    # Mismo paciente en train y valid (59/60 real): deben acabar juntos.
    add("train", "59.jpg", seed=900, labels="7 0.5 0.5 0.1 0.1\n", quality=95)
    add("valid", "60.jpg", seed=900, labels="7 0.5 0.5 0.1 0.1\n0 0.4 0.4 0.1 0.1\n", quality=60)
    r = _run(src, dst)
    assert r.returncode == 0, r.stderr

    manifest = {row.split(",")[2]: row.split(",") for row in
                (dst / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]}
    assert manifest["train/59.jpg"][1] == manifest["valid/60.jpg"][1]      # misma particion
    assert manifest["train/59.jpg"][3] == manifest["valid/60.jpg"][3]      # mismo grupo
    assert manifest["test/v1.jpg"][1] == "test" and manifest["externa/60.jpg"][1] == "externa"
    assert set(r[1] for r in manifest.values()) == {"train", "val", "test", "externa"}

    # Clases traducidas: 7 Apical periodontitis -> lesion_periapical (1); 0 Implant -> implante (6).
    split60 = manifest["valid/60.jpg"][1]
    ids = [line.split()[0] for line in (dst / "labels" / split60 / "muresanu_valid_60.txt").read_text().splitlines()]
    assert ids == ["1", "6"]

    # Mascara aplicada tambien en test.
    img = np.array(Image.open(dst / "images/test/muresanu_test_v1.jpg"))
    assert img[:, :3].max() < 20 and img[-3:, :].max() < 20

    cfg = yaml.safe_load((dst / "dataset.yaml").read_text(encoding="utf-8"))
    assert cfg["names"][5] == "protesis_fija" and cfg["val"] == "images/val"
    assert Path(cfg["path"]).is_absolute()
    ext = yaml.safe_load((dst / "dataset_externa.yaml").read_text(encoding="utf-8"))
    assert ext["val"] == ext["test"] == "images/externa"


def test_fuga_hacia_test_detiene(tmp_path):
    src, dst = tmp_path / "interim", tmp_path / "master"
    add = _make_interim(src, n_train=5)
    add("train", "880.jpg", seed=42, quality=95)
    add("test", "v2.jpg", seed=42, quality=60)   # mismo paciente en test
    r = _run(src, dst)
    assert r.returncode != 0
    assert "fuga" in (r.stdout + r.stderr).lower()
    assert not dst.exists()


def test_dry_run_no_escribe(tmp_path):
    src, dst = tmp_path / "interim", tmp_path / "master"
    _make_interim(src, n_train=5)
    assert _run(src, dst, "--dry-run").returncode == 0
    assert not dst.exists()


def test_no_sobrescribe_una_salida_existente(tmp_path):
    src, dst = tmp_path / "interim", tmp_path / "master"
    _make_interim(src, n_train=5)
    dst.mkdir()
    (dst / "algo.txt").write_text("previo")
    assert _run(src, dst).returncode != 0
    assert (dst / "algo.txt").read_text() == "previo"
