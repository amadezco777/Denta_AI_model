"""Pruebas de la deduplicacion por consenso de Muresanu (opcion B, 2026-09-27)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dataset_conversion/muresanu_dedup.py"

spec = importlib.util.spec_from_file_location("muresanu_dedup", SCRIPT)
dedup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedup)


# --- consensus() --------------------------------------------------------------

def test_par_conserva_solo_lo_que_marcan_ambos():
    a = [(4, 0.50, 0.50, 0.10, 0.10), (7, 0.20, 0.20, 0.05, 0.05)]  # caries + periapical
    b = [(4, 0.51, 0.50, 0.10, 0.10)]                                # solo caries
    assert dedup.consensus([a, b]) == [pytest.approx((4, 0.505, 0.50, 0.10, 0.10))]


def test_misma_posicion_distinta_clase_no_coincide():
    a = [(4, 0.5, 0.5, 0.1, 0.1)]
    b = [(5, 0.5, 0.5, 0.1, 0.1)]
    assert dedup.consensus([a, b]) == []


def test_solapamiento_insuficiente_no_coincide():
    a = [(4, 0.50, 0.5, 0.1, 0.1)]
    b = [(4, 0.56, 0.5, 0.1, 0.1)]  # IoU = 0.4/1.6 = 0.25
    assert dedup.consensus([a, b]) == []


def test_emparejamiento_uno_a_uno():
    # Dos cajas en A casi en el mismo sitio, una sola en B: solo una sobrevive.
    a = [(8, 0.5, 0.5, 0.1, 0.1), (8, 0.501, 0.5, 0.1, 0.1)]
    b = [(8, 0.5, 0.5, 0.1, 0.1)]
    assert len(dedup.consensus([a, b])) == 1


def test_trio_acepta_lo_que_marcan_la_segunda_y_la_tercera():
    a = []
    b = [(7, 0.3, 0.3, 0.1, 0.1)]
    c = [(7, 0.3, 0.3, 0.1, 0.1)]
    assert dedup.consensus([a, b, c]) == [pytest.approx((7, 0.3, 0.3, 0.1, 0.1))]


def test_trio_rechaza_lo_que_marca_una_sola_copia():
    a = [(7, 0.3, 0.3, 0.1, 0.1)]
    assert dedup.consensus([a, [], []]) == []


# --- de principio a fin ----------------------------------------------------------

def _make_dataset(root):
    for rel in dedup.SPLITS.values():
        (root / rel / "images").mkdir(parents=True)
        (root / rel / "labels").mkdir(parents=True)
    (root / "legend.txt").write_text('[("Carious lesion", 4)]')

    def add(rel, name, color, labels):
        Image.new("L", (100, 50), color).save(root / rel / "images" / name)
        (root / rel / "labels" / f"{Path(name).stem}.txt").write_text(labels)

    # Numeros fuera de los bloques A (101-199) y B (1-30), salvo donde se indica.
    add("train", "301.jpg", 10, "4 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.1 0.1\n")  # duplicado de valid/390
    add("valid", "390.jpg", 10, "4 0.5 0.5 0.1 0.1\n")
    add("train", "302.jpg", 20, "4 0.5 0.5 0.1 0.1\n4 0.5 0.5 0 0.1\n")    # unica, con caja de tamano 0
    add("test", "3.jpg", 30, "4 0.5 0.5 0.1 0.1\n")
    # externa tiene numeracion propia: su 1.jpg NO pertenece al bloque B.
    add("test_alte_cabinete/Ext-validation", "1.jpg", 40, "4 0.5 0.5 0.1 0.1\n")


def test_extremo_a_extremo(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)], check=True)

    def names(split, sub):
        return sorted(p.name for p in (dst / split / sub).iterdir())

    # El duplicado train/valid queda solo en train; valid se vacia.
    assert names("train", "images") == ["301.jpg", "302.jpg"]
    assert names("valid", "images") == []
    assert names("test", "images") == ["3.jpg"]
    assert names("externa", "images") == ["1.jpg"]

    # Consenso: solo la caja que marcaron ambas copias.
    assert (dst / "train/labels/301.txt").read_text().splitlines() == ["4 0.500000 0.500000 0.100000 0.100000"]
    # La caja de tamano 0 se descarta; las unicas no se tocan en lo demas.
    assert len((dst / "train/labels/302.txt").read_text().splitlines()) == 1

    log = (dst / "dedup_log.csv").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2  # cabecera + 1 grupo
    assert (dst / "legend.txt").is_file()


# --- paso 0: correccion del mapa de clases por bloques ----------------------------

def test_bloque_a_se_remapea(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    Image.new("L", (100, 50), 60).save(src / "train/images/150.jpg")
    # ids del orden alternativo: 1 obturacion, 5 impactado, 12 quirurgico, 4 caries (igual)
    (src / "train/labels/150.txt").write_text(
        "1 0.1 0.1 0.1 0.1\n5 0.3 0.3 0.1 0.1\n12 0.5 0.5 0.1 0.1\n4 0.7 0.7 0.1 0.1\n")
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)], check=True)

    ids = [line.split()[0] for line in (dst / "train/labels/150.txt").read_text().splitlines()]
    assert ids == ["2", "6", "13", "4"]
    assert "train/150.jpg,remapear" in (dst / "mapping_log.csv").read_text(encoding="utf-8")


def test_bloque_a_se_remapea_antes_del_consenso(tmp_path):
    # Misma radiografia: una copia en el bloque A (id 1 = obturacion) y otra
    # fuera (id 2 = obturacion). Solo coinciden si se remapea antes de comparar.
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    for split, name, cls in (("train", "120.jpg", 1), ("train", "620.jpg", 2)):
        Image.new("L", (100, 50), 70).save(src / split / "images" / name)
        (src / split / "labels" / f"{Path(name).stem}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1\n")
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)], check=True)

    assert (dst / "train/labels/120.txt").read_text().split()[0] == "2"


def test_bloque_b_se_excluye_pero_su_gemela_sobrevive(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    Image.new("L", (100, 50), 80).save(src / "train/images/15.jpg")   # bloque B
    Image.new("L", (100, 50), 80).save(src / "train/images/615.jpg")  # misma imagen, fuera
    (src / "train/labels/15.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (src / "train/labels/615.txt").write_text("2 0.5 0.5 0.1 0.1\n")
    Image.new("L", (100, 50), 90).save(src / "valid/images/20.jpg")   # bloque B, sin gemela
    (src / "valid/labels/20.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)], check=True)

    assert not (dst / "train/images/15.jpg").exists()
    assert not (dst / "valid/images/20.jpg").exists()
    # La gemela conserva sus propias cajas, sin consenso con la copia excluida.
    assert (dst / "train/labels/615.txt").read_text().split()[0] == "2"
    log = (dst / "mapping_log.csv").read_text(encoding="utf-8")
    assert "train/15.jpg,excluir" in log and "valid/20.jpg,excluir" in log


def test_excluye_grupo_sin_consenso(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    # Nuevo par en test que no coincide en nada.
    for name, labels in (("v1.jpg", "4 0.2 0.2 0.1 0.1\n"), ("v2.jpg", "4 0.8 0.8 0.1 0.1\n")):
        Image.new("L", (100, 50), 50).save(src / "test/images" / name)
        (src / "test/labels" / f"{Path(name).stem}.txt").write_text(labels)
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)], check=True)

    assert sorted(p.name for p in (dst / "test/images").iterdir()) == ["3.jpg"]
    assert not (dst / "test/labels/v1.txt").exists()
    assert "EXCLUIDA" in (dst / "dedup_log.csv").read_text(encoding="utf-8")


def test_no_sobrescribe_una_salida_existente(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_dataset(src)
    dst.mkdir()
    (dst / "algo.txt").write_text("previo")
    result = subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst)])
    assert result.returncode != 0
    assert (dst / "algo.txt").read_text() == "previo"
