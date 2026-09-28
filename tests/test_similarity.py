"""Pruebas de la deteccion de radiografias casi iguales (src/data/similarity.py)."""

import numpy as np
from PIL import Image

from src.data.similarity import connected_groups, similar_pairs, thumbnail


def _save(path, arr, quality=95):
    Image.fromarray(arr).resize((240, 120), Image.BILINEAR).save(path, quality=quality)


def test_misma_imagen_con_otra_compresion_correlaciona_casi_1(tmp_path):
    arr = np.random.default_rng(0).integers(0, 255, (12, 24), dtype=np.uint8)
    _save(tmp_path / "a.jpg", arr, 95)
    _save(tmp_path / "b.jpg", arr, 50)
    assert float(thumbnail(tmp_path / "a.jpg") @ thumbnail(tmp_path / "b.jpg")) > 0.98


def test_imagenes_distintas_no_correlacionan(tmp_path):
    rng = np.random.default_rng(1)
    _save(tmp_path / "a.jpg", rng.integers(0, 255, (12, 24), dtype=np.uint8))
    _save(tmp_path / "b.jpg", rng.integers(0, 255, (12, 24), dtype=np.uint8))
    assert float(thumbnail(tmp_path / "a.jpg") @ thumbnail(tmp_path / "b.jpg")) < 0.5


def test_imagen_de_un_solo_color_no_se_parece_a_nada(tmp_path):
    Image.new("L", (100, 50), 10).save(tmp_path / "a.jpg")
    assert not thumbnail(tmp_path / "a.jpg").any()


def test_similar_pairs_umbral_estricto():
    v = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    assert similar_pairs(v, 0.9) == [(0, 1)]
    assert similar_pairs(v[:1], 0.9) == []


def test_connected_groups_es_transitivo():
    # 0~1 y 1~2 => {0,1,2} juntos aunque 0 y 2 no se comparen; 3 queda solo.
    assert connected_groups(4, [(0, 1), (1, 2)]) == [[0, 1, 2], [3]]
