"""Deteccion de radiografias casi iguales por correlacion de miniaturas.

El MD5 solo detecta archivos identicos byte a byte. Si la misma radiografia se
guardo dos veces (otra compresion JPEG, otro exportado), el MD5 cambia pero los
pixeles no. Aqui cada imagen se reduce a una miniatura en escala de grises, se
normaliza (media 0, norma 1) y se compara con todas las demas: el producto
escalar de dos miniaturas es su correlacion de Pearson.

Umbrales medidos sobre Muresanu (2026-09-28, reports/figures/muresanu_near_dups.jpg):
  - >= 0,98: la misma imagen reexportada (a ojo, identicas).
  - 0,90-0,98: el mismo paciente en otra visita (cambian los tratamientos).
  - < 0,90: pacientes distintos; es el parecido general de dos panoramicas
    (aparecen pares de centros distintos, que no pueden ser el mismo paciente).
"""

import cv2
import numpy as np

THUMB_SIZE = (96, 48)  # (ancho, alto): panoramicas ~2:1
REEXPORT_THRESHOLD = 0.98
SAME_PATIENT_THRESHOLD = 0.90


def thumbnail(path):
    """Miniatura normalizada de una imagen, como vector de norma 1.

    Una imagen de un solo color (sin varianza) devuelve un vector de ceros,
    que no se parece a nada.
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"No se puede leer la imagen: {path}")
    v = cv2.resize(im, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.zeros_like(v)


def similar_pairs(vectors, threshold):
    """Pares (i, j), con i < j, cuya correlacion supera threshold."""
    if len(vectors) < 2:
        return []
    x = np.asarray(vectors, dtype=np.float32)
    corr = x @ x.T
    i, j = np.where(np.triu(corr > threshold, 1))
    return list(zip(i.tolist(), j.tolist()))


def connected_groups(n, pairs):
    """Agrupa los indices 0..n-1 en componentes conexas (union-find).

    Si A se parece a B y B a C, los tres quedan en el mismo grupo aunque A y C
    no superen el umbral entre si.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values())
