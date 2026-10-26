"""Lectura estricta de cajas YOLO de deteccion, sin conversiones implicitas."""

import math
import re
from pathlib import Path


class AnnotationError(ValueError):
    """Una anotacion no cumple el contrato de deteccion."""


def read_yolo_boxes(path, class_ids=None, *, drop_degenerate=False):
    """Devuelve (cajas, degeneradas descartadas) o falla con archivo y linea.

    Un archivo vacio existe y representa cero anotaciones; un archivo ausente
    es un error. La cobertura y la revision de negativos se validan aparte.
    drop_degenerate es exclusivo de la limpieza historica: no permite NaN,
    poligonos ni cajas fuera de imagen, y nunca se usa para entrenar.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Falta el archivo de etiquetas: {path}")
    boxes, dropped = [], 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        location = f"{path}:{line_no}"
        if len(parts) != 5:
            raise AnnotationError(f"{location}: se esperan exactamente 5 campos de deteccion")
        if not re.fullmatch(r"[0-9]+", parts[0]):
            raise AnnotationError(f"{location}: clase entera no negativa requerida")
        cls = int(parts[0])
        if class_ids is not None and cls not in class_ids:
            raise AnnotationError(f"{location}: id de clase {cls} sin entrada en el mapeo")
        try:
            xc, yc, w, h = map(float, parts[1:])
        except ValueError as exc:
            raise AnnotationError(f"{location}: coordenada no numerica") from exc
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            raise AnnotationError(f"{location}: todas las coordenadas deben ser finitas")
        if w < 0 or h < 0 or not (0 <= xc <= 1 and 0 <= yc <= 1):
            raise AnnotationError(f"{location}: coordenadas normalizadas fuera de rango")
        if w == 0 or h == 0:
            if drop_degenerate:
                dropped += 1
                continue
            raise AnnotationError(f"{location}: ancho y alto deben ser positivos")
        # Margen exclusivo para redondeo decimal, no recorta ni repara cajas.
        eps = 1e-6
        if (w > 1 or h > 1 or xc - w / 2 < -eps or yc - h / 2 < -eps
                or xc + w / 2 > 1 + eps or yc + h / 2 > 1 + eps):
            raise AnnotationError(f"{location}: la caja sale de la imagen")
        boxes.append((cls, xc, yc, w, h))
    return boxes, dropped
