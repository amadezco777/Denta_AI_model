"""Mascara versionada compartida por preparacion e inferencia futura."""

import math

MASK_VERSION = "border-mask-v1"
DEFAULT_MASK = {"left": 0.03, "right": 0.03, "bottom": 0.04}


def validate_mask(mask):
    if not isinstance(mask, dict) or set(mask) != {"left", "right", "bottom"}:
        raise ValueError("La mascara requiere left, right y bottom")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0 <= v < 1 for v in mask.values()):
        raise ValueError("Fracciones de mascara finitas en [0, 1) requeridas")
    if mask["left"] + mask["right"] >= 1:
        raise ValueError("La mascara no puede ocultar todo el ancho")


def apply_mask(image, mask=DEFAULT_MASK):
    """Modifica image; cero pixeles de franja no debe ocultar el extremo opuesto."""
    validate_mask(mask)
    if image is None or image.ndim not in (2, 3) or not all(image.shape[:2]):
        raise ValueError("Imagen decodificada no vacia requerida")
    h, w = image.shape[:2]
    left, right, bottom = int(mask["left"] * w), int(mask["right"] * w), int(mask["bottom"] * h)
    if left:
        image[:, :left] = 0
    if right:
        image[:, w - right:] = 0
    if bottom:
        image[h - bottom:, :] = 0
    return image


def boxes_under_mask(boxes, mask=DEFAULT_MASK):
    validate_mask(mask)
    return [b for b in boxes
            if b[1] - b[3] / 2 < mask["left"]
            or b[1] + b[3] / 2 > 1 - mask["right"]
            or b[2] + b[4] / 2 > 1 - mask["bottom"]]
