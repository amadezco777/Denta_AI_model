"""Particionado reproducible por grupos; nunca divide un grupo de adquisiciones."""

import math
import random
from collections import Counter


def stratified_group_split(groups, n_classes, val_fraction=0.15, seed=42, n_trials=2000):
    if (not math.isfinite(val_fraction) or not 0 < val_fraction < 1
            or len(groups) < 2 or type(n_trials) is not int or n_trials <= 0
            or type(n_classes) is not int or n_classes <= 0):
        raise ValueError("Reparto requiere 0 < val_fraction < 1, dos grupos y parametros positivos")
    for n, counts in groups:
        if (type(n) is not int or n <= 0 or any(type(c) is not int or not 0 <= c < n_classes
                                               or type(v) is not int or v < 0 for c, v in counts.items())):
            raise ValueError("Grupos con imagenes positivas y conteos de clase validos requeridos")
    total = Counter()
    for _, counts in groups:
        total.update(counts)
    n_images = sum(n for n, _ in groups)
    classes = [c for c in range(n_classes) if total[c] > 0]
    rng = random.Random(seed)
    order = list(range(len(groups)))
    best, best_cost = set(), float("inf")
    for _ in range(n_trials):
        rng.shuffle(order)
        val, val_counts, val_images = set(), Counter(), 0
        for i in order[:-1]:
            if val_images >= val_fraction * n_images:
                break
            val.add(i)
            val_counts.update(groups[i][1])
            val_images += groups[i][0]
        cost = max([abs(val_images / n_images - val_fraction)]
                   + [abs(val_counts[c] / total[c] - val_fraction) for c in classes])
        if cost < best_cost:
            best, best_cost = val, cost
    return best


def three_way_split(groups, n_classes, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Reserva test primero; aproxima frecuencias sin prometer estratificacion perfecta."""
    if (len(groups) < 3 or not math.isfinite(val_fraction) or not math.isfinite(test_fraction)
            or min(val_fraction, test_fraction) <= 0 or val_fraction + test_fraction >= 1):
        raise ValueError("Train/val/test requiere tres grupos y fracciones positivas cuya suma sea < 1")
    # Reservar test debe dejar al menos dos grupos para train/val.
    test = stratified_group_split(groups, n_classes, test_fraction, seed)
    if len(groups) - len(test) < 2:
        test = {min(test)}
    remaining = [i for i in range(len(groups)) if i not in test]
    remaining_fraction = sum(groups[i][0] for i in remaining) / sum(n for n, _ in groups)
    val = {remaining[i] for i in stratified_group_split(
        [groups[i] for i in remaining], n_classes, min(val_fraction / remaining_fraction, 0.95), seed + 1)}
    return ["test" if i in test else "val" if i in val else "train" for i in range(len(groups))]
