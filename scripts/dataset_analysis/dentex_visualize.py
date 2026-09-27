"""Dibuja las cajas de diagnostico de DENTEX sobre una muestra de imagenes.

Por que una muestra dirigida y no aleatoria: con una clase tan minoritaria como
Periapical Lesion (158 cajas de 3529), una muestra puramente al azar podria no
mostrar ni un solo ejemplo de ella. En vez de eso, elegimos a proposito:

  - N imagenes por cada clase de diagnostico (para ver ejemplos de todas,
    incluida la minoritaria).
  - La imagen con mas cajas (el caso mas "cargado").
  - Una imagen sin ninguna caja (para confirmar que ese vacio tiene sentido).

Requiere Pillow (ya viene en requirements.txt).

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/dentex_visualize.py
    python scripts/dataset_analysis/dentex_visualize.py --per-class 3 --seed 7
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET = (
    REPO_ROOT
    / "data/raw/dentex/training_data/training_data/quadrant-enumeration-disease"
)
DEFAULT_JSON = DEFAULT_SUBSET / "train_quadrant_enumeration_disease.json"
DEFAULT_IMAGES = DEFAULT_SUBSET / "xrays"
DEFAULT_OUT = REPO_ROOT / "reports/figures/dentex_samples"

# Un color por clase, para reconocerlas de un vistazo en las imagenes.
CLASS_COLORS = {
    "Impacted": (255, 165, 0),          # naranja
    "Caries": (255, 0, 0),              # rojo
    "Periapical Lesion": (0, 200, 255), # celeste
    "Deep Caries": (200, 0, 200),       # magenta
}
BOX_WIDTH = 6


def load_dataset(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def pick_sample_image_ids(data, per_class, seed):
    """Elige que imagenes visualizar y por que, sin mirar todavia el disco."""
    images = {img["id"]: img for img in data["images"]}
    class_names = {c["id"]: c["name"] for c in data["categories_3"]}

    images_by_class = defaultdict(set)
    boxes_per_image = defaultdict(int)
    for ann in data["annotations"]:
        class_id = ann.get("category_id_3")
        image_id = ann["image_id"]
        # Igual que en el EDA: una anotacion "huerfana" (que apunta a una
        # imagen que no esta en el JSON) se descarta en vez de intentar
        # visualizarla.
        if class_id in class_names and image_id in images:
            images_by_class[class_id].add(image_id)
            boxes_per_image[image_id] += 1

    rng = random.Random(seed)
    chosen = {}  # image_id -> motivo de la eleccion

    for class_id, name in class_names.items():
        candidates = sorted(images_by_class[class_id])
        rng.shuffle(candidates)
        for image_id in candidates[:per_class]:
            chosen.setdefault(image_id, f"ejemplo de '{name}'")

    if boxes_per_image:
        busiest_id = max(boxes_per_image, key=boxes_per_image.get)
        chosen.setdefault(busiest_id, f"la mas cargada ({boxes_per_image[busiest_id]} cajas)")

    without_boxes = [i for i in images if boxes_per_image[i] == 0]
    if without_boxes:
        empty_id = rng.choice(without_boxes)
        chosen.setdefault(empty_id, "sin ninguna caja de diagnostico")

    return chosen, images


def draw_boxes(image_path, annotations, class_names, out_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        for ann in annotations:
            class_id = ann.get("category_id_3")
            name = class_names.get(class_id, f"clase {class_id}")
            color = CLASS_COLORS.get(name, (255, 255, 255))
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=BOX_WIDTH)
            draw.text((x, max(0, y - 22)), name, fill=color)
        img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualiza cajas de diagnostico de DENTEX")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-class", type=int, default=2, help="imagenes de ejemplo por clase")
    parser.add_argument("--seed", type=int, default=42, help="semilla para elegir siempre la misma muestra")
    args = parser.parse_args()

    data = load_dataset(args.json)
    class_names = {c["id"]: c["name"] for c in data["categories_3"]}
    chosen, images = pick_sample_image_ids(data, args.per_class, args.seed)

    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for image_id, reason in chosen.items():
        file_name = images[image_id]["file_name"]
        src = args.images / file_name
        if not src.is_file():
            print(f"AVISO: no se encontro {src}, se omite")
            continue
        out_path = args.out_dir / f"{file_name.rsplit('.', 1)[0]}_boxes.png"
        draw_boxes(src, anns_by_image[image_id], class_names, out_path)
        manifest.append((out_path.name, file_name, reason, len(anns_by_image[image_id])))

    print(f"Guardadas {len(manifest)} imagenes en {args.out_dir}\n")
    print(f"{'archivo generado':<32}{'original':<16}{'cajas':>6}  motivo")
    for out_name, orig_name, reason, n_boxes in manifest:
        print(f"{out_name:<32}{orig_name:<16}{n_boxes:>6}  {reason}")


if __name__ == "__main__":
    main()
