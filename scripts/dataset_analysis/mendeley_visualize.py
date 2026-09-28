"""Dibuja las cajas del dataset de Mendeley, separadas por valor de <name> (3, 4, 5).

Las lesiones de Mendeley son diminutas (mediana: 0,1% de la imagen, ~56 px de
lado menor en radiografias de ~2400 px). En una panoramica completa apenas se
ven, asi que el script genera dos tipos de figura:

  1. Panoramicas completas con las cajas dibujadas (color por valor de <name>),
     para ver DONDE estan las lesiones y que encierra la caja respecto al diente.
       - N imagenes por cada valor de <name>.
       - La imagen con mas cajas (el caso mas "cargado").
  2. Una galeria de recortes ampliados por cada valor de <name>
     (grade_3_crops.png, ...), para comparar como se ve cada grado de cerca.
     Cada recorte deja contexto alrededor de la caja (CONTEXT_FACTOR veces su
     tamano) para que se vea tambien la raiz y el diente.

Solo usa las imagenes ORIGINALES (no las aumentadas).
Requiere Pillow (ya viene en requirements.txt).

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/mendeley_visualize.py
    python scripts/dataset_analysis/mendeley_visualize.py --per-grade 3 --crops 16 --seed 7
"""

import argparse
import math
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data/raw/mendeley_periapical/Periapical Lesions"
DEFAULT_IMAGES = DEFAULT_ROOT / "Original JPG Images"
DEFAULT_ANNOTS = DEFAULT_ROOT / "Image Annots"
DEFAULT_OUT = REPO_ROOT / "reports/figures/mendeley_samples"

# Un color por valor de <name>, de menos a mas grave.
GRADE_COLORS = {
    "3": (255, 220, 0),   # amarillo
    "4": (255, 130, 0),   # naranja
    "5": (255, 0, 0),     # rojo
}
BOX_WIDTH = 6

CONTEXT_FACTOR = 4    # el recorte mide CONTEXT_FACTOR x el lado mayor de la caja
MIN_CROP_PX = 300     # ...pero nunca menos que esto, para ver el diente
TILE_PX = 320         # tamano de cada recorte en la galeria
GRID_COLS = 4


def load_boxes(images_dir, annots_dir):
    """Devuelve {nombre_imagen: [(grado, xmin, ymin, xmax, ymax), ...]}.

    Descarta en silencio lo que el EDA ya reporta como roto (XML ilegible,
    grado desconocido, cajas invalidas): aqui solo se visualiza lo sano.
    """
    boxes = {}
    for img_path in sorted(images_dir.glob("*.jpg")):
        xml_path = annots_dir / f"{img_path.stem}.xml"
        if not xml_path.is_file():
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        found = []
        for obj in root.iter("object"):
            grade = (obj.findtext("name") or "").strip()
            if grade not in GRADE_COLORS:
                continue
            try:
                coords = [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError):
                continue
            if coords[2] > coords[0] and coords[3] > coords[1]:
                found.append((grade, *coords))
        boxes[img_path.name] = found
    return boxes


def pick_full_images(boxes, per_grade, rng):
    """Elige que panoramicas completas dibujar y por que."""
    by_grade = defaultdict(list)
    for name, items in boxes.items():
        for grade in {b[0] for b in items}:
            by_grade[grade].append(name)

    chosen = {}  # nombre -> motivo
    for grade in sorted(GRADE_COLORS):
        candidates = sorted(by_grade[grade])
        rng.shuffle(candidates)
        for name in candidates[:per_grade]:
            chosen.setdefault(name, f"ejemplo de grado {grade}")

    if boxes:
        busiest = max(boxes, key=lambda n: len(boxes[n]))
        chosen.setdefault(busiest, f"la mas cargada ({len(boxes[busiest])} cajas)")
    return chosen


def draw_full(image_path, items, out_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for grade, x0, y0, x1, y1 in items:
        color = GRADE_COLORS[grade]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=BOX_WIDTH)
        draw.text((x0, max(0, y0 - 22)), grade, fill=color)
    img.save(out_path)


def crop_around(img, box):
    """Recorte cuadrado centrado en la caja, con contexto, ajustado a los bordes."""
    _, x0, y0, x1, y1 = box
    side = max(MIN_CROP_PX, CONTEXT_FACTOR * max(x1 - x0, y1 - y0))
    side = min(side, img.width, img.height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(0, cx - side / 2), img.width - side)
    top = min(max(0, cy - side / 2), img.height - side)
    tile = img.crop((int(left), int(top), int(left + side), int(top + side)))
    scale = TILE_PX / tile.width
    tile = tile.resize((TILE_PX, TILE_PX))
    # La caja, trasladada al sistema de coordenadas del recorte.
    ImageDraw.Draw(tile).rectangle(
        [(x0 - int(left)) * scale, (y0 - int(top)) * scale,
         (x1 - int(left)) * scale, (y1 - int(top)) * scale],
        outline=GRADE_COLORS[box[0]], width=2,
    )
    return tile


def draw_gallery(images_dir, boxes, grade, n_crops, rng, out_path):
    """Galeria de recortes de cajas de un grado. Devuelve cuantos recortes puso."""
    candidates = [(name, b) for name, items in boxes.items() for b in items if b[0] == grade]
    candidates.sort(key=lambda c: (c[0], c[1][1:]))
    rng.shuffle(candidates)
    picked = candidates[:n_crops]
    if not picked:
        return 0

    rows = math.ceil(len(picked) / GRID_COLS)
    label_h = 18
    sheet = Image.new("RGB", (GRID_COLS * TILE_PX, rows * (TILE_PX + label_h)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (name, box) in enumerate(picked):
        with Image.open(images_dir / name) as img:
            tile = crop_around(img.convert("RGB"), box)
        x, y = (i % GRID_COLS) * TILE_PX, (i // GRID_COLS) * (TILE_PX + label_h)
        sheet.paste(tile, (x, y + label_h))
        w, h = int(box[3] - box[1]), int(box[4] - box[2])
        draw.text((x + 4, y + 3), f"{name}  caja {w}x{h}px", fill=GRADE_COLORS[grade])
    sheet.save(out_path)
    return len(picked)


def main():
    parser = argparse.ArgumentParser(description="Visualiza las cajas de Mendeley por grado")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annots", type=Path, default=DEFAULT_ANNOTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-grade", type=int, default=2, help="panoramicas completas por grado")
    parser.add_argument("--crops", type=int, default=12, help="recortes por grado en la galeria")
    parser.add_argument("--seed", type=int, default=42, help="semilla para elegir siempre la misma muestra")
    args = parser.parse_args()

    boxes = load_boxes(args.images, args.annots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    chosen = pick_full_images(boxes, args.per_grade, rng)
    manifest = []
    for name, reason in chosen.items():
        out_path = args.out_dir / f"{Path(name).stem}_boxes.png"
        draw_full(args.images / name, boxes[name], out_path)
        grades = "".join(sorted({b[0] for b in boxes[name]}))
        manifest.append((out_path.name, name, len(boxes[name]), grades, reason))

    print(f"Panoramicas completas en {args.out_dir}\n")
    print(f"{'archivo generado':<24}{'original':<12}{'cajas':>6}{'grados':>8}  motivo")
    for out_name, orig, n, grades, reason in manifest:
        print(f"{out_name:<24}{orig:<12}{n:>6}{grades:>8}  {reason}")

    print("\nGalerias de recortes por grado:")
    for grade in sorted(GRADE_COLORS):
        out_path = args.out_dir / f"grade_{grade}_crops.png"
        n = draw_gallery(args.images, boxes, grade, args.crops, rng, out_path)
        print(f"  {out_path.name}: {n} recortes" if n else f"  grado {grade}: sin cajas, no se genera")


if __name__ == "__main__":
    main()
