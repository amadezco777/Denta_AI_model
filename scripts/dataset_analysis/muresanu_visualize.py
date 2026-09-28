"""Revision visual del dataset de Muresanu: mapa de clases y consenso de duplicados.

Genera dos tipos de figura en reports/figures/muresanu_samples/:

  1. class_<id>_<nombre>.png: galeria de recortes ampliados de una clase,
     tomados de data/interim (el dataset ya deduplicado). Sirve para
     confirmar que legend.txt asocia bien id y nombre: clases inconfundibles
     (implante, ortodoncia, dispositivo quirurgico) delatarian un mapa cruzado.
  2. pair_<imagen>.png: una radiografia que venia duplicada, dibujada tres
     veces, una debajo de otra: anotacion de la copia A, de la copia B
     (ambas de data/raw) y el consenso que quedo en data/interim. Se eligen
     a partir de dedup_log.csv: la que mas cajas perdio, una que cruzaba
     train/valid y una al azar.

Las cajas se colorean por clase; en las figuras de pares cada caja lleva
ademas el id de su clase, y arriba hay una leyenda id -> nombre.

Requiere Pillow (ya viene en requirements.txt). Ejecutar DESPUES de
scripts/dataset_conversion/muresanu_dedup.py.

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/muresanu_visualize.py
    python scripts/dataset_analysis/muresanu_visualize.py --crops 12 --seed 7
"""

import argparse
import csv
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data/raw/muresanu_cluj/muresanu_cluj_dataset"
DEFAULT_INTERIM = REPO_ROOT / "data/interim/muresanu_cluj"
DEFAULT_OUT = REPO_ROOT / "reports/figures/muresanu_samples"

RAW_SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
    "externa": "test_alte_cabinete/Ext-validation",
}

# 14 colores bien separados, uno por id de clase.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
]

CONTEXT_FACTOR = 3    # el recorte mide CONTEXT_FACTOR x el lado mayor de la caja
MIN_CROP_PX = 300
TILE_PX = 300
GRID_COLS = 4
PAIR_WIDTH = 1400     # ancho de cada panel en las figuras de pares


def load_legend(path):
    pairs = re.findall(r'\("([^"]+)",\s*(\d+)\)', path.read_text(encoding="utf-8"))
    return {int(i): name for name, i in pairs}


def read_boxes(label_path):
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5 and float(parts[3]) > 0 and float(parts[4]) > 0:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def to_pixels(box, width, height):
    _, xc, yc, w, h = box
    return (xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height


def color(cls):
    return PALETTE[cls % len(PALETTE)]


# --- 1. Galerias por clase ----------------------------------------------------

def crop_tile(img, box):
    x0, y0, x1, y1 = to_pixels(box, img.width, img.height)
    side = max(MIN_CROP_PX, CONTEXT_FACTOR * max(x1 - x0, y1 - y0))
    side = min(side, img.width, img.height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(min(max(0, cx - side / 2), img.width - side))
    top = int(min(max(0, cy - side / 2), img.height - side))
    tile = img.crop((left, top, left + int(side), top + int(side)))
    scale = TILE_PX / tile.width
    tile = tile.resize((TILE_PX, TILE_PX))
    ImageDraw.Draw(tile).rectangle(
        [(x0 - left) * scale, (y0 - top) * scale, (x1 - left) * scale, (y1 - top) * scale],
        outline=color(box[0]), width=2,
    )
    return tile


def draw_class_gallery(interim, cls, name, n_crops, rng, out_path):
    candidates = []
    for split in RAW_SPLITS:
        for lbl in sorted((interim / split / "labels").glob("*.txt")):
            for box in read_boxes(lbl):
                if box[0] == cls:
                    candidates.append((split, lbl.stem, box))
    rng.shuffle(candidates)
    picked = candidates[:n_crops]
    if not picked:
        return 0

    rows = math.ceil(len(picked) / GRID_COLS)
    label_h = 18
    sheet = Image.new("RGB", (GRID_COLS * TILE_PX, rows * (TILE_PX + label_h)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (split, stem, box) in enumerate(picked):
        img_path = next((interim / split / "images").glob(f"{stem}.*"))
        with Image.open(img_path) as img:
            tile = crop_tile(img.convert("RGB"), box)
        x, y = (i % GRID_COLS) * TILE_PX, (i // GRID_COLS) * (TILE_PX + label_h)
        sheet.paste(tile, (x, y + label_h))
        draw.text((x + 4, y + 3), f"{split}/{img_path.name}", fill=color(cls))
    draw.text((sheet.width - 8 * len(name) - 10, 3), f"{cls}: {name}", fill=(255, 255, 255))
    sheet.save(out_path)
    return len(picked)


# --- 2. Pares: copia A / copia B / consenso ----------------------------------------

def draw_panel(img_path, boxes, title, class_names):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
    scale = PAIR_WIDTH / img.width
    img = img.resize((PAIR_WIDTH, int(img.height * scale)))
    header_h = 22
    panel = Image.new("RGB", (img.width, img.height + header_h), (30, 30, 30))
    panel.paste(img, (0, header_h))
    draw = ImageDraw.Draw(panel)
    for box in boxes:
        x0, y0, x1, y1 = to_pixels(box, img.width, img.height)
        draw.rectangle([x0, y0 + header_h, x1, y1 + header_h], outline=color(box[0]), width=3)
        draw.text((x0 + 2, y0 + header_h + 1), str(box[0]), fill=color(box[0]))
    draw.text((6, 5), f"{title} - {len(boxes)} cajas", fill=(255, 255, 255))
    return panel


def legend_strip(class_names, width):
    strip = Image.new("RGB", (width, 22), (15, 15, 15))
    draw = ImageDraw.Draw(strip)
    x = 6
    for cls, name in sorted(class_names.items()):
        text = f"{cls} {name}"
        draw.text((x, 5), text, fill=color(cls))
        x += 7 * len(text) + 12
    return strip


def pick_pairs(log_rows, rng):
    """Elige que grupos duplicados dibujar y por que."""
    rows = [r for r in log_rows if r["particion_final"] != "EXCLUIDA"]
    chosen = {}

    def lost(r):
        per_copy = [int(n) for n in r["cajas_por_copia"].split(" | ")]
        return sum(per_copy) / len(per_copy) - int(r["cajas_consenso"])

    if rows:
        worst = max(rows, key=lost)
        chosen[worst["md5"]] = (worst, "la que mas cajas perdio en el consenso")
        crossing = [r for r in rows if "train/" in r["copias"] and "valid/" in r["copias"]]
        if crossing:
            r = rng.choice(crossing)
            chosen.setdefault(r["md5"], (r, "cruzaba train y valid (fuga de datos)"))
        r = rng.choice(rows)
        chosen.setdefault(r["md5"], (r, "al azar"))
    return list(chosen.values())


def draw_pair(raw, interim, row, class_names, out_path):
    copies = row["copias"].split(" | ")
    panels = [legend_strip(class_names, PAIR_WIDTH)]
    for label, copy in zip("ABCD", copies):
        split, name = copy.split("/", 1)
        img_path = raw / RAW_SPLITS[split] / "images" / name
        boxes = read_boxes(raw / RAW_SPLITS[split] / "labels" / f"{Path(name).stem}.txt")
        panels.append(draw_panel(img_path, boxes, f"Copia {label}: raw/{copy}", class_names))
    final_split, final_name = row["particion_final"], row["imagen_final"]
    boxes = read_boxes(interim / final_split / "labels" / f"{Path(final_name).stem}.txt")
    panels.append(draw_panel(interim / final_split / "images" / final_name, boxes,
                             f"CONSENSO: interim/{final_split}/{final_name}", class_names))

    sheet = Image.new("RGB", (PAIR_WIDTH, sum(p.height for p in panels)), (30, 30, 30))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Revision visual de Muresanu (mapa de clases y consenso)")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--crops", type=int, default=8, help="recortes por clase")
    parser.add_argument("--seed", type=int, default=42, help="semilla para elegir siempre la misma muestra")
    args = parser.parse_args()

    class_names = load_legend(args.interim / "legend.txt")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Figuras en {args.out_dir}\n")
    print("Galerias por clase:")
    for cls, name in sorted(class_names.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        out_path = args.out_dir / f"class_{cls:02d}_{slug}.png"
        n = draw_class_gallery(args.interim, cls, name, args.crops, rng, out_path)
        print(f"  {out_path.name:<42} {n} recortes")

    with open(args.interim / "dedup_log.csv", encoding="utf-8") as f:
        log_rows = list(csv.DictReader(f))
    print("\nPares (copia A / copia B / consenso):")
    for row, reason in pick_pairs(log_rows, rng):
        out_path = args.out_dir / f"pair_{Path(row['imagen_final']).stem}.png"
        draw_pair(args.raw, args.interim, row, class_names, out_path)
        print(f"  {out_path.name:<20} {row['copias']:<36} cajas {row['cajas_por_copia']} -> "
              f"{row['cajas_consenso']}  ({reason})")


if __name__ == "__main__":
    main()
