"""Analisis exploratorio (EDA) del dataset de Muresanu et al. (Cluj-Napoca, 14 clases).

Que hace: lee las etiquetas YOLO de las cuatro particiones que trae el dataset
y responde, sin modificar nada:
  1. Hay algo roto? (imagenes sin etiqueta, lineas mal formadas, clases
     desconocidas, coordenadas fuera de [0, 1], cajas de tamano 0)
  2. Hay imagenes repetidas ENTRE particiones? (se compara el contenido con
     MD5, no el nombre: seria fuga de datos entre train y test)
  3. Cuantas cajas hay de cada clase, en cada particion?
  4. Cuantas cajas por imagen, y de que tamano son? (para saber si la caja
     encierra el diente entero, como DENTEX, o solo la lesion, como Mendeley)
  5. De que resolucion son las imagenes?

Solo usa la libreria estandar de Python: no hace falta instalar nada.

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/muresanu_eda.py
    python scripts/dataset_analysis/muresanu_eda.py --save-report reports/muresanu_eda.txt

Notas sobre el formato:
  - YOLO: un .txt por imagen, una linea por caja:
        clase x_centro y_centro ancho alto
    con las cuatro coordenadas normalizadas entre 0 y 1 (fraccion del ancho y
    alto de la imagen). Para pasar a pixeles hace falta el tamano real de la
    imagen, que se lee de la cabecera del JPG/PNG.
  - Los nombres de clase salen de `legend.txt`, que asocia nombre e id. El
    archivo `test_alte_cabinete/labels (1).txt` es solo una lista de nombres en
    OTRO orden, sin ids: no se usa.
  - "test_alte_cabinete" (rumano: "test de otras clinicas") es la validacion
    externa: 180 radiografias de otros centros.
"""

import argparse
import hashlib
import re
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from mendeley_eda import jpeg_size

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data/raw/muresanu_cluj/muresanu_cluj_dataset"

# nombre de la particion -> carpeta relativa a la raiz del dataset
SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
    "externa": "test_alte_cabinete/Ext-validation",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Tolerancia para coordenadas normalizadas apenas fuera de [0, 1] por redondeo.
BOUNDS_TOLERANCE = 1e-3


def png_size(path):
    """Devuelve (ancho, alto) de un PNG leyendo su cabecera IHDR, o None."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def image_size(path):
    if path.suffix.lower() == ".png":
        return png_size(path)
    return jpeg_size(path)


def load_legend(legend_path):
    """Lee legend.txt y devuelve {id: nombre}."""
    text = legend_path.read_text(encoding="utf-8")
    pairs = re.findall(r'\("([^"]+)",\s*(\d+)\)', text)
    return {int(i): name for name, i in pairs}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values):
    """Devuelve 'min / mediana / media / max' de una lista de numeros."""
    if not values:
        return "sin datos"
    return (
        f"min {min(values):.1f} / mediana {statistics.median(values):.1f} / "
        f"media {statistics.mean(values):.1f} / max {max(values):.1f}"
    )


def analyze(root):
    class_names = load_legend(root / "legend.txt")

    # --- Acumuladores ---------------------------------------------------
    n_images = Counter()                              # particion -> imagenes
    boxes = defaultdict(Counter)                      # particion -> clase -> cajas
    images_with = defaultdict(lambda: defaultdict(set))  # particion -> clase -> imagenes
    boxes_per_image = []
    area_pct = defaultdict(list)                      # clase -> % de la imagen
    min_side_px = defaultdict(list)                   # clase -> lado menor en px
    resolutions = Counter()
    hashes = defaultdict(list)                        # md5 -> ["particion/archivo"]

    missing_label = []      # imagen sin su .txt
    orphan_label = []       # .txt sin imagen
    unreadable_img = []     # cabecera ilegible
    malformed = []          # linea que no tiene 5 numeros
    unknown_class = []      # id fuera de legend.txt
    out_of_bounds = []      # coordenadas fuera de [0, 1]
    invalid_boxes = []      # ancho o alto <= 0
    empty_labels = []       # .txt vacio (imagen sin hallazgos)

    for split, rel in SPLITS.items():
        img_dir, lbl_dir = root / rel / "images", root / rel / "labels"
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES} \
            if img_dir.is_dir() else {}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
        orphan_label += [f"{split}/{s}.txt" for s in sorted(labels.keys() - images.keys())]

        for stem, img_path in sorted(images.items()):
            tag = f"{split}/{img_path.name}"
            n_images[split] += 1
            hashes[md5(img_path)].append(tag)
            size = image_size(img_path)
            if size is None:
                unreadable_img.append(tag)
                continue
            width, height = size
            resolutions[size] += 1

            if stem not in labels:
                missing_label.append(tag)
                continue
            lines = [l for l in labels[stem].read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                empty_labels.append(tag)
            n_boxes = 0
            for i, line in enumerate(lines, 1):
                parts = line.split()
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except (ValueError, IndexError, TypeError):
                    malformed.append(f"{tag}:{i}")
                    continue
                if cls not in class_names:
                    unknown_class.append(f"{tag}:{cls}")
                    continue
                if w <= 0 or h <= 0:
                    invalid_boxes.append(f"{tag}:{i}")
                    continue
                tol = BOUNDS_TOLERANCE
                if xc - w / 2 < -tol or yc - h / 2 < -tol or xc + w / 2 > 1 + tol or yc + h / 2 > 1 + tol:
                    out_of_bounds.append(f"{tag}:{i}")

                n_boxes += 1
                boxes[split][cls] += 1
                images_with[split][cls].add(stem)
                area_pct[cls].append(100 * w * h)
                min_side_px[cls].append(min(w * width, h * height))
            boxes_per_image.append(n_boxes)

    duplicates = [tags for tags in hashes.values() if len(tags) > 1]
    cross_split = [tags for tags in duplicates if len({t.split("/")[0] for t in tags}) > 1]

    # --- Informe ---------------------------------------------------------
    out = []
    line = out.append

    line("=" * 78)
    line("EDA MURESANU - Cluj-Napoca, 14 condiciones dentales (etiquetas YOLO)")
    line("=" * 78)
    line(f"Raiz: {root}")
    line("")
    for split in SPLITS:
        line(f"  {split:<8} {n_images[split]:>5} imagenes")
    line(f"  {'TOTAL':<8} {sum(n_images.values()):>5} imagenes")

    line("")
    line("-" * 78)
    line("1. INTEGRIDAD (todo esto deberia ser 0, salvo lo marcado como informativo)")
    line("-" * 78)
    line(f"Imagenes sin .txt de etiquetas:                   {len(missing_label)}")
    line(f".txt sin imagen:                                  {len(orphan_label)}")
    line(f"Imagenes con cabecera ilegible:                   {len(unreadable_img)}")
    line(f"Lineas mal formadas (no son 5 numeros):           {len(malformed)}")
    line(f"Cajas con clase fuera de legend.txt:              {len(unknown_class)}")
    line(f"Cajas con ancho o alto <= 0:                      {len(invalid_boxes)}")
    line(f"Cajas que se salen de la imagen:                  {len(out_of_bounds)}")
    line(f"Imagenes repetidas ENTRE particiones (MD5):       {len(cross_split)}")
    line(f"Imagenes repetidas dentro de una particion (MD5): {len(duplicates) - len(cross_split)}")
    line(f"(informativo) Imagenes con .txt vacio:            {len(empty_labels)}")
    for label, items in (
        ("Sin .txt", missing_label),
        (".txt sin imagen", orphan_label),
        ("Cabecera ilegible", unreadable_img),
        ("Mal formadas", malformed),
        ("Clase desconocida", unknown_class),
        ("Cajas invalidas", invalid_boxes),
        ("Fuera de limites", out_of_bounds),
        ("Repetidas entre particiones", cross_split),
        ("Repetidas (todas)", duplicates),
        (".txt vacio", empty_labels),
    ):
        if items:
            line(f"  {label} (primeros 5): {items[:5]}")

    line("")
    line("-" * 78)
    line("2. CAJAS POR CLASE Y PARTICION (ids segun legend.txt)")
    line("-" * 78)
    header = f"{'id':>2}  {'clase':<24}" + "".join(f"{s:>9}" for s in SPLITS) + f"{'total':>8}{'% cajas':>9}"
    line(header)
    grand_total = sum(sum(c.values()) for c in boxes.values())
    for cls in sorted(class_names):
        per_split = [boxes[s][cls] for s in SPLITS]
        total = sum(per_split)
        line(
            f"{cls:>2}  {class_names[cls]:<24}" + "".join(f"{n:>9}" for n in per_split)
            + f"{total:>8}{100 * total / max(grand_total, 1):>8.1f}%"
        )
    line(f"    {'TOTAL':<24}" + "".join(f"{sum(boxes[s].values()):>9}" for s in SPLITS) + f"{grand_total:>8}")
    line("")
    line("Imagenes (de train+valid+test+externa) que contienen cada clase:")
    all_images = sum(n_images.values())
    for cls in sorted(class_names):
        n = sum(len(images_with[s][cls]) for s in SPLITS)
        line(f"  {class_names[cls]:<24}{n:>6}  ({100 * n / max(all_images, 1):.1f}%)")

    line("")
    line("-" * 78)
    line("3. CAJAS POR IMAGEN")
    line("-" * 78)
    line(summarize(boxes_per_image))

    line("")
    line("-" * 78)
    line("4. TAMANO DE LAS CAJAS, por clase (todas las particiones)")
    line("-" * 78)
    line("Referencia: DENTEX (caja = diente) ~1,1-1,5 % de la imagen, lado menor ~160-190 px;")
    line("            Mendeley (caja = apice) ~0,1 %, lado menor ~56 px.")
    for cls in sorted(class_names):
        if not area_pct[cls]:
            continue
        line(f"{class_names[cls]}:")
        line(f"  % de la imagen que ocupa: {summarize(area_pct[cls])}")
        line(f"  lado menor (px):          {summarize(min_side_px[cls])}")

    line("")
    line("-" * 78)
    line("5. RESOLUCION DE LAS IMAGENES (leida del archivo real)")
    line("-" * 78)
    for (w, h), n in resolutions.most_common(10):
        line(f"  {w} x {h}: {n} imagenes")
    if len(resolutions) > 10:
        line(f"  ... y {len(resolutions) - 10} resoluciones distintas mas")
    line(f"Resoluciones distintas en total: {len(resolutions)}")

    return out


def main():
    parser = argparse.ArgumentParser(description="EDA del dataset de Muresanu et al. (14 clases)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="carpeta con legend.txt y las particiones")
    parser.add_argument("--save-report", type=Path, default=None, help="guardar el informe en este archivo")
    args = parser.parse_args()

    report = analyze(args.root)
    text = "\n".join(report)
    print(text)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(text + "\n", encoding="utf-8")
        print(f"\nInforme guardado en {args.save_report}")


if __name__ == "__main__":
    main()
