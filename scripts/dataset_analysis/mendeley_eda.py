"""Analisis exploratorio (EDA) del dataset de Mendeley de lesiones periapicales.

Que hace: lee los XML de anotacion (Pascal VOC) de las imagenes ORIGINALES y
responde, sin modificar nada:
  1. Hay algo roto? (imagenes sin XML, XML ilegibles, cajas invalidas, tamanos
     declarados que no coinciden con el JPG real, nombres de origen repetidos)
  2. Cuantas cajas hay de cada valor de <name> (3, 4, 5)?
  3. Cuantas cajas por imagen, y de que tamano son?
  4. De que resolucion son las imagenes?

Solo usa la libreria estandar de Python: no hace falta instalar nada.

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/mendeley_eda.py
    python scripts/dataset_analysis/mendeley_eda.py --save-report reports/mendeley_eda.txt

Notas sobre el formato:
  - Pascal VOC: un XML por imagen, cajas en pixeles absolutos
    (xmin, ymin, xmax, ymax). Ancho = xmax - xmin.
  - La imagen y su XML se emparejan por nombre de archivo (00011.jpg <-> 00011.xml).
    El campo <filename> del XML NO sirve para esto: guarda el nombre original
    del autor (p. ej. PN0004.jpg). Se usa solo para detectar duplicados.
  - <name> solo deberia valer 3, 4 o 5 (hipotesis: indice PAI). Para el
    proyecto todas las cajas son `lesion_periapical`; se desglosan por valor
    solo para conocer el dataset.
  - "Image Annots" contiene tambien los XML de las imagenes aumentadas; aqui se
    ignoran. Las aumentadas no se usan (serian fuga de datos entre particiones).
"""

import argparse
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data/raw/mendeley_periapical/Periapical Lesions"
DEFAULT_IMAGES = DEFAULT_ROOT / "Original JPG Images"
DEFAULT_ANNOTS = DEFAULT_ROOT / "Image Annots"

EXPECTED_LABELS = {"3", "4", "5"}

# Una caja que se sale de la imagen por menos de esto (en pixeles) se tolera.
BOUNDS_TOLERANCE_PX = 1.0

# Marcadores SOF de JPEG que llevan alto y ancho (excluye DHT, JPG y DAC).
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path):
    """Devuelve (ancho, alto) leyendo la cabecera del JPEG, o None si no puede."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":  # SOI: todo JPEG empieza asi
            return None
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            while byte == b"\xff":  # puede haber bytes de relleno 0xFF
                byte = f.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue  # marcadores sin longitud
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                return None
            length = int.from_bytes(length_bytes, "big")
            if marker in _SOF_MARKERS:
                data = f.read(5)  # precision (1) + alto (2) + ancho (2)
                if len(data) < 5:
                    return None
                return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
            f.seek(length - 2, 1)


def summarize(values):
    """Devuelve 'min / mediana / media / max' de una lista de numeros."""
    if not values:
        return "sin datos"
    return (
        f"min {min(values):.1f} / mediana {statistics.median(values):.1f} / "
        f"media {statistics.mean(values):.1f} / max {max(values):.1f}"
    )


def analyze(images_dir, annots_dir):
    """Recorre las imagenes originales una sola vez y devuelve las lineas del informe."""
    image_paths = sorted(images_dir.glob("*.jpg")) if images_dir.is_dir() else []

    # --- Acumuladores ---------------------------------------------------
    boxes_per_label = Counter()
    images_per_label = defaultdict(set)
    boxes_per_image = {}
    area_pct_per_label = defaultdict(list)   # % de la imagen que ocupa la caja
    min_side_per_label = defaultdict(list)   # lado menor de la caja, en px
    resolutions = Counter()
    source_names = defaultdict(list)         # <filename> -> imagenes que lo usan

    missing_xml = []        # imagen sin su XML
    unreadable_xml = []     # XML que no se puede parsear
    unreadable_jpg = []     # no se pudo leer el tamano de la cabecera
    size_mismatch = []      # <size> del XML distinto al tamano real del JPG
    unknown_label = []      # <name> fuera de 3/4/5
    invalid_boxes = []      # ancho o alto <= 0, o coordenadas no numericas
    out_of_bounds = []      # caja fuera de los limites de la imagen

    for img_path in image_paths:
        stem = img_path.stem
        xml_path = annots_dir / f"{stem}.xml"
        if not xml_path.is_file():
            missing_xml.append(img_path.name)
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            unreadable_xml.append(xml_path.name)
            continue

        real = jpeg_size(img_path)
        if real is None:
            unreadable_jpg.append(img_path.name)
            continue
        width, height = real
        resolutions[real] += 1

        try:
            declared = (int(root.findtext("size/width")), int(root.findtext("size/height")))
        except (TypeError, ValueError):
            declared = None
        if declared != real:
            size_mismatch.append(f"{img_path.name} xml={declared} real={real}")

        source = root.findtext("filename")
        if source:
            source_names[source].append(img_path.name)

        n_boxes = 0
        for obj in root.iter("object"):
            label = (obj.findtext("name") or "").strip()
            if label not in EXPECTED_LABELS:
                unknown_label.append(f"{img_path.name}:{label!r}")
                continue
            try:
                xmin, ymin, xmax, ymax = (
                    float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")
                )
            except (TypeError, ValueError):
                invalid_boxes.append(img_path.name)
                continue
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                invalid_boxes.append(img_path.name)
                continue
            tol = BOUNDS_TOLERANCE_PX
            if xmin < -tol or ymin < -tol or xmax > width + tol or ymax > height + tol:
                out_of_bounds.append(img_path.name)

            n_boxes += 1
            boxes_per_label[label] += 1
            images_per_label[label].add(img_path.name)
            area_pct_per_label[label].append(100 * (w * h) / (width * height))
            min_side_per_label[label].append(min(w, h))
        boxes_per_image[img_path.name] = n_boxes

    images_without_boxes = [name for name, n in boxes_per_image.items() if n == 0]
    duplicated_sources = {s: imgs for s, imgs in source_names.items() if len(imgs) > 1}

    # --- Informe ---------------------------------------------------------
    out = []
    line = out.append

    line("=" * 70)
    line("EDA MENDELEY - lesiones periapicales (solo imagenes originales)")
    line("=" * 70)
    line(f"Imagenes:    {images_dir}")
    line(f"Anotaciones: {annots_dir}")
    line("")
    line(f"JPG originales en disco: {len(image_paths)}")
    line(f"Imagenes analizadas (con XML y JPG legibles): {len(boxes_per_image)}")

    line("")
    line("-" * 70)
    line("1. INTEGRIDAD (todo esto deberia ser 0)")
    line("-" * 70)
    line(f"Imagenes sin XML:                                 {len(missing_xml)}")
    line(f"XML ilegibles:                                    {len(unreadable_xml)}")
    line(f"JPG con cabecera ilegible:                        {len(unreadable_jpg)}")
    line(f"Tamano en el XML distinto del JPG real:           {len(size_mismatch)}")
    line(f"Imagenes sin ninguna caja:                        {len(images_without_boxes)}")
    line(f"Cajas con <name> fuera de {sorted(EXPECTED_LABELS)}:          {len(unknown_label)}")
    line(f"Cajas con ancho o alto <= 0 o no numericas:       {len(invalid_boxes)}")
    line(f"Cajas que se salen de la imagen:                  {len(out_of_bounds)}")
    line(f"Nombres de origen (<filename>) repetidos:         {len(duplicated_sources)}")
    if not images_dir.is_dir():
        line(f"AVISO: no existe la carpeta de imagenes {images_dir}")
    for label, items in (
        ("Sin XML", missing_xml),
        ("XML ilegibles", unreadable_xml),
        ("JPG ilegibles", unreadable_jpg),
        ("Tamano distinto", size_mismatch),
        ("Sin cajas", images_without_boxes),
        ("<name> inesperado", unknown_label),
        ("Cajas invalidas", invalid_boxes),
        ("Fuera de limites", out_of_bounds),
        ("Origen repetido", [f"{s} -> {i}" for s, i in duplicated_sources.items()]),
    ):
        if items:
            line(f"  {label} (primeros 5): {items[:5]}")

    line("")
    line("-" * 70)
    line("2. CAJAS POR VALOR DE <name> (todas mapean a lesion_periapical)")
    line("-" * 70)
    n_images = len(boxes_per_image)
    total_boxes = sum(boxes_per_label.values())
    line(f"{'name':<6}{'cajas':>7}{'% cajas':>9}{'imagenes':>10}{'% imag.':>9}")
    for label in sorted(EXPECTED_LABELS):
        n = boxes_per_label[label]
        n_img = len(images_per_label[label])
        line(
            f"{label:<6}{n:>7}{100 * n / max(total_boxes, 1):>8.1f}%"
            f"{n_img:>10}{100 * n_img / max(n_images, 1):>8.1f}%"
        )
    line(f"{'TOTAL':<6}{total_boxes:>7}")

    line("")
    line("-" * 70)
    line("3. CAJAS POR IMAGEN")
    line("-" * 70)
    per_image = list(boxes_per_image.values())
    line(summarize(per_image))
    for n, count in sorted(Counter(per_image).items()):
        line(f"  {n} caja(s): {count} imagenes")

    line("")
    line("-" * 70)
    line("4. TAMANO DE LAS CAJAS")
    line("-" * 70)
    all_area = [v for vs in area_pct_per_label.values() for v in vs]
    all_side = [v for vs in min_side_per_label.values() for v in vs]
    line("Todas (lesion_periapical):")
    line(f"  % de la imagen que ocupa: {summarize(all_area)}")
    line(f"  lado menor (px):          {summarize(all_side)}")
    for label in sorted(EXPECTED_LABELS):
        if boxes_per_label[label] == 0:
            continue
        line(f"<name> = {label}:")
        line(f"  % de la imagen que ocupa: {summarize(area_pct_per_label[label])}")
        line(f"  lado menor (px):          {summarize(min_side_per_label[label])}")

    line("")
    line("-" * 70)
    line("5. RESOLUCION DE LAS IMAGENES (leida del JPG real)")
    line("-" * 70)
    for (w, h), n in resolutions.most_common(10):
        line(f"  {w} x {h}: {n} imagenes")
    if len(resolutions) > 10:
        line(f"  ... y {len(resolutions) - 10} resoluciones distintas mas")
    line(f"Resoluciones distintas en total: {len(resolutions)}")

    return out


def main():
    parser = argparse.ArgumentParser(description="EDA del dataset de Mendeley (lesiones periapicales)")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="carpeta con los JPG originales")
    parser.add_argument("--annots", type=Path, default=DEFAULT_ANNOTS, help="carpeta con los XML")
    parser.add_argument("--save-report", type=Path, default=None, help="guardar el informe en este archivo")
    args = parser.parse_args()

    report = analyze(args.images, args.annots)
    text = "\n".join(report)
    print(text)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(text + "\n", encoding="utf-8")
        print(f"\nInforme guardado en {args.save_report}")


if __name__ == "__main__":
    main()
