"""Analisis exploratorio (EDA) del subconjunto de diagnostico de DENTEX.

Que hace: lee el JSON de anotaciones y responde, sin modificar nada:
  1. Hay algo roto? (archivos que faltan, cajas invalidas, ids desconocidos)
  2. Cuantas cajas hay de cada clase de diagnostico?
  3. Cuantas cajas por imagen, y de que tamano son?
  4. De que resolucion son las imagenes?

Solo usa la libreria estandar de Python: no hace falta instalar nada.

Uso (desde la raiz del repositorio):
    python scripts/dataset_analysis/dentex_eda.py
    python scripts/dataset_analysis/dentex_eda.py --save-report reports/dentex_eda.txt

Nota sobre el formato: el JSON de DENTEX es "COCO ampliado". Cada anotacion trae
tres ids de categoria: category_id_1 (cuadrante), category_id_2 (numero de
diente) y category_id_3 (diagnostico). Para este proyecto solo nos interesa el
diagnostico, es decir, category_id_3.
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET = (
    REPO_ROOT
    / "data/raw/dentex/training_data/training_data/quadrant-enumeration-disease"
)
DEFAULT_JSON = DEFAULT_SUBSET / "train_quadrant_enumeration_disease.json"
DEFAULT_IMAGES = DEFAULT_SUBSET / "xrays"

# Una caja que se sale de la imagen por menos de esto (en pixeles) se tolera.
BOUNDS_TOLERANCE_PX = 1.0


def summarize(values):
    """Devuelve 'min / mediana / media / max' de una lista de numeros."""
    if not values:
        return "sin datos"
    return (
        f"min {min(values):.1f} / mediana {statistics.median(values):.1f} / "
        f"media {statistics.mean(values):.1f} / max {max(values):.1f}"
    )


def analyze(json_path, images_dir):
    """Recorre el dataset una sola vez y devuelve las lineas del informe."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]
    # category_id_3 es el diagnostico; su lista de nombres esta en categories_3.
    class_names = {c["id"]: c["name"] for c in data["categories_3"]}

    # --- Acumuladores ---------------------------------------------------
    boxes_per_class = Counter()
    images_per_class = defaultdict(set)
    boxes_per_image = Counter()
    area_pct_per_class = defaultdict(list)   # % de la imagen que ocupa la caja
    min_side_per_class = defaultdict(list)   # lado menor de la caja, en px

    orphan_annotations = []      # apuntan a una imagen que no existe en el JSON
    unknown_class = []           # category_id_3 ausente o fuera de categories_3
    invalid_boxes = []           # ancho o alto <= 0
    out_of_bounds = []           # caja fuera de los limites de la imagen

    for ann in annotations:
        img = images.get(ann["image_id"])
        if img is None:
            orphan_annotations.append(ann["id"])
            continue

        class_id = ann.get("category_id_3")
        if class_id not in class_names:
            unknown_class.append(ann["id"])
            continue

        x, y, w, h = ann["bbox"]  # COCO: [x, y, ancho, alto] en pixeles
        if w <= 0 or h <= 0:
            invalid_boxes.append(ann["id"])
            continue
        tol = BOUNDS_TOLERANCE_PX
        if (
            x < -tol
            or y < -tol
            or x + w > img["width"] + tol
            or y + h > img["height"] + tol
        ):
            out_of_bounds.append(ann["id"])

        boxes_per_class[class_id] += 1
        images_per_class[class_id].add(img["id"])
        boxes_per_image[img["id"]] += 1
        area_pct_per_class[class_id].append(100 * (w * h) / (img["width"] * img["height"]))
        min_side_per_class[class_id].append(min(w, h))

    # --- Comprobaciones contra el disco ----------------------------------
    on_disk = {p.name for p in images_dir.glob("*.png")} if images_dir.is_dir() else set()
    in_json = {img["file_name"] for img in images.values()}
    missing_files = sorted(in_json - on_disk)
    extra_files = sorted(on_disk - in_json)
    images_without_boxes = [
        img["file_name"] for i, img in images.items() if boxes_per_image[i] == 0
    ]

    # --- Informe ---------------------------------------------------------
    out = []
    line = out.append

    line("=" * 70)
    line("EDA DENTEX - subconjunto quadrant-enumeration-disease")
    line("=" * 70)
    line(f"JSON:      {json_path}")
    line(f"Imagenes:  {images_dir}")
    line("")
    line(f"Imagenes en el JSON: {len(images)}")
    line(f"Anotaciones en el JSON: {len(annotations)}")

    line("")
    line("-" * 70)
    line("1. INTEGRIDAD (todo esto deberia ser 0)")
    line("-" * 70)
    line(f"Imagenes del JSON que NO estan en disco:          {len(missing_files)}")
    line(f"PNG en disco que NO estan en el JSON:             {len(extra_files)}")
    line(f"Imagenes sin ninguna caja de diagnostico:         {len(images_without_boxes)}")
    line(f"Anotaciones que apuntan a una imagen inexistente: {len(orphan_annotations)}")
    line(f"Anotaciones con clase desconocida:                {len(unknown_class)}")
    line(f"Cajas con ancho o alto <= 0:                      {len(invalid_boxes)}")
    line(f"Cajas que se salen de la imagen:                  {len(out_of_bounds)}")
    if not images_dir.is_dir():
        line(f"AVISO: no existe la carpeta de imagenes {images_dir}")
    for label, items in (
        ("Faltan en disco", missing_files),
        ("Sobran en disco", extra_files),
        ("Sin cajas", images_without_boxes),
        ("Fuera de limites (ids de anotacion)", out_of_bounds),
    ):
        if items:
            line(f"  {label} (primeros 5): {items[:5]}")

    line("")
    line("-" * 70)
    line("2. CAJAS POR CLASE (category_id_3)")
    line("-" * 70)
    total_boxes = sum(boxes_per_class.values())
    line(f"{'id':>2}  {'clase':<20}{'cajas':>7}{'% cajas':>9}{'imagenes':>10}{'% imag.':>9}")
    for class_id in sorted(class_names):
        n = boxes_per_class[class_id]
        n_img = len(images_per_class[class_id])
        line(
            f"{class_id:>2}  {class_names[class_id]:<20}{n:>7}"
            f"{100 * n / max(total_boxes, 1):>8.1f}%{n_img:>10}"
            f"{100 * n_img / max(len(images), 1):>8.1f}%"
        )
    line(f"    {'TOTAL':<20}{total_boxes:>7}")
    counts = [boxes_per_class[c] for c in class_names if boxes_per_class[c] > 0]
    if len(counts) > 1:
        line(f"Desbalance (clase mayor / clase menor): {max(counts) / min(counts):.1f}x")

    line("")
    line("-" * 70)
    line("3. CAJAS POR IMAGEN")
    line("-" * 70)
    line(summarize([boxes_per_image[i] for i in images]))

    line("")
    line("-" * 70)
    line("4. TAMANO DE LAS CAJAS, por clase")
    line("-" * 70)
    for class_id in sorted(class_names):
        if boxes_per_class[class_id] == 0:
            continue
        line(f"{class_names[class_id]}:")
        line(f"  % de la imagen que ocupa: {summarize(area_pct_per_class[class_id])}")
        line(f"  lado menor (px):          {summarize(min_side_per_class[class_id])}")

    line("")
    line("-" * 70)
    line("5. RESOLUCION DE LAS IMAGENES")
    line("-" * 70)
    resolutions = Counter((img["width"], img["height"]) for img in images.values())
    for (w, h), n in resolutions.most_common(10):
        line(f"  {w} x {h}: {n} imagenes")
    if len(resolutions) > 10:
        line(f"  ... y {len(resolutions) - 10} resoluciones distintas mas")
    line(f"Resoluciones distintas en total: {len(resolutions)}")
    line("(Este dato sale del JSON; aun no se ha comprobado contra el tamano real del PNG.)")

    return out


def main():
    parser = argparse.ArgumentParser(description="EDA del subconjunto de diagnostico de DENTEX")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="ruta al JSON de anotaciones")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="carpeta con los PNG")
    parser.add_argument("--save-report", type=Path, default=None, help="guardar el informe en este archivo")
    args = parser.parse_args()

    report = analyze(args.json, args.images)
    text = "\n".join(report)
    print(text)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(text + "\n", encoding="utf-8")
        print(f"\nInforme guardado en {args.save_report}")


if __name__ == "__main__":
    main()
