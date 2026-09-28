"""Limpia el dataset de Muresanu: corrige el mapa de clases por bloques y deduplica.

PASO 0 - Correccion del mapa de clases (revision del 2026-09-27):
  La revision visual y numerica encontro lotes de imagenes anotados con OTRO
  orden de clases que el de legend.txt. Solo afecta a train y valid, que
  comparten numeracion (valid son los multiplos de 10); test (v1..v100) y
  externa (1..180) tienen numeracion propia y no se tocan.
    - Bloque A (imagenes 101-199): anotado con el orden de "labels (1).txt".
      Se traduce con BLOCK_A_REMAP. Evidencia: tamano y brillo de las cajas,
      revision visual y ausencia total del id 13 (ese orden solo tiene 13 clases).
    - Bloque B (imagenes 1-30): otro orden, deducido solo en parte. Se EXCLUYE.
    - Bloque C (imagenes >= 1392, revision del 2026-09-28): mapa correcto, pero
      anotacion INCOMPLETA: en el 98 % de los casos trae una sola caja por clase
      (un ejemplo de cada hallazgo, no todos; frente al 30 % del resto). Los
      hallazgos sin caja ensenarian al modelo que son fondo. Se EXCLUYE.
  La correccion se aplica ANTES del consenso, porque este compara ids de clase.
  Queda registrada en mapping_log.csv.

Por que: el EDA (reports/muresanu_eda.txt) encontro que 1.808 archivos son en
realidad 1.611 radiografias distintas. Las copias de una misma radiografia
traen anotaciones INDEPENDIENTES que solo coinciden en ~55 % de las cajas
(probablemente, anotadores distintos). Dejarlas asi tiene dos problemas:
  - fuga de datos: 42 radiografias estan a la vez en train y en valid;
  - etiquetas contradictorias: la misma imagen dice "caries" y "no caries".

Que hace (decision del 2026-09-27, "opcion B: consenso"):
  1. Agrupa las imagenes por contenido, no por nombre: primero por MD5 (archivos
     identicos) y despues por pixeles (correlacion de miniaturas >=
     REEXPORT_THRESHOLD, ver src/data/similarity.py). Lo segundo detecta la
     misma radiografia guardada con otra compresion, que el MD5 no ve
     (revision del 2026-09-28: 20 pares, casi todos con una copia del bloque C).
  2. Imagen sin copias: se copia tal cual.
  3. Grupo de copias: se conserva UNA imagen, y solo las cajas en las que
     coinciden al menos MIN_VOTES copias (misma clase, IoU > IOU_THRESHOLD,
     emparejamiento 1 a 1). La caja resultante es el promedio de las que
     coinciden. En un par, eso equivale a "lo que marcaron ambos".
  4. Particion del grupo: la de mayor prioridad entre las de sus copias
     (test > externa > train > valid, ver PARTITION_PRIORITY). Un grupo
     train+valid queda en train y sale de valid.
  5. Descarta las cajas de ancho o alto <= 0 (3 en todo el dataset).
  6. Excluye los grupos cuyo consenso queda vacio (los anotadores no
     coincidieron en nada): como imagen "sin hallazgos" ensenarian algo falso.
     Quedan en el registro CSV con particion_final = EXCLUIDA.

No modifica data/raw/. Escribe en data/interim/muresanu_cluj/ la misma
estructura (particion/images, particion/labels) y dos registros CSV para
auditar: mapping_log.csv (paso 0) y dedup_log.csv (fusiones).

Uso (desde la raiz del repositorio):
    python scripts/dataset_conversion/muresanu_dedup.py
    python scripts/dataset_conversion/muresanu_dedup.py --dry-run   # solo informa, no escribe
"""

import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.data.similarity import REEXPORT_THRESHOLD, connected_groups, similar_pairs, thumbnail  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "data/raw/muresanu_cluj/muresanu_cluj_dataset"
DEFAULT_DST = REPO_ROOT / "data/interim/muresanu_cluj"

SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
    "externa": "test_alte_cabinete/Ext-validation",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

IOU_THRESHOLD = 0.5
MIN_VOTES = 2
# Un grupo con copias en varias particiones va a la que tenga menor numero aqui.
# test y externa van primero para no mover nunca imagenes fuera de ellas
# (en este dataset ningun grupo cruza test o externa con otra particion).
# Entre valid y train se elige train: valid pierde esas imagenes y el modelo
# no se valida con imagenes que vio al entrenar.
PARTITION_PRIORITY = {"test": 0, "externa": 1, "train": 2, "valid": 3}

# Paso 0: bloques con otro mapa de clases (ver docstring).
NUMBERED_SPLITS = {"train", "valid"}
BLOCK_A = range(101, 200)
# id en el archivo -> id correcto segun legend.txt; los ids no listados no cambian.
BLOCK_A_REMAP = {1: 2, 2: 1, 5: 6, 6: 5, 11: 12, 12: 13}
BLOCK_B = range(1, 31)
BLOCK_C_START = 1392


def block_action(split, img):
    """Devuelve "remapear", "excluir" o None para una imagen segun su bloque."""
    if split not in NUMBERED_SPLITS or not img.stem.isdigit():
        return None
    n = int(img.stem)
    if n in BLOCK_B or n >= BLOCK_C_START:
        return "excluir"
    if n in BLOCK_A:
        return "remapear"
    return None


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_boxes(label_path, remap=None):
    """Lee un .txt YOLO y devuelve [(clase, xc, yc, w, h)], sin cajas de tamano 0.

    remap: {id_en_archivo: id_correcto} para las imagenes de un bloque mal mapeado.
    """
    boxes, dropped = [], 0
    if not label_path.is_file():
        return boxes, dropped
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
        if remap:
            cls = remap.get(cls, cls)
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes, dropped


def iou(a, b):
    """IoU entre dos cajas YOLO (clase, xc, yc, w, h)."""
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def consensus(copies, min_votes=MIN_VOTES, threshold=IOU_THRESHOLD):
    """Fusiona las cajas de varias copias de una misma imagen.

    copies: lista de listas de cajas, una lista por copia.
    Cada caja de la primera copia "abre" un cluster y busca, en cada una de las
    demas copias, su mejor pareja de la misma clase con IoU > threshold (cada
    caja solo puede usarse una vez). Las cajas que sobran en copias posteriores
    abren sus propios clusters, para que un grupo de 3 copias pueda aceptar
    algo que marcaron la 2a y la 3a pero no la 1a.
    Se conservan los clusters con al menos min_votes cajas, promediados.
    """
    used = [set() for _ in copies]
    kept = []
    for i, boxes in enumerate(copies):
        for bi, box in enumerate(boxes):
            if bi in used[i]:
                continue
            used[i].add(bi)
            cluster = [box]
            for j in range(i + 1, len(copies)):
                best, best_iou = None, threshold
                for bj, other in enumerate(copies[j]):
                    if bj in used[j] or other[0] != box[0]:
                        continue
                    score = iou(box, other)
                    if score > best_iou:
                        best, best_iou = bj, score
                if best is not None:
                    used[j].add(best)
                    cluster.append(copies[j][best])
            if len(cluster) >= min_votes:
                n = len(cluster)
                kept.append((box[0], *(sum(c[k] for c in cluster) / n for k in range(1, 5))))
    return kept


def write_labels(path, boxes):
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Deduplica Muresanu y fusiona anotaciones por consenso")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--dry-run", action="store_true", help="solo informa, no escribe nada")
    args = parser.parse_args()

    # 1. Agrupar por contenido.
    groups = defaultdict(list)  # md5 -> [(particion, ruta_imagen)]
    for split, rel in SPLITS.items():
        for img in sorted((args.src / rel / "images").iterdir()):
            if img.suffix.lower() in IMAGE_SUFFIXES:
                groups[md5(img)].append((split, img))

    # 1b. Unir los grupos MD5 que son la misma radiografia reexportada.
    keys = list(groups)
    pairs = similar_pairs([thumbnail(groups[k][0][1]) for k in keys], REEXPORT_THRESHOLD)
    pixel_merged = set()
    for comp in connected_groups(len(keys), pairs):
        if len(comp) > 1:
            head = keys[comp[0]]
            for k in comp[1:]:
                groups[head].extend(groups.pop(keys[k]))
            pixel_merged.add(head)

    if not args.dry_run:
        if args.dst.exists() and any(args.dst.iterdir()):
            raise SystemExit(f"{args.dst} no esta vacia: borrala a mano antes de regenerar.")
        for split in SPLITS:
            (args.dst / split / "images").mkdir(parents=True, exist_ok=True)
            (args.dst / split / "labels").mkdir(parents=True, exist_ok=True)

    per_split = defaultdict(int)
    total_boxes_in = total_boxes_out = dropped_zero = 0
    moved = 0
    emptied = []  # grupos en los que el consenso no dejo ninguna caja
    log_rows = []
    mapping_rows = []  # paso 0: una fila por imagen remapeada o excluida

    for digest, members in sorted(groups.items(), key=lambda kv: (kv[1][0][0], kv[1][0][1].name)):
        # Paso 0: registrar la accion de cada copia y quitar las del bloque B.
        # Si una copia excluida tiene gemelas fuera del bloque, el grupo sigue con ellas.
        for split, img in members:
            action = block_action(split, img)
            if action:
                mapping_rows.append({"imagen": f"{split}/{img.name}", "accion": action, "md5": digest})
        members = [m for m in members if block_action(*m) != "excluir"]
        if not members:
            continue

        members.sort(key=lambda m: (PARTITION_PRIORITY[m[0]], m[1].name))
        target_split, keep_img = members[0]
        copies = []
        for split, img in members:
            remap = BLOCK_A_REMAP if block_action(split, img) == "remapear" else None
            boxes, dropped = read_boxes(args.src / SPLITS[split] / "labels" / f"{img.stem}.txt", remap)
            copies.append(boxes)
            dropped_zero += dropped
            total_boxes_in += len(boxes)

        merged = copies[0] if len(members) == 1 else consensus(copies)
        # Si los anotadores no coincidieron en nada, la imagen quedaria vacia y
        # el modelo la aprenderia como "sana" sin serlo: se excluye.
        excluded = len(members) > 1 and not merged
        if excluded:
            emptied.append(keep_img.name)
        else:
            total_boxes_out += len(merged)
            per_split[target_split] += 1
        if len({s for s, _ in members}) > 1:
            moved += 1

        if len(members) > 1:
            log_rows.append({
                "md5": digest,
                "criterio": "pixeles" if digest in pixel_merged else "md5",
                "particion_final": "EXCLUIDA" if excluded else target_split,
                "imagen_final": keep_img.name,
                "copias": " | ".join(f"{s}/{p.name}" for s, p in members),
                "cajas_por_copia": " | ".join(str(len(c)) for c in copies),
                "cajas_consenso": len(merged),
            })

        if not args.dry_run and not excluded:
            out_dir = args.dst / target_split
            shutil.copy2(keep_img, out_dir / "images" / keep_img.name)
            write_labels(out_dir / "labels" / f"{keep_img.stem}.txt", merged)

    if not args.dry_run:
        with open(args.dst / "dedup_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0]) if log_rows else ["md5"])
            writer.writeheader()
            writer.writerows(log_rows)
        with open(args.dst / "mapping_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["imagen", "accion", "md5"])
            writer.writeheader()
            writer.writerows(mapping_rows)
        legend = args.src / "legend.txt"
        if legend.is_file():
            shutil.copy2(legend, args.dst / "legend.txt")

    n_files = sum(len(m) for m in groups.values())
    print(f"Archivos de imagen leidos:          {n_files}")
    print(f"Radiografias distintas (MD5+pix.):  {len(groups)}")
    print(f"  grupos unidos por pixeles:        {len(pixel_merged)}")
    print(f"Paso 0 - archivos remapeados (A):   {sum(r['accion'] == 'remapear' for r in mapping_rows)}")
    print(f"Paso 0 - archivos excluidos (B, C): {sum(r['accion'] == 'excluir' for r in mapping_rows)}")
    print(f"Grupos con copias fusionados:       {len(log_rows)}")
    print(f"  de ellos, entre particiones:      {moved}")
    print(f"Cajas de tamano 0 descartadas:      {dropped_zero}")
    print(f"Cajas leidas (validas):             {total_boxes_in}")
    print(f"Cajas escritas:                     {total_boxes_out}")
    print(f"Grupos excluidos (consenso vacio):  {len(emptied)}"
          + (f"  {emptied}" if emptied else ""))
    print("Imagenes por particion resultante:")
    for split in SPLITS:
        print(f"  {split:<8} {per_split[split]:>5}")
    if args.dry_run:
        print("\n(--dry-run: no se ha escrito nada)")
    else:
        print(f"\nResultado en {args.dst}\nRegistro de fusiones en {args.dst / 'dedup_log.csv'}")


if __name__ == "__main__":
    main()
