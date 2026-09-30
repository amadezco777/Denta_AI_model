"""Construye el dataset maestro YOLO a partir de Muresanu limpio.

Entrada: data/interim/muresanu_cluj/ (salida de muresanu_dedup.py).
Salida:  data/processed/master/ con images/{train,val,test,externa},
         labels/{...}, dataset.yaml (train/val/test), dataset_externa.yaml
         (la prueba externa como val/test) y manifest.csv (trazabilidad).

Reglas (decisiones del usuario del 2026-09-28, ver docs/taxonomy.md):
  1. Particiones. test y externa se conservan TAL CUAL: son las que miden el
     resultado final y permiten compararse con el paper. train y valid se
     juntan y se reparten de nuevo en train / val (VAL_FRACTION), porque el
     valid original era pequeno (107) y tenia un paciente compartido con train.
  2. Reparto por paciente. Las radiografias del mismo paciente en otra visita
     (correlacion de miniaturas >= SAME_PATIENT_THRESHOLD, ver
     src/data/similarity.py) forman un grupo que va entero a train o a val.
     Si un grupo enlaza train/valid con test o externa, el script se detiene:
     seria una fuga hacia la evaluacion.
  3. Estratificacion. Se prueban N_TRIALS repartos aleatorios de grupos y se
     elige el que mas se acerca a VAL_FRACTION a la vez en imagenes y en cajas
     de cada clase, para que val no se lleve una parte desproporcionada de
     ninguna clase.
  4. Clases. Se aplica configs/data/mapping_muresanu_cluj.yaml: cada id de
     origen va a su clase (configs/data/taxonomy.yaml) o se descarta
     (target: null). Un id que no este en el mapeo detiene el script.
  5. Mascara. En TODAS las imagenes se pinta de negro una franja fija en los
     bordes izquierdo, derecho e inferior (MASK), donde van las marcas del
     equipo: "PaX-i", R/L, la linea con fecha y exposicion y el nombre de la
     clinica. Se tapa en todas, no solo en las que tienen marca, para que el
     rectangulo negro no se convierta en otra pista. Ninguna caja de las 7
     clases toca esas franjas (medido el 2026-09-28).

Procedencia. dataset.yaml lleva en meta.provenance el commit de git del codigo
(y si habia cambios sin commitear), la huella SHA-256 de la entrada
(data/interim, mapeo y taxonomia), la huella de la salida (imagenes, etiquetas
y manifest.csv) y las versiones de Python/OpenCV/NumPy/PyYAML. Para comprobar
que un dataset en disco es el registrado:
    python -m src.data.provenance data/processed/master/dataset.yaml
Ver src/data/provenance.py.

No modifica data/interim/. Los nombres de salida llevan el origen:
muresanu_<particion_original>_<nombre>.

Uso (desde la raiz del repositorio):
    python scripts/dataset_conversion/build_master.py
    python scripts/dataset_conversion/build_master.py --dry-run   # solo informa, no escribe
"""

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.data.provenance import DATASET_EXCLUDE, environment, fingerprint, git_state  # noqa: E402
from src.data.similarity import SAME_PATIENT_THRESHOLD, connected_groups, similar_pairs, thumbnail  # noqa: E402

DEFAULT_SRC = REPO_ROOT / "data/interim/muresanu_cluj"
DEFAULT_DST = REPO_ROOT / "data/processed/master"
DEFAULT_MAPPING = REPO_ROOT / "configs/data/mapping_muresanu_cluj.yaml"
DEFAULT_TAXONOMY = REPO_ROOT / "configs/data/taxonomy.yaml"

DATASET = "muresanu"
POOL_SPLITS = ("train", "valid")        # se reparten de nuevo en train / val
FIXED_SPLITS = {"test": "test", "externa": "externa"}  # se conservan tal cual
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

VAL_FRACTION = 0.15
SEED = 42
N_TRIALS = 2000  # repartos aleatorios que se prueban (ver stratified_group_split)
# Fraccion del ancho (izquierda, derecha) o del alto (abajo) que se tapa.
MASK = {"left": 0.03, "right": 0.03, "bottom": 0.04}
JPEG_QUALITY = 95


def load_mapping(mapping_path, taxonomy_path):
    """Devuelve ({id_origen: id_destino o None}, nombres de clase por id)."""
    taxonomy = yaml.safe_load(Path(taxonomy_path).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in taxonomy["names"].items()}
    by_name = {v: k for k, v in names.items()}
    spec = yaml.safe_load(Path(mapping_path).read_text(encoding="utf-8"))
    if spec.get("taxonomy_version") != taxonomy["version"]:
        raise SystemExit(f"El mapeo es de la taxonomia {spec.get('taxonomy_version')}, "
                         f"pero la vigente es {taxonomy['version']}.")
    mapping = {}
    for entry in spec["mapping"]:
        target = entry["target"]
        if target is not None and target not in by_name:
            raise SystemExit(f"Clase destino desconocida en el mapeo: {target!r}")
        mapping[int(entry["source"])] = None if target is None else by_name[target]
    return mapping, names, taxonomy["version"]


def map_boxes(label_path, mapping):
    """Lee un .txt YOLO y traduce sus ids. Devuelve (cajas, Counter de descartadas por id)."""
    boxes, dropped = [], Counter()
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        src_id = int(parts[0])
        if src_id not in mapping:
            raise SystemExit(f"{label_path}: id de clase {src_id} sin entrada en el mapeo.")
        target = mapping[src_id]
        if target is None:
            dropped[src_id] += 1
        else:
            boxes.append((target, *map(float, parts[1:5])))
    return boxes, dropped


def apply_mask(image, mask=MASK):
    """Pinta de negro las franjas de borde (modifica y devuelve la imagen)."""
    h, w = image.shape[:2]
    image[:, : int(mask["left"] * w)] = 0
    image[:, w - int(mask["right"] * w):] = 0
    image[h - int(mask["bottom"] * h):, :] = 0
    return image


def boxes_under_mask(boxes, mask=MASK):
    """Cajas que tocan alguna franja tapada."""
    return [b for b in boxes
            if b[1] - b[3] / 2 < mask["left"]
            or b[1] + b[3] / 2 > 1 - mask["right"]
            or b[2] + b[4] / 2 > 1 - mask["bottom"]]


def stratified_group_split(groups, n_classes, val_fraction=VAL_FRACTION, seed=SEED, n_trials=N_TRIALS):
    """Reparte grupos entre train y val manteniendo la proporcion de cada clase.

    groups: lista de (n_imagenes, Counter{clase: cajas}).
    Devuelve un set con los indices de los grupos que van a val.

    Busqueda aleatoria: se sortean n_trials repartos (grupos barajados, se llenan
    en val hasta alcanzar val_fraction de las imagenes) y se queda el que tiene
    menor desviacion maxima respecto a val_fraction, contando las imagenes y cada
    clase. Un grupo nunca se parte.

    Sustituye al reparto voraz del 2026-09-28, que empezaba por los grupos con
    mas cajas de clases escasas: acertaba el 15 % de imagenes pero se llevaba a
    val el 18-20 % de las cajas (28 % de los implantes), quitandoselas a train.
    """
    total = Counter()
    for _, counts in groups:
        total.update(counts)
    n_images = sum(n for n, _ in groups)
    classes = [c for c in range(n_classes) if total[c] > 0]
    target_images = val_fraction * n_images

    rng = random.Random(seed)
    order = list(range(len(groups)))
    best, best_cost = set(), float("inf")
    for _ in range(n_trials):
        rng.shuffle(order)
        val, val_counts, val_images = set(), Counter(), 0
        for i in order:
            if val_images >= target_images:
                break
            val.add(i)
            val_counts.update(groups[i][1])
            val_images += groups[i][0]
        cost = max([abs(val_images / n_images - val_fraction)]
                   + [abs(val_counts[c] / total[c] - val_fraction) for c in classes])
        if cost < best_cost:
            best, best_cost = val, cost
    return best


def fingerprint_file(path):
    """SHA-256 (hex) de un solo archivo."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(src):
    """Lista de dicts {split, image, label} de data/interim."""
    items = []
    for split in (*POOL_SPLITS, *FIXED_SPLITS):
        for img in sorted((src / split / "images").iterdir()):
            if img.suffix.lower() in IMAGE_SUFFIXES:
                items.append({"split": split, "image": img,
                              "label": src / split / "labels" / f"{img.stem}.txt"})
    return items


def main():
    parser = argparse.ArgumentParser(description="Construye el dataset maestro YOLO desde Muresanu limpio")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="solo informa, no escribe nada")
    args = parser.parse_args()

    git = git_state(REPO_ROOT)
    if git is None or git["dirty"]:
        print("AVISO: hay cambios sin commitear (o no hay git). El dataset quedara marcado como\n"
              "       no reproducible desde un commit (meta.provenance.code.dirty).\n")

    mapping, names, version = load_mapping(args.mapping, args.taxonomy)
    items = collect(args.src)
    dropped = Counter()
    under_mask = 0
    for it in items:
        it["boxes"], d = map_boxes(it["label"], mapping)
        dropped.update(d)
        under_mask += len(boxes_under_mask(it["boxes"]))

    # Grupos de paciente sobre TODAS las imagenes, para detectar tambien fugas hacia test/externa.
    pairs = similar_pairs([thumbnail(it["image"]) for it in items], SAME_PATIENT_THRESHOLD)
    groups = connected_groups(len(items), pairs)
    leaks = []
    for g in groups:
        kinds = {"pool" if items[k]["split"] in POOL_SPLITS else items[k]["split"] for k in g}
        if len(kinds) > 1:
            leaks.append(sorted(f"{items[k]['split']}/{items[k]['image'].name}" for k in g))
    if leaks:
        raise SystemExit("Mismo paciente en particiones de evaluacion distintas (fuga):\n"
                         + "\n".join("  " + " ~ ".join(g) for g in leaks))

    pool_ids = [gi for gi, g in enumerate(groups) if items[g[0]]["split"] in POOL_SPLITS]
    pool_groups = [groups[gi] for gi in pool_ids]
    summary = [(len(g), Counter(b[0] for k in g for b in items[k]["boxes"])) for g in pool_groups]
    val_groups = {pool_ids[i] for i in stratified_group_split(summary, len(names), args.val_fraction, args.seed)}
    for gi, g in enumerate(groups):
        for k in g:
            it = items[k]
            it["group"] = gi
            if it["split"] in POOL_SPLITS:
                it["master"] = "val" if gi in val_groups else "train"
            else:
                it["master"] = FIXED_SPLITS[it["split"]]
            it["out"] = f"{DATASET}_{it['split']}_{it['image'].stem}"

    # Informe.
    per_split = {s: Counter() for s in ("train", "val", "test", "externa")}
    n_images = Counter()
    for it in items:
        n_images[it["master"]] += 1
        per_split[it["master"]].update(b[0] for b in it["boxes"])
    multi = [g for g in pool_groups if len(g) > 1]
    print(f"Imagenes leidas:                    {len(items)}")
    print(f"Grupos de paciente con >1 imagen:   {len(multi)} ({sum(len(g) for g in multi)} imagenes)")
    print(f"Cajas descartadas por el mapeo:     {sum(dropped.values())}  {dict(sorted(dropped.items()))}")
    print(f"Cajas conservadas bajo la mascara:  {under_mask}")
    header = "".join(f"{names[c][:12]:>13}" for c in names)
    print(f"\n{'particion':<9}{'imagenes':>9}{header}")
    for s, counts in per_split.items():
        print(f"{s:<9}{n_images[s]:>9}" + "".join(f"{counts[c]:>13}" for c in names))
    pool_total = per_split["train"] + per_split["val"]
    print(f"{'val %':<9}{100 * n_images['val'] / (n_images['train'] + n_images['val']):>8.1f}%"
          + "".join(f"{100 * per_split['val'][c] / max(1, pool_total[c]):>12.1f}%" for c in names))

    if args.dry_run:
        print("\n(--dry-run: no se ha escrito nada)")
        return

    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"{args.dst} no esta vacia: borrala a mano antes de regenerar.")
    for s in per_split:
        (args.dst / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / s).mkdir(parents=True, exist_ok=True)

    rows = []
    for it in items:
        suffix = it["image"].suffix.lower()
        image = cv2.imread(str(it["image"]), cv2.IMREAD_GRAYSCALE)
        out_img = args.dst / "images" / it["master"] / f"{it['out']}{suffix}"
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if suffix in {".jpg", ".jpeg"} else []
        if not cv2.imwrite(str(out_img), apply_mask(image), params):
            raise SystemExit(f"No se pudo escribir {out_img}")
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in it["boxes"]]
        (args.dst / "labels" / it["master"] / f"{it['out']}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        rows.append({"archivo": out_img.name, "particion": it["master"],
                     "origen": f"{it['split']}/{it['image'].name}", "grupo_paciente": it["group"],
                     "cajas": len(it["boxes"])})

    with open(args.dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "code": git,
        "input_fingerprint": {"interim": fingerprint(args.src),
                              "mapping": fingerprint_file(args.mapping),
                              "taxonomy": fingerprint_file(args.taxonomy)},
        # Se calcula antes de escribir los yaml, que quedan fuera (DATASET_EXCLUDE).
        "output_fingerprint": fingerprint(args.dst, DATASET_EXCLUDE),
        "environment": environment(),
    }
    meta = {"taxonomy_version": version, "built_at": date.today().isoformat(),
            "source_datasets": ["muresanu_cluj"], "val_fraction": args.val_fraction, "seed": args.seed,
            "mask": MASK, "split_strategy": "test y externa originales; train/val por grupo de paciente, estratificado",
            "provenance": provenance}
    # Ultralytics resuelve un "path" relativo contra su propia carpeta de datasets,
    # no contra el yaml: se escribe la ruta absoluta.
    base = {"path": str(args.dst.resolve()), "names": names, "meta": meta}
    for fname, splits in (("dataset.yaml", {"train": "images/train", "val": "images/val", "test": "images/test"}),
                          ("dataset_externa.yaml", {"train": "images/train", "val": "images/externa",
                                                    "test": "images/externa"})):
        (args.dst / fname).write_text(yaml.safe_dump({**base, **splits}, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    print(f"\nResultado en {args.dst}")


if __name__ == "__main__":
    main()
