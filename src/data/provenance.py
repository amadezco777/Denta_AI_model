"""Procedencia de un dataset generado: de que codigo y de que datos sale.

Sirve para responder, meses despues, "¿este dataset (o este modelo) sale
exactamente de estos datos y de este codigo?". build_master.py lo escribe en el
bloque meta.provenance de dataset.yaml; el entrenamiento puede recalcular la
huella del dataset y compararla antes de empezar.

Huella de una carpeta (fingerprint): SHA-256 sobre la lista ORDENADA de
(ruta relativa, contenido) de sus archivos. Cambia si cambia un byte, si se
renombra, anade o borra un archivo; no depende del orden en que el sistema
de archivos los devuelva ni de las fechas de modificacion.
"""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import yaml

CHUNK = 1 << 20
# Paquetes cuya version puede cambiar el resultado (codificacion JPEG,
# miniaturas para detectar copias, lectura de YAML).
TRACKED_PACKAGES = ("opencv-python", "numpy", "pyyaml")


def fingerprint(root, exclude_suffixes=()):
    """SHA-256 (hex) del contenido de `root`, recorriendo sus archivos en orden.

    exclude_suffixes: sufijos a ignorar (p. ej. los .cache que crea Ultralytics).
    """
    root = Path(root)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and not p.name.endswith(tuple(exclude_suffixes)))
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix().encode("utf-8")
        # Longitud antes de la ruta: evita que dos listas distintas de
        # (ruta, contenido) produzcan el mismo flujo de bytes.
        h.update(len(rel).to_bytes(8, "big") + rel)
        h.update(p.stat().st_size.to_bytes(8, "big"))
        with open(p, "rb") as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)
    return h.hexdigest()


def git_state(repo_root):
    """{"commit": sha, "dirty": bool}, o None si no es un repo git o no hay git.

    dirty = hay cambios sin commitear en archivos versionados (los datos,
    ignorados por .gitignore, no cuentan).
    """
    def run(*args):
        return subprocess.run(["git", "-C", str(repo_root), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": dirty}


def environment():
    """Versiones de Python y de los paquetes que pueden cambiar el resultado."""
    versions = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


# Archivos de un dataset maestro que NO entran en su huella: los yaml (llevan
# la ruta absoluta, la fecha y la propia huella) y los .cache de Ultralytics.
DATASET_EXCLUDE = (".yaml", ".cache")


def verify_dataset(yaml_path):
    """Recalcula la huella del dataset y la compara con la de su yaml.

    Devuelve (coincide, huella_registrada, huella_actual).
    """
    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("meta", {})
    expected = meta.get("provenance", {}).get("output_fingerprint")
    actual = fingerprint(yaml_path.parent, DATASET_EXCLUDE)
    return expected == actual, expected, actual


if __name__ == "__main__":
    # Uso: python -m src.data.provenance data/processed/master/dataset.yaml
    import sys

    ok, expected, actual = verify_dataset(sys.argv[1])
    print(f"registrada: {expected}\nactual:     {actual}")
    print("OK: el dataset coincide con el registrado." if ok
          else "DISTINTO: el dataset no es el que registra su yaml.")
    sys.exit(0 if ok else 1)
