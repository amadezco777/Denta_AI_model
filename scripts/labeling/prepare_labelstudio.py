"""Prepara imagenes y sugerencias sin aprobar ninguna etiqueta."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.data.labelstudio import prepare_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path)
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=REPO_ROOT / "configs/data/taxonomy.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--document-root", type=Path, default=REPO_ROOT / "data/labelstudio/media")
    args = parser.parse_args()
    try:
        count = prepare_bundle(args.out, args.taxonomy, images=args.images, dataset=args.dataset, document_root=args.document_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"NO SE IMPORTA: {exc}") from exc
    print(f"{count} tareas listas en {args.out}; importa tasks.json y usa label_config.xml")


if __name__ == "__main__":
    main()
