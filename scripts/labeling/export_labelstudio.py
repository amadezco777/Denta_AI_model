"""Convierte JSON completo revisado a YOLO detect y un CSV de revision."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.data.labelstudio import export_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--annotation-ids", type=Path, help="JSON sample_id -> id elegido tras adjudicacion")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        selections = json.loads(args.annotation_ids.read_text(encoding="utf-8")) if args.annotation_ids else None
        count = export_bundle(args.bundle, args.export, args.out, groups_path=args.groups, annotation_ids=selections,
                              val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"NO SE EXPORTA: {exc}") from exc
    print(f"{count} revisiones exportadas a {args.out}; verificar grupos y crear contrato externo antes de entrenar")


if __name__ == "__main__":
    main()
