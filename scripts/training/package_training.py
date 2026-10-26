"""Empaqueta/restaura una corrida aprobada para WSL, Colab o una GPU alquilada."""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.portable import package_training, restore_training  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("create")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    restore = actions.add_parser("restore")
    restore.add_argument("--bundle", type=Path, required=True)
    restore.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(package_training(args.config, args.out) if args.action == "create" else restore_training(args.bundle, args.out))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"NO SE TRASLADA: {exc}") from exc
