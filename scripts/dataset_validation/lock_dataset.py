"""Crea o verifica un contrato externo; no aprueba revisiones automaticamente."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.contract import create_contract, verify_contract, write_review_template  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--template", type=Path, help="CSV para completar durante la revision")
    action.add_argument("--create", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    try:
        if args.template:
            write_review_template(args.yaml, args.template)
            print(f"Plantilla pendiente: {args.template}")
        elif args.create:
            if not args.lock or not args.review:
                parser.error("--create requiere --lock y --review")
            print(f"Contrato: {create_contract(args.yaml, args.lock, args.review)}")
        else:
            if not args.lock:
                parser.error("--verify requiere --lock")
            ok, expected, actual = verify_contract(args.yaml, args.lock)
            print(f"Registrada: {expected}\nActual: {actual}")
            if not ok:
                raise ValueError("El dataset o su configuracion han cambiado")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
