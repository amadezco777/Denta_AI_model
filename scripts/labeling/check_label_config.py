"""Valida el XML con el esquema del servidor Label Studio instalado."""

import argparse
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    os.environ["DEBUG"] = "false"
    os.environ["LABEL_STUDIO_COLLECT_ANALYTICS"] = "false"
    os.environ["LABEL_STUDIO_SENTRY_DSN"] = ""
    os.environ["LABEL_STUDIO_FRONTEND_SENTRY_DSN"] = ""
    import label_studio

    sys.path.insert(0, str(Path(label_studio.__file__).parent))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "label_studio.core.settings.label_studio")
    from label_studio.core.label_config import validate_label_config

    validate_label_config(args.config.read_text(encoding="utf-8"))
    print("OK: esquema Label Studio valido")
