"""Pruebas minimas de integridad del repositorio.

Verifican que la estructura de carpetas y la documentacion base siguen en su
sitio. Sirven sobre todo para confirmar que pytest esta bien configurado desde
el primer dia; las pruebas reales llegan con el codigo de datos y modelo.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "configs/data",
    "configs/training",
    "data/raw",
    "data/external",
    "data/interim",
    "data/processed",
    "docs",
    "models/checkpoints",
    "models/exports",
    "notebooks",
    "reports/figures",
    "scripts/dataset_analysis",
    "scripts/dataset_conversion",
    "scripts/dataset_validation",
    "src/data",
    "src/training",
    "src/evaluation",
    "src/inference",
    "src/utils",
]

EXPECTED_DOCS = [
    "docs/dataset_registry.md",
    "docs/taxonomy.md",
    "docs/glossary.md",
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
]


@pytest.mark.parametrize("relative_path", EXPECTED_DIRS)
def test_expected_directory_exists(relative_path):
    assert (ROOT / relative_path).is_dir(), f"Falta el directorio {relative_path}"


@pytest.mark.parametrize("relative_path", EXPECTED_DOCS)
def test_expected_file_exists(relative_path):
    assert (ROOT / relative_path).is_file(), f"Falta el archivo {relative_path}"


def test_src_is_importable():
    import src  # noqa: F401
