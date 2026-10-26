"""YAML seguro que rechaza claves repetidas en lugar de pisarlas."""

from pathlib import Path

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Clave YAML repetida: {key!r}, linea {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml(path):
    value = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: se requiere un objeto YAML")
    return value
