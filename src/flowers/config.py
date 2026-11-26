"""Rutas y clases del corpus de flores.

Las rutas se leen en cada acceso para que las variables de entorno
puedan cambiarlas sin reiniciar el proceso.
"""

from __future__ import annotations

import os
from pathlib import Path

CLASSES: list[str] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

_DEFAULT_IMAGES_DIR = "~/Downloads/archive/train"
_DEFAULT_DOCS_DIR = "~/Downloads/documentos"
_DEFAULT_ARTIFACTS_DIR = "./artifacts"


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def images_dir() -> Path:
    """Directorio de imágenes. Variable: FLOWERS_IMAGES_DIR."""
    return _path_from_env("FLOWERS_IMAGES_DIR", _DEFAULT_IMAGES_DIR)


def docs_dir() -> Path:
    """Directorio de documentos. Variable: FLOWERS_DOCS_DIR."""
    return _path_from_env("FLOWERS_DOCS_DIR", _DEFAULT_DOCS_DIR)


def artifacts_dir() -> Path:
    """Directorio de artefactos. Variable: ARTIFACTS_DIR."""
    return _path_from_env("ARTIFACTS_DIR", _DEFAULT_ARTIFACTS_DIR)
