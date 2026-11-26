"""Pruebas de rutas y clases en flowers.config."""

from pathlib import Path

import pytest

from flowers.config import CLASSES, artifacts_dir, docs_dir, images_dir


def test_classes_son_las_cinco_flores() -> None:
    assert CLASSES == ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def test_rutas_por_defecto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOWERS_IMAGES_DIR", raising=False)
    monkeypatch.delenv("FLOWERS_DOCS_DIR", raising=False)
    monkeypatch.delenv("ARTIFACTS_DIR", raising=False)

    assert images_dir() == Path.home() / "Downloads" / "archive" / "train"
    assert docs_dir() == Path.home() / "Downloads" / "documentos"
    assert artifacts_dir() == Path("./artifacts")


def test_variables_de_entorno_pisan_el_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    docs = tmp_path / "docs"
    artifacts = tmp_path / "out"
    monkeypatch.setenv("FLOWERS_IMAGES_DIR", str(images))
    monkeypatch.setenv("FLOWERS_DOCS_DIR", str(docs))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))

    assert images_dir() == images
    assert docs_dir() == docs
    assert artifacts_dir() == artifacts


def test_tilde_se_expande(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOWERS_IMAGES_DIR", "~/corpus/train")
    monkeypatch.setenv("FLOWERS_DOCS_DIR", "~/corpus/docs")

    assert images_dir() == Path.home() / "corpus" / "train"
    assert docs_dir() == Path.home() / "corpus" / "docs"
