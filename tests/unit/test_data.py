"""Pruebas del lector de imágenes, documentos y split holdout."""

from collections import Counter
from pathlib import Path

import pytest

from flowers.config import CLASSES, docs_dir, images_dir
from flowers.data import Muestra, Particion, listar_documentos, listar_imagenes, split_estratificado


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    imagenes = tmp_path / "imagenes"
    for clase, nombres in {
        "daisy": ["00.jpg", "01.PNG"],
        "rose": ["a.jpg", "B.JPEG", "nota.txt", ".oculta.jpg"],
        "tulip": ["00.jpg"],
    }.items():
        carpeta = imagenes / clase
        carpeta.mkdir(parents=True)
        for nombre in nombres:
            (carpeta / nombre).write_bytes(b"")
    (imagenes / "daisy" / "extra").mkdir()
    (imagenes / "daisy" / "extra" / "z.jpg").write_bytes(b"")
    (imagenes / "otra").mkdir()
    (imagenes / "otra" / "a.jpg").write_bytes(b"")
    (imagenes / "suelta.jpg").write_bytes(b"")

    documentos = tmp_path / "docs"
    documentos.mkdir()
    for nombre in (
        "rose3.txt",
        "sunflower12.txt",
        "daisy.txt",
        "notes.txt",
        "tulip4.md",
        ".escondido.txt",
        "dandelion2.TXT",
    ):
        (documentos / nombre).write_text("texto", encoding="utf-8")
    return imagenes, documentos


def _muestras(n_por_clase: dict[str, int]) -> list[Muestra]:
    muestras: list[Muestra] = []
    for clase, n in n_por_clase.items():
        for i in range(n):
            muestras.append(
                Muestra(id=f"{clase}-{i:03d}", ruta=Path(f"/{clase}/{i}.jpg"), clase=clase)
            )
    return muestras


def test_listar_imagenes_usa_la_subcarpeta_como_clase(tmp_path: Path) -> None:
    imagenes, _ = _corpus(tmp_path)

    encontradas = listar_imagenes(imagenes)

    assert [m.id for m in encontradas] == [
        "daisy/00.jpg",
        "daisy/01.PNG",
        "rose/B.JPEG",
        "rose/a.jpg",
        "tulip/00.jpg",
    ]
    assert [m.clase for m in encontradas] == ["daisy", "daisy", "rose", "rose", "tulip"]
    assert all(m.ruta.is_file() for m in encontradas)
    assert all(m.ruta.parent.name == m.clase for m in encontradas)


def test_listar_documentos_quita_digitos_del_nombre(tmp_path: Path) -> None:
    _, documentos = _corpus(tmp_path)

    encontradas = listar_documentos(documentos)

    assert [(m.id, m.clase) for m in encontradas] == [
        ("daisy.txt", "daisy"),
        ("dandelion2.TXT", "dandelion"),
        ("rose3.txt", "rose"),
        ("sunflower12.txt", "sunflower"),
    ]
    assert all(m.ruta.is_file() for m in encontradas)


def test_directorio_ausente_falla(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        listar_imagenes(tmp_path / "no-existe")
    with pytest.raises(FileNotFoundError):
        listar_documentos(tmp_path / "no-existe")


def test_listar_sin_argumento_usa_las_variables_de_entorno(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    imagenes, documentos = _corpus(tmp_path)
    monkeypatch.setenv("FLOWERS_IMAGES_DIR", str(imagenes))
    monkeypatch.setenv("FLOWERS_DOCS_DIR", str(documentos))

    assert listar_imagenes() == listar_imagenes(imagenes)
    assert listar_documentos() == listar_documentos(documentos)


def test_split_estratificado_reserva_15_por_ciento_por_clase() -> None:
    tamanos = {"daisy": 20, "dandelion": 4, "rose": 10, "sunflower": 1, "tulip": 7}
    muestras = _muestras(tamanos)

    particion = split_estratificado(muestras)

    assert isinstance(particion, Particion)
    holdout = Counter(m.clase for m in particion.holdout)
    train = Counter(m.clase for m in particion.train)
    for clase, n in tamanos.items():
        assert holdout[clase] == round(n * 0.15)
        assert train[clase] == n - holdout[clase]
    ids = [m.id for m in muestras]
    assert sorted(m.id for m in (*particion.train, *particion.holdout)) == sorted(ids)
    assert set(m.id for m in particion.train).isdisjoint(m.id for m in particion.holdout)
    assert list(particion.train) == sorted(particion.train, key=lambda m: m.id)
    assert list(particion.holdout) == sorted(particion.holdout, key=lambda m: m.id)


def test_split_es_reproducible_y_no_depende_del_orden() -> None:
    muestras = _muestras({"daisy": 20, "rose": 10, "tulip": 7})

    primera = split_estratificado(muestras)
    segunda = split_estratificado(list(reversed(muestras)))
    explicita = split_estratificado(muestras, fraccion=0.15, semilla=42)

    assert primera == segunda == explicita


def test_otra_semilla_cambia_el_holdout() -> None:
    muestras = _muestras({"daisy": 20, "rose": 10, "tulip": 7, "dandelion": 4})

    base = split_estratificado(muestras, semilla=42)
    otra = split_estratificado(muestras, semilla=99)

    assert {m.id for m in base.holdout} != {m.id for m in otra.holdout}
    assert len(base.holdout) == len(otra.holdout)


def test_split_vacio_y_fraccion_invalida() -> None:
    vacio = split_estratificado([])

    assert vacio == Particion(train=(), holdout=())
    with pytest.raises(ValueError):
        split_estratificado(_muestras({"rose": 4}), fraccion=0)
    with pytest.raises(ValueError):
        split_estratificado(_muestras({"rose": 4}), fraccion=1)


@pytest.mark.integration
def test_corpus_real_tiene_2746_imagenes_25_docs_y_5_clases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLOWERS_IMAGES_DIR", raising=False)
    monkeypatch.delenv("FLOWERS_DOCS_DIR", raising=False)
    if not images_dir().is_dir() or not docs_dir().is_dir():
        pytest.skip("corpus real no está en Downloads")

    imagenes = listar_imagenes()
    documentos = listar_documentos()

    assert len(imagenes) == 2746
    assert len(documentos) == 25
    assert sorted({m.clase for m in imagenes}) == CLASSES
    assert sorted({m.clase for m in documentos}) == CLASSES
    assert len({m.id for m in imagenes}) == 2746
    assert len({m.id for m in documentos}) == 25
    particion = split_estratificado(imagenes)
    conteo = Counter(m.clase for m in imagenes)
    holdout = Counter(m.clase for m in particion.holdout)
    for clase, n in conteo.items():
        assert holdout[clase] == round(n * 0.15)
    assert len(particion.train) + len(particion.holdout) == 2746
