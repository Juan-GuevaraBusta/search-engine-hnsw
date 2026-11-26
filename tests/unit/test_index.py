"""Pruebas del constructor de índices con codificadores falsos y datos sintéticos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from flowers import embeddings, index
from flowers.config import CLASSES

DIM = 16


def _vector_falso(clave: str) -> np.ndarray:
    """Vector determinista (según el texto) y de norma 1."""
    semilla = int(hashlib.sha1(clave.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(semilla).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class ImagenFalsa:
    """Imita a ImageEncoder: el vector depende de la ruta; cuenta las llamadas."""

    model_id = "falso/imagen"

    def __init__(self) -> None:
        self.codificadas = 0

    def encode_images(self, rutas, batch_size: int = 8) -> np.ndarray:
        self.codificadas += len(rutas)
        return np.stack([_vector_falso(Path(r).parent.name + "/" + Path(r).name) for r in rutas])


class TextoFalso:
    backend = "falso"

    def encode(self, textos, batch_size: int = 32) -> np.ndarray:
        return np.stack([_vector_falso(t) for t in textos])


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    imagenes = tmp_path / "imagenes"
    for clase in CLASSES:
        (imagenes / clase).mkdir(parents=True)
        for i in range(10):
            (imagenes / clase / f"{i:02d}.jpg").write_bytes(b"x")  # el falso no lee el contenido
    docs = tmp_path / "docs"
    docs.mkdir()
    for clase in CLASSES:
        for i in (1, 2):
            (docs / f"{clase}{i}.txt").write_text(f"texto sobre {clase} numero {i}", encoding="utf-8")
    return imagenes, docs, tmp_path / "artifacts"


def _construir(corpus, imagen=None, texto=None, **kw):
    imagenes, docs, artefactos = corpus
    return index.build_all(
        progress=False,
        artifacts_dir=artefactos,
        images_dir=imagenes,
        docs_dir=docs,
        image_encoder=imagen or ImagenFalsa(),
        text_encoder=texto or TextoFalso(),
        **kw,
    )


def test_contrato_de_archivos(corpus) -> None:
    _construir(corpus)
    art = corpus[2]
    for nombre in (
        "images_vectors.npy",
        "images_meta.json",
        "images_hnsw.npz",
        "images_holdout_vectors.npy",
        "images_holdout_meta.json",
        "docs_vectors.npy",
        "docs_meta.json",
        "docs_hnsw.npz",
    ):
        assert (art / nombre).is_file(), nombre

    vec = np.load(art / "images_vectors.npy")
    meta = json.loads((art / "images_meta.json").read_text())
    assert vec.dtype == np.float32 and vec.shape == (len(meta), DIM)
    assert np.allclose(np.linalg.norm(vec, axis=1), 1, atol=1e-5)
    assert [m["id"] for m in meta] == list(range(len(meta)))
    assert set(meta[0]) >= {"id", "ruta", "clase"}

    # 50 fotos: 15 % por clase -> round(1.5)=2 holdout por clase = 10; 40 train.
    hold = json.loads((art / "images_holdout_meta.json").read_text())
    assert (len(meta), len(hold)) == (40, 10)
    assert {m["ruta"] for m in meta}.isdisjoint({m["ruta"] for m in hold})
    assert np.load(art / "images_holdout_vectors.npy").shape == (10, DIM)

    docs_meta = json.loads((art / "docs_meta.json").read_text())
    assert len(docs_meta) == 10
    assert set(docs_meta[0]) >= {"id", "archivo", "clase"}


def test_construir_guardar_cargar_y_consultar_devuelve_el_propio_vector(corpus) -> None:
    _construir(corpus)
    for nombre in ("images", "docs"):
        cargado = index.load_index(nombre, artifacts_dir=corpus[2])
        assert len(cargado.meta) == len(cargado.vectors) == len(cargado.hnsw._vectors)
        # Orden de meta = orden de ids del HNSW: el vecino 1 de cada vector es él mismo.
        for i in range(len(cargado.meta)):
            ids, sims = cargado.hnsw.knn_query(cargado.vectors[i], k=1, ef_search=100)
            assert ids[0] == i
            assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_meta_apunta_a_la_foto_correcta(corpus) -> None:
    _construir(corpus)
    cargado = index.load_index("images", artifacts_dir=corpus[2])
    for i, m in enumerate(cargado.meta):
        ruta = Path(m["ruta"])
        assert ruta.parent.name == m["clase"]
        esperado = _vector_falso(ruta.parent.name + "/" + ruta.name)
        assert np.allclose(cargado.vectors[i], esperado, atol=1e-6)


def test_holdout_no_esta_indexado(corpus) -> None:
    _construir(corpus)
    cargado = index.load_index("images", artifacts_dir=corpus[2])
    vec_h, meta_h = index.load_holdout(corpus[2])
    assert len(meta_h) == len(vec_h) == 10
    assert len(cargado.hnsw._vectors) == 40
    rutas_train = {m["ruta"] for m in cargado.meta}
    assert all(m["ruta"] not in rutas_train for m in meta_h)


def test_reanuda_con_cache_sin_recodificar(corpus) -> None:
    primero = ImagenFalsa()
    _construir(corpus, imagen=primero)
    assert primero.codificadas == 50

    segundo = ImagenFalsa()
    _construir(corpus, imagen=segundo)
    assert segundo.codificadas == 0

    (corpus[0] / "rose" / "00.jpg").write_bytes(b"foto editada")
    despues = ImagenFalsa()
    _construir(corpus, imagen=despues)
    assert despues.codificadas == 1


def test_reanuda_tras_interrupcion(corpus) -> None:
    class Interrumpida(ImagenFalsa):
        def encode_images(self, rutas, batch_size: int = 8):
            if self.codificadas >= 20:
                raise KeyboardInterrupt
            return super().encode_images(rutas, batch_size)

    with pytest.raises(KeyboardInterrupt):
        _construir(corpus, imagen=Interrumpida(), tamano_lote=10)
    assert not (corpus[2] / "images_hnsw.npz").exists()

    despues = ImagenFalsa()
    _construir(corpus, imagen=despues, tamano_lote=10)
    assert despues.codificadas == 30  # solo las que faltaban
    assert index.load_index("images", artifacts_dir=corpus[2]).hnsw._vectors


def test_max_por_clase(corpus) -> None:
    resumen = _construir(corpus, max_por_clase=4)
    assert resumen["imagenes"]["train"] + resumen["imagenes"]["holdout"] == 20


def test_build_all_usa_los_codificadores_por_defecto(corpus, monkeypatch) -> None:
    def solo_embeddinggemma(*, fallback):
        assert fallback == "error"
        return TextoFalso()

    monkeypatch.setattr(embeddings, "ImageEncoder", ImagenFalsa)
    monkeypatch.setattr(embeddings, "TextEncoder", solo_embeddinggemma)
    imagenes, docs, art = corpus
    monkeypatch.setenv("FLOWERS_IMAGES_DIR", str(imagenes))
    monkeypatch.setenv("FLOWERS_DOCS_DIR", str(docs))
    monkeypatch.setenv("ARTIFACTS_DIR", str(art))
    index.build_all(progress=False)
    assert index.load_index("docs").vectors.shape == (10, DIM)


def test_load_index_valida_nombre_y_existencia(tmp_path) -> None:
    with pytest.raises(ValueError):
        index.load_index("otra", artifacts_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        index.load_index("images", artifacts_dir=tmp_path)


def test_load_index_detecta_artefactos_inconsistentes(corpus) -> None:
    _construir(corpus)
    meta = corpus[2] / "docs_meta.json"
    datos = json.loads(meta.read_text())
    meta.write_text(json.dumps(datos[:-1]))
    with pytest.raises(ValueError, match="inconsistentes"):
        index.load_index("docs", artifacts_dir=corpus[2])
