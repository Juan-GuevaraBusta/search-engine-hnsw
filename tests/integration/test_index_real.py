"""Construye el índice con fotos y documentos reales (subconjunto) y verifica el contrato."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from flowers import data, index
from flowers.config import CLASSES, docs_dir, images_dir

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def construido(tmp_path_factory):
    if not images_dir().is_dir() or not docs_dir().is_dir():
        pytest.skip("no hay corpus real en Downloads")
    art = tmp_path_factory.mktemp("artifacts")
    resumen = index.build_all(progress=False, artifacts_dir=art, max_por_clase=20)
    assert resumen["documentos"]["backend_texto"] == "embeddinggemma"
    return art, resumen


def test_contrato_con_corpus_real(construido) -> None:
    art, resumen = construido
    total = resumen["imagenes"]["train"] + resumen["imagenes"]["holdout"]
    assert total == 20 * len(CLASSES)

    img = index.load_index("images", artifacts_dir=art)
    assert img.vectors.dtype == np.float32 and img.vectors.shape[1] == 768
    assert np.allclose(np.linalg.norm(img.vectors, axis=1), 1, atol=1e-4)
    assert len(img.meta) == resumen["imagenes"]["train"]
    assert {m["clase"] for m in img.meta} == set(CLASSES)

    vec_h, meta_h = index.load_holdout(art)
    assert len(meta_h) == resumen["imagenes"]["holdout"] == len(vec_h)
    assert vec_h.dtype == np.float32 and vec_h.shape == (15, 768)
    assert np.allclose(np.linalg.norm(vec_h, axis=1), 1, atol=1e-4)
    assert Counter(m["clase"] for m in img.meta) == dict.fromkeys(CLASSES, 17)
    assert Counter(m["clase"] for m in meta_h) == dict.fromkeys(CLASSES, 3)
    assert {m["ruta"] for m in img.meta}.isdisjoint(m["ruta"] for m in meta_h)
    assert not (art / "images_holdout_hnsw.npz").exists()

    docs = index.load_index("docs", artifacts_dir=art)
    documentos = data.listar_documentos()
    assert docs.vectors.dtype == np.float32 and docs.vectors.shape == (len(documentos), 768)
    assert [(m["archivo"], m["clase"]) for m in docs.meta] == [
        (muestra.id, muestra.clase) for muestra in documentos
    ]
    assert np.allclose(np.linalg.norm(docs.vectors, axis=1), 1, atol=1e-4)


def test_consulta_devuelve_el_propio_vector(construido) -> None:
    art, _ = construido
    for nombre in ("images", "docs"):
        cargado = index.load_index(nombre, artifacts_dir=art)
        with np.load(art / f"{nombre}_hnsw.npz", allow_pickle=False) as archivo:
            np.testing.assert_allclose(archivo["vectors"], cargado.vectors, atol=1e-6)
        assert [m["id"] for m in cargado.meta] == list(range(len(cargado.vectors)))
        for i in range(0, len(cargado.meta), 7):
            ids, sims = cargado.hnsw.knn_query(cargado.vectors[i], k=1, ef_search=100)
            # El corpus real tiene duplicados exactos (p. ej. dandelion2.txt y
            # dandelion3.txt son iguales): en un empate vale cualquiera de los dos.
            np.testing.assert_allclose(cargado.vectors[ids[0]], cargado.vectors[i], atol=1e-6)
            assert sims[0] == pytest.approx(1.0, abs=1e-5)


def test_foto_holdout_encuentra_su_clase(construido) -> None:
    art, _ = construido
    img = index.load_index("images", artifacts_dir=art)
    vec_h, meta_h = index.load_holdout(art)
    aciertos = 0
    for v, m in zip(vec_h, meta_h):
        ids, _ = img.hnsw.knn_query(v, k=5, ef_search=100)
        clases = [img.meta[i]["clase"] for i in ids]
        aciertos += max(set(clases), key=clases.count) == m["clase"]
    assert aciertos / len(meta_h) >= 0.8


def test_info_json(construido) -> None:
    art, _ = construido
    info = json.loads((art / "index_info.json").read_text())
    assert info["M"] == 16 and info["ef_construction"] == 200
