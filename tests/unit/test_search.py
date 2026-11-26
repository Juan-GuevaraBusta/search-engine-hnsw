"""Pruebas del motor de búsqueda con artefactos sintéticos.

Se generan en `tmp_path` siguiendo el contrato de `flowers.search`:
vectores aleatorios alrededor de un centroide por clase.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from flowers.config import CLASSES
from flowers.hnsw import HNSWIndex
from flowers.metrics import recall_ann
from flowers.search import SearchEngine

DIM = 32
POR_CLASE = 40


def _normalizar(m: np.ndarray) -> np.ndarray:
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def _escribir(carpeta: Path, prefijo: str, vectores: np.ndarray, meta: list[dict]) -> None:
    np.save(carpeta / f"{prefijo}_vectors.npy", vectores)
    (carpeta / f"{prefijo}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    indice = HNSWIndex(M=8, ef_construction=100, seed=1)
    indice.add_batch(vectores)
    indice.save(carpeta / f"{prefijo}_hnsw.npz")


@pytest.fixture(scope="module")
def centroides() -> np.ndarray:
    # Una dirección casi ortogonal por clase: centroides bien separados.
    return _normalizar(np.random.default_rng(0).normal(size=(len(CLASSES), DIM)))


@pytest.fixture(scope="module")
def motor(tmp_path_factory: pytest.TempPathFactory, centroides: np.ndarray) -> SearchEngine:
    carpeta = tmp_path_factory.mktemp("artefactos")
    rng = np.random.default_rng(1)

    vectores, meta = [], []
    for c, clase in enumerate(CLASSES):
        ruido = rng.normal(scale=0.08, size=(POR_CLASE, DIM))
        vectores.append(_normalizar(centroides[c] + ruido))
        meta += [
            {"id": f"{clase}_{i}", "ruta": f"train/{clase}/{i}.jpg", "clase": clase}
            for i in range(POR_CLASE)
        ]
    _escribir(carpeta, "images", np.vstack(vectores), meta)

    # Documentos: uno por clase, en otro espacio de dimensión distinta.
    docs_dim = 16
    docs_c = _normalizar(rng.normal(size=(len(CLASSES), docs_dim)))
    docs_meta = [{"clase": c, "archivo": f"{c}.txt"} for c in CLASSES]
    _escribir(carpeta, "docs", docs_c, docs_meta)
    return SearchEngine.load(carpeta)


def _consulta(centroides: np.ndarray, clase: str, semilla: int) -> np.ndarray:
    ruido = np.random.default_rng(semilla).normal(scale=0.08, size=DIM)
    return centroides[CLASSES.index(clase)] + ruido


def test_clasificar_confianzas_suman_uno(motor: SearchEngine) -> None:
    ids, sims, _ = motor.knn_hnsw(np.ones(DIM), k=10)
    confianzas = motor.clasificar(ids, sims)
    assert sum(confianzas.values()) == pytest.approx(1.0)
    assert all(0.0 <= c <= 1.0 for c in confianzas.values())


def test_ejemplo_del_prd_rosa_83_por_ciento(motor: SearchEngine) -> None:
    # Ids de ejemplo: 4 rosas y 1 tulipán (el orden del meta es el de los ids).
    meta = motor._imagenes.meta
    rosas = [i for i, m in enumerate(meta) if m["clase"] == "rose"][:4]
    tulipan = next(i for i, m in enumerate(meta) if m["clase"] == "tulip")
    ids = [rosas[0], rosas[1], rosas[2], tulipan, rosas[3]]
    sims = [0.91, 0.89, 0.85, 0.70, 0.84]

    confianzas = motor.clasificar(ids, sims)

    assert confianzas["rose"] == pytest.approx(3.49 / 4.19)  # ~83 %
    assert confianzas["tulip"] == pytest.approx(0.70 / 4.19)  # ~17 %
    assert round(confianzas["rose"] * 100) == 83
    assert next(iter(confianzas)) == "rose"


def test_clasificar_ignora_similitudes_negativas_y_casos_limite(motor: SearchEngine) -> None:
    meta = motor._imagenes.meta
    rosa = next(i for i, m in enumerate(meta) if m["clase"] == "rose")
    tulipan = next(i for i, m in enumerate(meta) if m["clase"] == "tulip")

    # Un vecino con similitud negativa no aporta puntaje.
    asi = motor.clasificar([rosa, tulipan], [0.8, -0.5])
    assert asi == {"rose": pytest.approx(1.0), "tulip": pytest.approx(0.0)}
    # Si todas son <= 0, el voto se reparte por igual en vez de dividir por cero.
    parejo = motor.clasificar([rosa, tulipan], [-0.2, 0.0])
    assert parejo == {"rose": pytest.approx(0.5), "tulip": pytest.approx(0.5)}
    assert motor.clasificar([], []) == {}


@pytest.mark.parametrize("clase", CLASSES)
def test_la_clase_correcta_gana(motor: SearchEngine, centroides: np.ndarray, clase: str) -> None:
    resultado = motor.buscar_imagen(_consulta(centroides, clase, semilla=7), k=10)

    assert resultado.clase == clase
    assert resultado.confianzas[clase] > 0.9
    assert sum(resultado.confianzas.values()) == pytest.approx(1.0)
    assert len(resultado.vecinos) == 10
    assert all(v.clase == clase and clase in v.ruta for v in resultado.vecinos)
    sims = [v.sim for v in resultado.vecinos]
    assert sims == sorted(sims, reverse=True)
    assert resultado.ms_hnsw >= 0 and resultado.ms_exacto >= 0
    assert 0 < resultado.comparaciones_hnsw
    assert resultado.comparaciones_exacto == len(CLASSES) * POR_CLASE


def test_hnsw_se_parece_al_exacto(motor: SearchEngine, centroides: np.ndarray) -> None:
    ids_h, ids_e = [], []
    for semilla in range(30):
        q = _consulta(centroides, CLASSES[semilla % len(CLASSES)], semilla)
        h, sims_h, _ = motor.knn_hnsw(q, k=10, ef_search=100)
        e, sims_e = motor.knn_exacto(q, k=10)
        ids_h.append(h)
        ids_e.append(e)
        # Mismas similitudes coseno: ambos métodos miden lo mismo.
        np.testing.assert_allclose(np.sort(sims_h), np.sort(sims_e), atol=1e-4)

    assert recall_ann(ids_h, ids_e) >= 0.95


def test_knn_exacto_ordena_por_similitud(motor: SearchEngine, centroides: np.ndarray) -> None:
    ids, sims = motor.knn_exacto(centroides[0], k=5)
    assert len(ids) == 5
    assert np.all(np.diff(sims) <= 1e-6)
    assert 0.9 < sims[0] <= 1.0 + 1e-6


def test_buscar_docs(motor: SearchEngine) -> None:
    # La consulta es el vector del documento de "tulip": debe salir primero.
    docs = motor._docs
    i = [m["clase"] for m in docs.meta].index("tulip")
    resultado = motor.buscar_docs(docs.vectores[i], k=3)

    assert len(resultado) == 3
    assert resultado[0].archivo == "tulip.txt"
    assert resultado[0].clase == "tulip"
    assert resultado[0].sim == pytest.approx(1.0, abs=1e-5)
    assert [d.sim for d in resultado] == sorted((d.sim for d in resultado), reverse=True)


def test_load_sin_documentos(tmp_path: Path, motor: SearchEngine) -> None:
    img = motor._imagenes
    _escribir(tmp_path, "images", img.vectores, img.meta)

    sin_docs = SearchEngine.load(tmp_path)

    assert sin_docs.buscar_imagen(img.vectores[0], k=3).clase == img.meta[0]["clase"]
    with pytest.raises(RuntimeError):
        sin_docs.buscar_docs(np.ones(4))


def test_load_usa_artifacts_dir_de_config(
    tmp_path: Path, motor: SearchEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = motor._imagenes
    _escribir(tmp_path, "images", img.vectores, img.meta)
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))

    assert SearchEngine.load().buscar_imagen(img.vectores[0], k=1).vecinos[0].sim > 0.99
