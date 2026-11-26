"""Pruebas de evaluación y benchmark con artefactos sintéticos (sin modelos)."""

import json
from pathlib import Path

import numpy as np
import pytest

from flowers.config import CLASSES
from flowers.evaluation import (
    bench_crecimiento,
    bench_ef_search,
    evaluar_clasificacion,
    evaluar_texto,
    graficar_benchmark,
    graficar_confusion,
)
from flowers.hnsw import HNSWIndex
from flowers.search import SearchEngine

DIM = 32
POR_CLASE = 40
DIM_DOCS = 16


def _norm(m: np.ndarray) -> np.ndarray:
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def _escribir(carpeta: Path, prefijo: str, vectores: np.ndarray, meta: list[dict]) -> None:
    np.save(carpeta / f"{prefijo}_vectors.npy", vectores)
    (carpeta / f"{prefijo}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    indice = HNSWIndex(M=8, ef_construction=100, seed=1)
    indice.add_batch(vectores)
    indice.save(carpeta / f"{prefijo}_hnsw.npz")


@pytest.fixture(scope="module")
def mundo(tmp_path_factory: pytest.TempPathFactory):
    """Motor sintético + holdout + vectores de 'texto' alrededor de los centroides."""
    carpeta = tmp_path_factory.mktemp("eval")
    rng = np.random.default_rng(0)
    cent = _norm(rng.normal(size=(len(CLASSES), DIM)))

    vectores, meta, holdout, meta_h = [], [], [], []
    for c, clase in enumerate(CLASSES):
        vectores.append(_norm(cent[c] + rng.normal(scale=0.08, size=(POR_CLASE, DIM))))
        meta += [{"id": len(meta) + i, "ruta": f"{clase}/{i}.jpg", "clase": clase} for i in range(POR_CLASE)]
        holdout.append(_norm(cent[c] + rng.normal(scale=0.08, size=(6, DIM))))
        meta_h += [{"clase": clase} for _ in range(6)]
    _escribir(carpeta, "images", np.vstack(vectores), meta)

    docs_c = _norm(rng.normal(size=(len(CLASSES), DIM_DOCS)))
    # dandelion tiene dos documentos IDÉNTICOS (como dandelion2.txt y dandelion3.txt).
    docs = np.vstack([docs_c, docs_c[CLASSES.index("dandelion")][None, :]])
    docs_meta = [{"clase": c, "archivo": f"{c}.txt"} for c in CLASSES]
    docs_meta.append({"clase": "dandelion", "archivo": "dandelion_copia.txt"})
    _escribir(carpeta, "docs", docs, docs_meta)

    return {
        "motor": SearchEngine.load(carpeta),
        "holdout": np.vstack(holdout),
        "reales": [m["clase"] for m in meta_h],
        "cent": cent,
        "docs_c": docs_c,
        "vectores": np.vstack(vectores),
    }


def test_clasificacion_perfecta_en_datos_separables(mundo) -> None:
    r = evaluar_clasificacion(mundo["motor"], mundo["holdout"], mundo["reales"], k=10, ef_search=50)

    assert r["accuracy_top1"] == 1.0
    assert r["recall_at_5_clase"] == 1.0
    assert r["recall_hnsw_vs_exacto_at_10"] >= 0.95
    assert r["n_consultas"] == len(CLASSES) * 6
    # Matriz diagonal: cada clase tiene 6 aciertos y nada fuera.
    m = np.array(r["matriz_confusion"])
    assert (m == 6 * np.eye(len(CLASSES))).all()
    assert r["comparaciones_exacto"] == len(CLASSES) * POR_CLASE
    assert 0 < r["comparaciones_hnsw_media"]
    assert r["ms_hnsw_media"] > 0 and r["ms_exacto_media"] > 0
    assert r["por_clase"]["rose"] == {"soporte": 6, "recall": 1.0, "precision": 1.0}


def test_clasificacion_detecta_errores(mundo) -> None:
    # Se declara que todas las fotos son 'daisy': solo 1/5 acierta.
    reales = ["daisy"] * len(mundo["holdout"])
    r = evaluar_clasificacion(mundo["motor"], mundo["holdout"], reales, k=10)
    assert r["accuracy_top1"] == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        evaluar_clasificacion(mundo["motor"], mundo["holdout"], ["daisy"])


def test_texto_aciertos_y_empate_de_documentos_identicos(mundo) -> None:
    consultas = [(f"frase de {c}", c) for c in CLASSES]
    r = evaluar_texto(mundo["motor"], consultas, mundo["cent"], mundo["docs_c"])

    assert r["aciertos_imagenes"] == r["aciertos_documentos"] == r["total"] == 5
    dand = next(f for f in r["consultas"] if f["clase_esperada"] == "dandelion")
    # Los dos documentos de dandelion empatan en sim 1.0 y ambos son de la clase correcta.
    assert dand["documentos"]["sim_top1"] == pytest.approx(1.0, abs=1e-5)
    assert dand["documentos"]["clases_empatadas"] == ["dandelion"]
    assert len(dand["documentos"]["top"]) == 5


def test_texto_marca_fallo_y_docs_opcionales(mundo) -> None:
    # La consulta dice 'rose' pero el vector apunta a tulip.
    cent = mundo["cent"]
    r = evaluar_texto(mundo["motor"], [("x", "rose")], cent[[CLASSES.index("tulip")]], None)
    assert r["aciertos_imagenes"] == 0
    assert "aciertos_documentos" not in r
    with pytest.raises(ValueError):
        evaluar_texto(mundo["motor"], [("x", "rose")], cent[:2], None)


def test_bench_ef_search_estructura_y_recall(mundo) -> None:
    motor = mundo["motor"]
    hnsw = motor._imagenes.hnsw
    r = bench_ef_search(hnsw, mundo["vectores"], mundo["holdout"][:10], k=10, ef_values=(10, 50, 200))

    assert [f["ef_search"] for f in r["hnsw"]] == [10, 50, 200]
    assert r["comparaciones_exacto"] == len(mundo["vectores"])
    assert r["exacto_sklearn_ms"]["media"] > 0 and r["exacto_numpy_ms"]["media"] > 0
    for f in r["hnsw"]:
        assert 0.0 <= f["recall"] <= 1.0
        assert f["ms"]["media"] > 0 and f["comparaciones_media"] > 0
    # Más ef_search = más comparaciones y recall que no baja.
    comps = [f["comparaciones_media"] for f in r["hnsw"]]
    assert comps == sorted(comps)
    assert r["hnsw"][-1]["recall"] >= r["hnsw"][0]["recall"]
    assert r["hnsw"][-1]["recall"] >= 0.95


def test_bench_crecimiento_reconstruye_por_n(mundo) -> None:
    total = len(mundo["vectores"])
    filas = bench_crecimiento(
        mundo["vectores"], mundo["holdout"][:5], [50, 100, 10_000],
        k=5, ef_search=50, M=8, ef_construction=50,
        hnsw_completo=mundo["motor"]._imagenes.hnsw,
    )
    # 10 000 se recorta al total; no hay n repetidos y salen ordenados.
    assert [f["n"] for f in filas] == [50, 100, total]
    assert filas[0]["segundos_construccion"] is not None  # reconstruido
    assert filas[-1]["segundos_construccion"] is None  # reutilizó el índice cargado
    assert all(f["comparaciones_exacto"] == f["n"] for f in filas)
    assert all(0 <= f["recall"] <= 1 and f["hnsw_ms"] > 0 for f in filas)


def test_graficas_se_guardan(mundo, tmp_path: Path) -> None:
    graficar_confusion(np.eye(5, dtype=int) * 3, CLASSES, tmp_path / "c.png")
    ef = bench_ef_search(
        mundo["motor"]._imagenes.hnsw, mundo["vectores"], mundo["holdout"][:4], k=5, ef_values=(10, 50)
    )
    cr = bench_crecimiento(mundo["vectores"], mundo["holdout"][:4], [50], k=5, M=8, ef_construction=50)
    graficar_benchmark(ef, cr, tmp_path / "b.png")
    assert (tmp_path / "c.png").stat().st_size > 1000
    assert (tmp_path / "b.png").stat().st_size > 1000
