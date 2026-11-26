"""Lógica de evaluación y benchmark (PRD 4.9 y 8).

Aquí vive todo el cálculo; `scripts/evaluate.py` y `scripts/bench.py` solo
cargan artefactos, llaman a estas funciones y guardan los resultados. Así se
puede probar con vectores sintéticos, sin modelos ni corpus real.

Tres bloques:

* Clasificación del holdout: ¿HNSW + voto ponderado adivina la flor?
* Consultas de texto: ¿"una flor amarilla que sigue al sol" cae en girasol?
* Benchmark: tiempo por consulta de fuerza bruta vs HNSW, variando
  ``ef_search`` y el tamaño ``n`` del índice.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors

from flowers import metrics
from flowers.config import CLASSES
from flowers.hnsw import HNSWIndex
from flowers.search import SearchEngine

# Cinco consultas fijas en inglés, una por clase (PRD 8: "texto -> clase 5 de 5").
CONSULTAS_TEXTO: list[tuple[str, str]] = [
    ("a white flower with a yellow center and thin petals", "daisy"),
    ("a tall yellow flower with a large dark center that follows the sun", "sunflower"),
    ("a round yellow wildflower that turns into a white puffball of seeds", "dandelion"),
    ("a red flower with soft velvety petals and thorns on the stem", "rose"),
    ("a cup-shaped spring flower on a straight stem", "tulip"),
]

# Consulta extra, NO cuenta en el 5/5: la frase puramente "de comportamiento".
# SigLIP se entrenó con pies de foto que describen lo que SE VE; "sigue al sol"
# no se ve, y sin "large dark center" queda en empate dandelion/sunflower (50/50).
CONSULTAS_EXTRA: list[tuple[str, str]] = [
    ("a yellow flower that follows the sun", "sunflower"),
]

# Dos documentos con sim >= tope - TOL se consideran empatados (p. ej. textos idénticos).
TOL_EMPATE = 1e-6


# ----------------------------------------------------------------------------
# 1. Clasificación del holdout
# ----------------------------------------------------------------------------


def evaluar_clasificacion(
    motor: SearchEngine,
    vectores: NDArray[np.float32],
    clases_reales: Sequence[str],
    *,
    k: int = 10,
    ef_search: int = 50,
    k_recall_clase: int = 5,
) -> dict[str, Any]:
    """Clasifica cada foto del holdout con HNSW y calcula las métricas de 4.9.

    Por cada foto: k vecinos con HNSW -> voto ponderado -> clase ganadora.
    Además se piden los mismos k vecinos por fuerza bruta para medir el recall
    del índice aproximado (``metrics.recall_ann``) y comparar tiempos.
    """
    if len(vectores) != len(clases_reales):
        raise ValueError("vectores y clases_reales deben tener la misma longitud")

    pred: list[str] = []
    clases_vecinos: list[list[str]] = []
    ids_hnsw, ids_exactos = [], []
    ms_hnsw, ms_exacto, comparaciones = [], [], []

    for q in vectores:
        t0 = time.perf_counter()
        ids, sims, comps = motor.knn_hnsw(q, k, ef_search)
        ms_hnsw.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        ids_ex, _ = motor.knn_exacto(q, k)
        ms_exacto.append((time.perf_counter() - t0) * 1000)

        confianzas = motor.clasificar(ids, sims)
        pred.append(next(iter(confianzas), ""))
        # Clases de los vecinos de más a menos parecido (para recall@5 de clase).
        clases_vecinos.append([motor._imagenes.meta[int(i)]["clase"] for i in ids])
        ids_hnsw.append(ids)
        ids_exactos.append(ids_ex)
        comparaciones.append(comps)

    reales = list(clases_reales)
    matriz = metrics.matriz_confusion(reales, pred)
    return {
        "n_consultas": len(reales),
        "k": k,
        "ef_search": ef_search,
        "accuracy_top1": metrics.accuracy(reales, pred),
        f"recall_at_{k_recall_clase}_clase": metrics.recall_at_k_clase(
            reales, clases_vecinos, k_recall_clase
        ),
        f"recall_hnsw_vs_exacto_at_{k}": metrics.recall_ann(
            np.array(ids_hnsw), np.array(ids_exactos)
        ),
        "ms_hnsw_media": float(np.mean(ms_hnsw)),
        "ms_hnsw_p95": float(np.percentile(ms_hnsw, 95)),
        "ms_exacto_media": float(np.mean(ms_exacto)),
        "speedup_exacto_sobre_hnsw": float(np.mean(ms_exacto) / np.mean(ms_hnsw)),
        "comparaciones_hnsw_media": float(np.mean(comparaciones)),
        "comparaciones_exacto": len(motor._imagenes.vectores),
        "clases": list(CLASSES),
        "matriz_confusion": matriz.tolist(),
        "por_clase": _metricas_por_clase(matriz),
    }


def _metricas_por_clase(matriz: NDArray[np.int64]) -> dict[str, dict[str, float]]:
    """Precisión y recall por clase a partir de la matriz de confusión."""
    resultado: dict[str, dict[str, float]] = {}
    for i, clase in enumerate(CLASSES):
        verdaderos = float(matriz[i, i])
        total_real = float(matriz[i].sum())
        total_pred = float(matriz[:, i].sum())
        resultado[clase] = {
            "soporte": int(total_real),
            "recall": verdaderos / total_real if total_real else 0.0,
            "precision": verdaderos / total_pred if total_pred else 0.0,
        }
    return resultado


# ----------------------------------------------------------------------------
# 2. Consultas de texto
# ----------------------------------------------------------------------------


def evaluar_texto(
    motor: SearchEngine,
    consultas: Sequence[tuple[str, str]],
    vectores_siglip: NDArray[np.float32],
    vectores_docs: NDArray[np.float32] | None,
    *,
    k_imagenes: int = 10,
    k_docs: int = 5,
    ef_search: int = 50,
) -> dict[str, Any]:
    """Texto -> imágenes (SigLIP) y texto -> documentos (EmbeddingGemma).

    ``consultas`` es una lista de (frase, clase esperada). Para imágenes se usa
    el mismo voto ponderado que con una foto. Para documentos se mira el
    documento más parecido; si varios empatan en la cima (textos idénticos,
    como dandelion2.txt y dandelion3.txt) basta con que una clase empatada
    sea la correcta.
    """
    if len(vectores_siglip) != len(consultas):
        raise ValueError("Falta un vector SigLIP por consulta")
    if vectores_docs is not None and len(vectores_docs) != len(consultas):
        raise ValueError("Falta un vector de documentos por consulta")

    filas: list[dict[str, Any]] = []
    for i, (frase, esperada) in enumerate(consultas):
        ids, sims, _ = motor.knn_hnsw(vectores_siglip[i], k_imagenes, ef_search)
        confianzas = motor.clasificar(ids, sims)
        top_img = next(iter(confianzas), "")
        fila: dict[str, Any] = {
            "consulta": frase,
            "clase_esperada": esperada,
            "imagenes": {
                "clase_top1": top_img,
                "acierto": top_img == esperada,
                "confianza": float(confianzas.get(top_img, 0.0)),
            },
        }
        if vectores_docs is not None:
            docs = motor.buscar_docs(vectores_docs[i], k=k_docs)
            tope = docs[0].sim if docs else 0.0
            empatados = [d for d in docs if d.sim >= tope - TOL_EMPATE]
            fila["documentos"] = {
                "archivo_top1": docs[0].archivo if docs else "",
                "clase_top1": docs[0].clase if docs else "",
                "sim_top1": float(tope),
                "clases_empatadas": sorted({d.clase for d in empatados}),
                "acierto": any(d.clase == esperada for d in empatados),
                "top": [
                    {"archivo": d.archivo, "clase": d.clase, "sim": d.sim} for d in docs
                ],
            }
        filas.append(fila)

    resumen: dict[str, Any] = {
        "consultas": filas,
        "aciertos_imagenes": sum(f["imagenes"]["acierto"] for f in filas),
        "total": len(filas),
    }
    if vectores_docs is not None:
        resumen["aciertos_documentos"] = sum(f["documentos"]["acierto"] for f in filas)
    return resumen


# ----------------------------------------------------------------------------
# 3. Benchmark
# ----------------------------------------------------------------------------


def _ids_exactos(vectores: NDArray[np.float32], consultas: NDArray[np.float32], k: int) -> NDArray:
    """Verdad terreno: k vecinos exactos (coseno) calculados de golpe con numpy."""
    sims = consultas @ vectores.T
    return np.argsort(-sims, axis=1)[:, :k]


def _tiempo_ms(funcion: Callable[[NDArray[np.float32]], Any], consultas: NDArray[np.float32]) -> list[float]:
    """Ejecuta ``funcion`` por cada consulta y devuelve los ms de cada una."""
    tiempos = []
    for q in consultas:
        t0 = time.perf_counter()
        funcion(q)
        tiempos.append((time.perf_counter() - t0) * 1000)
    return tiempos


def _fuerza_bruta(vectores: NDArray[np.float32], k: int) -> tuple[Callable, Callable]:
    """Dos referencias exactas: sklearn (la del producto) y numpy puro (más rápida)."""
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(vectores)
    k_real = min(k, len(vectores))

    def con_sklearn(q: NDArray[np.float32]) -> Any:
        return nn.kneighbors(q.reshape(1, -1), n_neighbors=k_real)

    def con_numpy(q: NDArray[np.float32]) -> Any:
        sims = vectores @ q
        top = np.argpartition(-sims, k_real - 1)[:k_real]
        return top[np.argsort(-sims[top])]

    return con_sklearn, con_numpy


def _estadisticas(ms: Sequence[float]) -> dict[str, float]:
    return {"media": float(np.mean(ms)), "p95": float(np.percentile(ms, 95))}


def bench_ef_search(
    hnsw: HNSWIndex,
    vectores: NDArray[np.float32],
    consultas: NDArray[np.float32],
    *,
    k: int = 10,
    ef_values: Sequence[int] = (10, 20, 50, 100, 200),
) -> dict[str, Any]:
    """Latencia y recall de HNSW para cada ``ef_search``, frente a fuerza bruta.

    ``ef_search`` es el tamaño de la lista de candidatos al buscar: más grande
    = más comparaciones, más lento, mejor recall. Es la perilla velocidad/calidad.
    """
    verdad = _ids_exactos(vectores, consultas, k)
    con_sklearn, con_numpy = _fuerza_bruta(vectores, k)
    ms_sklearn = _tiempo_ms(con_sklearn, consultas)
    ms_numpy = _tiempo_ms(con_numpy, consultas)

    filas = []
    for ef in ef_values:
        ids_encontrados, ms, comps = [], [], []
        for q in consultas:
            t0 = time.perf_counter()
            ids, _ = hnsw.knn_query(q, k, ef_search=ef)
            ms.append((time.perf_counter() - t0) * 1000)
            ids_encontrados.append(ids)
            comps.append(hnsw.last_query_comparisons)
        filas.append(
            {
                "ef_search": int(ef),
                "ms": _estadisticas(ms),
                "recall": metrics.recall_ann(np.array(ids_encontrados), verdad),
                "comparaciones_media": float(np.mean(comps)),
                "speedup_vs_sklearn": float(np.mean(ms_sklearn) / np.mean(ms)),
                "speedup_vs_numpy": float(np.mean(ms_numpy) / np.mean(ms)),
            }
        )
    return {
        "n": len(vectores),
        "k": k,
        "n_consultas": len(consultas),
        "exacto_sklearn_ms": _estadisticas(ms_sklearn),
        "exacto_numpy_ms": _estadisticas(ms_numpy),
        "comparaciones_exacto": len(vectores),
        "hnsw": filas,
    }


def bench_crecimiento(
    vectores: NDArray[np.float32],
    consultas: NDArray[np.float32],
    tamanos: Sequence[int],
    *,
    k: int = 10,
    ef_search: int = 50,
    M: int = 16,
    ef_construction: int = 200,
    semilla: int = 42,
    hnsw_completo: HNSWIndex | None = None,
    progreso: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Cómo crece el tiempo por consulta al crecer n.

    Para cada n se toma una submuestra aleatoria de los vectores reales y se
    reconstruye un HNSW propio con ella (si n es el total y se pasa
    ``hnsw_completo``, se reutiliza el ya construido). Fuerza bruta es O(n·d);
    HNSW debería crecer mucho más despacio, casi logarítmico.
    """
    rng = np.random.default_rng(semilla)
    resultados = []
    for n in sorted({min(int(t), len(vectores)) for t in tamanos}):
        if n == len(vectores):
            sub = vectores
        else:
            elegidos = np.sort(rng.choice(len(vectores), size=n, replace=False))
            sub = vectores[elegidos]

        if n == len(vectores) and hnsw_completo is not None:
            indice, seg_build = hnsw_completo, None
        else:
            t0 = time.perf_counter()
            indice = HNSWIndex(M=M, ef_construction=ef_construction)
            indice.add_batch(sub)
            seg_build = time.perf_counter() - t0
        if progreso:
            progreso(f"  n={n}: HNSW listo")

        fila = bench_ef_search(indice, sub, consultas, k=k, ef_values=[ef_search])
        h = fila["hnsw"][0]
        resultados.append(
            {
                "n": n,
                "segundos_construccion": seg_build,
                "exacto_sklearn_ms": fila["exacto_sklearn_ms"]["media"],
                "exacto_numpy_ms": fila["exacto_numpy_ms"]["media"],
                "hnsw_ms": h["ms"]["media"],
                "recall": h["recall"],
                "comparaciones_hnsw": h["comparaciones_media"],
                "comparaciones_exacto": n,
                "speedup_vs_sklearn": h["speedup_vs_sklearn"],
                "speedup_vs_numpy": h["speedup_vs_numpy"],
            }
        )
    return resultados


# ----------------------------------------------------------------------------
# Gráficas (matplotlib se importa aquí dentro: las pruebas no lo necesitan)
# ----------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: solo guarda archivos
    import matplotlib.pyplot as plt

    return plt


def graficar_confusion(
    matriz: ArrayLike, clases: Sequence[str], ruta: str | Any, titulo: str = "Matriz de confusión"
) -> None:
    """Guarda la matriz de confusión: fila = clase real, columna = predicha."""
    plt = _pyplot()
    m = np.asarray(matriz)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(clases)), labels=clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases)), labels=clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(titulo)
    umbral = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, int(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > umbral else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_benchmark(ef: dict[str, Any], crecimiento: list[dict[str, Any]], ruta: str | Any) -> None:
    """Tres paneles: latencia vs ef_search, recall vs ef_search, latencia vs n."""
    plt = _pyplot()
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.5))

    efs = [f["ef_search"] for f in ef["hnsw"]]
    ax = ejes[0]
    ax.plot(efs, [f["ms"]["media"] for f in ef["hnsw"]], "o-", label="HNSW")
    ax.axhline(ef["exacto_sklearn_ms"]["media"], color="C3", ls="--", label="exacto (sklearn)")
    ax.axhline(ef["exacto_numpy_ms"]["media"], color="C2", ls=":", label="exacto (numpy)")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("ms por consulta")
    ax.set_title(f"Latencia vs ef_search (n={ef['n']}, k={ef['k']})")
    ax.legend()

    ax = ejes[1]
    ax.plot(efs, [f["recall"] * 100 for f in ef["hnsw"]], "o-", color="C1")
    ax.axhline(95, color="gray", ls="--", label="meta 95 %")
    ax.set_xlabel("ef_search")
    ax.set_ylabel(f"recall@{ef['k']} vs exacto (%)")
    ax.set_title("Recall vs ef_search")
    ax.legend()

    ax = ejes[2]
    ns = [c["n"] for c in crecimiento]
    ax.plot(ns, [c["hnsw_ms"] for c in crecimiento], "o-", label="HNSW")
    ax.plot(ns, [c["exacto_sklearn_ms"] for c in crecimiento], "s--", color="C3", label="exacto (sklearn)")
    ax.plot(ns, [c["exacto_numpy_ms"] for c in crecimiento], "^:", color="C2", label="exacto (numpy)")
    ax.set_xlabel("n (fotos indexadas)")
    ax.set_ylabel("ms por consulta")
    ax.set_title("Crecimiento con n")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
