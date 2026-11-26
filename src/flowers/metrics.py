"""Métricas para saber si el buscador funciona (PRD, sección 4.9).

Son funciones puras sobre listas y arreglos: no dependen del índice, así que
se pueden probar y reutilizar en cualquier evaluación.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flowers.config import CLASSES


def accuracy(y_real: Sequence[str], y_pred: Sequence[str]) -> float:
    """Accuracy top-1: fracción de fotos cuya clase se adivinó bien."""
    if len(y_real) != len(y_pred):
        raise ValueError("y_real y y_pred deben tener la misma longitud")
    if len(y_real) == 0:
        raise ValueError("No hay muestras para evaluar")
    aciertos = sum(real == pred for real, pred in zip(y_real, y_pred))
    return aciertos / len(y_real)


def recall_at_k_clase(
    y_real: Sequence[str], clases_vecinos: Sequence[Sequence[str]], k: int
) -> float:
    """Recall@k por clase: ¿la clase correcta aparece entre los k primeros vecinos?

    `clases_vecinos[i]` son las clases de los vecinos de la consulta i, de
    más a menos parecido. Devuelve la fracción de consultas con al menos un
    vecino de la clase real dentro de los k primeros.
    """
    if k < 1:
        raise ValueError("k debe ser >= 1")
    if len(y_real) != len(clases_vecinos):
        raise ValueError("y_real y clases_vecinos deben tener la misma longitud")
    if len(y_real) == 0:
        raise ValueError("No hay muestras para evaluar")
    aciertos = sum(real in list(vecinos)[:k] for real, vecinos in zip(y_real, clases_vecinos))
    return aciertos / len(y_real)


def matriz_confusion(
    y_real: Sequence[str],
    y_pred: Sequence[str],
    clases: Sequence[str] | None = None,
) -> NDArray[np.int64]:
    """Matriz de confusión: fila = clase real, columna = clase predicha.

    La celda [i, j] cuenta cuántas fotos de la clase i se clasificaron como j.
    La diagonal son los aciertos; fuera de ella se ve qué flores se confunden.
    """
    if len(y_real) != len(y_pred):
        raise ValueError("y_real y y_pred deben tener la misma longitud")
    orden = list(CLASSES if clases is None else clases)
    posicion = {clase: i for i, clase in enumerate(orden)}
    matriz = np.zeros((len(orden), len(orden)), dtype=np.int64)
    for real, pred in zip(y_real, y_pred):
        if real not in posicion or pred not in posicion:
            raise ValueError(f"Clase desconocida: {real!r} / {pred!r}")
        matriz[posicion[real], posicion[pred]] += 1
    return matriz


def recall_ann(ids_hnsw: ArrayLike, ids_exactos: ArrayLike) -> float:
    """Recall del índice aproximado frente a la búsqueda exacta.

    Por consulta: |vecinos de HNSW ∩ vecinos exactos| / |vecinos exactos|.
    Acepta una consulta (arreglo 1D) o varias (2D, una fila por consulta) y
    devuelve el promedio. 1.0 significa que HNSW encontró los mismos vecinos.
    """
    hnsw = np.atleast_2d(np.asarray(ids_hnsw))
    exactos = np.atleast_2d(np.asarray(ids_exactos))
    if hnsw.shape[0] != exactos.shape[0]:
        raise ValueError("Debe haber el mismo número de consultas en ambos")
    if exactos.shape[1] == 0:
        raise ValueError("No hay vecinos exactos con los que comparar")
    recalls = [
        len(set(fila_h.tolist()) & set(fila_e.tolist())) / len(set(fila_e.tolist()))
        for fila_h, fila_e in zip(hnsw, exactos)
    ]
    return float(np.mean(recalls))
