"""Pruebas de las métricas de evaluación."""

import numpy as np
import pytest

from flowers.metrics import accuracy, matriz_confusion, recall_ann, recall_at_k_clase


def test_accuracy() -> None:
    assert accuracy(["rose", "tulip", "daisy", "rose"], ["rose", "rose", "daisy", "rose"]) == 0.75
    with pytest.raises(ValueError):
        accuracy(["rose"], [])


def test_recall_at_k_clase() -> None:
    vecinos = [["tulip", "rose", "rose"], ["daisy", "daisy", "tulip"]]
    reales = ["rose", "tulip"]
    # A k=1 ninguna consulta tiene su clase primero; a k=3 ambas la tienen.
    assert recall_at_k_clase(reales, vecinos, k=1) == 0.0
    assert recall_at_k_clase(reales, vecinos, k=2) == 0.5
    assert recall_at_k_clase(reales, vecinos, k=3) == 1.0


def test_matriz_confusion_filas_reales_columnas_predichas() -> None:
    clases = ["daisy", "rose", "tulip"]
    real = ["rose", "rose", "tulip", "daisy"]
    pred = ["rose", "tulip", "tulip", "daisy"]

    matriz = matriz_confusion(real, pred, clases)

    np.testing.assert_array_equal(matriz, [[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    assert matriz.sum() == len(real)
    with pytest.raises(ValueError):
        matriz_confusion(["cactus"], ["rose"], clases)


def test_matriz_confusion_usa_las_clases_del_proyecto_por_defecto() -> None:
    assert matriz_confusion(["rose"], ["rose"]).shape == (5, 5)


def test_recall_ann() -> None:
    assert recall_ann([1, 2, 3], [3, 2, 1]) == 1.0
    assert recall_ann([1, 2, 9], [1, 2, 3]) == pytest.approx(2 / 3)
    # Varias consultas: se promedia el recall de cada una.
    assert recall_ann([[1, 2], [5, 6]], [[1, 2], [7, 8]]) == 0.5
    with pytest.raises(ValueError):
        recall_ann([[1]], [[1], [2]])
