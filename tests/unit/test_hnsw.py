"""Pruebas del índice HNSW con vectores sintéticos."""

from pathlib import Path

import numpy as np
import pytest

from flowers.hnsw import HNSWIndex


def test_busqueda_ordenada_por_coseno_con_ids_de_insercion() -> None:
    index = HNSWIndex(ef_search=1)
    assert index.add([0, 2]) == 0
    np.testing.assert_array_equal(index.add_batch([[30, 40], [-4, -3], [4, 3]]), [1, 2, 3])

    ids, scores = index.knn_query([4, 3], k=10)

    np.testing.assert_array_equal(ids, [3, 1, 0, 2])
    np.testing.assert_allclose(scores, [1.0, 0.96, 0.60, -1.0])


def test_la_consulta_cuenta_sus_comparaciones() -> None:
    index = HNSWIndex()
    index.add([1, 0])
    before = index.distance_comparisons

    ids, _ = index.knn_query([1, 0], k=1)
    np.testing.assert_array_equal(ids, [0])
    assert index.last_query_comparisons > 0
    assert index.distance_comparisons - before == index.last_query_comparisons

    ids, scores = index.knn_query([1, 0], k=0)
    assert len(ids) == len(scores) == index.last_query_comparisons == 0


def test_las_capas_son_anidadas_y_respetan_el_limite_de_grado() -> None:
    index = HNSWIndex(M=4, seed=12)
    index.add_batch(np.random.default_rng(2).normal(size=(128, 8)))

    assert len(index._layers) > 1
    assert set(index._layers[0]) == set(range(128))
    for layer, links in enumerate(index._layers):
        if layer > 0:
            assert links.keys() <= index._layers[layer - 1].keys()
        limit = 2 * index.M if layer == 0 else index.M
        for node, neighbors in links.items():
            assert len(neighbors) == len(set(neighbors)) <= limit
            assert node not in neighbors
            assert set(neighbors) <= links.keys()


@pytest.mark.parametrize("vector", [[0, 0], [1, 2, 3], [np.nan, 1], [[1, 2]]])
def test_rechaza_vectores_invalidos(vector: list) -> None:
    index = HNSWIndex()
    index.add([1, 0])

    with pytest.raises(ValueError):
        index.add(vector)
    with pytest.raises(ValueError):
        index.knn_query(vector, k=1)
    assert index.add([0, 1]) == 1


def test_save_load_restaura_busqueda_y_continuacion(tmp_path: Path) -> None:
    vectors = np.random.default_rng(13).normal(size=(100, 8))
    index = HNSWIndex(M=6, ef_construction=40, ef_search=15, seed=4)
    index.add_batch(vectors[:70])
    path = tmp_path / "flowers.npz"
    index.save(path)

    restored = HNSWIndex.load(path)
    assert (restored.M, restored.ef_construction, restored.ef_search) == (6, 40, 15)
    np.testing.assert_array_equal(restored.knn_query(vectors[0], 10), index.knn_query(vectors[0], 10))
    np.testing.assert_array_equal(restored.add_batch(vectors[70:]), index.add_batch(vectors[70:]))
    np.testing.assert_array_equal(restored.knn_query(vectors[-1], 10), index.knn_query(vectors[-1], 10))
    assert restored.last_query_comparisons == index.last_query_comparisons


def test_save_load_de_indice_vacio(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    HNSWIndex().save(path)

    restored = HNSWIndex.load(path)
    ids, scores = restored.knn_query([1, 0], k=1)
    assert ids.size == scores.size == 0
    assert restored.add([1, 0]) == 0
