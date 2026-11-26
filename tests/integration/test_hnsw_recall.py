"""Recall@10 en 2 000 vectores con scikit-learn y hnswlib como referencia."""

import hnswlib
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from flowers.hnsw import HNSWIndex


def recall_at_k(actual: NDArray, expected: NDArray) -> float:
    matches = sum(
        len(set(found) & set(exact))
        for found, exact in zip(actual, expected, strict=True)
    )
    return matches / expected.size


def test_hnsw_numpy_y_hnswlib_frente_a_coseno_exacto() -> None:
    seed = 42
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((2000, 64), dtype=np.float32)
    queries = rng.standard_normal((64, 64), dtype=np.float32)
    k, ef, degree, ef_construction = 10, 128, 16, 128

    brute = NearestNeighbors(metric="cosine", algorithm="brute").fit(vectors)
    expected = brute.kneighbors(queries, n_neighbors=k, return_distance=False)

    index = HNSWIndex(M=degree, ef_construction=ef_construction, seed=seed)
    index.add_batch(vectors)
    answers = [index.knn_query(query, k=k, ef_search=ef) for query in queries]
    ids, scores = map(np.stack, zip(*answers, strict=True))
    assert ids.shape == (len(queries), k)
    assert np.all((ids >= 0) & (ids < len(vectors)))
    assert np.all(np.diff(scores, axis=1) <= 0)

    reference = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    reference.init_index(
        max_elements=len(vectors), ef_construction=ef_construction, M=degree, random_seed=seed
    )
    reference.add_items(vectors, num_threads=1)
    reference.set_ef(ef)
    reference_ids, reference_distances = reference.knn_query(queries, k=k, num_threads=1)

    # hnswlib devuelve distancia 1 - coseno. Recalculamos ambos resultados
    # sobre los IDs devueltos, que pueden diferir entre los dos grafos.
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    np.testing.assert_allclose(
        scores, np.einsum("qd,qkd->qk", unit_queries, unit_vectors[ids]), atol=1e-6
    )
    np.testing.assert_allclose(
        1 - reference_distances,
        np.einsum("qd,qkd->qk", unit_queries, unit_vectors[reference_ids]),
        atol=1e-6,
    )

    own_recall = recall_at_k(ids, expected)
    reference_recall = recall_at_k(reference_ids, expected)
    comparison = f"recall@10 NumPy={own_recall:.4f}, hnswlib={reference_recall:.4f}"
    print(comparison)
    assert own_recall >= 0.95 and reference_recall >= 0.95, comparison
