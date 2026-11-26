"""HNSW de Malkov y Yashunin (https://arxiv.org/abs/1603.09320).

Como en una skip list, cada nodo empieza en la capa 0 y tiene una
probabilidad decreciente de aparecer en cada capa adicional. Arriba se
avanza a saltos entre unos pocos nodos y abajo se afina la búsqueda.
En una skip list se sigue un orden; aquí cada capa es un grafo y se
eligen las aristas según la cercanía a la consulta.
"""

from __future__ import annotations

import heapq
import math
import pickle
from collections.abc import Iterable
from operator import index
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


class HNSWIndex:
    """Índice incremental de similitud coseno con IDs consecutivos desde cero.

    `distance_comparisons` acumula los productos punto de construcción y
    búsqueda. `last_query_comparisons` cuenta los de la última consulta.
    """

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int | None = 42,
    ) -> None:
        self.M = index(M)
        self.ef_construction = index(ef_construction)
        self.ef_search = index(ef_search)
        if self.M < 2 or self.ef_construction < 1 or self.ef_search < 1:
            raise ValueError("M debe ser >= 2 y ambos parámetros ef deben ser >= 1")
        self._rng = np.random.default_rng(seed)
        self._level_scale = 1.0 / math.log(self.M)
        self._vectors: list[NDArray[np.float32]] = []
        # El ID ubica el vector y su lista de adyacencia en cada capa.
        self._layers: list[dict[int, list[int]]] = []
        self._entry: int | None = None
        self.distance_comparisons = 0
        self.last_query_comparisons = 0

    def _normalize(self, vector: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(vector, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Se requiere un vector unidimensional, finito y no vacío")
        if self._vectors and values.shape != self._vectors[0].shape:
            raise ValueError("La dimensión del vector debe coincidir con la del índice")
        # Escalar antes de calcular la norma evita desbordamientos al elevar al cuadrado.
        scale = np.max(np.abs(values))
        if scale == 0:
            raise ValueError("El vector cero no tiene similitud coseno definida")
        scaled = values / scale
        return (scaled / np.linalg.norm(scaled)).astype(np.float32)

    def _similarity(self, query: NDArray[np.float32], node: int) -> float:
        self.distance_comparisons += 1
        # Con norma L2 igual a uno, el coseno se reduce a un producto punto.
        return float(np.dot(query, self._vectors[node]))

    def _search_layer(
        self,
        query: NDArray[np.float32],
        entries: Iterable[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Algoritmo 2 del paper. Devuelve pares (coseno, ID) de mejor a peor."""
        links = self._layers[layer]
        visited = set(entries)
        # El heap de candidatos saca primero al de mayor coseno. El heap de
        # resultados conserva arriba al peor, justo el que se poda al llenarse.
        # En ambos heaps, un coseno igual favorece al menor ID.
        candidates = [(-self._similarity(query, node), node) for node in visited]
        best = [(-score, -node) for score, node in candidates]
        heapq.heapify(candidates)
        heapq.heapify(best)
        while candidates:
            negative_score, node = heapq.heappop(candidates)
            if len(best) == ef and (-negative_score, -node) < best[0]:
                # La frontera ya quedó por debajo de los ef mejores hallazgos.
                break
            for neighbor in links[node]:
                if neighbor in visited:
                    continue
                # Varias aristas pueden llegar al mismo nodo. Basta medirlo una vez por capa.
                visited.add(neighbor)
                score = self._similarity(query, neighbor)
                result = (score, -neighbor)
                if len(best) < ef or result > best[0]:
                    heapq.heappush(candidates, (-score, neighbor))
                    heapq.heappush(best, result)
                    if len(best) > ef:
                        heapq.heappop(best)
        return [(score, -node) for score, node in sorted(best, reverse=True)]

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], limit: int
    ) -> list[int]:
        """Algoritmo 4, con relleno de aristas descartadas si sobran cupos."""
        ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
        if len(ordered) <= limit:
            return [node for _, node in ordered]
        selected: list[int] = []
        discarded: list[int] = []
        for score, node in ordered:
            # Si el candidato está más cerca de un vecino ya elegido que del
            # centro, aporta poca cobertura en esa dirección del grafo.
            vector = self._vectors[node]
            if any(self._similarity(vector, other) > score for other in selected):
                discarded.append(node)
            else:
                selected.append(node)
                if len(selected) == limit:
                    return selected
        return selected + discarded[: limit - len(selected)]

    def add(self, vector: ArrayLike) -> int:
        """Inserta un vector y devuelve su ID."""
        query = self._normalize(vector)
        node = len(self._vectors)
        self._vectors.append(query)
        # Con esta escala, P(nivel >= l) = 1 / M**l, como en una skip list.
        level = int(self._rng.exponential() * self._level_scale)
        top = len(self._layers) - 1

        if self._entry is not None:
            entries = [self._entry]
            for layer in range(top, level, -1):
                # Con ef=1 el recorrido superior es voraz. El nodo al que
                # llegamos también existe en la siguiente capa.
                entries = [self._search_layer(query, entries, 1, layer)[0][1]]

            ef = max(self.M, self.ef_construction)
            for layer in range(min(level, top), -1, -1):
                links = self._layers[layer]
                candidates = self._search_layer(query, entries, ef, layer)
                neighbors = self._select_neighbors(candidates, self.M)
                links[node] = neighbors
                # Se agregan aristas en ambos sentidos. La poda limita la
                # lista saliente del vecino, así que puede quedar una arista de ida.
                degree_limit = 2 * self.M if layer == 0 else self.M
                for neighbor in neighbors:
                    links[neighbor].append(node)
                    if len(links[neighbor]) > degree_limit:
                        center = self._vectors[neighbor]
                        options = [
                            (self._similarity(center, other), other)
                            for other in links[neighbor]
                        ]
                        links[neighbor] = self._select_neighbors(options, degree_limit)
                # Al construir, se baja con todos los hallazgos de esta capa.
                entries = [candidate for _, candidate in candidates]

        if level > top:
            # Si es el primer nodo o supera la altura actual, abre las capas
            # nuevas y se convierte en la puerta de entrada para futuras búsquedas.
            self._layers.extend({node: []} for _ in range(top + 1, level + 1))
            self._entry = node
        return node

    def add_batch(self, vectors: Iterable[ArrayLike]) -> NDArray[np.intp]:
        """Inserta los vectores en orden y devuelve sus IDs en ese orden."""
        return np.fromiter((self.add(vector) for vector in vectors), dtype=np.intp)

    def knn_query(
        self, q: ArrayLike, k: int, ef_search: int | None = None
    ) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
        """Devuelve hasta k IDs y sus cosenos en orden de similitud descendente."""
        k = index(k)
        ef = self.ef_search if ef_search is None else index(ef_search)
        if k < 0 or ef < 1:
            raise ValueError("k debe ser >= 0 y ef_search debe ser >= 1")
        self.last_query_comparisons = 0
        if self._entry is None or k == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32)

        query = self._normalize(q)
        previous_comparisons = self.distance_comparisons
        entries = [self._entry]
        for layer in range(len(self._layers) - 1, 0, -1):
            entries = [self._search_layer(query, entries, 1, layer)[0][1]]

        k = min(k, len(self._vectors))
        results = self._search_layer(query, entries, max(k, ef), 0)[:k]
        self.last_query_comparisons = self.distance_comparisons - previous_comparisons
        ids = np.array([node for _, node in results], dtype=np.intp)
        scores = np.array([score for score, _ in results], dtype=np.float32)
        # Corrige el pequeño exceso de ±1 que puede producir el redondeo.
        return ids, np.clip(scores, -1.0, 1.0)

    def save(self, path: str | Path) -> None:
        """Guarda los vectores y el grafo en un solo archivo NPZ."""
        graph = pickle.dumps(
            {
                "layers": self._layers,
                "entry": self._entry,
                "params": (self.M, self.ef_construction, self.ef_search),
                "rng_state": self._rng.bit_generator.state,
            }
        )
        # Guardar el pickle como bytes mantiene el NPZ libre de arrays de objetos.
        with Path(path).open("wb") as target:
            np.savez(
                target,
                vectors=np.asarray(self._vectors, dtype=np.float32),
                graph=np.frombuffer(graph, dtype=np.uint8),
            )

    @classmethod
    def load(cls, path: str | Path) -> HNSWIndex:
        """Restaura un índice guardado. El pickle requiere un archivo de confianza."""
        with np.load(path, allow_pickle=False) as archive:
            graph = pickle.loads(archive["graph"].tobytes())
            restored = cls(*graph["params"])
            restored._vectors = list(archive["vectors"])
        restored._layers = graph["layers"]
        restored._entry = graph["entry"]
        # Continuar las inserciones conserva la secuencia de niveles aleatorios.
        restored._rng.bit_generator.state = graph["rng_state"]
        return restored
