"""Motor de búsqueda: k-NN con HNSW, k-NN exacto y voto ponderado por clase.

Lee los artefactos que deja la construcción del índice (ver contrato abajo)
y responde dos preguntas sobre un vector de consulta:

* ¿Qué flor es y qué tan seguro estoy? -> `buscar_imagen` (PRD 4.8).
* ¿Qué documentos hablan de esto?       -> `buscar_docs`.

Contrato de `artifacts_dir` (el orden de `*_meta.json` es el orden de IDs
del HNSW correspondiente):

* images_vectors.npy  (float32 n x d, solo train, normalizados)
* images_meta.json    (lista de {id, ruta, clase})
* images_hnsw.npz     (HNSWIndex.save)
* docs_vectors.npy, docs_meta.json (clase y archivo), docs_hnsw.npz
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors

from flowers import config
from flowers.hnsw import HNSWIndex


@dataclass(frozen=True, slots=True)
class Vecino:
    """Una foto parecida a la consulta."""

    ruta: str
    clase: str
    sim: float


@dataclass(frozen=True, slots=True)
class ResultadoImagen:
    """Respuesta completa de `buscar_imagen`."""

    vecinos: list[Vecino]
    confianzas: dict[str, float]  # clase -> confianza, de mayor a menor
    clase: str  # clase ganadora
    ms_hnsw: float
    ms_exacto: float
    comparaciones_hnsw: int  # productos punto que hizo HNSW
    comparaciones_exacto: int  # el exacto compara contra todo el corpus


@dataclass(frozen=True, slots=True)
class DocumentoVecino:
    """Un documento parecido a la consulta."""

    archivo: str
    clase: str
    sim: float


@dataclass
class _Coleccion:
    """Vectores + metadatos + HNSW de un corpus (imágenes o documentos)."""

    vectores: NDArray[np.float32]
    meta: list[dict]
    hnsw: HNSWIndex
    _exacto: NearestNeighbors | None = field(default=None, repr=False)

    @classmethod
    def cargar(cls, carpeta: Path, prefijo: str) -> _Coleccion:
        vectores = np.load(carpeta / f"{prefijo}_vectors.npy")
        meta = json.loads((carpeta / f"{prefijo}_meta.json").read_text(encoding="utf-8"))
        hnsw = HNSWIndex.load(carpeta / f"{prefijo}_hnsw.npz")
        if len(meta) != len(vectores):
            raise ValueError(f"{prefijo}: meta y vectores tienen distinto tamaño")
        return cls(vectores=vectores, meta=meta, hnsw=hnsw)

    def exacto(self) -> NearestNeighbors:
        """Fuerza bruta con coseno; se ajusta una sola vez y se reutiliza."""
        if self._exacto is None:
            self._exacto = NearestNeighbors(metric="cosine", algorithm="brute")
            self._exacto.fit(self.vectores)
        return self._exacto


class SearchEngine:
    """Busca vecinos en las colecciones de imágenes y documentos."""

    def __init__(self, imagenes: _Coleccion, docs: _Coleccion | None = None) -> None:
        self._imagenes = imagenes
        self._docs = docs

    @classmethod
    def load(cls, artifacts_dir: str | Path | None = None) -> SearchEngine:
        """Carga los artefactos. Los documentos son opcionales."""
        carpeta = Path(artifacts_dir) if artifacts_dir is not None else config.artifacts_dir()
        imagenes = _Coleccion.cargar(carpeta, "images")
        hay_docs = all(
            (carpeta / nombre).exists()
            for nombre in ("docs_vectors.npy", "docs_meta.json", "docs_hnsw.npz")
        )
        docs = _Coleccion.cargar(carpeta, "docs") if hay_docs else None
        return cls(imagenes, docs)

    # ------------------------------------------------------------------
    # k-NN sobre imágenes
    # ------------------------------------------------------------------

    def knn_hnsw(
        self, q: ArrayLike, k: int, ef_search: int = 50
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], int]:
        """k vecinos aproximados con HNSW.

        Devuelve (ids, similitudes coseno, nº de comparaciones del grafo).
        """
        return self._knn_hnsw(self._imagenes, q, k, ef_search)

    def knn_exacto(self, q: ArrayLike, k: int) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
        """k vecinos exactos por fuerza bruta (referencia para medir HNSW)."""
        return self._knn_exacto(self._imagenes, q, k)

    @staticmethod
    def _knn_hnsw(
        coleccion: _Coleccion, q: ArrayLike, k: int, ef_search: int
    ) -> tuple[NDArray[np.intp], NDArray[np.float32], int]:
        ids, sims = coleccion.hnsw.knn_query(q, k, ef_search=ef_search)
        return ids, sims, coleccion.hnsw.last_query_comparisons

    @staticmethod
    def _knn_exacto(
        coleccion: _Coleccion, q: ArrayLike, k: int
    ) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
        consulta = np.asarray(q, dtype=np.float32).reshape(1, -1)
        k = min(int(k), len(coleccion.vectores))
        distancias, ids = coleccion.exacto().kneighbors(consulta, n_neighbors=k)
        # sklearn devuelve distancia coseno = 1 - similitud.
        return ids[0].astype(np.intp), (1.0 - distancias[0]).astype(np.float32)

    # ------------------------------------------------------------------
    # Voto ponderado (PRD 4.8)
    # ------------------------------------------------------------------

    def clasificar(self, ids: ArrayLike, sims: ArrayLike) -> dict[str, float]:
        """Voto ponderado: cada vecino vota por su clase con peso = similitud.

        puntaje[clase]   = suma de max(0, sim) de los vecinos de esa clase
        confianza[clase] = puntaje[clase] / suma de todos los puntajes

        Devuelve solo las clases que aparecen entre los vecinos, de mayor a
        menor confianza; las confianzas suman 1. Si todas las similitudes
        son <= 0 no hay información para ponderar y se reparte el voto por
        igual entre los vecinos.
        """
        ids = np.asarray(ids, dtype=np.intp)
        pesos = np.maximum(0.0, np.asarray(sims, dtype=np.float64))
        if len(ids) == 0:
            return {}
        if len(ids) != len(pesos):
            raise ValueError("ids y sims deben tener la misma longitud")
        if pesos.sum() == 0.0:
            pesos = np.ones_like(pesos)

        puntajes: dict[str, float] = {}
        for i, peso in zip(ids, pesos):
            clase = self._imagenes.meta[int(i)]["clase"]
            puntajes[clase] = puntajes.get(clase, 0.0) + float(peso)
        total = sum(puntajes.values())
        ordenado = sorted(puntajes.items(), key=lambda par: par[1], reverse=True)
        return {clase: puntaje / total for clase, puntaje in ordenado}

    # ------------------------------------------------------------------
    # Consultas completas
    # ------------------------------------------------------------------

    def buscar_imagen(self, vector: ArrayLike, k: int = 10, ef_search: int = 50) -> ResultadoImagen:
        """¿Qué flor es? Vecinos, confianza por clase, ganadora y tiempos."""
        t0 = time.perf_counter()
        ids, sims, comparaciones = self.knn_hnsw(vector, k, ef_search)
        ms_hnsw = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.knn_exacto(vector, k)
        ms_exacto = (time.perf_counter() - t0) * 1000

        vecinos = [
            Vecino(
                ruta=str(self._imagenes.meta[int(i)]["ruta"]),
                clase=self._imagenes.meta[int(i)]["clase"],
                sim=float(s),
            )
            for i, s in zip(ids, sims)
        ]
        confianzas = self.clasificar(ids, sims)
        return ResultadoImagen(
            vecinos=vecinos,
            confianzas=confianzas,
            clase=next(iter(confianzas), ""),
            ms_hnsw=ms_hnsw,
            ms_exacto=ms_exacto,
            comparaciones_hnsw=comparaciones,
            comparaciones_exacto=len(self._imagenes.vectores),
        )

    def buscar_docs(self, vector: ArrayLike, k: int = 3) -> list[DocumentoVecino]:
        """Documentos más parecidos a la consulta, con similitud y clase."""
        if self._docs is None:
            raise RuntimeError("No hay artefactos de documentos cargados")
        ids, sims, _ = self._knn_hnsw(self._docs, vector, k, ef_search=max(k, 50))
        resultado = []
        for i, s in zip(ids, sims):
            meta = self._docs.meta[int(i)]
            # El contrato dice "clase y archivo"; se tolera ruta o id como nombre.
            archivo = meta.get("archivo") or meta.get("ruta") or meta.get("id", "")
            resultado.append(DocumentoVecino(str(archivo), meta["clase"], float(s)))
        return resultado
