"""Capa de servicio del dashboard: junta el motor de búsqueda y los codificadores.

La interfaz (Streamlit) NO crea ni conoce los modelos: le pide un
``Servicio`` a la función de fábrica :func:`crear_servicio`. Así:

* La app es fácil de probar: en los tests se inyecta un servicio con
  codificadores falsos (rápidos, sin descargar modelos).
* La lógica "qué comparar y qué medir" vive aquí, no mezclada con botones.

Cómo sustituir el servicio sin tocar la app
-------------------------------------------
Pon la variable de entorno ``FLOWERS_SERVICE_FACTORY="paquete.modulo:funcion"``.
La función recibe la carpeta de artefactos (``Path``) y devuelve un
:class:`Servicio`. Si la variable no existe se usa el servicio real
(SigLIP + EmbeddingGemma).

Los codificadores solo necesitan cumplir este contrato (duck typing):

* ``imagen_enc.encode_images(pil_o_ruta) -> ndarray (1, d)``
* ``imagen_enc.encode_text_siglip([texto]) -> ndarray (1, d)``
* ``texto_enc.encode([texto]) -> ndarray (1, d_docs)`` (opcional: ``backend``)
"""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flowers import config
from flowers.index import load_holdout
from flowers.search import ResultadoImagen, SearchEngine

# El proyecto es 100 % local: sin red una vez descargados los modelos.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

VARIABLE_FABRICA = "FLOWERS_SERVICE_FACTORY"

# Nombres en español para mostrar (las clases internas están en inglés).
NOMBRES_ES: dict[str, str] = {
    "daisy": "Margarita",
    "dandelion": "Diente de león",
    "rose": "Rosa",
    "sunflower": "Girasol",
    "tulip": "Tulipán",
}

# Preguntas de ejemplo por clase (en inglés: SigLIP se entrenó con texto en inglés).
PREGUNTAS_EJEMPLO: dict[str, str] = {
    "daisy": "a small white flower with a yellow center",
    "dandelion": "a fluffy yellow dandelion in a green field",
    "rose": "a red rose with velvet petals",
    "sunflower": "a tall sunflower with big yellow petals facing the sun",
    "tulip": "a colorful tulip blooming in spring",
}


class ArtefactosFaltantes(RuntimeError):
    """Faltan los archivos que deja `make index`."""


# --------------------------------------------------------------------------
# Resultados que consume la interfaz
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Consulta:
    """Todo lo que pasó en una búsqueda de imágenes (foto o texto -> fotos)."""

    tipo: str  # "foto" | "texto"
    etiqueta: str  # nombre de la foto o la frase escrita
    dim: int  # tamaño del vector de consulta
    norma: float  # norma L2 del vector (debe ser ~1)
    ms_codificar: float  # tiempo del codificador (0 si el vector ya existía)
    k: int
    ef_search: int
    resultado: ResultadoImagen
    coinciden: bool  # ¿HNSW devolvió exactamente los mismos vecinos que el exacto?
    recall: float  # fracción de los k exactos que HNSW también encontró
    n_corpus: int
    capas: int  # capas del grafo HNSW
    M: int


@dataclass(frozen=True, slots=True)
class DocEncontrado:
    archivo: str
    clase: str
    sim: float
    texto: str


@dataclass(frozen=True, slots=True)
class ConsultaDocs:
    """Resultado de buscar documentos con EmbeddingGemma."""

    documentos: list[DocEncontrado]
    backend: str  # "embeddinggemma" o "siglip" (respaldo)
    dim: int
    norma: float
    ms_codificar: float
    aviso: str | None = None  # p. ej. el vector no cabe en el índice de docs


@dataclass
class Servicio:
    """Motor de búsqueda + codificadores + carpeta de artefactos."""

    engine: SearchEngine
    imagen_enc: Any
    texto_enc: Any
    artifacts_dir: Path
    _holdout: tuple[np.ndarray, list[dict]] | None = field(default=None, repr=False)
    _holdout_leido: bool = field(default=False, repr=False)

    # ---- información del índice -------------------------------------

    @property
    def n_imagenes(self) -> int:
        return len(self.engine._imagenes.vectores)

    @property
    def dim_imagenes(self) -> int:
        return int(self.engine._imagenes.vectores.shape[1])

    @property
    def hay_docs(self) -> bool:
        return self.engine._docs is not None

    @property
    def n_docs(self) -> int:
        return len(self.engine._docs.vectores) if self.hay_docs else 0

    @property
    def dim_docs(self) -> int:
        return int(self.engine._docs.vectores.shape[1]) if self.hay_docs else 0

    def holdout(self) -> tuple[np.ndarray, list[dict]] | None:
        """Fotos que NO están en el índice (para demos). None si no existen."""
        if not self._holdout_leido:
            self._holdout_leido = True
            try:
                self._holdout = load_holdout(self.artifacts_dir)
            except (FileNotFoundError, ValueError):
                self._holdout = None
        return self._holdout

    # ---- codificar ----------------------------------------------------

    def codificar_foto(self, imagen: Any) -> tuple[np.ndarray, float]:
        """Foto (PIL o ruta) -> (vector SigLIP 1-D, milisegundos)."""
        t0 = time.perf_counter()
        v = np.asarray(self.imagen_enc.encode_images(imagen), dtype=np.float32)[0]
        return v, (time.perf_counter() - t0) * 1000

    def codificar_texto_siglip(self, texto: str) -> tuple[np.ndarray, float]:
        """Frase -> (vector SigLIP, ms). Vive en el mismo espacio que las fotos."""
        t0 = time.perf_counter()
        v = np.asarray(self.imagen_enc.encode_text_siglip([texto]), dtype=np.float32)[0]
        return v, (time.perf_counter() - t0) * 1000

    # ---- buscar -------------------------------------------------------

    def consultar_imagenes(
        self,
        vector: np.ndarray,
        *,
        tipo: str,
        etiqueta: str,
        k: int,
        ef_search: int,
        ms_codificar: float = 0.0,
    ) -> Consulta:
        """k-NN con HNSW + k-NN exacto + voto ponderado sobre un vector."""
        k = max(1, min(int(k), self.n_imagenes))
        resultado = self.engine.buscar_imagen(vector, k=k, ef_search=ef_search)

        # Para saber si HNSW acertó comparamos los ids contra el exacto.
        ids_h, _, _ = self.engine.knn_hnsw(vector, k, ef_search)
        ids_e, _ = self.engine.knn_exacto(vector, k)
        recall = len(set(map(int, ids_h)) & set(map(int, ids_e))) / max(1, len(ids_e))

        hnsw = self.engine._imagenes.hnsw
        return Consulta(
            tipo=tipo,
            etiqueta=etiqueta,
            dim=int(np.asarray(vector).shape[-1]),
            norma=float(np.linalg.norm(vector)),
            ms_codificar=ms_codificar,
            k=k,
            ef_search=int(ef_search),
            resultado=resultado,
            coinciden=set(map(int, ids_h)) == set(map(int, ids_e)),
            recall=recall,
            n_corpus=self.n_imagenes,
            capas=len(getattr(hnsw, "_layers", [])) or 1,
            M=int(getattr(hnsw, "M", 0)),
        )

    def consultar_docs(self, texto: str, k: int = 3) -> ConsultaDocs:
        """Texto -> EmbeddingGemma -> documentos más cercanos con su texto."""
        if not self.hay_docs:
            raise ArtefactosFaltantes("No hay índice de documentos. Ejecuta `make index`.")
        t0 = time.perf_counter()
        v = np.asarray(self.texto_enc.encode([texto]), dtype=np.float32)[0]
        ms = (time.perf_counter() - t0) * 1000
        backend = str(getattr(self.texto_enc, "backend", None) or "embeddinggemma")
        base = dict(backend=backend, dim=int(v.shape[-1]), norma=float(np.linalg.norm(v)))

        indexado_con = self._backend_indice_docs()
        if v.shape[-1] != self.dim_docs or (indexado_con and backend != indexado_con):
            # Pasa si EmbeddingGemma (gated) no cargó y se usó el respaldo SigLIP:
            # aunque las dimensiones coincidan, son espacios distintos.
            aviso = (
                f"La pregunta se codificó con '{backend}' (vector de {v.shape[-1]}) pero el "
                f"índice de documentos se construyó con '{indexado_con or '?'}' "
                f"({self.dim_docs}). No se comparan espacios distintos; revisa que "
                "EmbeddingGemma (modelo con licencia) se pueda cargar."
            )
            return ConsultaDocs([], ms_codificar=ms, aviso=aviso, **base)

        k = max(1, min(int(k), self.n_docs))
        vecinos = self.engine.buscar_docs(v, k)
        docs = [
            DocEncontrado(d.archivo, d.clase, d.sim, self._texto_documento(d.archivo))
            for d in vecinos
        ]
        return ConsultaDocs(docs, ms_codificar=ms, **base)

    def _backend_indice_docs(self) -> str | None:
        """Codificador con el que se construyó el índice de docs (index_info.json)."""
        info = self.artifacts_dir / "index_info.json"
        try:
            datos = json.loads(info.read_text(encoding="utf-8"))
            return datos.get("documentos", {}).get("backend_texto")
        except (OSError, ValueError, AttributeError):
            return None

    def _texto_documento(self, archivo: str) -> str:
        """Texto del documento: del meta si está guardado; si no, del disco."""
        for meta in self.engine._docs.meta:
            if meta.get("archivo") == archivo:
                if meta.get("texto"):
                    return str(meta["texto"])
                for clave in ("ruta",):
                    ruta = Path(str(meta.get(clave, "")))
                    if ruta.is_file():
                        return ruta.read_text(encoding="utf-8", errors="replace")
        candidato = config.docs_dir() / archivo
        if candidato.is_file():
            return candidato.read_text(encoding="utf-8", errors="replace")
        return "(texto no disponible)"


# --------------------------------------------------------------------------
# Fábrica
# --------------------------------------------------------------------------


def _verificar_artefactos(carpeta: Path) -> None:
    faltan = [
        n
        for n in ("images_vectors.npy", "images_meta.json", "images_hnsw.npz")
        if not (carpeta / n).is_file()
    ]
    if faltan:
        raise ArtefactosFaltantes(
            f"Faltan artefactos en {carpeta}: {', '.join(faltan)}. Ejecuta `make index`."
        )


def _crear_real(carpeta: Path) -> Servicio:
    # Importación diferida: torch/transformers solo se tocan en la app real.
    from flowers.embeddings import ImageEncoder, TextEncoder

    return Servicio(SearchEngine.load(carpeta), ImageEncoder(), TextEncoder(), carpeta)


def crear_servicio(
    artifacts_dir: str | Path | None = None, fabrica: str | None = None
) -> Servicio:
    """Fábrica del servicio. Respeta ``FLOWERS_SERVICE_FACTORY`` para pruebas."""
    carpeta = Path(artifacts_dir) if artifacts_dir else config.artifacts_dir()
    _verificar_artefactos(carpeta)
    fabrica = fabrica if fabrica is not None else os.environ.get(VARIABLE_FABRICA, "")
    if fabrica:
        modulo, _, nombre = fabrica.partition(":")
        funcion = getattr(importlib.import_module(modulo), nombre)
        return funcion(carpeta)
    return _crear_real(carpeta)
