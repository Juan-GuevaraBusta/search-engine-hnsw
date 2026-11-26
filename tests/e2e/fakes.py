"""Servicio falso para probar el dashboard sin modelos ni corpus real.

Se activa con::

    FLOWERS_SERVICE_FACTORY=tests.e2e.fakes:crear_falso

* ``construir_artefactos(carpeta)`` escribe artefactos sintéticos con el
  contrato de ``flowers.search`` (vectores alrededor de un centroide por clase).
* Los codificadores falsos deciden la clase por una regla simple y
  determinista (color de la foto o palabra clave del texto), así los tests
  pueden afirmar un resultado exacto.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from flowers.config import CLASSES
from flowers.hnsw import HNSWIndex
from flowers.search import SearchEngine
from flowers.service import Servicio

DIM = 32
DIM_DOCS = 16
POR_CLASE = 30
POR_CLASE_HOLDOUT = 2


def _normalizar(m: np.ndarray) -> np.ndarray:
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def centroides(dim: int = DIM, semilla: int = 0) -> np.ndarray:
    """Una dirección casi ortogonal por clase."""
    return _normalizar(np.random.default_rng(semilla).normal(size=(len(CLASSES), dim)))


def _escribir(carpeta: Path, prefijo: str, vectores: np.ndarray, meta: list[dict]) -> None:
    np.save(carpeta / f"{prefijo}_vectors.npy", vectores)
    (carpeta / f"{prefijo}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if prefijo != "images_holdout":
        indice = HNSWIndex(M=8, ef_construction=100, seed=1)
        indice.add_batch(vectores)
        indice.save(carpeta / f"{prefijo}_hnsw.npz")


def construir_artefactos(carpeta: Path, *, con_docs: bool = True) -> Path:
    """Crea imágenes (train + holdout con fotos reales en disco) y documentos."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    cent = centroides()

    vectores, meta, vh, mh = [], [], [], []
    fotos = carpeta / "fotos"
    fotos.mkdir(exist_ok=True)
    for c, clase in enumerate(CLASSES):
        vectores.append(_normalizar(cent[c] + rng.normal(scale=0.08, size=(POR_CLASE, DIM))))
        for i in range(POR_CLASE):
            ruta = fotos / f"{clase}_{i}.jpg"
            Image.new("RGB", (16, 16), (c * 50, 100, 100)).save(ruta)
            meta.append({"id": len(meta), "ruta": str(ruta), "clase": clase})
        vh.append(_normalizar(cent[c] + rng.normal(scale=0.08, size=(POR_CLASE_HOLDOUT, DIM))))
        for i in range(POR_CLASE_HOLDOUT):
            ruta = fotos / f"holdout_{clase}_{i}.jpg"
            Image.new("RGB", (16, 16), (c * 50, 100, 100)).save(ruta)
            mh.append({"id": len(mh), "ruta": str(ruta), "clase": clase})
    _escribir(carpeta, "images", np.vstack(vectores), meta)
    _escribir(carpeta, "images_holdout", np.vstack(vh), mh)

    if con_docs:
        cd = centroides(DIM_DOCS, semilla=5)
        docs_meta = [
            {
                "id": str(i),
                "archivo": f"{clase}1.txt",
                "clase": clase,
                "texto": f"Documento falso sobre {clase}. " * 3,
            }
            for i, clase in enumerate(CLASSES)
        ]
        _escribir(carpeta, "docs", cd, docs_meta)
    return carpeta


# --------------------------------------------------------------------------
# Codificadores falsos
# --------------------------------------------------------------------------

_PALABRAS = {"daisy": "daisy", "dandelion": "dandelion", "rose": "rose", "sun": "sunflower", "tulip": "tulip"}


def _clase_de_texto(texto: str) -> int:
    for palabra, clase in _PALABRAS.items():
        if palabra in texto.lower():
            return CLASSES.index(clase)
    return 0


class ImageEncoderFalso:
    """Foto -> centroide según el rojo medio (R = 50 * indice de clase)."""

    def encode_images(self, imagen) -> np.ndarray:
        rojo = float(np.asarray(imagen.convert("RGB"))[..., 0].mean())
        clase = min(len(CLASSES) - 1, int(round(rojo / 50)))
        return centroides()[clase][None, :]

    def encode_text_siglip(self, textos) -> np.ndarray:
        return centroides()[_clase_de_texto(textos[0])][None, :]


class TextEncoderFalso:
    backend = "embeddinggemma"

    def encode(self, textos) -> np.ndarray:
        return centroides(DIM_DOCS, semilla=5)[_clase_de_texto(textos[0])][None, :]


class TextEncoderRespaldo:
    """Simula el respaldo SigLIP: vector de otra dimensión que el índice de docs."""

    backend = "siglip"

    def encode(self, textos) -> np.ndarray:
        return centroides()[_clase_de_texto(textos[0])][None, :]


# --------------------------------------------------------------------------
# Fábricas inyectables por FLOWERS_SERVICE_FACTORY
# --------------------------------------------------------------------------


def crear_falso(carpeta: Path) -> Servicio:
    return Servicio(SearchEngine.load(carpeta), ImageEncoderFalso(), TextEncoderFalso(), carpeta)


def crear_falso_con_respaldo(carpeta: Path) -> Servicio:
    return Servicio(SearchEngine.load(carpeta), ImageEncoderFalso(), TextEncoderRespaldo(), carpeta)
