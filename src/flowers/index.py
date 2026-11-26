"""Construcción, guardado y carga de los índices HNSW (fotos y documentos).

Esta es la **Fase 1** del PRD (sección 5.3): se ejecuta una sola vez.

    fotos / .txt  ->  codificador  ->  vectores normalizados  ->  HNSW  ->  artifacts/

Se construyen DOS índices separados, porque SigLIP (fotos) y EmbeddingGemma
(documentos) producen vectores que no se pueden comparar entre sí.

Contrato de archivos en ``config.artifacts_dir()``
--------------------------------------------------
(otra tarea los lee; no cambiar nombres ni formas)::

    images_vectors.npy          float32 (n, d), SOLO split train, norma 1
    images_meta.json            [{id, ruta, clase}, ...]  (id = posición)
    images_hnsw.npz             HNSWIndex.save
    images_holdout_vectors.npy  float32 (m, d), split holdout (NO indexado)
    images_holdout_meta.json    [{id, ruta, clase}, ...]
    docs_vectors.npy            float32 (p, d2)
    docs_meta.json              [{id, archivo, clase, ruta, texto}, ...]
    docs_hnsw.npz               HNSWIndex.save

Regla de oro: **el orden de ``meta`` es el orden de ids del HNSW**. Si
``meta[5]`` es una rosa, el vecino con id 5 en el HNSW es esa rosa.

Reanudar: codificar fotos es lo lento (minutos). Cada vector se guarda en
``artifacts/cache/`` apenas se calcula; si el proceso se interrumpe, la
siguiente corrida salta las fotos que ya tienen su vector.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from flowers import config
from flowers.data import (
    Muestra,
    Particion,
    listar_documentos,
    listar_imagenes,
    split_estratificado,
)
from flowers.embeddings import l2_normalize
from flowers.hnsw import HNSWIndex

# Parámetros HNSW de la construcción (PRD 4.6).
M = 16
EF_CONSTRUCTION = 200

# Cuántas fotos se codifican antes de guardar en el cache a disco.
TAMANO_LOTE = 32

# Nombres de archivos del contrato.
IMAGENES = "images"
IMAGENES_HOLDOUT = "images_holdout"
DOCUMENTOS = "docs"


@dataclass(frozen=True, slots=True)
class IndiceCargado:
    """Un índice listo para consultar.

    ``vectors[i]`` y ``meta[i]`` corresponden al id ``i`` del ``hnsw``.
    """

    hnsw: HNSWIndex
    vectors: np.ndarray
    meta: list[dict[str, Any]]


# --------------------------------------------------------------------------
# Rutas del contrato
# --------------------------------------------------------------------------


def _ruta_vectores(nombre: str, directorio: Path) -> Path:
    """Ruta de ``{nombre}_vectors.npy`` dentro del directorio de artefactos."""
    return directorio / f"{nombre}_vectors.npy"


def _ruta_meta(nombre: str, directorio: Path) -> Path:
    """Ruta de ``{nombre}_meta.json`` dentro del directorio de artefactos."""
    return directorio / f"{nombre}_meta.json"


def _ruta_hnsw(nombre: str, directorio: Path) -> Path:
    """Ruta de ``{nombre}_hnsw.npz`` dentro del directorio de artefactos."""
    return directorio / f"{nombre}_hnsw.npz"


# --------------------------------------------------------------------------
# Escritura atómica: se escribe a un archivo temporal y luego se renombra,
# así una corrida interrumpida nunca deja un archivo a medias.
# --------------------------------------------------------------------------


def _archivo_temporal(ruta: Path):
    """Abre un temporal junto a ``ruta``, para poder renombrarlo de forma atómica."""
    return NamedTemporaryFile(dir=ruta.parent, prefix=f".{ruta.stem}-", delete=False)


def _guardar_npy(ruta: Path, vectores: np.ndarray) -> None:
    """Escribe una matriz ``.npy`` y la publica con un renombre atómico."""
    with _archivo_temporal(ruta) as destino:
        np.save(destino, vectores)
    os.replace(destino.name, ruta)


def _guardar_json(ruta: Path, datos: Any) -> None:
    """Escribe JSON en UTF-8 y lo publica con un renombre atómico."""
    with _archivo_temporal(ruta) as destino:
        tmp = Path(destino.name)
    tmp.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, ruta)


def _guardar_hnsw(ruta: Path, indice: HNSWIndex) -> None:
    """Guarda el índice HNSW en un NPZ y lo publica con un renombre atómico.

    Siguiente paso: `src/flowers/hnsw.py` → `HNSWIndex.save()`.
    """
    with _archivo_temporal(ruta) as destino:
        tmp = Path(destino.name)
    indice.save(tmp)
    os.replace(tmp, ruta)


# --------------------------------------------------------------------------
# Cache a disco de vectores de fotos (para reanudar)
# --------------------------------------------------------------------------


def _dir_cache(directorio: Path, codificador: Any) -> Path:
    """Carpeta de cache, separada por modelo (otro modelo = otros vectores)."""
    modelo = str(getattr(codificador, "model_id", "desconocido")).replace("/", "_")
    return directorio / "cache" / modelo


def _archivo_cache(cache: Path, muestra: Muestra) -> Path:
    """Nombre de cache de una foto: hash de ruta, tamaño y fechas del archivo.

    La posición puede cambiar entre builds. La ruta y el estado del archivo
    distinguen corpus con nombres iguales y fotos que se editaron.
    """
    ruta = muestra.ruta.resolve()
    estado = ruta.stat()
    clave = json.dumps([str(ruta), estado.st_size, estado.st_mtime_ns, estado.st_ctime_ns])
    return cache / (hashlib.sha256(clave.encode("utf-8")).hexdigest() + ".npy")


def _leer_cache(archivo: Path) -> np.ndarray | None:
    """Lee un vector del cache; si está corrupto o no existe devuelve None."""
    if not archivo.is_file():
        return None
    try:
        vector = np.load(archivo, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if vector.ndim == 1 and vector.dtype == np.float32 and np.isclose(np.linalg.norm(vector), 1):
        return vector
    return None


def _normalizar(vectores: np.ndarray, filas: int) -> np.ndarray:
    """Normaliza a norma 1 y exige exactamente ``filas`` vectores finitos.

    Siguiente paso: `src/flowers/embeddings.py` → `l2_normalize()`. Luego vuelve a quien llamó (`_vectorizar_imagenes()` o `construir_documentos()`).
    """
    matriz = l2_normalize(vectores)
    if matriz.shape[0] != filas or matriz.shape[1] == 0:
        raise ValueError(f"Se esperaban {filas} vectores, llegó una matriz {matriz.shape}")
    if not np.allclose(np.linalg.norm(matriz, axis=1), 1, atol=1e-5):
        raise ValueError("Cada embedding debe ser un vector finito con norma 1")
    return matriz


def _vectorizar_imagenes(
    muestras: list[Muestra],
    codificador: Any,
    directorio: Path,
    *,
    tamano_lote: int,
    progress: bool,
) -> np.ndarray:
    """Codifica fotos por lotes, guardando cada vector en el cache.

    Devuelve una matriz ``(n, d)`` float32 en el mismo orden que ``muestras``.

    Siguiente paso:
      1. `src/flowers/embeddings.py` → `ImageEncoder.encode_images()`.
      2. Al volver: `_normalizar()` (este archivo) y guarda cada vector en la caché.
    """
    cache = _dir_cache(directorio, codificador)
    cache.mkdir(parents=True, exist_ok=True)

    archivos_cache = [_archivo_cache(cache, muestra) for muestra in muestras]
    vectores: list[np.ndarray | None] = [_leer_cache(archivo) for archivo in archivos_cache]
    pendientes = [i for i, v in enumerate(vectores) if v is None]
    _log(
        progress,
        f"fotos: {len(muestras)} en total, {len(muestras) - len(pendientes)} ya en cache, "
        f"{len(pendientes)} por codificar",
    )

    inicio = time.perf_counter()
    for desde in range(0, len(pendientes), tamano_lote):
        lote = pendientes[desde : desde + tamano_lote]
        rutas = [muestras[i].ruta for i in lote]
        codificados = _normalizar(codificador.encode_images(rutas), len(lote))
        for i, vector in zip(lote, codificados, strict=True):
            vectores[i] = vector
            _guardar_npy(archivos_cache[i], vector)
        hechas = desde + len(lote)
        if progress:
            seg = time.perf_counter() - inicio
            restante = seg / hechas * (len(pendientes) - hechas)
            _log(True, f"  fotos codificadas {hechas}/{len(pendientes)} ({seg:.0f}s, faltan ~{restante:.0f}s)")

    if not muestras:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vectores).astype(np.float32, copy=False)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Construcción
# --------------------------------------------------------------------------


def _construir_hnsw(vectores: np.ndarray, *, M: int, ef_construction: int) -> HNSWIndex:
    """Inserta los vectores EN ORDEN: la fila i obtiene el id i.

    Siguiente paso: `src/flowers/hnsw.py` → `HNSWIndex.add_batch()`.
    """
    indice = HNSWIndex(M=M, ef_construction=ef_construction)
    if len(vectores):
        indice.add_batch(vectores)
    return indice


def _meta_imagenes(muestras: tuple[Muestra, ...] | list[Muestra]) -> list[dict[str, Any]]:
    """Metadatos de fotos en el mismo orden que los ids del HNSW: id, ruta y clase."""
    return [{"id": i, "ruta": str(m.ruta), "clase": m.clase} for i, m in enumerate(muestras)]


def _limitar_por_clase(muestras: list[Muestra], max_por_clase: int | None) -> list[Muestra]:
    """Se queda con las primeras ``max_por_clase`` fotos de cada clase (para pruebas)."""
    if max_por_clase is None:
        return muestras
    contador: dict[str, int] = {}
    elegidas: list[Muestra] = []
    for m in muestras:  # ya vienen ordenadas por id
        if contador.get(m.clase, 0) < max_por_clase:
            elegidas.append(m)
            contador[m.clase] = contador.get(m.clase, 0) + 1
    return elegidas


def construir_imagenes(
    directorio: Path,
    *,
    codificador: Any,
    imagenes_dir: Path | None = None,
    max_por_clase: int | None = None,
    tamano_lote: int = TAMANO_LOTE,
    M: int = M,
    ef_construction: int = EF_CONSTRUCTION,
    progress: bool = True,
) -> dict[str, Any]:
    """Construye y guarda el índice de fotos (train) y los vectores del holdout.

    Siguiente paso:
      1. `src/flowers/data.py` → `listar_imagenes()`.
      2. `src/flowers/data.py` → `split_estratificado()`.
      3. `_vectorizar_imagenes()` (este archivo).
      4. `_construir_hnsw()` (este archivo).
    """
    todas = _limitar_por_clase(listar_imagenes(imagenes_dir), max_por_clase)
    if not todas:
        raise ValueError("No se encontraron imágenes para indexar")
    particion: Particion = split_estratificado(todas)  # 15 % holdout, semilla 42
    train, holdout = list(particion.train), list(particion.holdout)
    _log(progress, f"imágenes: {len(train)} train + {len(holdout)} holdout")

    # Se codifica todo junto (train + holdout) para reutilizar el cache.
    vectores = _vectorizar_imagenes(
        train + holdout, codificador, directorio, tamano_lote=tamano_lote, progress=progress
    )
    v_train, v_holdout = vectores[: len(train)], vectores[len(train) :]

    _log(progress, f"construyendo HNSW de fotos (M={M}, efConstruction={ef_construction})...")
    inicio = time.perf_counter()
    indice = _construir_hnsw(v_train, M=M, ef_construction=ef_construction)
    seg_hnsw = time.perf_counter() - inicio
    _log(progress, f"  HNSW de fotos listo en {seg_hnsw:.1f}s")

    _guardar_npy(_ruta_vectores(IMAGENES, directorio), v_train)
    _guardar_json(_ruta_meta(IMAGENES, directorio), _meta_imagenes(train))
    _guardar_hnsw(_ruta_hnsw(IMAGENES, directorio), indice)
    _guardar_npy(_ruta_vectores(IMAGENES_HOLDOUT, directorio), v_holdout)
    _guardar_json(_ruta_meta(IMAGENES_HOLDOUT, directorio), _meta_imagenes(holdout))
    return {
        "train": len(train),
        "holdout": len(holdout),
        "dim": int(v_train.shape[1]),
        "segundos_hnsw": round(seg_hnsw, 2),
    }


def construir_documentos(
    directorio: Path,
    *,
    codificador: Any,
    documentos_dir: Path | None = None,
    M: int = M,
    ef_construction: int = EF_CONSTRUCTION,
    progress: bool = True,
) -> dict[str, Any]:
    """Construye y guarda el índice de documentos ``.txt`` (todos se indexan).

    Siguiente paso:
      1. `src/flowers/data.py` → `listar_documentos()`.
      2. `src/flowers/embeddings.py` → `TextEncoder.encode()`.
      3. `_normalizar()` y `_construir_hnsw()` (este archivo).
    """
    docs = listar_documentos(documentos_dir)
    if not docs:
        raise ValueError("No se encontraron documentos .txt para indexar")
    textos = [d.ruta.read_text(encoding="utf-8") for d in docs]
    _log(progress, f"documentos: {len(docs)}; codificando...")
    vectores = _normalizar(codificador.encode(textos), len(textos))

    indice = _construir_hnsw(vectores, M=M, ef_construction=ef_construction)
    meta = [
        {"id": i, "archivo": d.id, "clase": d.clase, "ruta": str(d.ruta), "texto": t}
        for i, (d, t) in enumerate(zip(docs, textos, strict=True))
    ]
    _guardar_npy(_ruta_vectores(DOCUMENTOS, directorio), vectores)
    _guardar_json(_ruta_meta(DOCUMENTOS, directorio), meta)
    _guardar_hnsw(_ruta_hnsw(DOCUMENTOS, directorio), indice)
    return {
        "documentos": len(docs),
        "dim": int(vectores.shape[1]),
        "backend_texto": getattr(codificador, "backend", None),
    }


def build_all(
    progress: bool = True,
    *,
    artifacts_dir: Path | None = None,
    images_dir: Path | None = None,
    docs_dir: Path | None = None,
    max_por_clase: int | None = None,
    tamano_lote: int = TAMANO_LOTE,
    image_encoder: Any = None,
    text_encoder: Any = None,
) -> dict[str, Any]:
    """Construye los dos índices y los guarda en ``artifacts_dir``.

    Todos los argumentos salvo ``progress`` son opcionales: por defecto se usan
    las rutas de ``config`` y los codificadores reales. ``max_por_clase`` limita
    las fotos por clase (útil para pruebas rápidas). Devuelve un resumen.

    Siguiente paso:
      1. `src/flowers/embeddings.py` → `ImageEncoder()` y `TextEncoder()` (solo se crean; los pesos cargan después).
      2. `construir_imagenes()` (este archivo): índice de fotos.
      3. `construir_documentos()` (este archivo): índice de documentos.
    """
    directorio = Path(artifacts_dir) if artifacts_dir is not None else config.artifacts_dir()
    directorio.mkdir(parents=True, exist_ok=True)

    if image_encoder is None or text_encoder is None:
        # Import perezoso: así los módulos que solo cargan índices no importan torch.
        from flowers.embeddings import ImageEncoder, TextEncoder

        if image_encoder is None:
            image_encoder = ImageEncoder()
        if text_encoder is None:
            text_encoder = TextEncoder(fallback="error")

    inicio = time.perf_counter()
    resumen_img = construir_imagenes(
        directorio,
        codificador=image_encoder,
        imagenes_dir=images_dir,
        max_por_clase=max_por_clase,
        tamano_lote=tamano_lote,
        progress=progress,
    )
    resumen_docs = construir_documentos(
        directorio, codificador=text_encoder, documentos_dir=docs_dir, progress=progress
    )
    resumen = {
        "imagenes": resumen_img,
        "documentos": resumen_docs,
        "M": M,
        "ef_construction": EF_CONSTRUCTION,
        "segundos_total": round(time.perf_counter() - inicio, 1),
        "artifacts_dir": str(directorio),
    }
    _guardar_json(directorio / "index_info.json", resumen)
    _log(progress, f"listo en {resumen['segundos_total']}s -> {directorio}")
    return resumen


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def load_index(nombre: str = IMAGENES, artifacts_dir: Path | None = None) -> IndiceCargado:
    """Carga un índice guardado: ``"images"`` o ``"docs"``.

    Comprueba que vectores, meta e índice HNSW tengan el mismo tamaño.
    """
    if nombre not in (IMAGENES, DOCUMENTOS):
        raise ValueError(f"nombre debe ser 'images' o 'docs', no {nombre!r}")
    directorio = Path(artifacts_dir) if artifacts_dir is not None else config.artifacts_dir()
    ruta_hnsw = _ruta_hnsw(nombre, directorio)
    if not ruta_hnsw.is_file():
        raise FileNotFoundError(f"No existe {ruta_hnsw}. Corre `make index` primero.")
    hnsw = HNSWIndex.load(ruta_hnsw)
    vectores = np.load(_ruta_vectores(nombre, directorio), allow_pickle=False)
    meta = json.loads(_ruta_meta(nombre, directorio).read_text(encoding="utf-8"))
    if not (len(hnsw._vectors) == len(vectores) == len(meta)):
        raise ValueError(
            f"Artefactos inconsistentes en {directorio}: HNSW={len(hnsw._vectors)}, "
            f"vectores={len(vectores)}, meta={len(meta)}. Reconstruye con `make index`."
        )
    return IndiceCargado(hnsw=hnsw, vectors=vectores, meta=meta)


def load_holdout(artifacts_dir: Path | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Carga vectores y meta de las fotos holdout (no están en el HNSW)."""
    directorio = Path(artifacts_dir) if artifacts_dir is not None else config.artifacts_dir()
    vectores = np.load(_ruta_vectores(IMAGENES_HOLDOUT, directorio), allow_pickle=False)
    meta = json.loads(_ruta_meta(IMAGENES_HOLDOUT, directorio).read_text(encoding="utf-8"))
    if len(vectores) != len(meta):
        raise ValueError("Holdout inconsistente: vectores y meta tienen distinto tamaño")
    return vectores, meta


def _log(progress: bool, mensaje: str) -> None:
    """Imprime el avance en stderr cuando ``progress`` está activo."""
    if progress:
        print(mensaje, file=sys.stderr, flush=True)
