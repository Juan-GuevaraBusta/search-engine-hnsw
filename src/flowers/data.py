"""Lee el corpus de flores y separa train/holdout.

Las imágenes viven en subcarpetas de ``FLOWERS_IMAGES_DIR``; el nombre de
la subcarpeta es la clase. Los documentos viven en ``FLOWERS_DOCS_DIR``;
la clase es el nombre del archivo sin dígitos (``rose3.txt`` → ``rose``).

El holdout se queda con cerca del 15 % de cada clase. El reparto usa una
semilla fija, así que el mismo corpus siempre da el mismo corte.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from flowers.config import CLASSES, docs_dir, images_dir

FRACCION_HOLDOUT = 0.15
SEMILLA = 42

_CLASES = set(CLASSES)
_SUFIJOS_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_DIGITOS = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class Muestra:
    """Una imagen o un documento del corpus."""

    id: str
    ruta: Path
    clase: str


@dataclass(frozen=True, slots=True)
class Particion:
    """Corte train/holdout. Las tuplas quedan ordenadas por ``id``."""

    train: tuple[Muestra, ...]
    holdout: tuple[Muestra, ...]


def listar_imagenes(directorio: Path | None = None) -> list[Muestra]:
    """Lista las imágenes de un nivel: subcarpeta = clase.

    Solo entran las clases de :data:`flowers.config.CLASSES` y los archivos
    con sufijo de imagen. El ``id`` es la ruta relativa con ``/``.
    """
    raiz = _exigir_directorio(images_dir() if directorio is None else directorio, "imágenes")
    encontradas: list[Muestra] = []
    for clase_dir in raiz.iterdir():
        if not clase_dir.is_dir() or clase_dir.name.startswith("."):
            continue
        clase = clase_dir.name
        if clase not in _CLASES:
            continue
        for archivo in clase_dir.iterdir():
            if not _es_archivo_visible(archivo):
                continue
            if archivo.suffix.lower() not in _SUFIJOS_IMAGEN:
                continue
            encontradas.append(
                Muestra(id=archivo.relative_to(raiz).as_posix(), ruta=archivo, clase=clase)
            )
    encontradas.sort(key=lambda muestra: muestra.id)
    return encontradas


def listar_documentos(directorio: Path | None = None) -> list[Muestra]:
    """Lista los ``.txt``. La clase es el nombre sin dígitos, en minúsculas.

    ``rose3.txt`` → clase ``rose``. Se descarta un archivo cuya clase no
    esté en :data:`flowers.config.CLASSES`. El ``id`` es el nombre del archivo.
    """
    raiz = _exigir_directorio(docs_dir() if directorio is None else directorio, "documentos")
    encontradas: list[Muestra] = []
    for archivo in raiz.iterdir():
        if not _es_archivo_visible(archivo) or archivo.suffix.lower() != ".txt":
            continue
        clase = _DIGITOS.sub("", archivo.stem).lower()
        if clase not in _CLASES:
            continue
        encontradas.append(Muestra(id=archivo.name, ruta=archivo, clase=clase))
    encontradas.sort(key=lambda muestra: muestra.id)
    return encontradas


def split_estratificado(
    registros: list[Muestra] | tuple[Muestra, ...],
    *,
    fraccion: float = FRACCION_HOLDOUT,
    semilla: int = SEMILLA,
) -> Particion:
    """Separa cada clase en train y holdout.

    Por clase se reserva ``round(n * fraccion)`` muestras para el holdout
    (``round`` de Python: empate al entero par). Dentro de cada clase el
    orden de entrada no importa: se ordena por ``id`` y luego se baraja
    con ``semilla``. Train y holdout salen ordenados por ``id``.
    """
    if not 0 < fraccion < 1:
        raise ValueError(f"fraccion debe estar entre 0 y 1, no {fraccion}")

    grupos: dict[str, list[Muestra]] = {}
    for registro in registros:
        grupos.setdefault(registro.clase, []).append(registro)

    rng = random.Random(semilla)
    train: list[Muestra] = []
    holdout: list[Muestra] = []
    for clase in sorted(grupos):
        grupo = sorted(grupos[clase], key=lambda muestra: muestra.id)
        rng.shuffle(grupo)
        n_holdout = round(len(grupo) * fraccion)
        holdout.extend(grupo[:n_holdout])
        train.extend(grupo[n_holdout:])

    train.sort(key=lambda muestra: muestra.id)
    holdout.sort(key=lambda muestra: muestra.id)
    return Particion(train=tuple(train), holdout=tuple(holdout))


def _exigir_directorio(directorio: Path, etiqueta: str) -> Path:
    raiz = Path(directorio)
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe el directorio de {etiqueta}: {raiz}")
    return raiz


def _es_archivo_visible(archivo: Path) -> bool:
    return archivo.is_file() and not archivo.name.startswith(".")
