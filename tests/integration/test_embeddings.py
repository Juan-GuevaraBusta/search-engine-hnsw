"""Pruebas de integración de SigLIP y EmbeddingGemma."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flowers.config import images_dir
from flowers.embeddings import ImageEncoder, TextEncoder

SIGLIP_DIM = 768
_NORM_ATOL = 1e-4


def _assert_unit_batch(vectors: np.ndarray, n: int, dim: int | None = None) -> None:
    assert isinstance(vectors, np.ndarray)
    assert vectors.dtype == np.float32
    assert vectors.ndim == 2
    assert vectors.shape[0] == n
    if dim is None:
        assert vectors.shape[1] > 0
    else:
        assert vectors.shape[1] == dim
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, np.ones(n, dtype=np.float32), atol=_NORM_ATOL)


def _rose_path() -> Path:
    roses = sorted((images_dir() / "rose").glob("*.jpg"))
    if not roses:
        pytest.skip(f"no hay fotos de rosa en {images_dir() / 'rose'}")
    return roses[0]


@pytest.fixture(scope="module")
def image_encoder() -> ImageEncoder:
    return ImageEncoder()


def test_dispositivo_es_mps_si_existe_si_no_cpu() -> None:
    encoder = ImageEncoder()
    if torch.backends.mps.is_available():
        assert encoder.device == "mps"
    else:
        assert encoder.device == "cpu"


def test_carga_perezosa_no_crea_el_modelo_al_construir() -> None:
    encoder = ImageEncoder()
    assert encoder.model is None


def test_encode_images_forma_norma_y_acepta_ruta_o_pil(image_encoder: ImageEncoder) -> None:
    rose = _rose_path()
    with Image.open(rose) as raw:
        pil = raw.convert("RGB")
    vectors = image_encoder.encode_images([rose, pil], batch_size=1)
    _assert_unit_batch(vectors, n=2, dim=SIGLIP_DIM)
    assert float(np.dot(vectors[0], vectors[1])) > 0.99


def test_encode_text_siglip_forma_y_norma(image_encoder: ImageEncoder) -> None:
    vectors = image_encoder.encode_text_siglip(["a rose", "a sunflower"])
    _assert_unit_batch(vectors, n=2, dim=SIGLIP_DIM)


def test_foto_de_rosa_mas_cerca_de_a_rose_que_de_a_sunflower(
    image_encoder: ImageEncoder,
) -> None:
    image = image_encoder.encode_images([_rose_path()], batch_size=1)
    texts = image_encoder.encode_text_siglip(["a rose", "a sunflower"])
    rose_sim = float(np.dot(image[0], texts[0]))
    sunflower_sim = float(np.dot(image[0], texts[1]))
    assert rose_sim > sunflower_sim


def test_text_encoder_devuelve_vectores_normalizados() -> None:
    encoder = TextEncoder()
    vectors = encoder.encode(["a rose", "a sunflower"])
    _assert_unit_batch(vectors, n=2)
    assert encoder.backend in {"embeddinggemma", "siglip"}
    if encoder.backend == "siglip":
        assert encoder.load_error is not None


def test_embeddinggemma_forma_y_norma() -> None:
    encoder = TextEncoder(fallback="error")
    try:
        vectors = encoder.encode(["a rose is a woody perennial"])
    except Exception as exc:
        pytest.skip(
            "EmbeddingGemma no disponible: falta licencia o token. "
            f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        )
    assert encoder.backend == "embeddinggemma"
    _assert_unit_batch(vectors, n=1)
