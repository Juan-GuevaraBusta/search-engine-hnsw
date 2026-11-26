"""Codificadores de fotos y texto.

SigLIP (`google/siglip-base-patch16-224`) pone fotos y frases en el mismo
espacio. EmbeddingGemma (`google/embeddinggemma-300m`) codifica solo texto.

Los dos devuelven ``float32`` con norma L2 = 1 y forma ``(n, d)``.
La carga de pesos es perezosa: construir la clase no descarga nada.
El dispositivo es ``mps`` si existe; si no, ``cpu``.

Respaldo de EmbeddingGemma
--------------------------
El modelo es gated. Sin licencia aceptada o sin token de Hugging Face
la carga falla. ``TextEncoder`` no se queda trabado: con
``FLOWERS_TEXT_FALLBACK=siglip`` (el valor por defecto) codifica el texto
con SigLIP. Esos vectores viven en el espacio de las fotos, no en el de
los documentos, así que no se mezclan con un índice ya hecho con Gemma.

``FLOWERS_TEXT_FALLBACK=error`` propaga la excepción original.
El mismo valor se puede pasar como ``fallback=`` al constructor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
GEMMA_MODEL_ID = "google/embeddinggemma-300m"

_ACCESS_SNIPPETS = (
    "gated",
    "401 client error",
    "403 client error",
    "401 unauthorized",
    "403 forbidden",
    "is restricted",
    "not in the authorized list",
    "pass a token",
    "provide a token",
    "hf_token",
    "huggingface-cli login",
    "must be authenticated",
    "authentication required",
    "invalid username or password",
    "private repository",
    "awaiting a review",
)

# Una copia por (modelo, dispositivo). Los pesos pesan cientos de MB.
_SIGLIP_CACHE: dict[tuple[str, str], tuple[object, object]] = {}
_GEMMA_CACHE: dict[tuple[str, str], object] = {}
_GEMMA_ERRORS: dict[str, BaseException] = {}

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(device: str | None = None) -> str:
    """``mps`` si el proceso puede usarlo; si no, ``cpu``."""
    if device is not None:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Devuelve una copia ``float32`` con cada fila de norma 1."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"se esperaba una matriz 2D, llegó ndim={matrix.ndim}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, np.float32(1e-12))
    return (matrix / norms).astype(np.float32, copy=False)


def is_model_access_error(exc: BaseException) -> bool:
    """True si ``exc`` (o su causa) es licencia, repo gated o falta de token."""
    try:
        from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError

        access_types: tuple[type, ...] = (GatedRepoError, LocalTokenNotFoundError)
    except ImportError:
        access_types = ()

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if access_types and isinstance(current, access_types):
            return True
        text = f"{type(current).__name__}: {current}".lower()
        if any(snippet in text for snippet in _ACCESS_SNIPPETS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _pooled(output: object) -> object:
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    raise TypeError(
        "SigLIP no devolvió un embedding. "
        f"Tipo recibido: {type(output).__name__}."
    )


def _to_numpy(pooled: object) -> np.ndarray:
    import torch

    if not isinstance(pooled, torch.Tensor):
        raise TypeError(f"se esperaba un tensor, llegó {type(pooled).__name__}")
    return pooled.detach().float().cpu().numpy()


class ImageEncoder:
    """SigLIP. ``encode_images`` y ``encode_text_siglip`` comparten espacio."""

    def __init__(
        self,
        model_id: str = SIGLIP_MODEL_ID,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self._model: object | None = None
        self._processor: object | None = None

    @property
    def model(self) -> object | None:
        """Pesos ya cargados, o ``None`` si nadie ha codificado todavía."""
        return self._model

    def encode_images(
        self,
        paths_or_pil: str | Path | Image.Image | Sequence[str | Path | Image.Image],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Codifica rutas o imágenes PIL. Forma ``(n, d)``, norma 1."""
        images = [_to_rgb(item) for item in _as_items(paths_or_pil)]
        if not images:
            self._ensure_loaded()
            return np.zeros((0, self._width()), dtype=np.float32)
        self._ensure_loaded()
        parts: list[np.ndarray] = []
        for batch in _chunks(images, batch_size):
            parts.append(self._encode_image_batch(batch))
        return l2_normalize(np.concatenate(parts, axis=0))

    def encode_text_siglip(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Codifica frases con el texto de SigLIP. Forma ``(n, d)``, norma 1.

        El relleno es ``max_length``: así se entrenó el modelo (el id de
        pad es el mismo que el de fin de frase).
        """
        rows = _as_strings(texts)
        if not rows:
            self._ensure_loaded()
            return np.zeros((0, self._width()), dtype=np.float32)
        self._ensure_loaded()
        parts: list[np.ndarray] = []
        for batch in _chunks(rows, batch_size):
            parts.append(self._encode_text_batch(batch))
        return l2_normalize(np.concatenate(parts, axis=0))

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        model, processor = _load_siglip(self.model_id, self.device)
        self._model = model
        self._processor = processor

    def _width(self) -> int:
        config = getattr(self._model, "config", None)
        for value in (
            getattr(config, "projection_dim", None),
            getattr(getattr(config, "text_config", None), "projection_size", None),
            getattr(getattr(config, "vision_config", None), "hidden_size", None),
        ):
            if isinstance(value, int) and value > 0:
                return value
        raise RuntimeError(f"no se pudo leer la dimensión de {self.model_id}")

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        processor = self._processor
        model = self._model
        encoded = processor(images=list(images), return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(self.device)
        with torch.inference_mode():
            output = model.get_image_features(pixel_values=pixel_values)
        return _to_numpy(_pooled(output))

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        processor = self._processor
        model = self._model
        encoded = processor(
            text=list(texts),
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        # SigLIP se entrenó con relleno hasta 64 tokens y SIN máscara de atención:
        # pasarla degrada los vectores, así que solo enviamos input_ids.
        with torch.inference_mode():
            output = model.get_text_features(input_ids=input_ids)
        return _to_numpy(_pooled(output))


class TextEncoder:
    """EmbeddingGemma, con respaldo configurable a texto SigLIP.

    ``backend`` queda en ``embeddinggemma`` o ``siglip`` después de la
    primera llamada a ``encode``. Si hubo que usar el respaldo,
    ``load_error`` guarda la excepción original.
    """

    def __init__(
        self,
        model_id: str = GEMMA_MODEL_ID,
        device: str | None = None,
        fallback: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.fallback = _resolve_fallback(fallback)
        self.backend: str | None = None
        self.load_error: BaseException | None = None
        self._gemma: object | None = None
        self._siglip: ImageEncoder | None = None
        self._ready = False

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Codifica textos. Forma ``(n, d)``, norma 1."""
        rows = _as_strings(texts)
        self._ensure_loaded()
        if not rows:
            width = self._width()
            return np.zeros((0, width), dtype=np.float32)
        if self.backend == "embeddinggemma":
            return self._encode_gemma(rows, batch_size)
        assert self._siglip is not None
        return self._siglip.encode_text_siglip(rows, batch_size=batch_size)

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        model, error = _try_load_gemma(self.model_id, self.device)
        if model is not None:
            self._gemma = model
            self.backend = "embeddinggemma"
            self.load_error = None
            self._ready = True
            return
        assert error is not None
        self.load_error = error
        if self.fallback != "siglip":
            raise error
        self._siglip = ImageEncoder(device=self.device)
        self.backend = "siglip"
        self._ready = True

    def _encode_gemma(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        model = self._gemma
        raw = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(raw))

    def _width(self) -> int:
        if self.backend == "embeddinggemma" and self._gemma is not None:
            dimension = getattr(self._gemma, "get_embedding_dimension", None)
            if callable(dimension):
                return int(dimension())
        if self._siglip is not None:
            self._siglip._ensure_loaded()
            return self._siglip._width()
        return 0


def _resolve_fallback(fallback: str | None) -> str:
    chosen = fallback if fallback is not None else os.environ.get("FLOWERS_TEXT_FALLBACK", "siglip")
    if chosen not in {"siglip", "error"}:
        raise ValueError(
            "FLOWERS_TEXT_FALLBACK debe ser 'siglip' o 'error'. "
            f"Llegó {chosen!r}."
        )
    return chosen


def _load_siglip(model_id: str, device: str) -> tuple[object, object]:
    key = (model_id, device)
    cached = _SIGLIP_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    _SIGLIP_CACHE[key] = (model, processor)
    return model, processor


def _try_load_gemma(model_id: str, device: str) -> tuple[object | None, BaseException | None]:
    key = (model_id, device)
    cached = _GEMMA_CACHE.get(key)
    if cached is not None:
        return cached, None
    previous = _GEMMA_ERRORS.get(model_id)
    if previous is not None:
        return None, previous
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id, device=device)
    except Exception as exc:
        if is_model_access_error(exc):
            _GEMMA_ERRORS[model_id] = exc
            return None, exc
        raise
    _GEMMA_CACHE[key] = model
    return model, None


def _as_items(
    paths_or_pil: str | Path | Image.Image | Sequence[str | Path | Image.Image],
) -> list[str | Path | Image.Image]:
    if isinstance(paths_or_pil, (str, Path, Image.Image)):
        return [paths_or_pil]
    return list(paths_or_pil)


def _to_rgb(item: str | Path | Image.Image) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    with Image.open(item) as image:
        return image.convert("RGB")


def _as_strings(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("se esperaba una lista de textos, no un str suelto")
    rows = list(texts)
    if not all(isinstance(row, str) for row in rows):
        raise TypeError("cada texto debe ser str")
    return rows


def _chunks(items: Sequence, batch_size: int) -> list[Sequence]:
    if batch_size < 1:
        raise ValueError(f"batch_size debe ser >= 1, llegó {batch_size}")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
