"""Descarga SigLIP y EmbeddingGemma a la caché de Hugging Face.

Uso (sin instalar el paquete)::

    PYTHONPATH=src python scripts/download_models.py

Si EmbeddingGemma está gated y falta la licencia o el token, el script
no aborta. Imprime el error exacto y deja el respaldo documentado:
``TextEncoder`` usa el texto de SigLIP cuando
``FLOWERS_TEXT_FALLBACK=siglip`` (valor por defecto).
``FLOWERS_TEXT_FALLBACK=error`` hace que el codificador propague el fallo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flowers.embeddings import GEMMA_MODEL_ID, SIGLIP_MODEL_ID, is_model_access_error

_IGNORE = [
    "*.h5",
    "*.ot",
    "*.msgpack",
    "flax_model*",
    "tf_model*",
    "onnx/*",
    "openvino/*",
    "rust_model*",
    "pytorch_model.bin",
]


def download_models() -> int:
    """Baja los dos modelos. Devuelve 0 si SigLIP quedó en caché."""
    from huggingface_hub import snapshot_download

    print(f"Descargando {SIGLIP_MODEL_ID} ...", flush=True)
    snapshot_download(SIGLIP_MODEL_ID, ignore_patterns=_IGNORE)
    print(f"Listo: {SIGLIP_MODEL_ID}", flush=True)

    print(f"Descargando {GEMMA_MODEL_ID} ...", flush=True)
    try:
        snapshot_download(GEMMA_MODEL_ID, ignore_patterns=_IGNORE)
    except Exception as exc:
        if not is_model_access_error(exc):
            raise
        print(
            "EmbeddingGemma no se descargó. El modelo es gated: falta aceptar "
            "la licencia o un token de Hugging Face.\n"
            "Respaldo activo: TextEncoder codifica texto con SigLIP "
            "(FLOWERS_TEXT_FALLBACK=siglip). Esos vectores no se comparan "
            "con un índice construido con EmbeddingGemma.\n"
            f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 0
    print(f"Listo: {GEMMA_MODEL_ID}", flush=True)
    return 0


def main() -> None:
    sys.exit(download_models())


if __name__ == "__main__":
    main()
