"""Construye los índices HNSW de fotos y documentos (``make index``).

Uso::

    PYTHONPATH=src python scripts/build_index.py
    PYTHONPATH=src python scripts/build_index.py --max-por-clase 20   # prueba rápida

Si se interrumpe, volver a correrlo reanuda: las fotos ya codificadas
quedan en ``artifacts/cache/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flowers.index import build_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-por-clase", type=int, default=None, help="limita las fotos por clase (pruebas)"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="destino (por defecto config)")
    parser.add_argument("--silencioso", action="store_true", help="sin mensajes de progreso")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    resumen = build_all(
        progress=not args.silencioso,
        artifacts_dir=args.artifacts_dir,
        max_por_clase=args.max_por_clase,
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
