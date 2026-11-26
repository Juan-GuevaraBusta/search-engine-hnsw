"""Evaluación del buscador (PRD 4.9 y 8): accuracy, recall, confusión y texto.

Uso::

    ARTIFACTS_DIR=artifacts PYTHONPATH=src python scripts/evaluate.py

Guarda ``docs/results/evaluation.json`` y ``docs/results/confusion_matrix.png``.
Toda la lógica está en ``flowers.evaluation``; aquí solo se cargan artefactos,
se codifican las 5 consultas de texto y se escriben los resultados.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from flowers import config
from flowers.evaluation import (
    CONSULTAS_EXTRA,
    CONSULTAS_TEXTO,
    evaluar_clasificacion,
    evaluar_texto,
    graficar_confusion,
)
from flowers.index import load_holdout
from flowers.search import SearchEngine


def _codificar_consultas(artifacts: Path, frases: list[str]):
    """Vectores de las 5 frases: SigLIP (texto->fotos) y EmbeddingGemma (texto->docs).

    Los documentos se indexaron con el backend que dice ``index_info.json``; la
    consulta DEBE usar el mismo, o los vectores vivirían en espacios distintos.
    """
    from flowers.embeddings import ImageEncoder, TextEncoder

    siglip = ImageEncoder().encode_text_siglip(frases)

    info = json.loads((artifacts / "index_info.json").read_text(encoding="utf-8"))
    backend_indice = info.get("documentos", {}).get("backend_texto")
    codificador = TextEncoder(fallback="error" if backend_indice == "embeddinggemma" else "siglip")
    docs = codificador.encode(frases)
    if backend_indice and codificador.backend != backend_indice:
        raise RuntimeError(
            f"El índice de docs usa {backend_indice} pero la consulta usó {codificador.backend}"
        )
    return siglip, docs, codificador.backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=_RAIZ / "docs" / "results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--sin-texto", action="store_true", help="omite la evaluación de texto")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    artifacts = args.artifacts_dir or config.artifacts_dir()
    args.salida.mkdir(parents=True, exist_ok=True)

    motor = SearchEngine.load(artifacts)
    vectores, meta = load_holdout(artifacts)
    reales = [m["clase"] for m in meta]

    print(f"holdout: {len(vectores)} fotos; k={args.k}, ef_search={args.ef_search}", file=sys.stderr)
    resultado = {
        "clasificacion": evaluar_clasificacion(
            motor, vectores, reales, k=args.k, ef_search=args.ef_search
        )
    }
    c = resultado["clasificacion"]
    graficar_confusion(
        c["matriz_confusion"], c["clases"], args.salida / "confusion_matrix.png",
        titulo=f"Confusión HNSW (k={args.k}, ef={args.ef_search}, acc={c['accuracy_top1']:.1%})",
    )

    if not args.sin_texto:
        tiene_docs = motor._docs is not None
        for clave, consultas in (("texto", CONSULTAS_TEXTO), ("texto_extra", CONSULTAS_EXTRA)):
            siglip, docs, backend = _codificar_consultas(Path(artifacts), [f for f, _ in consultas])
            resultado[clave] = evaluar_texto(
                motor, consultas, siglip, docs if tiene_docs else None, ef_search=args.ef_search
            )
            resultado[clave]["backend_documentos"] = backend

    destino = args.salida / "evaluation.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"accuracy top-1      : {c['accuracy_top1']:.2%}")
    print(f"recall@5 de clase   : {c['recall_at_5_clase']:.2%}")
    print(f"recall HNSW@{args.k}     : {c[f'recall_hnsw_vs_exacto_at_{args.k}']:.2%}")
    print(f"latencia HNSW/exacta: {c['ms_hnsw_media']:.2f} ms / {c['ms_exacto_media']:.2f} ms")
    print(f"comparaciones       : {c['comparaciones_hnsw_media']:.0f} vs {c['comparaciones_exacto']}")
    if "texto" in resultado:
        t = resultado["texto"]
        print(f"texto->fotos        : {t['aciertos_imagenes']}/{t['total']}")
        if "aciertos_documentos" in t:
            print(f"texto->documentos   : {t['aciertos_documentos']}/{t['total']}")
    if "texto_extra" in resultado:
        e = resultado["texto_extra"]["consultas"][0]
        print(f"extra '{e['consulta']}': fotos={e['imagenes']['clase_top1']} "
              f"({e['imagenes']['confianza']:.0%}), docs={e['documentos']['clase_top1']}")
    print(f"guardado en {destino}")


if __name__ == "__main__":
    main()
