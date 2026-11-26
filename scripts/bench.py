"""Benchmark: fuerza bruta vs HNSW (PRD 8) variando ef_search y n.

Uso::

    ARTIFACTS_DIR=artifacts PYTHONPATH=src python scripts/bench.py

Guarda ``docs/results/benchmark.json`` y ``docs/results/benchmark.png``.
Las consultas son las fotos del holdout (no están en el índice).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from flowers import config
from flowers.evaluation import bench_crecimiento, bench_ef_search, graficar_benchmark
from flowers.index import load_holdout, load_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=_RAIZ / "docs" / "results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=50, help="ef para la curva de n")
    parser.add_argument("--tamanos", type=int, nargs="+", default=[500, 1000, 2000])
    args = parser.parse_args()

    artifacts = args.artifacts_dir or config.artifacts_dir()
    args.salida.mkdir(parents=True, exist_ok=True)

    indice = load_index("images", artifacts)
    consultas, _ = load_holdout(artifacts)
    n = len(indice.vectors)
    print(f"n={n}, consultas={len(consultas)}, k={args.k}", file=sys.stderr)

    ef = bench_ef_search(indice.hnsw, indice.vectors, consultas, k=args.k)
    crecimiento = bench_crecimiento(
        indice.vectors, consultas, [*args.tamanos, n],
        k=args.k, ef_search=args.ef_search, hnsw_completo=indice.hnsw,
        progreso=lambda m: print(m, file=sys.stderr, flush=True),
    )

    resultado = {"ef_search": ef, "crecimiento_n": crecimiento}
    (args.salida / "benchmark.json").write_text(
        json.dumps(resultado, indent=2) + "\n", encoding="utf-8"
    )
    graficar_benchmark(ef, crecimiento, args.salida / "benchmark.png")

    print(f"exacto sklearn: {ef['exacto_sklearn_ms']['media']:.2f} ms | numpy: {ef['exacto_numpy_ms']['media']:.2f} ms | comparaciones: {n}")
    for f in ef["hnsw"]:
        print(f"ef={f['ef_search']:>3}: {f['ms']['media']:.2f} ms (p95 {f['ms']['p95']:.2f}), recall {f['recall']:.2%}, "
              f"comps {f['comparaciones_media']:.0f}, x{f['speedup_vs_sklearn']:.2f} sklearn, x{f['speedup_vs_numpy']:.2f} numpy")
    for c in crecimiento:
        print(f"n={c['n']:>5}: hnsw {c['hnsw_ms']:.2f} ms, sklearn {c['exacto_sklearn_ms']:.2f} ms, numpy {c['exacto_numpy_ms']:.2f} ms, recall {c['recall']:.2%}, comps {c['comparaciones_hnsw']:.0f}")


if __name__ == "__main__":
    main()
