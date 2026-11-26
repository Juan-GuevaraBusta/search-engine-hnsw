# search-engine-hnsw

Buscador **100 % local** de flores. Escribes una pregunta o subes una foto, y el sistema te dice **qué flor es** y **qué tan seguro está**, buscando entre 2 334 fotos y 25 documentos con un índice **HNSW** (vecinos más cercanos por similitud coseno) hecho a mano en NumPy.

- Embeddings de Google, locales: **SigLIP** (fotos y texto) y **EmbeddingGemma** (documentos).
- Búsqueda: HNSW propio + comparación contra `hnswlib` y la búsqueda exacta de `scikit-learn`.
- Dashboard: **Streamlit**.
- Todo explicado paso a paso, para principiantes, en **[`docs/PRD.md`](docs/PRD.md)** (con diagramas Mermaid).

## Resultados (holdout de 412 fotos que el índice nunca vio)

| Métrica | Resultado |
|---|---|
| Accuracy top-1 | **97,09 %** |
| Recall@5 de clase | 98,06 % |
| Recall HNSW vs búsqueda exacta (k=10) | 99,85 % |
| Latencia HNSW | 0,55 ms (4–5× más rápido que scikit-learn) |
| Texto → clase correcta | 5/5 en fotos y 5/5 en documentos |

Con este tamaño de corpus, una multiplicación de matrices con NumPy puro es aún más rápida que el HNSW propio; HNSW se justifica al crecer `n`. Detalles y hallazgos honestos en la sección 8 del PRD.

## Cómo correrlo

Requisitos: macOS/Linux, [`uv`](https://docs.astral.sh/uv/), Python 3.12 y los datos en `~/Downloads/archive/train/<clase>/*.jpg` y `~/Downloads/documentos/*.txt` (rutas configurables con `FLOWERS_IMAGES_DIR`, `FLOWERS_DOCS_DIR` y `ARTIFACTS_DIR`).

```bash
make setup     # crea el entorno en ~/.venvs/search-engine-hnsw e instala dependencias
make models    # descarga SigLIP y EmbeddingGemma (una vez; EmbeddingGemma pide aceptar su licencia y `hf auth login`)
make index     # vectoriza todo y construye los índices HNSW (~3 min)
make test      # corre las pruebas
make evaluate  # accuracy, recall y matriz de confusión -> docs/results/
make bench     # fuerza bruta vs HNSW -> docs/results/
make demo      # abre el dashboard en http://localhost:8501
```

> El entorno virtual vive **fuera** de `~/Documents` a propósito: esa carpeta se sincroniza con iCloud y PyTorch tardaba 75 s en importarse (contra 5 s fuera).
>
> Si no aceptas la licencia de EmbeddingGemma, usa `FLOWERS_TEXT_FALLBACK=siglip` (por defecto) y los documentos se vectorizan con el texto de SigLIP.

## Estructura

```
docs/PRD.md          PRD didáctico con teoría, arquitectura, demo paso a paso y resultados
docs/results/        evaluation.json, benchmark.json, matriz de confusión y gráficas
src/flowers/         config, data, embeddings, hnsw, index, search, metrics, evaluation, service
app/streamlit_app.py dashboard
scripts/             build_index, evaluate, bench, download_models
tests/               unit, integration, e2e (80 pruebas)
```
