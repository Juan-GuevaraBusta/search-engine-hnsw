# search-engine-hnsw

## Proyecto final — Asignatura Modelos Matemáticos para IA

**Presentado por:** Juan Bustamante, María Antonia Gil e Isabela Ramírez

---

## ¿De qué trata el proyecto?

Un buscador **100 % local** de flores. Escribes una pregunta (por ejemplo, *"a red rose flower with soft petals"*) o subes la foto de una flor, y el sistema te dice **qué flor es** (margarita, diente de león, rosa, girasol o tulipán) y **qué tan seguro está**, mostrando las fotos y los documentos más parecidos que tiene guardados.

La idea central de la asignatura que aplicamos aquí: convertir fotos y textos en **vectores** (listas de números), medir su parecido con la **similitud coseno** y encontrar los vecinos más cercanos con un índice **HNSW** (*Hierarchical Navigable Small World*), en vez de comparar contra todo el corpus.

- **Embeddings de Google, corriendo local:** SigLIP (fotos y texto) y EmbeddingGemma (documentos).
- **Índice HNSW propio**, escrito en NumPy y comentado para poder leerlo como material de clase. Se valida contra `hnswlib` y contra la búsqueda exacta de `scikit-learn`.
- **Dashboard** en Streamlit para la demo.
- **Explicación completa** para principiantes, con analogías de Estructuras de Datos y Algoritmos y diagramas Mermaid, en **[`docs/PRD.md`](docs/PRD.md)**.

### Resultados (holdout de 412 fotos que el índice nunca vio)

| Métrica | Resultado | Meta |
|---|---|---|
| Accuracy top-1 | **97,09 %** | ≥ 90 % |
| Recall@5 de clase | 98,06 % | ≥ 97 % |
| Recall del HNSW frente a la búsqueda exacta (k = 10) | 99,85 % | ≥ 95 % |
| Latencia de consulta HNSW | 0,55 ms | < 50 ms |
| Texto → clase correcta (5 consultas) | 5/5 en fotos y 5/5 en documentos | 5/5 |

Con 2 334 fotos, una multiplicación de matrices con NumPy puro todavía es más rápida que nuestro HNSW (que está escrito en Python para que se pueda leer). HNSW se justifica cuando el corpus crece. Los hallazgos completos, incluidos los que no salen bien, están en la sección 8 del PRD.

---

## Cómo levantar el proyecto

### 1. Requisitos

| Requisito | Detalle |
|---|---|
| Sistema | macOS o Linux (probado en macOS) |
| Python | 3.12 (el proyecto exige `>=3.12,<3.14`) |
| [`uv`](https://docs.astral.sh/uv/) | gestor de entornos; instala Python 3.12 si falta |
| `make` | para los atajos |
| Espacio | ~3 GB (PyTorch y los modelos) |
| Internet | **solo** la primera vez, para descargar dependencias y modelos |

### 2. Obtener el código

```bash
git clone https://github.com/Juan-GuevaraBusta/search-engine-hnsw.git
cd search-engine-hnsw
```

### 3. Preparar los datos

Los datos **no están en el repositorio** (pesan ~216 MB). Se leen de estas rutas, que se pueden cambiar con variables de entorno:

| Qué | Ruta por defecto | Variable para cambiarla |
|---|---|---|
| Fotos de entrenamiento | `~/Downloads/archive/train/<clase>/*.jpg` | `FLOWERS_IMAGES_DIR` |
| Documentos | `~/Downloads/documentos/*.txt` | `FLOWERS_DOCS_DIR` |
| Índices generados | `./artifacts` | `ARTIFACTS_DIR` |

Las clases son `daisy`, `dandelion`, `rose`, `sunflower` y `tulip` (una subcarpeta por clase). Los documentos se llaman `rose1.txt`, `daisy3.txt`, etc.: la clase es el nombre sin el número. El programa **solo lee** estas carpetas, nunca las modifica.

### 4. Crear el entorno

```bash
make setup
```

Crea el entorno virtual en `~/.venvs/search-engine-hnsw` e instala las dependencias (PyTorch, transformers, sentence-transformers, hnswlib, scikit-learn, Streamlit, plotly, pytest…).

> **¿Por qué el entorno está fuera del proyecto?** En macOS, `~/Documents` se sincroniza con iCloud. Con el entorno dentro del proyecto, importar PyTorch tardaba 75 s; fuera tarda 5 s. Si quieres otra ubicación: `make setup VENV=/otra/ruta`.

### 5. Descargar los modelos

```bash
make models
```

- **SigLIP** (`google/siglip-base-patch16-224`) se descarga directamente.
- **EmbeddingGemma** (`google/embeddinggemma-300m`) es un modelo *gated*: antes hay que
  1. aceptar la licencia en https://huggingface.co/google/embeddinggemma-300m con tu cuenta de Hugging Face, y
  2. iniciar sesión con un token de lectura (créalo en https://huggingface.co/settings/tokens):
     ```bash
     curl -LsSf https://hf.co/cli/install.sh | bash   # instala el comando `hf`
     hf auth login --token TU_TOKEN
     ```

  Sin eso, el sistema **sigue funcionando**: los documentos se vectorizan con el texto de SigLIP (`FLOWERS_TEXT_FALLBACK=siglip`, que es el valor por defecto), aunque con menor calidad semántica.

Después de descargarlos, el proyecto corre sin internet (`HF_HUB_OFFLINE=1`).

### 6. Construir los índices

```bash
make index
```

Vectoriza las 2 746 fotos y los 25 documentos y construye los dos índices HNSW (`M=16`, `efConstruction=200`). Tarda unos **3 minutos**. Se separa un 15 % de las fotos (412) como *holdout* para evaluar, y no se indexan. Si se interrumpe, al volver a correr **reanuda** desde la caché en `artifacts/cache/`.

Resultado en `artifacts/`: `images_hnsw.npz`, `docs_hnsw.npz`, los vectores `.npy` y los metadatos `.json`.

### 7. Abrir el dashboard

```bash
make demo
```

Abre http://localhost:8501. Qué se puede hacer:

- **Subir foto:** clase ganadora, gráfico de confianza por clase y los vecinos con miniatura, similitud coseno y clase.
- **Escribir pregunta** (mejor en inglés): fotos más cercanas (SigLIP) y documentos más cercanos (EmbeddingGemma).
- **Panel lateral:** ajustar `k` y `efSearch`, y ver HNSW contra búsqueda exacta (milisegundos, comparaciones y si los vecinos coinciden).
- **Botones de ejemplo:** una foto del holdout y una pregunta por clase, para una demo rápida.
- **Cómo funciona:** qué pasó por detrás en la última consulta.

Si faltan los índices, la app lo avisa y pide correr `make index`.

---

## Cómo hacer las pruebas

El proyecto tiene unas **80 pruebas** en tres niveles. Se corren con `pytest` a través de `make`:

```bash
make test
```

| Nivel | Carpeta | Qué verifica |
|---|---|---|
| **Unitarias** | `tests/unit/` | coseno y normalización, el HNSW con datos sintéticos (orden, ids válidos, guardar/cargar), lector de datos y split estratificado, voto ponderado y confianza, métricas, construcción de índices con codificadores falsos |
| **Integración** | `tests/integration/` | embeddings reales (forma, norma 1, "foto de rosa" más cerca de *"a rose"* que de *"a sunflower"*), recall del HNSW propio contra scikit-learn y `hnswlib`, construcción de índices con un subconjunto de fotos reales |
| **E2E de la interfaz** | `tests/e2e/` | el dashboard con `streamlit.testing.AppTest`: subir una foto, escribir una pregunta, cambiar `k`, pregunta vacía y falta de artefactos |

Comandos útiles:

```bash
# Solo lo rápido (sin modelos ni datos reales)
PYTHONPATH=src ~/.venvs/search-engine-hnsw/bin/python -m pytest tests/unit

# Una carpeta o un archivo concreto
PYTHONPATH=src ~/.venvs/search-engine-hnsw/bin/python -m pytest tests/e2e -v

# Todo excepto las pruebas que usan los datos reales
PYTHONPATH=src ~/.venvs/search-engine-hnsw/bin/python -m pytest -m "not integration"

# Todo, incluida la prueba de la app con los índices reales
ARTIFACTS_DIR=$PWD/artifacts PYTHONPATH=src ~/.venvs/search-engine-hnsw/bin/python -m pytest
```

Las pruebas marcadas `integration` usan el corpus real y se **omiten solas** si no están las fotos, los modelos o los índices. La prueba de EmbeddingGemma se omite si no aceptaste la licencia. Las pruebas de integración con modelos cargan PyTorch, por eso la suite completa tarda cerca de **1,5 minutos**.

### Evaluar la calidad y la velocidad

Estas dos corridas no son pruebas de `pytest`: miden el sistema y guardan los resultados en `docs/results/`.

```bash
make evaluate   # accuracy, recall@5, matriz de confusión, consultas de texto
make bench      # fuerza bruta vs HNSW al variar efSearch y el tamaño del corpus
```

Generan `evaluation.json`, `benchmark.json`, `confusion_matrix.png` y `benchmark.png`. Necesitan que `make index` ya se haya corrido.

### Si algo falla

| Síntoma | Causa probable | Solución |
|---|---|---|
| `GatedRepoError 401` | falta aceptar la licencia de EmbeddingGemma o iniciar sesión | pasos de la sección 5, o usar el respaldo `FLOWERS_TEXT_FALLBACK=siglip` |
| `SiglipTokenizer requires the SentencePiece library` | entorno incompleto | `make setup` de nuevo |
| Imports de PyTorch muy lentos (> 1 min) | entorno dentro de una carpeta sincronizada con iCloud | usar el entorno por defecto en `~/.venvs/` |
| La app dice "ejecuta `make index`" | no existen los índices | `make index` |
| Pruebas `integration` omitidas | no hay datos, modelos o índices | es normal; revisa las secciones 3, 5 y 6 |

---

## Estructura del repositorio

```
docs/PRD.md            PRD didáctico: teoría, arquitectura, demo paso a paso, resultados
docs/results/          evaluation.json, benchmark.json, matriz de confusión y gráficas
src/flowers/           config, data, embeddings, hnsw, index, search, metrics, evaluation, service
app/streamlit_app.py   dashboard
scripts/               build_index, evaluate, bench, download_models
tests/                 unit, integration, e2e
Makefile               setup, models, index, test, evaluate, bench, demo
```

## Documentación

- **[`docs/PRD.md`](docs/PRD.md):** explica todo el proceso, desde qué es un embedding hasta cómo se construye y se recorre el grafo HNSW, comparado con skip lists, grafos y búsqueda lineal.
- **[`docs/results/`](docs/results/):** los números reales de la evaluación y el benchmark.
