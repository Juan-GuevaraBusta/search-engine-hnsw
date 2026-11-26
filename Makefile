# El entorno virtual vive FUERA de ~/Documents: esa carpeta se sincroniza con
# iCloud y torch (cientos de MB) volvia lentisimos los imports (75 s vs 5 s).
VENV ?= $(HOME)/.venvs/search-engine-hnsw
PY := $(VENV)/bin/python

.PHONY: setup test models index evaluate bench demo

setup:
	uv venv --python 3.12 $(VENV) && VIRTUAL_ENV=$(VENV) uv pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PY) -m pytest

models:
	PYTHONPATH=src $(PY) scripts/download_models.py

index:
	PYTHONPATH=src $(PY) scripts/build_index.py

evaluate:
	PYTHONPATH=src $(PY) scripts/evaluate.py

bench:
	PYTHONPATH=src $(PY) scripts/bench.py

demo:
	PYTHONPATH=src $(VENV)/bin/streamlit run app/streamlit_app.py
