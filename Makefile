.PHONY: setup test models index evaluate bench demo

setup:
	uv venv --python 3.12 .venv && uv pip install -e ".[dev]"

test:
	.venv/bin/pytest

models:
	.venv/bin/python scripts/download_models.py

index:
	.venv/bin/python scripts/build_index.py

evaluate:
	.venv/bin/python scripts/evaluate.py

bench:
	.venv/bin/python scripts/bench.py

demo:
	.venv/bin/streamlit run app/streamlit_app.py
