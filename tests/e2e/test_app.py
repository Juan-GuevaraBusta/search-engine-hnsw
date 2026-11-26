"""Pruebas de extremo a extremo del dashboard con `streamlit.testing.v1.AppTest`.

AppTest ejecuta el script de Streamlit en el mismo proceso, sin navegador.
Los tests inyectan un servicio con codificadores falsos mediante
``FLOWERS_SERVICE_FACTORY`` (ver ``tests/e2e/fakes.py``); solo la prueba
marcada ``integration`` usa los artefactos y modelos reales.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from flowers.service import crear_servicio
from tests.e2e import fakes

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
TIMEOUT = 60


@pytest.fixture
def app_falsa(tmp_path, monkeypatch) -> AppTest:
    """App lista con artefactos sintéticos y codificadores falsos."""
    fakes.construir_artefactos(tmp_path / "artifacts")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FLOWERS_SERVICE_FACTORY", "tests.e2e.fakes:crear_falso")
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def _metricas(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_arranca_sin_errores(app_falsa: AppTest) -> None:
    assert not app_falsa.exception
    assert [t.label for t in app_falsa.tabs] == [
        "📷 Subir foto",
        "✍️ Escribir pregunta",
        "🔧 Cómo funciona",
    ]
    assert app_falsa.slider(key="k").value == 10
    assert app_falsa.slider(key="ef_search").value == 50


def test_sin_artefactos_pide_make_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "vacio"))
    monkeypatch.delenv("FLOWERS_SERVICE_FACTORY", raising=False)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("make index" in e.value for e in at.error)


def test_foto_de_ejemplo_clasifica_y_muestra_vecinos(app_falsa: AppTest) -> None:
    app_falsa.button(key="ej_foto_rose").click().run()
    assert not app_falsa.exception
    metricas = _metricas(app_falsa)
    assert metricas["Clase ganadora"] == "Rosa"
    # Panel HNSW vs exacto y pestaña "Cómo funciona" aparecen tras la consulta.
    assert {"HNSW", "Exacto", "Tamaño del vector", "Norma del vector"} <= set(metricas)
    assert metricas["Tamaño del vector"] == f"{fakes.DIM} números"
    assert metricas["Norma del vector"] == "1.0000"
    assert any("coinciden" in s.value for s in app_falsa.success)


def test_cambiar_k_cambia_los_vecinos(app_falsa: AppTest) -> None:
    app_falsa.button(key="ej_foto_tulip").click().run()
    assert "Los 10 vecinos" in " ".join(s.value for s in app_falsa.subheader)
    app_falsa.slider(key="k").set_value(3).run()
    assert not app_falsa.exception
    assert "Los 3 vecinos" in " ".join(s.value for s in app_falsa.subheader)
    assert _metricas(app_falsa)["Clase ganadora"] == "Tulipán"


def test_pregunta_de_texto_da_fotos_y_documentos(app_falsa: AppTest) -> None:
    app_falsa.text_input(key="texto_pregunta").set_value("a red rose").run()
    app_falsa.button(key="buscar_texto").click().run()
    assert not app_falsa.exception
    assert _metricas(app_falsa)["Clase ganadora"] == "Rosa"
    # El documento más cercano es el de la clase rose y se muestra su texto.
    textos = " ".join(m.value for m in app_falsa.markdown)
    assert "rose1.txt" in textos
    assert any("Documento falso sobre rose" in m.value for m in app_falsa.markdown)


def test_pregunta_de_ejemplo(app_falsa: AppTest) -> None:
    app_falsa.button(key="ej_txt_sunflower").click().run()
    assert not app_falsa.exception
    assert app_falsa.text_input(key="texto_pregunta").value.startswith("a tall sunflower")
    assert _metricas(app_falsa)["Clase ganadora"] == "Girasol"


def test_pregunta_vacia_avisa(app_falsa: AppTest) -> None:
    app_falsa.button(key="buscar_texto").click().run()
    assert not app_falsa.exception
    assert any("Escribe algo" in w.value for w in app_falsa.warning)


def test_respaldo_de_texto_no_mezcla_espacios(tmp_path, monkeypatch) -> None:
    fakes.construir_artefactos(tmp_path / "artifacts")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FLOWERS_SERVICE_FACTORY", "tests.e2e.fakes:crear_falso_con_respaldo")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.button(key="ej_txt_rose").click().run()
    assert not at.exception
    assert any("No se comparan espacios distintos" in w.value for w in at.warning)
    # Aun así las fotos se buscan bien (SigLIP texto -> fotos).
    assert _metricas(at)["Clase ganadora"] == "Rosa"


def test_subir_foto(app_falsa: AppTest) -> None:
    """Sube un PNG (rojo 100 -> 'rose' para el codificador falso) por el uploader."""
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (100, 100, 100)).save(buffer, format="PNG")
    app_falsa.file_uploader(key="subida").set_value(("rosa.png", buffer.getvalue(), "image/png"))
    app_falsa.run()
    assert not app_falsa.exception
    assert _metricas(app_falsa)["Clase ganadora"] == "Rosa"
    assert _metricas(app_falsa)["Tamaño del vector"] == f"{fakes.DIM} números"


def test_servicio_compara_hnsw_con_exacto(tmp_path) -> None:
    fakes.construir_artefactos(tmp_path)
    s = crear_servicio(tmp_path, "tests.e2e.fakes:crear_falso")
    vector, _ = s.codificar_foto(Image.new("RGB", (16, 16), (100, 100, 100)))
    c = s.consultar_imagenes(vector, tipo="foto", etiqueta="x", k=5, ef_search=50)
    assert c.resultado.clase == "rose"
    assert c.coinciden and c.recall == 1.0
    assert c.resultado.comparaciones_hnsw < c.resultado.comparaciones_exacto


@pytest.mark.integration
def test_app_con_artefactos_reales_consulta_de_texto(monkeypatch) -> None:
    """Corre la app real (SigLIP + EmbeddingGemma + índices reales)."""
    from flowers import config

    carpeta = config.artifacts_dir()
    if not (carpeta / "images_hnsw.npz").is_file():
        pytest.skip("no hay artefactos reales; ejecuta `make index` o define ARTIFACTS_DIR")
    monkeypatch.delenv("FLOWERS_SERVICE_FACTORY", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    at.button(key="ej_txt_sunflower").click().run()
    assert not at.exception
    assert _metricas(at)["Clase ganadora"] == "Girasol"
    assert not at.warning, [w.value for w in at.warning]
    assert any("sunflower" in m.value for m in at.markdown)  # documento cercano
