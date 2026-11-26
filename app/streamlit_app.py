"""Dashboard Streamlit del buscador de flores con HNSW.

Cómo correrlo:  ``make demo``  (o ``streamlit run app/streamlit_app.py``).

Idea para leer este archivo
---------------------------
Streamlit vuelve a ejecutar TODO el script cada vez que tocas un control.
Por eso:

* lo pesado (índice y modelos) se carga una sola vez con ``st.cache_resource``;
* lo que el usuario eligió (la foto, la pregunta) se recuerda en
  ``st.session_state``, que sobrevive entre ejecuciones;
* la lógica de búsqueda vive en ``flowers.service``; aquí solo se dibuja.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Sin red por defecto (RNF1). Debe fijarse antes de importar transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Permite `streamlit run app/streamlit_app.py` sin PYTHONPATH=src.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from flowers import config  # noqa: E402
from flowers.service import (  # noqa: E402
    NOMBRES_ES,
    PREGUNTAS_EJEMPLO,
    ArtefactosFaltantes,
    Consulta,
    Servicio,
    VARIABLE_FABRICA,
    crear_servicio,
)

# Un color por clase, usado igual en barras y etiquetas.
COLORES = {
    "daisy": "#8e9aaf",
    "dandelion": "#e9b824",
    "rose": "#d1495b",
    "sunflower": "#f28f3b",
    "tulip": "#7b4b94",
}
COLUMNAS_VECINOS = 5


def nombre(clase: str) -> str:
    """Nombre en español de una clase interna (daisy -> Margarita)."""
    return NOMBRES_ES.get(clase, clase)


# --------------------------------------------------------------------------
# Carga única de recursos pesados
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Cargando índices y modelos (solo la primera vez)...")
def obtener_servicio(carpeta: str, fabrica: str) -> Servicio:
    """Crea el servicio una vez por (carpeta, fábrica) y lo reutiliza.

    Siguiente paso: `src/flowers/service.py` → `crear_servicio()`.
    """
    return crear_servicio(carpeta, fabrica)


@st.cache_data(show_spinner=False, max_entries=256)
def miniatura(ruta: str, lado: int = 220) -> Image.Image | None:
    """Miniatura de una foto del corpus; None si el archivo ya no existe."""
    try:
        with Image.open(ruta) as im:
            im = im.convert("RGB")
            im.thumbnail((lado, lado))
            return im.copy()
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Piezas visuales
# --------------------------------------------------------------------------


def grafico_confianza(confianzas: dict[str, float], ganadora: str) -> go.Figure:
    """Barras horizontales con el % de confianza de las 5 clases."""
    clases = list(NOMBRES_ES)  # siempre las 5, aunque alguna tenga 0 %
    valores = [100 * confianzas.get(c, 0.0) for c in clases]
    fig = go.Figure(
        go.Bar(
            x=valores,
            y=[nombre(c) for c in clases],
            orientation="h",
            marker_color=[COLORES.get(c, "#888") for c in clases],
            marker_line_width=[3 if c == ganadora else 0 for c in clases],
            marker_line_color="#222",
            text=[f"{v:.1f} %" for v in valores],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis={"range": [0, 115], "title": "Confianza (%)", "showgrid": False},
        yaxis={"autorange": "reversed"},
        height=300,
        margin={"l": 0, "r": 10, "t": 10, "b": 10},
        showlegend=False,
    )
    return fig


def mostrar_resultado(c: Consulta) -> None:
    """Clase ganadora, gráfico de confianza y cuadrícula de vecinos.

    Siguiente paso: Fin del recorrido: pinta barras de confianza, vecinos y comparación con el exacto.
    """
    r = c.resultado
    gana = r.confianzas.get(r.clase, 0.0) * 100
    izq, der = st.columns([1, 2])
    with izq:
        st.metric("Clase ganadora", nombre(r.clase), f"{gana:.1f} % de confianza")
        st.caption(
            f"Voto ponderado de los {c.k} vecinos más parecidos: "
            "cada uno vota por su clase con peso = su similitud coseno."
        )
    with der:
        st.plotly_chart(grafico_confianza(r.confianzas, r.clase), width="stretch")

    st.subheader(f"Los {c.k} vecinos más parecidos")
    for inicio in range(0, len(r.vecinos), COLUMNAS_VECINOS):
        cols = st.columns(COLUMNAS_VECINOS)
        for col, v in zip(cols, r.vecinos[inicio : inicio + COLUMNAS_VECINOS]):
            with col:
                mini = miniatura(v.ruta)
                if mini is not None:
                    st.image(mini, width="stretch")
                else:
                    st.caption("(foto no disponible)")
                st.markdown(f"**{nombre(v.clase)}**  \nsimilitud `{v.sim:.3f}`")


def mostrar_docs(servicio: Servicio, pregunta: str, k_docs: int) -> None:
    """Documentos más cercanos con EmbeddingGemma.

    Siguiente paso: `src/flowers/service.py` → `Servicio.consultar_docs()`.
    """
    st.subheader("Documentos más cercanos (EmbeddingGemma)")
    if not servicio.hay_docs:
        st.info("No hay índice de documentos. Ejecuta `make index`.")
        return
    cd = servicio.consultar_docs(pregunta, k_docs)
    if cd.aviso:
        st.warning(cd.aviso)
        return
    if cd.backend != "embeddinggemma":
        st.caption(f"Codificador de texto usado: {cd.backend}")
    for d in cd.documentos:
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**{d.archivo}** · {nombre(d.clase)}")
            b.markdown(f"similitud `{d.sim:.3f}`")
            st.progress(max(0.0, min(1.0, d.sim)))
            st.write(d.texto[:500] + ("..." if len(d.texto) > 500 else ""))
            if len(d.texto) > 500:
                with st.expander("Ver documento completo"):
                    st.write(d.texto)


# --------------------------------------------------------------------------
# Pestaña 1: subir foto
# --------------------------------------------------------------------------


def _usar_holdout(servicio: Servicio, idx: int) -> None:
    """Callback de los botones de ejemplo: usa el vector ya guardado del holdout."""
    vectores, meta = servicio.holdout()
    st.session_state["foto_entrada"] = {
        "vector": vectores[idx],
        "etiqueta": Path(meta[idx]["ruta"]).name,
        "imagen": meta[idx]["ruta"],  # ruta en disco
        "ms_codificar": 0.0,
        "real": meta[idx]["clase"],
    }
    st.session_state["ultima_tab"] = "foto"


def pestana_foto(servicio: Servicio, k: int, ef: int) -> Consulta | None:
    """Pestaña de foto: codifica la foto subida y muestra la clase y los vecinos.

    Siguiente paso:
      1. `src/flowers/service.py` → `Servicio.codificar_foto()`.
      2. `src/flowers/service.py` → `Servicio.consultar_imagenes()`.
      3. `mostrar_resultado()` (este archivo).
    """
    st.markdown("Sube una foto de una flor y mira qué vecinos encuentra HNSW.")
    subida = st.file_uploader("Foto (jpg, png...)", type=["jpg", "jpeg", "png", "webp"], key="subida")

    # Foto nueva -> se codifica una sola vez y se recuerda su vector.
    if subida is not None and st.session_state.get("foto_subida_id") != subida.file_id:
        st.session_state["foto_subida_id"] = subida.file_id
        with st.spinner("Codificando la foto con SigLIP..."):
            imagen = Image.open(subida).convert("RGB")
            vector, ms = servicio.codificar_foto(imagen)
        st.session_state["foto_entrada"] = {
            "vector": vector,
            "etiqueta": subida.name,
            "imagen": imagen,
            "ms_codificar": ms,
            "real": None,
        }
        st.session_state["ultima_tab"] = "foto"

    holdout = servicio.holdout()
    if holdout is not None:
        st.markdown("**O prueba con una foto de ejemplo** (no están en el índice):")
        _vectores, meta = holdout
        cols = st.columns(len(NOMBRES_ES))
        for col, clase in zip(cols, NOMBRES_ES):
            idx = next((i for i, m in enumerate(meta) if m["clase"] == clase), None)
            if idx is None:
                continue
            with col:
                mini = miniatura(meta[idx]["ruta"], 160)
                if mini is not None:
                    st.image(mini, width="stretch")
                st.button(
                    nombre(clase),
                    key=f"ej_foto_{clase}",
                    on_click=_usar_holdout,
                    args=(servicio, idx),
                    width="stretch",
                )

    entrada = st.session_state.get("foto_entrada")
    if entrada is None:
        st.info("Sube una foto o elige una de ejemplo para empezar.")
        return None

    st.divider()
    col_img, col_txt = st.columns([1, 3])
    with col_img:
        imagen = entrada["imagen"]
        if isinstance(imagen, str):
            imagen = miniatura(imagen, 320)
        if imagen is not None:
            st.image(imagen, caption=entrada["etiqueta"], width="stretch")
    with col_txt:
        if entrada["real"]:
            st.caption(f"Clase real de esta foto de ejemplo: **{nombre(entrada['real'])}**")
        st.write(
            "La foto se convirtió en un vector; HNSW busca los vectores más cercanos "
            "y cada uno vota por su clase."
        )

    consulta = servicio.consultar_imagenes(
        entrada["vector"],
        tipo="foto",
        etiqueta=entrada["etiqueta"],
        k=k,
        ef_search=ef,
        ms_codificar=entrada["ms_codificar"],
    )
    mostrar_resultado(consulta)
    return consulta


# --------------------------------------------------------------------------
# Pestaña 2: escribir pregunta
# --------------------------------------------------------------------------


def _pedir_busqueda_texto() -> None:
    st.session_state["texto_pendiente"] = True
    st.session_state["ultima_tab"] = "texto"


def _poner_pregunta(texto: str) -> None:
    st.session_state["texto_pregunta"] = texto
    _pedir_busqueda_texto()


def pestana_texto(servicio: Servicio, k: int, ef: int) -> Consulta | None:
    """Pestaña de preguntas: codifica la frase y muestra fotos y documentos cercanos.

    Siguiente paso:
      1. `src/flowers/service.py` → `Servicio.codificar_texto_siglip()`.
      2. `src/flowers/service.py` → `Servicio.consultar_imagenes()`, y `mostrar_resultado()` (este archivo).
      3. `mostrar_docs()` (este archivo): la rama de documentos.
    """
    st.markdown(
        "Escribe una pregunta o descripción. Funciona mejor en **inglés** "
        "(SigLIP se entrenó con texto en inglés y los documentos están en inglés)."
    )
    st.text_input(
        "Pregunta",
        key="texto_pregunta",
        placeholder="a yellow flower that follows the sun",
        on_change=_pedir_busqueda_texto,
    )
    st.button("Buscar", key="buscar_texto", type="primary", on_click=_pedir_busqueda_texto)

    st.markdown("**O prueba un ejemplo:**")
    cols = st.columns(len(PREGUNTAS_EJEMPLO))
    for col, (clase, frase) in zip(cols, PREGUNTAS_EJEMPLO.items()):
        col.button(
            nombre(clase),
            key=f"ej_txt_{clase}",
            help=frase,
            on_click=_poner_pregunta,
            args=(frase,),
            width="stretch",
        )

    # Búsqueda pedida -> se codifica la frase con SigLIP una sola vez.
    if st.session_state.pop("texto_pendiente", False):
        frase = st.session_state.get("texto_pregunta", "").strip()
        if frase:
            with st.spinner("Codificando la pregunta..."):
                vector, ms = servicio.codificar_texto_siglip(frase)
            st.session_state["texto_entrada"] = {"frase": frase, "vector": vector, "ms": ms}
        else:
            st.warning("Escribe algo para buscar.")

    entrada = st.session_state.get("texto_entrada")
    if entrada is None:
        st.info("Escribe una pregunta o elige un ejemplo para empezar.")
        return None

    st.divider()
    st.markdown(f"Pregunta: _{entrada['frase']}_")
    st.header("Fotos más cercanas (SigLIP texto → imágenes)")
    consulta = servicio.consultar_imagenes(
        entrada["vector"],
        tipo="texto",
        etiqueta=entrada["frase"],
        k=k,
        ef_search=ef,
        ms_codificar=entrada["ms"],
    )
    mostrar_resultado(consulta)
    st.divider()
    mostrar_docs(servicio, entrada["frase"], k_docs=3)
    return consulta


# --------------------------------------------------------------------------
# Pestaña 3: cómo funciona
# --------------------------------------------------------------------------


def pestana_como_funciona(servicio: Servicio, c: Consulta | None) -> None:
    st.markdown(
        """
### El recorrido de una consulta, paso a paso

1. **Codificar.** SigLIP convierte la foto (o la frase) en un *vector* de números.
   Se normaliza para que su largo (norma) sea 1.
2. **Buscar vecinos.** Con vectores de norma 1, el producto punto es la *similitud coseno*:
   cerca de 1 = muy parecidos.
3. **HNSW.** En vez de comparar con todas las fotos, baja por un grafo de capas:
   arriba salta lejos, abajo afina. Compara con muy pocas fotos.
4. **Exacto.** Para comparar, también se busca por fuerza bruta (contra todas).
5. **Voto.** Cada vecino vota por su clase con peso = su similitud;
   `confianza = puntaje de la clase / suma de puntajes`.
"""
    )
    st.subheader("Qué pasó en la última consulta")
    if c is None:
        st.info("Todavía no hiciste ninguna consulta. Prueba una foto o una pregunta.")
        return
    r = c.resultado
    ahorro = 100 * (1 - r.comparaciones_hnsw / max(1, r.comparaciones_exacto))
    a, b, d = st.columns(3)
    a.metric("Tamaño del vector", f"{c.dim} números")
    b.metric("Norma del vector", f"{c.norma:.4f}", help="Debe ser 1: vector normalizado")
    d.metric("Capas del grafo HNSW", c.capas)
    a, b, d = st.columns(3)
    a.metric("Comparaciones HNSW", r.comparaciones_hnsw)
    b.metric("Comparaciones exacto", r.comparaciones_exacto)
    d.metric("Comparaciones evitadas", f"{ahorro:.1f} %")
    st.markdown(
        f"- Consulta: **{c.etiqueta}** ({'foto' if c.tipo == 'foto' else 'texto'}).\n"
        f"- Codificarla tardó **{c.ms_codificar:.1f} ms**; buscar con HNSW, "
        f"**{r.ms_hnsw:.2f} ms**; búsqueda exacta, **{r.ms_exacto:.2f} ms**.\n"
        f"- Parámetros: k = {c.k}, efSearch = {c.ef_search}, M = {c.M}.\n"
        f"- Resultado: **{nombre(r.clase)}** con {100 * r.confianzas.get(r.clase, 0):.1f} %."
    )
    st.markdown("**Votos por clase**")
    for clase, conf in r.confianzas.items():
        st.progress(conf, text=f"{nombre(clase)}: {100 * conf:.1f} %")


# --------------------------------------------------------------------------
# Panel lateral
# --------------------------------------------------------------------------


def panel_hnsw_vs_exacto(c: Consulta | None) -> None:
    st.subheader("HNSW vs exacto")
    if c is None:
        st.caption("Haz una consulta para ver la comparación.")
        return
    r = c.resultado
    col1, col2 = st.columns(2)
    col1.metric("HNSW", f"{r.ms_hnsw:.2f} ms")
    col2.metric("Exacto", f"{r.ms_exacto:.2f} ms")
    col1.metric("Comparaciones", r.comparaciones_hnsw)
    col2.metric("Comparaciones", r.comparaciones_exacto)
    if c.coinciden:
        st.success("Los vecinos coinciden con la búsqueda exacta.")
    else:
        st.warning(f"HNSW encontró {c.recall:.0%} de los vecinos exactos. Sube efSearch.")


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada del dashboard: carga el servicio y dibuja la barra lateral y las pestañas.

    Siguiente paso:
      1. `obtener_servicio()` (este archivo): carga índices y modelos una sola vez.
      2. `pestana_foto()`, `pestana_texto()` o `pestana_como_funciona()` (este archivo), según la pestaña.
    """
    st.set_page_config(page_title="Buscador de flores con HNSW", page_icon="🌼", layout="wide")
    st.title("🌼 Buscador de flores con HNSW")
    st.caption("Todo corre en tu computadora: SigLIP + EmbeddingGemma + un HNSW hecho en NumPy.")

    carpeta = config.artifacts_dir()
    try:
        servicio = obtener_servicio(str(carpeta), os.environ.get(VARIABLE_FABRICA, ""))
    except ArtefactosFaltantes as exc:
        st.error(f"No encontré los índices. Ejecuta `make index` primero.\n\n{exc}")
        st.stop()
        return

    # Controles del panel lateral (se definen antes porque las pestañas los usan).
    with st.sidebar:
        st.header("Parámetros")
        k = st.slider("k (vecinos)", 1, 30, 10, key="k", help="Cuántos vecinos votan.")
        ef = st.slider(
            "efSearch", 10, 300, 50, key="ef_search",
            help="Más alto = más preciso pero más lento.",
        )
        st.caption(
            f"Índice: {servicio.n_imagenes} fotos ({servicio.dim_imagenes} dim)"
            + (f" y {servicio.n_docs} documentos" if servicio.hay_docs else "")
        )
        panel = st.container()

    tab_foto, tab_texto, tab_como = st.tabs(["📷 Subir foto", "✍️ Escribir pregunta", "🔧 Cómo funciona"])
    with tab_foto:
        c_foto = pestana_foto(servicio, k, ef)
    with tab_texto:
        c_texto = pestana_texto(servicio, k, ef)

    # "Última consulta" = la pestaña que el usuario tocó más recientemente.
    por_tab = {"foto": c_foto, "texto": c_texto}
    ultima = por_tab.get(st.session_state.get("ultima_tab", ""))
    ultima = ultima or c_foto or c_texto

    with tab_como:
        pestana_como_funciona(servicio, ultima)
    with panel:
        panel_hnsw_vs_exacto(ultima)


main()
