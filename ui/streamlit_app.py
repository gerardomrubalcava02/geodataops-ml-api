"""Interfaz mínima de demostración. Solo consume la API REST (no importa el backend).

Ejecutar:  uv run streamlit run ui/streamlit_app.py
Variable:  GEODATAOPS_API_URL (por defecto http://127.0.0.1:8000)
"""

from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("GEODATAOPS_API_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"
st.set_page_config(page_title="GeoDataOps ML", layout="wide")


def call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, API + path, timeout=30, **kw)
    except httpx.HTTPError as exc:
        st.error(f"No se pudo conectar con la API ({API}): {exc}")
        st.stop()
    if r.status_code >= 400:
        err = r.json().get("error", {})
        st.warning(f"{r.status_code} · {err.get('codigo')}: {err.get('mensaje')}")
        if err.get("detalle"):
            st.json(err["detalle"])
        return None
    return r.json()


salud = call("GET", "/salud")
st.title("GeoDataOps · Indicadores y modelo predictivo")
fuente = salud["fuente_datos"]
st.caption(
    f"Fuente: **{fuente['tipo']}** · {fuente['ubicacion']} · "
    f"{'⚠️ SINTÉTICA' if fuente['sintetico'] else 'datos reales'} · "
    f"modelo: `{salud['modelo_version'] or 'sin entrenar'}` · LLM: {salud['llm_proveedor']}"
)

tab_ind, tab_pred, tab_model, tab_chat = st.tabs(["Indicadores", "Predicción", "Modelo", "Asistente"])

with tab_ind:
    municipios = call("GET", "/municipios") or []
    catalogo = call("GET", "/indicadores") or []
    st.dataframe(pd.DataFrame(catalogo), hide_index=True, use_container_width=True)
    opciones = {f"{m['cvegeo']} {m.get('nombre') or ''}".strip(): m["cvegeo"] for m in municipios}
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Municipio", list(opciones))
    inds = c2.multiselect("Indicadores", [c["indicador"] for c in catalogo],
                          default=[c["indicador"] for c in catalogo][:1])
    if sel:
        data = call("GET", "/observaciones", params={"cvegeo": opciones[sel], "indicador": inds, "limit": 1000})
        if data and data["items"]:
            df = pd.DataFrame(data["items"])
            st.line_chart(df.pivot(index="periodo", columns="indicador", values="valor"))
            st.dataframe(df, hide_index=True, use_container_width=True)

with tab_pred:
    st.write("Crecimiento poblacional esperado de t a t+1.")
    c1, c2, c3 = st.columns(3)
    cvegeo = c1.text_input("cvegeo", "99001")
    periodo = c2.number_input("Año base (t)", 1000, 9999, 2023)
    fuente_datos = c3.radio("Historial", ["entrenamiento", "operativa"], horizontal=True)
    if st.button("Predecir"):
        body = {"cvegeo": cvegeo, "periodo": int(periodo), "fuente_datos": fuente_datos}
        res = call("POST", "/predicciones", json=body)
        if res:
            if res.get("advertencia"):
                st.warning(res["advertencia"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Crecimiento esperado", f"{res['crecimiento_esperado']:.2%}")
            m2.metric(f"Población {res['periodo_base']}", f"{res['poblacion_base']:,}")
            m3.metric(f"Estimada {res['periodo_objetivo']}", f"{res['poblacion_estimada']:,}")
            st.json(res["caracteristicas"])

with tab_model:
    meta = call("GET", "/modelos/actual")
    if meta:
        if meta["datos"]["advertencia"]:
            st.warning(meta["datos"]["advertencia"])
        st.write(f"Versión `{meta['version']}` · algoritmo **{meta['algoritmo']}** · "
                 f"mejora MAE vs mejor baseline: **{meta['mejora_mae_vs_mejor_baseline_pct']}%**")
        st.subheader("Prueba (holdout temporal)")
        st.dataframe(pd.DataFrame(meta["prueba"]).T, use_container_width=True)
        st.subheader("Validación cruzada temporal (entrenamiento)")
        st.dataframe(pd.DataFrame(meta["validacion_cruzada"]).T, use_container_width=True)
        st.subheader("Importancia por permutación")
        imp = pd.DataFrame(meta["importancia_permutacion_prueba"]).set_index("caracteristica")
        st.bar_chart(imp)
        with st.expander("Metadatos completos"):
            st.json(meta)

with tab_chat:
    if not salud["llm_configurado"]:
        st.info("Asistente no configurado. Defina GEODATAOPS_ML_LLM_PROVIDER=anthropic y una API key.")
    pregunta = st.text_input("Pregunta", "¿Qué indicadores hay disponibles y de qué fuente provienen?")
    if st.button("Preguntar", disabled=not salud["llm_configurado"]):
        out, log = st.empty(), st.container()
        texto = ""
        with httpx.stream("POST", API + "/chat/stream", json={"pregunta": pregunta}, timeout=180) as r:
            evento = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    evento = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if evento == "text":
                        texto += data["delta"]
                        out.markdown(texto)
                    elif evento in ("tool_call", "tool_result"):
                        detalle = json.dumps(data.get("argumentos", data.get("ok")), ensure_ascii=False)
                        log.caption(f"{evento}: {data.get('nombre')} {detalle}")
                    elif evento == "error":
                        st.error(f"{data['codigo']}: {data['mensaje']}")
