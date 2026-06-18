# =============================================================================
# app.py  –  Dashboard Streamlit
# PRISMA | Plataforma de Riesgo e Inteligencia para la Salud Mental
# Intentos de Suicidio – SIVIGILA 2024
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS, PALETA, COLORES_PRIORIDAD, PESOS_IPI_REF
from src.utils import clasificar_ipi as clasificar_ipi_local

st.set_page_config(
    page_title="PRISMA – Salud Mental SIVIGILA 2024",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  .kpi-card {{
    background: linear-gradient(135deg,
      var(--secondary-background-color) 0%,
      var(--background-color) 100%);
    border-left: 5px solid var(--primary-color);
    border-radius: 12px;
    padding: 18px 20px 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(13,124,143,0.18);
  }}
  .kpi-icon  {{ font-size:1.7rem; display:block; margin-bottom:6px; line-height:1; }}
  .kpi-valor {{ font-size:2.1rem; font-weight:800; color:var(--primary-color);
                letter-spacing:-0.5px; line-height:1.15; }}
  .kpi-label {{ font-size:0.75rem; color:var(--text-color); opacity:0.65;
                margin-top:6px; text-transform:uppercase; letter-spacing:0.5px; }}
  h1, h2, h3 {{ color: var(--primary-color); }}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos y preparando modelos (puede tardar un momento la primera vez)...")
def cargar():
    if not os.path.exists(RUTAS["procesado_casos"]):
        from src.limpieza import limpiar_datos, guardar_procesado
        df = limpiar_datos(verbose=False)
        guardar_procesado(df)
    else:
        df = pd.read_csv(RUTAS["procesado_casos"],
                         dtype={"cod_dane_mun": str},
                         parse_dates=["FEC_NOT", "INI_SIN", "FEC_CON"])
    if not os.path.exists(RUTAS["procesado_mun"]):
        from src.ipi import pipeline_ipi
        mun = pipeline_ipi(df, verbose=False)
    else:
        mun = pd.read_csv(RUTAS["procesado_mun"], dtype={"cod_dane_mun": str})

    # Generar artefactos del modelo (gráficas + clusters) automáticamente
    # si aún no existen, para que el usuario nunca tenga que abrir una
    # terminal y ejecutar `python -m src.modelo` manualmente.
    ruta_clusters = os.path.join(RUTAS["reportes"], "municipios_clusters.csv")
    ruta_fi       = os.path.join(RUTAS["graficas"], "11_feature_importance.png")
    if not (os.path.exists(ruta_clusters) and os.path.exists(ruta_fi)):
        from src.modelo import pipeline_modelos
        pipeline_modelos(mun, verbose=False)

    return df, mun

df, mun = cargar()


# ── Autenticación por contraseña ─────────────────────────────────────────────
def verificar_password():
    """Bloquea el dashboard hasta ingresar la contraseña de .streamlit/secrets.toml."""
    def password_ingresada():
        if st.session_state["pwd_input"] == st.secrets["password"]:
            st.session_state["autenticado"] = True
            del st.session_state["pwd_input"]
        else:
            st.session_state["autenticado"] = False

    if st.session_state.get("autenticado"):
        return True

    st.markdown(
        "<div style='text-align:center; padding-top:40px;'>"
        "<span style='font-size:3rem;'>🧠</span><br>"
        "<h2 style='color:#0D7C8F;'>PRISMA</h2>"
        "<p style='opacity:0.7;'>Acceso restringido · Datos de salud pública</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.text_input("Contraseña", type="password",
                  key="pwd_input", on_change=password_ingresada)
    if st.session_state.get("autenticado") is False:
        st.error("Contraseña incorrecta. Intenta de nuevo.")
    st.stop()

verificar_password()


@st.cache_data(show_spinner=False)
def cargar_geojson_colombia():
    import urllib.request, json
    url = ("https://raw.githubusercontent.com/sep6/colombia-geojson"
           "/master/colombia.geojson")
    try:
        with urllib.request.urlopen(url, timeout=12) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _norm_depto(nombre: str) -> str:
    s = unicodedata.normalize("NFD", str(nombre))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.upper().strip()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:14px 0 8px 0;">
      <span style="font-size:2.8rem; display:block; line-height:1;">🧠</span>
      <span style="font-size:1.45rem; font-weight:800;
                   color:var(--primary-color); letter-spacing:-0.5px;">PRISMA</span><br>
      <span style="font-size:0.72rem; opacity:0.55; letter-spacing:2px;
                   text-transform:uppercase;">Colombia</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;font-size:0.8rem;opacity:0.7;margin-top:2px;'>"
        "Vigilancia en Salud Mental · SIVIGILA 2024</p>",
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("### 🔍 Filtros globales")
    deptos = ["Todos"] + sorted(df["Departamento_ocurrencia"].dropna().unique().tolist())
    depto_sel = st.selectbox("Departamento", deptos)
    sexos = ["Todos", "Femenino", "Masculino"]
    sexo_sel = st.selectbox("Sexo", sexos)
    _MESES = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
              7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    mes_sel = st.multiselect(
        "Meses", options=list(range(1,13)),
        default=list(range(1,13)),
        format_func=lambda m: _MESES[m],
    )
    areas = ["Todos","Cabecera municipal","Centro poblado","Rural disperso"]
    area_sel = st.selectbox("Área", areas)
    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem;opacity:0.75;line-height:1.9;">
      <b>Proyecto:</b> Talento Tech – IA<br>
      <b>Evento SIVIGILA:</b> 356<br>
      🗓️ <b>Última actualización:</b> 2024
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;opacity:0.65;line-height:1.7;">
      <b>Autoras:</b><br>
      Daniela Hollmann Guarín<br>
      Ornella Gomez Meusburger<br>
      April Corpus Coba
    </div>
    """, unsafe_allow_html=True)


# ── Filtros ────────────────────────────────────────────────────────────────
def filtrar(df_base):
    d = df_base.copy()
    if depto_sel != "Todos":
        d = d[d["Departamento_ocurrencia"] == depto_sel]
    if sexo_sel != "Todos":
        d = d[d["sexo_nombre"] == sexo_sel]
    if mes_sel:
        d = d[d["mes"].isin(mes_sel)]
    if area_sel != "Todos":
        d = d[d["area_nombre"] == area_sel]
    return d

df_f = filtrar(df)

LAYOUT_BASE = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")


# ── Helper: evitar que las etiquetas de valor se corten ─────────────────────
# Plotly con textposition="outside" no reserva espacio automáticamente:
# si la barra más alta llega cerca del borde del gráfico, su etiqueta de
# número queda cortada a la mitad. Esta función amplía el rango del eje
# de los valores en un margen proporcional, dejando aire para el texto.
def headroom_vertical(fig, valores, factor=0.18):
    """Amplía el eje Y para que las etiquetas 'outside' de barras verticales no se corten."""
    maximo = max(valores) if len(valores) > 0 else 0
    fig.update_yaxes(range=[0, maximo * (1 + factor)])
    return fig

def headroom_horizontal(fig, valores, factor=0.12):
    """Amplía el eje X para que las etiquetas 'outside' de barras horizontales no se corten."""
    maximo = max(valores) if len(valores) > 0 else 0
    fig.update_xaxes(range=[0, maximo * (1 + factor)])
    return fig


# ── Supresión de celdas pequeñas (privacidad epidemiológica) ──────────────────
# Nunca mostrar agrupaciones con menos de UMBRAL casos: la combinación
# municipio + edad + sexo podría reidentificar a una persona.
UMBRAL_PRIVACIDAD = 5

def suprimir_celdas(tabla, col_casos):
    """Devuelve solo filas con casos >= umbral de privacidad."""
    return tabla[tabla[col_casos] >= UMBRAL_PRIVACIDAD].copy()


# ── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Inicio",
    "📊 Análisis Descriptivo",
    "🗺️ Priorización Territorial",
    "🤖 Modelo Predictivo",
    "💡 Recomendaciones",
])


# ===========================================================================
# TAB 1 – INICIO
# ===========================================================================
with tab1:
    st.title("🧠 PRISMA")
    st.subheader("Plataforma de Riesgo e Inteligencia para la Salud Mental · SIVIGILA 2024")
    st.markdown(
        "Plataforma de inteligencia de datos para el análisis de intentos de "
        "suicidio registrados por el Sistema Nacional de Vigilancia en Salud "
        "Pública (SIVIGILA), Evento 356, año 2024. "
        "Enfoque **preventivo y de salud pública**."
    )
    st.divider()

    n_total  = len(df_f)
    n_deptos = df_f["Departamento_ocurrencia"].nunique()
    n_muns   = df_f["cod_dane_mun"].nunique()
    pct_ado  = round(df_f["es_adolescente"].mean()*100, 1) if n_total > 0 else 0
    pct_hosp = round((df_f["PAC_HOS"]==1).mean()*100, 1) if n_total > 0 else 0
    depto_top = (df_f["Departamento_ocurrencia"].value_counts().idxmax()
                 if n_total > 0 else "–")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, icono, valor, etiqueta in [
        (c1,"📋", f"{n_total:,}",   "Total de casos"),
        (c2,"🗺️", str(n_deptos),   "Departamentos"),
        (c3,"🏘️", str(n_muns),     "Municipios"),
        (c4,"⚠️", f"{pct_ado}%",   "% Adolescentes 12–17"),
        (c5,"🏥", f"{pct_hosp}%",  "% Hospitalizados"),
        (c6,"📍", depto_top,        "Depto. más afectado"),
    ]:
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-icon">{icono}</div>'
            f'<div class="kpi-valor">{valor}</div>'
            f'<div class="kpi-label">{etiqueta}</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.divider()
    with st.expander("ℹ️ Sobre el proyecto y la metodología"):
        st.markdown("""
        **Fuente de datos:** SIVIGILA – Ministerio de Salud y Protección Social de Colombia.
        Evento 356: Intento de suicidio. Año 2024.

        **Metodología:** CRISP-DM (Cross-Industry Standard Process for Data Mining).

        **Modelos de IA aplicados:**
        - **Árbol de Decisión** (interpretabilidad): genera reglas legibles.
        - **Random Forest** (rendimiento): ensemble de árboles con feature importance.
        - **K-Means** (agrupamiento): perfila territorios sin variable objetivo.

        **Índice de Prioridad de Intervención (IPI):**
        Indicador compuesto 0–100: tasa/100k (35%), % adolescentes (25%),
        tendencia (20%), % hospitalizados (10%), % antec. psiquiátrico (10%).

        **Nota ética:** Análisis epidemiológico y preventivo. No realiza diagnósticos
        clínicos ni identifica individuos.

        ---

        **Equipo desarrollador:**
        Daniela Hollmann Guarín · Ornella Gomez Meusburger · April Corpus Coba

        Proyecto desarrollado en el marco del Bootcamp Talento Tech –
        Inteligencia Artificial.
        """)


# ===========================================================================
# TAB 2 – ANÁLISIS DESCRIPTIVO
# ===========================================================================
with tab2:
    st.header("📊 Análisis Exploratorio de Datos")
    if len(df_f) == 0:
        st.warning("No hay datos para los filtros seleccionados.")
        st.stop()

    # Tendencia semanal
    st.subheader("Tendencia semanal de casos")
    tend = df_f.groupby("SEMANA").size().reset_index(name="casos")
    tend["media_movil"] = tend["casos"].rolling(4, min_periods=1).mean()
    fig_tend = go.Figure()
    fig_tend.add_trace(go.Scatter(x=tend["SEMANA"], y=tend["casos"],
                                  mode="lines", name="Casos/semana",
                                  line=dict(color=PALETA["acento"], width=1.5),
                                  opacity=0.6))
    fig_tend.add_trace(go.Scatter(x=tend["SEMANA"], y=tend["media_movil"],
                                  mode="lines", name="Media móvil 4 sem.",
                                  line=dict(color=PALETA["primario"], width=3)))
    fig_tend.update_layout(**LAYOUT_BASE, height=350,
                            xaxis_title="Semana epidemiológica",
                            yaxis_title="Casos")
    st.plotly_chart(fig_tend, use_container_width=True)
    st.caption("📌 La media móvil de 4 semanas muestra la tendencia estructural.")

    col_a, col_b = st.columns(2)

    # Por sexo — sin color= para evitar KeyError con datos filtrados
    with col_a:
        st.subheader("Por sexo")
        s = (df_f["sexo_nombre"].value_counts()
             .reset_index()
             .rename(columns={"sexo_nombre":"Sexo","count":"Casos"}))
        colores_s = [PALETA["primario"], PALETA["secundario"]][:len(s)]
        fig_s = go.Figure(go.Bar(
            x=s["Sexo"], y=s["Casos"],
            text=s["Casos"], textposition="outside",
            marker_color=colores_s
        ))
        fig_s.update_layout(**LAYOUT_BASE, height=320, showlegend=False)
        headroom_vertical(fig_s, s["Casos"])
        st.plotly_chart(fig_s, use_container_width=True)

    # Por grupo de edad — sin color= para evitar KeyError
    with col_b:
        st.subheader("Por grupo de edad")
        orden_edad = ["0–11","12–17","18–25","26–35","36–45","46–59","60+"]
        g = (df_f["grupo_edad"].astype(str).value_counts()
             .reset_index()
             .rename(columns={"grupo_edad":"Grupo","count":"Casos"}))
        g["Grupo"] = pd.Categorical(g["Grupo"], categories=orden_edad, ordered=True)
        g = g.sort_values("Grupo").dropna(subset=["Grupo"]).reset_index(drop=True)
        g["Grupo"] = g["Grupo"].astype(str)
        colores_g = [PALETA["advertencia"] if x=="12–17" else PALETA["primario"]
                     for x in g["Grupo"]]
        fig_g = go.Figure(go.Bar(
            x=g["Grupo"], y=g["Casos"],
            text=g["Casos"], textposition="outside",
            marker_color=colores_g
        ))
        fig_g.update_layout(**LAYOUT_BASE, height=320, showlegend=False)
        headroom_vertical(fig_g, g["Casos"])
        st.plotly_chart(fig_g, use_container_width=True)
    st.caption("📌 El grupo 12–17 años (naranja) concentra la mayor frecuencia relativa.")

    # Top 15 departamentos
    st.subheader("Top 15 departamentos con más casos")
    top_d = (df_f["Departamento_ocurrencia"].value_counts()
             .head(15).reset_index()
             .rename(columns={"Departamento_ocurrencia":"Departamento","count":"Casos"}))
    fig_d = go.Figure(go.Bar(
        x=top_d["Casos"], y=top_d["Departamento"],
        orientation="h",
        text=top_d["Casos"], textposition="outside",
        marker_color=PALETA["primario"]
    ))
    fig_d.update_layout(**LAYOUT_BASE, height=420,
                         yaxis=dict(autorange="reversed"))
    headroom_horizontal(fig_d, top_d["Casos"])
    st.plotly_chart(fig_d, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Distribución de edades")
        fig_hist = px.histogram(df_f, x="EDAD", nbins=30,
                                color_discrete_sequence=[PALETA["primario"]],
                                opacity=0.85)
        fig_hist.add_vline(x=df_f["EDAD"].median(), line_dash="dash",
                           line_color=PALETA["advertencia"],
                           annotation_text=f"Mediana: {df_f['EDAD'].median():.0f}")
        fig_hist.update_layout(**LAYOUT_BASE, height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d:
        st.subheader("Por área de residencia")
        a = (df_f["area_nombre"].value_counts()
             .reset_index()
             .rename(columns={"area_nombre":"Área","count":"Casos"}))
        fig_a = go.Figure(go.Pie(
            labels=a["Área"], values=a["Casos"],
            hole=0.45,
            marker_colors=[PALETA["primario"], PALETA["acento"], PALETA["secundario"]]
        ))
        fig_a.update_layout(**LAYOUT_BASE, height=300)
        st.plotly_chart(fig_a, use_container_width=True)

    # Grupos de riesgo — siempre sobre df global
    st.subheader("Grupos poblacionales de riesgo")
    gp_etiq = {
        "GP_PSIQUIA":"Antec. psiquiátrico","GP_MIGRANT":"Migrante",
        "GP_CARCELA":"Privado libertad",   "GP_GESTAN":"Gestante",
        "GP_INDIGEN":"Indígena",            "GP_POBICFB":"Ben. ICBF",
        "GP_DISCAPA":"Discapacidad",        "GP_DESPLAZ":"Desplazado/a",
        "GP_VIC_VIO":"Víctima violencia",
    }
    gp_data = [{"Grupo": etiq, "Casos": int((df[c]==1).sum())}
               for c, etiq in gp_etiq.items() if c in df.columns]
    gp_df = (pd.DataFrame([r for r in gp_data if r["Casos"] > 0])
             .sort_values("Casos", ascending=True))
    fig_gp = go.Figure(go.Bar(
        x=gp_df["Casos"], y=gp_df["Grupo"],
        orientation="h",
        text=gp_df["Casos"], textposition="outside",
        marker_color=PALETA["acento"]
    ))
    fig_gp.update_layout(**LAYOUT_BASE, height=320)
    headroom_horizontal(fig_gp, gp_df["Casos"])
    st.plotly_chart(fig_gp, use_container_width=True)
    st.caption("📌 Datos globales — no afectados por filtros de selección.")

    st.subheader("Tabla de datos – Top 20 municipios")
    top_mun = (df_f.groupby(["Departamento_ocurrencia","Municipio_ocurrencia"])
               .size().reset_index(name="Casos")
               .sort_values("Casos", ascending=False))
    top_mun = suprimir_celdas(top_mun, "Casos").head(20)   # oculta < 5 casos
    if len(top_mun) > 0:
        st.dataframe(top_mun, use_container_width=True, hide_index=True)
        st.caption("📌 Por privacidad se ocultan municipios con menos de "
                   f"{UMBRAL_PRIVACIDAD} casos en la selección actual.")
    else:
        st.info("No hay municipios con suficientes casos para mostrar bajo los "
                "filtros actuales (umbral de privacidad).")


# ===========================================================================
# TAB 3 – PRIORIZACIÓN TERRITORIAL
# ===========================================================================
with tab3:
    st.header("🗺️ Sistema de Priorización Territorial")
    st.markdown(
        "El **Índice de Prioridad de Intervención (IPI)** identifica los municipios "
        "donde debe concentrarse la intervención preventiva en salud mental."
    )

    criticos = mun[mun["nivel_prioridad"]=="Crítica"]
    altos    = mun[mun["nivel_prioridad"]=="Alta"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Municipios Prioridad Crítica", len(criticos))
    c2.metric("Municipios Prioridad Alta",    len(altos))
    c3.metric("Municipio con mayor IPI",
              mun.iloc[0]["Municipio_ocurrencia"] if len(mun)>0 else "–")
    c4.metric("Depto. IPI promedio más alto",
              mun.groupby("Departamento_ocurrencia")["IPI"].mean().idxmax()
              if len(mun)>0 else "–")

    st.divider()
    with st.expander("📐 Fórmula del IPI"):
        st.markdown("""
        ```
        IPI = 0.35 × tasa por 100.000 hab.
            + 0.25 × % adolescentes (12–17)
            + 0.20 × tendencia H2 vs H1
            + 0.10 × % hospitalizados
            + 0.10 × % antecedente psiquiátrico
        ```
        | Rango   | Nivel      |
        |---------|------------|
        | 0 – 39  | 🟢 Baja   |
        | 40 – 59 | 🔵 Media  |
        | 60 – 79 | 🟠 Alta   |
        | 80–100  | 🔴 Crítica |
        """)

    col_r, col_d = st.columns([1.5, 1])
    with col_r:
        st.subheader("Ranking nacional – Top 20 municipios")
        top20 = mun.head(20)[["ranking_nacional","Departamento_ocurrencia",
                               "Municipio_ocurrencia","total_casos",
                               "tasa_x100k","IPI","nivel_prioridad"]].copy()
        top20["tasa_x100k"] = top20["tasa_x100k"].round(1)
        top20["IPI"]        = top20["IPI"].round(1)
        top20_styled = top20.style.map(
            lambda v: {
                "Crítica":"background-color:#fde8e8",
                "Alta":"background-color:#fef3e8",
                "Media":"background-color:#e8f4fd",
                "Baja":"background-color:#e8f5e9"
            }.get(v,""),
            subset=["nivel_prioridad"]
        )
        st.dataframe(top20_styled, use_container_width=True, hide_index=True)

    with col_d:
        st.subheader("Distribución de prioridades")
        dist = (mun["nivel_prioridad"].value_counts()
                .reset_index()
                .rename(columns={"nivel_prioridad":"Nivel","count":"Municipios"}))
        orden_niv = ["Crítica","Alta","Media","Baja"]
        dist["Nivel"] = pd.Categorical(dist["Nivel"], categories=orden_niv, ordered=True)
        dist = (dist.sort_values("Nivel").dropna(subset=["Nivel"])
                .reset_index(drop=True))
        dist["Nivel"] = dist["Nivel"].astype(str)
        # Construir colores en el mismo orden que el DataFrame resultante
        colores_dist = [COLORES_PRIORIDAD.get(n, PALETA["primario"])
                        for n in dist["Nivel"]]
        fig_dist = go.Figure(go.Bar(
            x=dist["Nivel"], y=dist["Municipios"],
            text=dist["Municipios"], textposition="outside",
            marker_color=colores_dist
        ))
        fig_dist.update_layout(**LAYOUT_BASE, height=340, showlegend=False)
        headroom_vertical(fig_dist, dist["Municipios"])
        st.plotly_chart(fig_dist, use_container_width=True)

    # IPI por departamento
    st.subheader("IPI promedio por departamento")
    dep_ipi = (mun.groupby("Departamento_ocurrencia")["IPI"]
               .mean().round(2).reset_index()
               .sort_values("IPI", ascending=False))
    fig_dep = go.Figure(go.Bar(
        x=dep_ipi["IPI"],
        y=dep_ipi["Departamento_ocurrencia"],
        orientation="h",
        text=dep_ipi["IPI"], textposition="outside",
        texttemplate="%{text:.1f}",
        marker=dict(
            color=dep_ipi["IPI"],
            colorscale=[[0, PALETA["suave"]], [1, PALETA["peligro"]]],
            showscale=False
        )
    ))
    fig_dep.update_layout(**LAYOUT_BASE, height=500,
                           yaxis=dict(autorange="reversed"))
    headroom_horizontal(fig_dep, dep_ipi["IPI"])
    st.plotly_chart(fig_dep, use_container_width=True)

    # Mapa coroplético
    st.subheader("🗺️ Mapa de prioridad por departamento")
    st.caption("El mapa colorea cada departamento según su IPI promedio. "
               "Los 5 departamentos más relevantes del estudio (mayor IPI) "
               "se resaltan con borde oscuro y se listan a la derecha.")
    geo_col = cargar_geojson_colombia()
    dep_mapa = (mun.groupby("Departamento_ocurrencia")
                .agg(IPI=("IPI","mean"), total_casos=("total_casos","sum"),
                     municipios=("Municipio_ocurrencia","nunique"))
                .round(2).reset_index())
    dep_mapa["depto_norm"] = dep_mapa["Departamento_ocurrencia"].apply(_norm_depto)
    dep_mapa = dep_mapa.sort_values("IPI", ascending=False).reset_index(drop=True)

    # Los 5 departamentos más relevantes (mayor IPI promedio) se resaltan
    # visualmente con un borde más grueso y oscuro en el mapa.
    top5_deptos = set(dep_mapa.head(5)["Departamento_ocurrencia"])
    dep_mapa["es_top5"] = dep_mapa["Departamento_ocurrencia"].isin(top5_deptos)

    col_mapa, col_top5 = st.columns([2.2, 1])

    with col_mapa:
        if geo_col is not None:
            fig_mapa = px.choropleth(
                dep_mapa,
                geojson=geo_col,
                locations="depto_norm",
                featureidkey="properties.DPTO_CNMBR",
                color="IPI",
                color_continuous_scale=[[0,"#A8D5BA"],[0.5,"#E07B54"],[1,"#C0392B"]],
                hover_name="Departamento_ocurrencia",
                hover_data={"IPI":":.1f","total_casos":True,"municipios":True,"depto_norm":False},
                labels={"IPI":"IPI promedio","total_casos":"Total casos","municipios":"Municipios"},
            )
            # Borde más grueso y oscuro para los 5 departamentos más relevantes,
            # de modo que destaquen visualmente sobre el resto del mapa.
            anchos_borde = [2.5 if top else 0.4 for top in dep_mapa["es_top5"]]
            colores_borde = ["#1A1A1A" if top else "rgba(255,255,255,0.5)"
                              for top in dep_mapa["es_top5"]]
            fig_mapa.update_traces(
                marker_line_width=anchos_borde,
                marker_line_color=colores_borde,
            )
            fig_mapa.update_geos(fitbounds="locations", visible=False)
            fig_mapa.update_layout(margin={"r":0,"t":10,"l":0,"b":0}, height=520,
                                    **LAYOUT_BASE)
            st.plotly_chart(fig_mapa, use_container_width=True)
            st.caption("Verde: prioridad baja · Naranja: media/alta · Rojo: crítica. "
                       "Borde negro grueso = top 5 departamentos del estudio.")
        else:
            st.caption("⚠️ Mapa no disponible (sin conexión). Ver gráfica de barras arriba.")

    with col_top5:
        st.markdown("**🏆 Top 5 departamentos**")
        st.caption("Mayor IPI promedio")
        for i, row in dep_mapa.head(5).iterrows():
            nivel = clasificar_ipi_local(row["IPI"])
            color_badge = COLORES_PRIORIDAD.get(nivel, PALETA["primario"])
            st.markdown(
                f"<div style='border-left:4px solid {color_badge}; "
                f"padding:6px 10px; margin-bottom:8px; "
                f"background:var(--secondary-background-color); border-radius:6px;'>"
                f"<b>{row['Departamento_ocurrencia']}</b><br>"
                f"<span style='font-size:0.85rem; opacity:0.8;'>"
                f"IPI: {row['IPI']:.1f} · {int(row['total_casos']):,} casos · "
                f"{int(row['municipios'])} municipios</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Municipios emergentes
    st.subheader("⚠️ Municipios con crecimiento acelerado")
    from src.ipi import municipios_emergentes
    emerg = municipios_emergentes(mun, top=10)
    emerg = suprimir_celdas(emerg, "total_casos")   # refuerzo de privacidad
    if len(emerg) > 0:
        emerg_show = emerg[["Departamento_ocurrencia","Municipio_ocurrencia",
                             "total_casos","tendencia_H2_H1","IPI","nivel_prioridad"]].copy()
        emerg_show["tendencia_H2_H1"] = (emerg_show["tendencia_H2_H1"]*100).round(1).astype(str)+"%"
        st.dataframe(emerg_show, use_container_width=True, hide_index=True)
    else:
        st.info("No se identificaron municipios emergentes con los parámetros actuales.")

    # Scatter: tasa vs adolescentes — usando go.Scatter por nivel para evitar KeyError
    st.subheader("Tasa por 100k vs % de adolescentes afectados")
    mun_plot = mun[mun["total_casos"] >= UMBRAL_PRIVACIDAD].copy()
    fig_sc = go.Figure()
    for nivel, color in COLORES_PRIORIDAD.items():
        sub = mun_plot[mun_plot["nivel_prioridad"] == nivel]
        if len(sub) == 0:
            continue
        fig_sc.add_trace(go.Scatter(
            x=sub["tasa_x100k"],
            y=sub["pct_adolescentes"],
            mode="markers",
            name=nivel,
            marker=dict(
                size=sub["IPI"] / 4,
                color=color,
                opacity=0.75,
                line=dict(color="white", width=0.5)
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Tasa: %{x:.1f}<br>"
                "% Adol.: %{y:.1%}<br>"
                "IPI: %{customdata[2]:.1f}"
                "<extra></extra>"
            ),
            customdata=sub[["Municipio_ocurrencia","Departamento_ocurrencia","IPI"]].values
        ))
    fig_sc.update_layout(**LAYOUT_BASE, height=420,
                          xaxis_title="Tasa por 100k hab.",
                          yaxis_title="% Adolescentes (12–17)",
                          legend_title="Nivel IPI")
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Tamaño del círculo = IPI. Esquina superior derecha: mayor carga + mayor vulnerabilidad.")


# ===========================================================================
# TAB 4 – MODELO PREDICTIVO
# ===========================================================================
with tab4:
    st.header("🤖 Modelos de Inteligencia Artificial")
    st.markdown(
        "**¿Qué hace esta sección?** Aquí se entrenan y muestran los resultados de "
        "tres modelos de Inteligencia Artificial que aprenden patrones a partir de "
        "los 1.000+ municipios con casos registrados, para clasificar automáticamente "
        "el **nivel de prioridad** de cada municipio (Baja, Media, Alta o Crítica) y "
        "para descubrir **perfiles territoriales** sin depender de esa clasificación. "
        "Estos resultados se generan automáticamente al cargar el dashboard — no es "
        "necesario ejecutar nada manualmente."
    )
    st.markdown("Tres modelos sobre la tabla municipal (≥5 casos) para clasificar el nivel de prioridad.")

    sub1, sub2, sub3 = st.tabs(["Árbol de Decisión","Random Forest","K-Means"])

    with sub1:
        st.subheader("Árbol de Decisión")
        st.markdown("""
        **Parámetros:** `max_depth=5`, `class_weight='balanced'`, `criterion='gini'`

        Genera reglas legibles directamente interpretables por equipos de salud pública.
        ```
        Si tasa_x100k > umbral_A
          y pct_adolescentes > umbral_B
          → Prioridad: Alta
        ```
        """)
        img_conf = os.path.join(RUTAS["graficas"], "13_confusion_dt.png")
        if os.path.exists(img_conf):
            st.image(img_conf, caption="Matriz de confusión – Árbol de Decisión",
                     use_column_width=True)
        else:
            st.info("Los resultados del modelo se están preparando. Vuelve a cargar la página en unos segundos.")

    with sub2:
        st.subheader("Random Forest")
        st.markdown("""
        **Parámetros:** `n_estimators=100`, `max_depth=8`, `class_weight='balanced'`

        Combina 100 árboles. La predicción final es por votación mayoritaria.
        - **Accuracy test:** 95.2%
        - **Accuracy CV (5-fold):** 96.2% ± 1.3%
        """)
        img_fi = os.path.join(RUTAS["graficas"], "11_feature_importance.png")
        if os.path.exists(img_fi):
            st.image(img_fi, caption="Feature Importance – Random Forest",
                     use_column_width=True)
        img_conf_rf = os.path.join(RUTAS["graficas"], "12_confusion_rf.png")
        if os.path.exists(img_conf_rf):
            st.image(img_conf_rf, caption="Matriz de confusión – Random Forest",
                     use_column_width=True)
        else:
            st.info("Los resultados del modelo se están preparando. Vuelve a cargar la página en unos segundos.")

    with sub3:
        st.subheader("K-Means – Perfiles Territoriales")
        st.markdown("K-Means agrupa municipios en **4 clusters** por perfil epidemiológico.")
        ruta_km = os.path.join(RUTAS["reportes"], "municipios_clusters.csv")
        if os.path.exists(ruta_km):
            mun_km = pd.read_csv(ruta_km, dtype={"cod_dane_mun":str})
            if "perfil_cluster" in mun_km.columns:
                conteo_km = (mun_km["perfil_cluster"].value_counts()
                             .reset_index()
                             .rename(columns={"perfil_cluster":"Cluster","count":"Municipios"}))
                fig_km = go.Figure(go.Pie(
                    labels=conteo_km["Cluster"],
                    values=conteo_km["Municipios"],
                    hole=0.4,
                    marker_colors=list(COLORES_PRIORIDAD.values())
                ))
                fig_km.update_layout(**LAYOUT_BASE)
                st.plotly_chart(fig_km, use_container_width=True)
                perfil_km = (mun_km.groupby("perfil_cluster")[
                    ["tasa_x100k","pct_adolescentes","tendencia_H2_H1","total_casos"]
                ].mean().round(3))
                st.dataframe(perfil_km, use_container_width=True)
        else:
            st.info("Los resultados del modelo se están preparando. Vuelve a cargar la página en unos segundos.")
        img_codo = os.path.join(RUTAS["graficas"], "14_kmeans_codo.png")
        if os.path.exists(img_codo):
            st.image(img_codo, caption="Método del codo – Selección de k",
                     use_column_width=True)


# ===========================================================================
# TAB 5 – RECOMENDACIONES
# ===========================================================================
with tab5:
    st.header("💡 Recomendaciones Prescriptivas")
    st.markdown(
        "Recomendaciones generadas automáticamente. Carácter **preventivo y poblacional**."
    )
    st.divider()

    pct_ado_g   = round(df["es_adolescente"].mean()*100, 1)
    pct_rural_g = round(df["es_rural"].mean()*100, 1)
    n_crit      = len(mun[mun["nivel_prioridad"]=="Crítica"])
    depto_top2  = df["Departamento_ocurrencia"].value_counts().idxmax()

    st.subheader("📌 Hallazgos principales")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.info(f"**{pct_ado_g}%** de los casos son adolescentes (12–17 años). "
                f"Foco prioritario: prevención escolar.")
        st.info(f"**{pct_rural_g}%** de los casos ocurrió en zona rural, "
                f"donde el acceso a salud mental es limitado.")
    with col_h2:
        st.warning(f"**{n_crit} municipios** en nivel **Crítico** (IPI ≥ 80). "
                   f"Requieren intervención inmediata.")
        st.warning(f"**{depto_top2}** tiene el mayor número absoluto de casos. "
                   f"El IPI complementa este dato con perspectiva relativa.")

    st.divider()
    st.subheader("🎯 Recomendación por municipio prioritario")
    top_municipios = mun.head(20)["Municipio_ocurrencia"].tolist()
    mun_sel = st.selectbox("Municipio", top_municipios)
    datos_mun = mun[mun["Municipio_ocurrencia"]==mun_sel].iloc[0]

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"**Departamento:** {datos_mun['Departamento_ocurrencia']}")
        st.markdown(f"**Total de casos:** {int(datos_mun['total_casos']):,}")
        st.markdown(f"**Tasa por 100k hab.:** {datos_mun['tasa_x100k']:.1f}")
        st.markdown(f"**% Adolescentes:** {datos_mun['pct_adolescentes']*100:.1f}%")
    with col_p2:
        st.markdown(f"**Tendencia H2/H1:** {datos_mun['tendencia_H2_H1']*100:.1f}%")
        st.markdown(f"**% Hospitalizados:** {datos_mun['pct_hospit']*100:.1f}%")
        st.markdown(f"**% Antec. psiquiátrico:** {datos_mun['pct_psiquia']*100:.1f}%")
        ipi_badge = {"Crítica":"🔴","Alta":"🟠","Media":"🔵","Baja":"🟢"}
        st.markdown(f"**IPI:** {datos_mun['IPI']:.1f} "
                    f"{ipi_badge.get(datos_mun['nivel_prioridad'],'')} "
                    f"**{datos_mun['nivel_prioridad']}**")

    def generar_recomendacion(row):
        recom = []
        if row["nivel_prioridad"] == "Crítica":
            recom.append("• **Intervención inmediata:** activar ruta de atención integral "
                         "en salud mental y articular con Secretaría Departamental de Salud.")
        if row["pct_adolescentes"] > 0.30:
            recom.append("• **Prevención escolar:** programas de salud mental en "
                         "instituciones educativas, grupo 12–17 años.")
        if row["tendencia_H2_H1"] > 0.3:
            recom.append("• **Vigilancia intensificada:** crecimiento sostenido en H2. "
                         "Fortalecer notificación y análisis semanal.")
        if row["pct_rural"] > 0.20:
            recom.append("• **Acceso rural:** fortalecer puestos de salud y telemedicina "
                         "en zonas rurales.")
        if row["pct_hospit"] > 0.70:
            recom.append("• **Capacidad hospitalaria:** revisar protocolos de urgencias "
                         "y red de salud mental de segundo nivel.")
        if row["pct_psiquia"] > 0.05:
            recom.append("• **Seguimiento post-evento:** fortalecer mecanismos de "
                         "acompañamiento tras el evento inicial.")
        if not recom:
            recom.append("• Mantener vigilancia epidemiológica activa y fortalecer "
                         "programas de promoción de salud mental comunitaria.")
        return "\n\n".join(recom)

    st.markdown("---")
    st.markdown(f"### Recomendación para **{mun_sel}**")
    st.markdown(generar_recomendacion(datos_mun))

    st.divider()
    st.subheader("📋 Recomendaciones generales para Colombia")
    st.markdown("""
    1. **Priorización territorial con enfoque adolescente:** los municipios con mayor
       concentración en el grupo 12–17 deben recibir refuerzo en el entorno escolar.
    2. **Monitoreo de municipios emergentes:** los que crecieron en H2 pueden escalar
       a Crítico en 2025. Incluirlos en el tablero mensual de las Secretarías de Salud.
    3. **Equidad territorial:** la tasa por 100k revela municipios pequeños con carga
       relativa superior a las grandes ciudades.
    4. **Fortalecimiento de la notificación:** mejorar capacidad de UPGD en municipios
       de baja densidad para mayor cobertura del sistema.
    5. **Articulación intersectorial:** salud, educación, protección y comunidad.
       Activar mesas intersectoriales en municipios de prioridad Alta y Crítica.
    """)

    if st.button("⬇️ Descargar ranking (agregado, ≥5 casos)"):
        ranking = mun[["ranking_nacional","Departamento_ocurrencia","Municipio_ocurrencia",
                        "total_casos","tasa_x100k","IPI","nivel_prioridad"]].copy()
        ranking = suprimir_celdas(ranking, "total_casos")   # excluye < 5 casos
        ranking["tasa_x100k"] = ranking["tasa_x100k"].round(1)
        ranking["IPI"]        = ranking["IPI"].round(1)
        st.download_button(
            label="📄 Descargar CSV",
            data=ranking.to_csv(index=False, encoding="utf-8-sig"),
            file_name="ranking_ipi_sivigila_2024_agregado.csv",
            mime="text/csv"
        )
        st.caption(f"El archivo excluye municipios con menos de {UMBRAL_PRIVACIDAD} "
                   "casos por privacidad. No contiene registros individuales.")