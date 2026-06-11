# =============================================================================
# visualizaciones.py
# Fase 4 CRISP-DM: todas las gráficas estáticas del EDA.
# Paleta visual: Celadon (#A8D5BA) y Azul Bondi (#0D7C8F).
# Diseño sobrio y moderno para salud pública.
# Cada función guarda el PNG en outputs/graficas/ y retorna la figura.
#
# Uso desde la raíz del proyecto:
#   python -m src.visualizaciones
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # backend sin pantalla para entornos de servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS, PALETA, SECUENCIA_COLORES

# --- Estilo global ---
plt.rcParams.update({
    "figure.facecolor":  PALETA["fondo"],
    "axes.facecolor":    PALETA["fondo"],
    "axes.edgecolor":    PALETA["neutro"],
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "sans-serif",
    "legend.fontsize":   9,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})
sns.set_style("whitegrid")


# -----------------------------------------------------------------------------
# HELPER INTERNO
# -----------------------------------------------------------------------------

def _guardar(fig: plt.Figure, nombre: str) -> None:
    """Guarda la figura en outputs/graficas/."""
    ruta = os.path.join(RUTAS["graficas"], nombre)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=PALETA["fondo"])
    print(f"  ✓ Gráfica guardada: {ruta}")


# -----------------------------------------------------------------------------
# 1. BARRAS – Casos por departamento (Top N)
# -----------------------------------------------------------------------------

def grafica_departamentos(df: pd.DataFrame, top: int = 15) -> plt.Figure:
    """Barras horizontales con los N departamentos con más casos."""
    from src.eda import casos_por_departamento
    tabla = casos_por_departamento(df, top=top)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        tabla["Departamento_ocurrencia"],
        tabla["total_casos"],
        color=PALETA["primario"],
        edgecolor="white",
        linewidth=0.5
    )
    # Etiquetas de valor
    for bar in bars:
        ax.text(
            bar.get_width() + 30,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width()):,}",
            va="center", ha="left", fontsize=8, color=PALETA["neutro"]
        )
    ax.invert_yaxis()
    ax.set_title(f"Top {top} departamentos con mayor número de casos\n"
                 f"Intentos de suicidio – Colombia 2024")
    ax.set_xlabel("Número de casos notificados")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _guardar(fig, "01_casos_por_departamento.png")
    return fig


# -----------------------------------------------------------------------------
# 2. BARRAS – Casos por grupo de edad
# -----------------------------------------------------------------------------

def grafica_grupo_edad(df: pd.DataFrame) -> plt.Figure:
    """Barras verticales por grupo de edad estandarizado."""
    from src.eda import casos_por_grupo_edad
    tabla = casos_por_grupo_edad(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    colores = [PALETA["advertencia"] if "12–17" in str(g) else PALETA["primario"]
               for g in tabla["grupo_edad"]]
    bars = ax.bar(
        tabla["grupo_edad"].astype(str),
        tabla["total_casos"],
        color=colores,
        edgecolor="white",
        linewidth=0.5
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{int(bar.get_height()):,}",
            ha="center", fontsize=8, color=PALETA["neutro"]
        )
    ax.set_title("Distribución de casos por grupo de edad\nIntentos de suicidio – Colombia 2024")
    ax.set_xlabel("Grupo de edad")
    ax.set_ylabel("Número de casos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    # Nota sobre adolescentes
    ax.annotate("↑ Grupo más afectado",
                xy=(1, tabla.loc[tabla["grupo_edad"].astype(str) == "12–17",
                                  "total_casos"].values[0]),
                xytext=(2.5, tabla["total_casos"].max() * 0.85),
                arrowprops=dict(arrowstyle="->", color=PALETA["advertencia"]),
                color=PALETA["advertencia"], fontsize=9)
    fig.tight_layout()
    _guardar(fig, "02_casos_por_grupo_edad.png")
    return fig


# -----------------------------------------------------------------------------
# 3. PIE / BARRAS – Casos por sexo
# -----------------------------------------------------------------------------

def grafica_sexo(df: pd.DataFrame) -> plt.Figure:
    """Donut chart y barras de porcentaje por sexo."""
    from src.eda import casos_por_sexo
    tabla = casos_por_sexo(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Donut
    colores_sexo = [PALETA["primario"], PALETA["secundario"]]
    wedges, texts, autotexts = ax1.pie(
        tabla["total_casos"],
        labels=tabla["sexo_nombre"],
        autopct="%1.1f%%",
        colors=colores_sexo,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white")
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax1.set_title("Por sexo (%)")

    # Barras absolutas
    ax2.bar(tabla["sexo_nombre"], tabla["total_casos"],
            color=colores_sexo, edgecolor="white")
    for i, row in tabla.iterrows():
        ax2.text(i, row["total_casos"] + 100,
                 f"{row['total_casos']:,}", ha="center", fontsize=10)
    ax2.set_title("Por sexo (n)")
    ax2.set_ylabel("Casos")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Distribución por sexo – Intentos de suicidio Colombia 2024",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, "03_casos_por_sexo.png")
    return fig


# -----------------------------------------------------------------------------
# 4. LÍNEA TEMPORAL – Tendencia semanal
# -----------------------------------------------------------------------------

def grafica_tendencia_semanal(df: pd.DataFrame) -> plt.Figure:
    """Línea de casos por semana epidemiológica con media móvil."""
    from src.eda import tendencia_semanal
    tabla = tendencia_semanal(df)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tabla["SEMANA"], tabla["casos"],
            color=PALETA["acento"], linewidth=1, alpha=0.6, label="Casos por semana")
    ax.plot(tabla["SEMANA"], tabla["media_movil_4sem"],
            color=PALETA["primario"], linewidth=2.5, label="Media móvil 4 semanas")
    ax.fill_between(tabla["SEMANA"], tabla["casos"],
                    alpha=0.1, color=PALETA["primario"])

    # Marcar semana pico
    pico = tabla.loc[tabla["casos"].idxmax()]
    ax.annotate(
        f"Pico: sem. {int(pico['SEMANA'])} ({int(pico['casos'])} casos)",
        xy=(pico["SEMANA"], pico["casos"]),
        xytext=(pico["SEMANA"] - 8, pico["casos"] + 30),
        arrowprops=dict(arrowstyle="->", color=PALETA["advertencia"]),
        color=PALETA["advertencia"], fontsize=9
    )

    ax.set_title("Tendencia temporal de casos por semana epidemiológica\n"
                 "Intentos de suicidio – Colombia 2024")
    ax.set_xlabel("Semana epidemiológica")
    ax.set_ylabel("Número de casos")
    ax.legend()
    ax.set_xlim(1, 52)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _guardar(fig, "04_tendencia_semanal.png")
    return fig


# -----------------------------------------------------------------------------
# 5. MAPA DE CALOR – Cruce mes × sexo
# -----------------------------------------------------------------------------

def grafica_mapa_calor_mes_sexo(df: pd.DataFrame) -> plt.Figure:
    """Heatmap de casos: meses (filas) × sexo (columnas)."""
    from src.eda import cruce_mes_sexo
    tabla = cruce_mes_sexo(df)

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        tabla,
        annot=True, fmt=",d",
        cmap=sns.light_palette(PALETA["primario"], as_cmap=True),
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Número de casos"}
    )
    ax.set_title("Mapa de calor: mes de notificación × sexo\n"
                 "Intentos de suicidio – Colombia 2024")
    ax.set_xlabel("Sexo")
    ax.set_ylabel("Mes")
    fig.tight_layout()
    _guardar(fig, "05_heatmap_mes_sexo.png")
    return fig


# -----------------------------------------------------------------------------
# 6. BOXPLOT – Edad por sexo
# -----------------------------------------------------------------------------

def grafica_boxplot_edad_sexo(df: pd.DataFrame) -> plt.Figure:
    """Boxplot de distribución de edad por sexo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data_f = df[df["SEXO"] == "F"]["EDAD"].dropna()
    data_m = df[df["SEXO"] == "M"]["EDAD"].dropna()

    bp = ax.boxplot(
        [data_f, data_m],
        labels=["Femenino", "Masculino"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=PALETA["neutro"]),
        capprops=dict(color=PALETA["neutro"]),
        flierprops=dict(marker="o", markerfacecolor=PALETA["acento"],
                        markersize=3, alpha=0.4)
    )
    bp["boxes"][0].set_facecolor(PALETA["primario"])
    bp["boxes"][1].set_facecolor(PALETA["secundario"])

    ax.set_title("Distribución de edad por sexo\nIntentos de suicidio – Colombia 2024")
    ax.set_ylabel("Edad (años)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _guardar(fig, "06_boxplot_edad_sexo.png")
    return fig


# -----------------------------------------------------------------------------
# 7. HISTOGRAMA – Distribución de edades
# -----------------------------------------------------------------------------

def grafica_histograma_edad(df: pd.DataFrame) -> plt.Figure:
    """Histograma de edades con curva de densidad."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["EDAD"].dropna(), bins=30,
            color=PALETA["primario"], edgecolor="white",
            alpha=0.85, density=False)

    # Líneas verticales de referencia
    mediana = df["EDAD"].median()
    media   = df["EDAD"].mean()
    ax.axvline(mediana, color=PALETA["advertencia"], linestyle="--",
               linewidth=1.5, label=f"Mediana: {mediana:.0f}")
    ax.axvline(media,   color=PALETA["neutro"],      linestyle=":",
               linewidth=1.5, label=f"Media: {media:.1f}")

    # Sombrear grupo adolescente
    ax.axvspan(12, 17, alpha=0.15, color=PALETA["advertencia"],
               label="Adolescentes (12–17)")

    ax.set_title("Distribución de edades\nIntentos de suicidio – Colombia 2024")
    ax.set_xlabel("Edad (años)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _guardar(fig, "07_histograma_edad.png")
    return fig


# -----------------------------------------------------------------------------
# 8. BARRAS – Casos por área (urbano/rural)
# -----------------------------------------------------------------------------

def grafica_area(df: pd.DataFrame) -> plt.Figure:
    """Barras por área de residencia."""
    from src.eda import casos_por_area
    tabla = casos_por_area(df)

    fig, ax = plt.subplots(figsize=(7, 4))
    colores = [PALETA["primario"], PALETA["acento"], PALETA["secundario"]]
    bars = ax.bar(tabla["area_nombre"], tabla["total_casos"],
                  color=colores[:len(tabla)], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{int(bar.get_height()):,}", ha="center", fontsize=10)
    ax.set_title("Casos por área de residencia\nIntentos de suicidio – Colombia 2024")
    ax.set_ylabel("Número de casos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _guardar(fig, "08_casos_por_area.png")
    return fig


# -----------------------------------------------------------------------------
# 9. BARRAS AGRUPADAS – Cruce departamento × sexo (Top 10)
# -----------------------------------------------------------------------------

def grafica_depto_sexo(df: pd.DataFrame) -> plt.Figure:
    """Barras agrupadas: top 10 departamentos por sexo."""
    from src.eda import cruce_departamento_sexo
    tabla = cruce_departamento_sexo(df, top=10)
    # Quitar fila/columna Total
    tabla = tabla.drop("Total", axis=0, errors="ignore").drop("Total", axis=1, errors="ignore")

    x     = range(len(tabla))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], tabla.get("Femenino", [0]*len(tabla)),
           width, label="Femenino", color=PALETA["primario"], edgecolor="white")
    ax.bar([i + width/2 for i in x], tabla.get("Masculino", [0]*len(tabla)),
           width, label="Masculino", color=PALETA["secundario"], edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(tabla.index, rotation=30, ha="right")
    ax.set_title("Top 10 departamentos: casos por sexo\nIntentos de suicidio – Colombia 2024")
    ax.set_ylabel("Número de casos")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _guardar(fig, "09_depto_por_sexo.png")
    return fig


# -----------------------------------------------------------------------------
# 10. BARRAS – Grupos de riesgo (GP_*)
# -----------------------------------------------------------------------------

def grafica_grupos_riesgo(df: pd.DataFrame) -> plt.Figure:
    """Barras horizontales de grupos de riesgo con al menos 10 casos."""
    from src.eda import perfil_grupos_riesgo
    tabla = perfil_grupos_riesgo(df)
    tabla = tabla[tabla["casos"] >= 10]

    # Etiquetas legibles para los grupos
    etiquetas = {
        "GP_PSIQUIA":  "Antecedente psiquiátrico",
        "GP_MIGRANT":  "Población migrante",
        "GP_CARCELA":  "Privado de libertad",
        "GP_GESTAN":   "Gestante",
        "GP_INDIGEN":  "Población indígena",
        "GP_POBICFB":  "Beneficiario ICBF",
        "GP_DISCAPA":  "Con discapacidad",
        "GP_DESPLAZ":  "Desplazado/a",
        "GP_VIC_VIO":  "Víctima de violencia",
        "GP_OTROS":    "Otros grupos",
    }
    tabla["grupo_label"] = tabla["grupo"].map(etiquetas).fillna(tabla["grupo"])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(tabla["grupo_label"], tabla["casos"],
                   color=PALETA["acento"], edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 5,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}",
                va="center", fontsize=8, color=PALETA["neutro"])
    ax.invert_yaxis()
    ax.set_title("Casos según grupo poblacional de riesgo\nIntentos de suicidio – Colombia 2024")
    ax.set_xlabel("Número de casos")
    fig.tight_layout()
    _guardar(fig, "10_grupos_riesgo.png")
    return fig


# -----------------------------------------------------------------------------
# GENERAR TODAS LAS GRÁFICAS EN UN SOLO LLAMADO
# -----------------------------------------------------------------------------

def generar_todas(df: pd.DataFrame) -> None:
    """Ejecuta todas las funciones de visualización en secuencia."""
    print("\n Generando visualizaciones...\n")
    grafica_departamentos(df)
    grafica_grupo_edad(df)
    grafica_sexo(df)
    grafica_tendencia_semanal(df)
    grafica_mapa_calor_mes_sexo(df)
    grafica_boxplot_edad_sexo(df)
    grafica_histograma_edad(df)
    grafica_area(df)
    grafica_depto_sexo(df)
    grafica_grupos_riesgo(df)
    print("\n  ✓ Todas las gráficas generadas en outputs/graficas/\n")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from src.limpieza import limpiar_datos
    df = limpiar_datos(verbose=False)
    generar_todas(df)
