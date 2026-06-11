# =============================================================================
# eda.py
# Fase 3 CRISP-DM: Análisis Exploratorio de Datos (EDA).
# Genera todas las tablas resumen usando groupby y crosstab.
# Cada función retorna un DataFrame listo para visualizar o exportar.
#
# Uso desde la raíz del proyecto:
#   python -m src.eda
# =============================================================================

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS


# -----------------------------------------------------------------------------
# FUNCIONES DE RESUMEN
# Cada función recibe el DataFrame limpio y retorna una tabla.
# -----------------------------------------------------------------------------

def resumen_general(df: pd.DataFrame) -> dict:
    """
    KPIs globales del dataset. Retorna un dict con valores únicos.
    Usado para las tarjetas KPI del dashboard.
    """
    return {
        "total_casos":          len(df),
        "departamentos":        df["Departamento_ocurrencia"].nunique(),
        "municipios":           df["cod_dane_mun"].nunique(),
        "edad_promedio":        round(df["EDAD"].mean(), 1),
        "edad_mediana":         df["EDAD"].median(),
        "pct_femenino":         round((df["SEXO"] == "F").mean() * 100, 1),
        "pct_masculino":        round((df["SEXO"] == "M").mean() * 100, 1),
        "pct_menores":          round(df["es_menor"].mean() * 100, 1),
        "pct_adolescentes":     round(df["es_adolescente"].mean() * 100, 1),
        "pct_hospitalizados":   round((df["PAC_HOS"] == 1).mean() * 100, 1),
        "pct_rural":            round(df["es_rural"].mean() * 100, 1),
        "depto_mas_casos":      df["Departamento_ocurrencia"].value_counts().idxmax(),
        "mes_mas_casos":        df["mes"].value_counts().idxmax(),
        "grupo_edad_mas_casos": str(df["grupo_edad"].value_counts().idxmax()),
    }


def casos_por_departamento(df: pd.DataFrame, top: int = None) -> pd.DataFrame:
    """
    Cuenta de casos por departamento de ocurrencia.
    top: si se especifica, retorna solo los N primeros.
    """
    tabla = (
        df.groupby("Departamento_ocurrencia", observed=True)
        .agg(
            total_casos=("CONSECUTIVE", "count"),
            pct_menores=("es_menor", "mean"),
            pct_hospit=("PAC_HOS", lambda x: (x == 1).mean()),
        )
        .reset_index()
        .sort_values("total_casos", ascending=False)
    )
    tabla["pct_menores"] = (tabla["pct_menores"] * 100).round(1)
    tabla["pct_hospit"]  = (tabla["pct_hospit"]  * 100).round(1)
    if top:
        tabla = tabla.head(top)
    return tabla


def casos_por_municipio(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Top N municipios por número de casos."""
    tabla = (
        df.groupby(
            ["cod_dane_mun", "Departamento_ocurrencia", "Municipio_ocurrencia"],
            observed=True
        )
        .agg(total_casos=("CONSECUTIVE", "count"))
        .reset_index()
        .sort_values("total_casos", ascending=False)
        .head(top)
    )
    return tabla


def casos_por_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución por sexo."""
    tabla = (
        df.groupby("sexo_nombre", observed=True)
        .size()
        .reset_index(name="total_casos")
        .sort_values("total_casos", ascending=False)
    )
    tabla["porcentaje"] = (tabla["total_casos"] / tabla["total_casos"].sum() * 100).round(1)
    return tabla


def casos_por_grupo_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución por grupo de edad estandarizado."""
    tabla = (
        df.groupby("grupo_edad", observed=True)
        .size()
        .reset_index(name="total_casos")
    )
    tabla["porcentaje"] = (tabla["total_casos"] / tabla["total_casos"].sum() * 100).round(1)
    return tabla


def casos_por_mes(df: pd.DataFrame) -> pd.DataFrame:
    """Casos por mes de notificación (FEC_NOT)."""
    # Orden cronológico garantizado
    tabla = (
        df.groupby("mes", observed=True)
        .size()
        .reset_index(name="total_casos")
        .sort_values("mes")
    )
    meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
             7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    tabla["mes_nombre"] = tabla["mes"].map(meses)
    return tabla


def casos_por_area(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución por área de residencia (urbano/rural)."""
    tabla = (
        df.groupby("area_nombre", observed=True)
        .size()
        .reset_index(name="total_casos")
        .sort_values("total_casos", ascending=False)
    )
    tabla["porcentaje"] = (tabla["total_casos"] / tabla["total_casos"].sum() * 100).round(1)
    return tabla


def casos_por_trimestre(df: pd.DataFrame) -> pd.DataFrame:
    """Casos por trimestre epidemiológico."""
    tabla = (
        df.groupby("trimestre", observed=True)
        .size()
        .reset_index(name="total_casos")
    )
    return tabla


def tendencia_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serie de tiempo semanal (semanas 1–52).
    Incluye media móvil de 4 semanas para suavizar la curva.
    """
    tabla = (
        df.groupby("SEMANA", observed=True)
        .size()
        .reset_index(name="casos")
        .sort_values("SEMANA")
    )
    tabla["media_movil_4sem"] = tabla["casos"].rolling(window=4, min_periods=1).mean().round(1)
    return tabla


# -----------------------------------------------------------------------------
# CRUCES (CROSSTAB)
# -----------------------------------------------------------------------------

def cruce_sexo_grupo_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla cruzada: sexo × grupo de edad."""
    return pd.crosstab(
        df["grupo_edad"],
        df["sexo_nombre"],
        margins=True,
        margins_name="Total"
    )


def cruce_departamento_sexo(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    Tabla cruzada: top N departamentos × sexo.
    Ordenada por total de casos.
    """
    top_deptos = (
        df["Departamento_ocurrencia"]
        .value_counts()
        .head(top)
        .index
    )
    df_filtrado = df[df["Departamento_ocurrencia"].isin(top_deptos)]
    return pd.crosstab(
        df_filtrado["Departamento_ocurrencia"],
        df_filtrado["sexo_nombre"],
        margins=True,
        margins_name="Total"
    )


def cruce_mes_sexo(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla cruzada: mes × sexo."""
    tabla = pd.crosstab(df["mes"], df["sexo_nombre"])
    meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
             7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    tabla.index = tabla.index.map(meses)
    return tabla


def perfil_grupos_riesgo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen de los grupos poblacionales de riesgo (GP_*).
    Cuenta cuántos casos pertenecen a cada grupo (código 1 = sí).
    """
    gp_cols = [c for c in df.columns if c.startswith("GP_")]
    registros = []
    for col in gp_cols:
        n_si = (df[col] == 1).sum()
        registros.append({
            "grupo":      col,
            "casos":      n_si,
            "porcentaje": round(n_si / len(df) * 100, 2),
        })
    return (
        pd.DataFrame(registros)
        .sort_values("casos", ascending=False)
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA – muestra todas las tablas en consola
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.limpieza import limpiar_datos

    df = limpiar_datos(verbose=False)

    print("\n=== KPIs GLOBALES ===")
    kpis = resumen_general(df)
    for k, v in kpis.items():
        print(f"  {k:<30s}: {v}")

    print("\n=== CASOS POR DEPARTAMENTO (top 10) ===")
    print(casos_por_departamento(df, top=10).to_string(index=False))

    print("\n=== CASOS POR SEXO ===")
    print(casos_por_sexo(df).to_string(index=False))

    print("\n=== CASOS POR GRUPO DE EDAD ===")
    print(casos_por_grupo_edad(df).to_string(index=False))

    print("\n=== CASOS POR MES ===")
    print(casos_por_mes(df).to_string(index=False))

    print("\n=== GRUPOS DE RIESGO ===")
    print(perfil_grupos_riesgo(df).to_string(index=False))

    print("\n=== CRUCE SEXO × GRUPO EDAD ===")
    print(cruce_sexo_grupo_edad(df).to_string())
