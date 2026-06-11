# =============================================================================
# limpieza.py
# Fase 2 CRISP-DM: Preparación de los datos.
# Normaliza nombres, convierte fechas, construye el código DANE de 5 dígitos,
# crea variables derivadas, descarta columnas vacías y guarda el dataset limpio.
#
# Uso desde la raíz del proyecto:
#   python -m src.limpieza
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    RUTAS, BINS_EDAD, ETIQUETAS_EDAD, BINS_SEMANA, ETIQUETAS_TRIMESTRE,
    ETIQUETAS_SEXO, ETIQUETAS_AREA, ETIQUETAS_HOSPITALIZACION
)
from src.carga import cargar_datos


# -----------------------------------------------------------------------------
# COLUMNAS QUE SE DESCARTAN
# Vacías (>95% nulos) o irrelevantes para el análisis territorial.
# -----------------------------------------------------------------------------

COLUMNAS_ELIMINAR = [
    "GRU_POB",      # 100% nulo
    "FEC_DEF",      # 100% nulo (no hay fallecidos en esta base)
    "CBMTE",        # 100% nulo
    "CER_DEF",      # 100% nulo
    "FM_FUERZA",    # 99.5% nulo (fuerzas militares, irrelevante aquí)
    "FM_UNIDAD",    # 99.5% nulo
    "FM_GRADO",     # 99.5% nulo
    "COD_PAIS_O",   # Toda Colombia, sin variabilidad útil
    "COD_PAIS_R",   # Ídem
    "COD_PAIS_N",   # No existe en la base (COD_PAIS_R cubre residencia)
    "COD_PRE",      # Código de prestador, no relevante para análisis territorial
    "COD_SUB",      # Código de subgrupo de evento, sin variabilidad
    "COD_EVE",      # Todos son 356; se mantiene Nombre_evento
    "ANO",          # Todos son 2024; constante
    "CON_FIN",      # Todos son 1 (confirmado); sin variabilidad
    "nom_est_f_caso", # Idem texto
    "AJUSTE",       # Código de ajuste de notificación, no analítico
    "va_sispro",    # Variable interna SIVIGILA
    "FEC_AJU",      # Fecha de ajuste, no analítica
    "FEC_ARC_XL",   # Fecha de archivo, no analítica
    "confirmados",  # Siempre 1
]


# -----------------------------------------------------------------------------
# PIPELINE PRINCIPAL DE LIMPIEZA
# -----------------------------------------------------------------------------

def limpiar_datos(df: pd.DataFrame = None, verbose: bool = True) -> pd.DataFrame:
    """
    Aplica el pipeline completo de limpieza y transformación.

    Parámetros
    ----------
    df : pd.DataFrame, opcional
        DataFrame en bruto. Si None, llama a cargar_datos().
    verbose : bool
        Si True, imprime resumen de cambios.

    Retorna
    -------
    pd.DataFrame limpio con variables derivadas.
    """
    if df is None:
        df = cargar_datos(verbose=False)

    df = df.copy()
    filas_iniciales = len(df)

    if verbose:
        print("=" * 65)
        print("  LIMPIEZA DE DATOS – SIVIGILA 2024")
        print("=" * 65)

    # ------------------------------------------------------------------
    # PASO 1: Normalizar nombres de columnas
    # Se conservan los nombres originales de SIVIGILA para trazabilidad,
    # pero se agrega una versión limpia para columnas derivadas.
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip()
    if verbose:
        print("\n[1/9] Nombres de columnas normalizados (espacios eliminados)")

    # ------------------------------------------------------------------
    # PASO 2: Eliminar duplicados
    # ------------------------------------------------------------------
    antes = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"[2/9] Duplicados eliminados: {antes - len(df)}")

    # ------------------------------------------------------------------
    # PASO 3: Descartar columnas de alta nulidad o sin variabilidad
    # ------------------------------------------------------------------
    cols_a_eliminar = [c for c in COLUMNAS_ELIMINAR if c in df.columns]
    df = df.drop(columns=cols_a_eliminar)
    if verbose:
        print(f"[3/9] Columnas descartadas: {len(cols_a_eliminar)}")

    # ------------------------------------------------------------------
    # PASO 4: Construir código DANE de 5 dígitos (clave municipal)
    # CRÍTICO: COD_MUN_O solo no es único entre departamentos.
    # El código DANE correcto es DPTO (2 dígitos) + MUN (3 dígitos).
    # ------------------------------------------------------------------
    df["cod_dane_mun"] = (
        df["COD_DPTO_O"].astype(str).str.zfill(2) +
        df["COD_MUN_O"].astype(str).str.zfill(3)
    )
    if verbose:
        print(f"[4/9] Código DANE 5 dígitos construido | "
              f"Municipios únicos: {df['cod_dane_mun'].nunique()}")

    # ------------------------------------------------------------------
    # PASO 5: Convertir fechas (vienen como texto 'YYYY-MM-DD')
    # ------------------------------------------------------------------
    columnas_fecha = ["FEC_NOT", "INI_SIN", "FEC_CON", "FEC_HOS", "FECHA_NTO"]
    for col in columnas_fecha:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")

    if verbose:
        nulos_fecha = df["FEC_NOT"].isna().sum()
        print(f"[5/9] Fechas convertidas | FEC_NOT nulos: {nulos_fecha}")

    # ------------------------------------------------------------------
    # PASO 6: Variables temporales derivadas
    # ------------------------------------------------------------------
    _MESES_ES = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                 7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
                 11:"Noviembre",12:"Diciembre"}
    df["mes"]           = df["FEC_NOT"].dt.month
    df["mes_nombre"]    = df["mes"].map(_MESES_ES)
    df["trimestre"]     = pd.cut(
                              df["SEMANA"],
                              bins=BINS_SEMANA,
                              labels=ETIQUETAS_TRIMESTRE,
                              right=True
                          )
    df["semestre"]      = df["SEMANA"].apply(lambda s: "H1 (Sem 1–26)" if s <= 26 else "H2 (Sem 27–52)")

    # Edad en el momento del reporte (calculada desde FECHA_NTO si existe)
    # Usamos directamente EDAD ya que UNI_MED = 1 (años) en toda la base.
    if verbose:
        print("[6/9] Variables temporales derivadas: mes, trimestre, semestre")

    # ------------------------------------------------------------------
    # PASO 7: Variables demográficas derivadas
    # ------------------------------------------------------------------
    # Grupo de edad estandarizado para salud pública
    df["grupo_edad"] = pd.cut(
        df["EDAD"],
        bins=BINS_EDAD,
        labels=ETIQUETAS_EDAD,
        right=True
    )

    # Etiquetas legibles para sexo, área, hospitalización
    df["sexo_nombre"]   = df["SEXO"].map(ETIQUETAS_SEXO).fillna("Otro/Ind.")
    df["area_nombre"]   = df["AREA"].map(ETIQUETAS_AREA).fillna("Desconocida")
    df["hospitalizado"] = df["PAC_HOS"].map(ETIQUETAS_HOSPITALIZACION).fillna("Desconocido")

    # Flag booleano para menores de 18 y adolescentes (12–17)
    df["es_menor"]       = df["EDAD"] < 18
    df["es_adolescente"] = df["EDAD"].between(12, 17)

    # Flag: ¿zona rural? (AREA == 3)
    df["es_rural"] = df["AREA"] == 3

    if verbose:
        print("[7/9] Variables demográficas derivadas: grupo_edad, flags menor/rural")

    # ------------------------------------------------------------------
    # PASO 8: Normalizar nom_grupo (etnia) – quitar espacios excesivos
    # ------------------------------------------------------------------
    if "nom_grupo" in df.columns:
        df["nom_grupo"] = df["nom_grupo"].str.strip().replace("", "Sin especificar")
        df["nom_grupo"] = df["nom_grupo"].fillna("Sin especificar")

    if verbose:
        print("[8/9] Campo nom_grupo (etnia) normalizado")

    # ------------------------------------------------------------------
    # PASO 9: Reporte final
    # ------------------------------------------------------------------
    filas_finales = len(df)
    if verbose:
        print(f"[9/9] Dataset limpio listo")
        print(f"\n  Filas iniciales   : {filas_iniciales:,}")
        print(f"  Filas finales     : {filas_finales:,}")
        print(f"  Columnas finales  : {df.shape[1]}")
        print(f"  Columnas nuevas   : cod_dane_mun, mes, mes_nombre, trimestre, "
              f"semestre, grupo_edad, sexo_nombre, area_nombre, hospitalizado, "
              f"es_menor, es_adolescente, es_rural")
        _detectar_outliers(df)

    return df


# -----------------------------------------------------------------------------
# DETECCIÓN DE OUTLIERS (para diagnóstico, no para eliminación)
# El tema de salud pública requiere conservar todos los registros extremos.
# -----------------------------------------------------------------------------

def _detectar_outliers(df: pd.DataFrame) -> None:
    """
    Reporta posibles valores atípicos en EDAD usando IQR.
    No elimina: en salud pública los extremos son clínicamente válidos.
    """
    print("\n  Análisis de outliers – EDAD (método IQR):")
    q1  = df["EDAD"].quantile(0.25)
    q3  = df["EDAD"].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    outliers = df[(df["EDAD"] < lim_inf) | (df["EDAD"] > lim_sup)]
    print(f"    Límite inferior: {lim_inf:.1f} | Límite superior: {lim_sup:.1f}")
    print(f"    Casos fuera de rango: {len(outliers):,} "
          f"({len(outliers)/len(df)*100:.1f}%)")
    print(f"    → Conservados: representan casos clínicamente válidos "
          f"(niños < 12 y adultos mayores).")


# -----------------------------------------------------------------------------
# GUARDAR DATASET LIMPIO
# -----------------------------------------------------------------------------

def guardar_procesado(df: pd.DataFrame, ruta: str = None) -> None:
    """
    Guarda el DataFrame limpio como CSV en data/processed/.
    """
    if ruta is None:
        ruta = RUTAS["procesado_casos"]
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ Dataset limpio guardado en: {ruta}")
    print(f"    {len(df):,} filas × {df.shape[1]} columnas")


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df_limpio = limpiar_datos(verbose=True)
    guardar_procesado(df_limpio)
