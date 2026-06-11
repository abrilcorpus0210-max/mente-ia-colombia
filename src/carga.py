# =============================================================================
# carga.py
# Fase 1 CRISP-DM: Comprensión de los datos.
# Lee el archivo Excel de SIVIGILA, detecta columnas automáticamente,
# y genera un perfil completo: shape, tipos, nulos, duplicados.
#
# Uso desde la raíz del proyecto:
#   python -m src.carga
# =============================================================================

import pandas as pd
import sys
import os

# Importar rutas y paleta desde utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE CARGA
# -----------------------------------------------------------------------------

def cargar_datos(ruta: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Lee el archivo Excel de SIVIGILA (Evento 356 – 2024).

    Parámetros
    ----------
    ruta : str, opcional
        Ruta al archivo .xlsx. Si no se indica, usa RUTAS['raw_xlsx'].
    verbose : bool
        Si True, imprime el perfil completo en consola.

    Retorna
    -------
    pd.DataFrame con los datos en bruto, sin modificar.
    """
    if ruta is None:
        ruta = RUTAS["raw_xlsx"]

    # --- Lectura ---
    # dtype=str en columnas de código para evitar que pandas interprete
    # códigos DANE como enteros y pierda los ceros a la izquierda.
    df = pd.read_excel(
        ruta,
        sheet_name="Hoja1",
        dtype={
            "COD_DPTO_O": str,
            "COD_MUN_O":  str,
            "COD_DPTO_R": str,
            "COD_MUN_R":  str,
            "COD_DPTO_N": str,
            "COD_MUN_N":  str,
        }
    )

    if verbose:
        _imprimir_perfil(df)

    return df


# -----------------------------------------------------------------------------
# PERFILADO AUTOMÁTICO
# -----------------------------------------------------------------------------

def _imprimir_perfil(df: pd.DataFrame) -> None:
    """
    Imprime un perfil estructurado del DataFrame:
    shape, head, info, describe y análisis de calidad.
    """
    sep = "=" * 65

    print(sep)
    print("  PERFIL DEL DATASET – SIVIGILA 2024 (Evento 356)")
    print(sep)

    # Shape
    print(f"\n▶ Dimensiones      : {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"▶ Evento           : {df['Nombre_evento'].iloc[0]}")
    print(f"▶ Año              : {df['ANO'].iloc[0]}")

    # Primeras filas
    print("\n" + "-" * 65)
    print("  PRIMERAS 3 FILAS (df.head(3))")
    print("-" * 65)
    print(df.head(3).T.to_string())   # transpuesto para legibilidad

    # Tipos de datos
    print("\n" + "-" * 65)
    print("  TIPOS DE DATOS (df.dtypes)")
    print("-" * 65)
    tipos = df.dtypes.reset_index()
    tipos.columns = ["Columna", "Tipo"]
    print(tipos.to_string(index=False))

    # Estadísticas descriptivas (solo numéricas relevantes)
    print("\n" + "-" * 65)
    print("  ESTADÍSTICAS DESCRIPTIVAS – columnas numéricas clave")
    print("-" * 65)
    cols_desc = ["EDAD", "SEMANA", "SEXO"]
    print(df[["EDAD", "SEMANA"]].describe().round(2).to_string())

    # Calidad de datos
    print("\n" + "-" * 65)
    print("  CALIDAD DE DATOS")
    print("-" * 65)

    total = len(df)
    duplicados_fila  = df.duplicated().sum()
    duplicados_cons  = df["CONSECUTIVE"].duplicated().sum()

    print(f"  Filas totales          : {total:,}")
    print(f"  Filas duplicadas       : {duplicados_fila}")
    print(f"  CONSECUTIVE duplicado  : {duplicados_cons}")

    # Porcentaje de nulos por columna (solo las que tienen > 0 %)
    nulos = (df.isna().sum() / total * 100).round(1)
    nulos = nulos[nulos > 0].sort_values(ascending=False)

    if nulos.empty:
        print("  Valores nulos          : ninguno en ninguna columna ✓")
    else:
        print(f"\n  Columnas con valores nulos ({len(nulos)}):")
        for col, pct in nulos.items():
            barra = "█" * int(pct / 5)
            print(f"    {col:<20s}  {pct:5.1f}%  {barra}")

    # Distribuciones rápidas de variables clave
    print("\n" + "-" * 65)
    print("  DISTRIBUCIÓN VARIABLES CLAVE")
    print("-" * 65)
    print("  SEXO:")
    print(df["SEXO"].value_counts(dropna=False).to_string())
    print("\n  ÁREA:")
    print(df["AREA"].value_counts(dropna=False).to_string())
    print("\n  PAC_HOS (Hospitalizado):")
    print(df["PAC_HOS"].value_counts(dropna=False).to_string())
    print("\n  Top 5 Departamentos de Ocurrencia:")
    print(df["Departamento_ocurrencia"].value_counts().head(5).to_string())

    print("\n" + sep + "\n")


def obtener_columnas(df: pd.DataFrame) -> dict:
    """
    Detecta y clasifica automáticamente las columnas disponibles
    en el DataFrame. Útil para validar antes de ejecutar cada fase.

    Retorna un diccionario con categorías de columnas.
    """
    columnas = {
        "identificacion":  ["CONSECUTIVE", "COD_EVE"],
        "temporales":      [c for c in df.columns if c in
                            ["FEC_NOT", "INI_SIN", "FEC_CON", "FEC_HOS",
                             "FECHA_NTO", "SEMANA", "ANO", "FEC_ARC_XL"]],
        "demograficas":    [c for c in df.columns if c in
                            ["EDAD", "UNI_MED", "SEXO", "PER_ETN",
                             "OCUPACION", "estrato", "nacionalidad"]],
        "geograficas":     [c for c in df.columns if c in
                            ["COD_DPTO_O", "COD_MUN_O", "AREA",
                             "Departamento_ocurrencia", "Municipio_ocurrencia",
                             "COD_DPTO_R", "COD_MUN_R"]],
        "clinicas":        [c for c in df.columns if c in
                            ["PAC_HOS", "CON_FIN", "TIP_CAS", "TIP_SS"]],
        "grupos_riesgo":   [c for c in df.columns if c.startswith("GP_")],
        "alta_nulidad":    ["GRU_POB", "FEC_DEF", "CBMTE", "CER_DEF",
                            "FM_FUERZA", "FM_UNIDAD", "FM_GRADO"],
    }
    return columnas


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df = cargar_datos(verbose=True)
    cols = obtener_columnas(df)
    print("Clasificación de columnas detectadas:")
    for categoria, lista in cols.items():
        print(f"  {categoria:<18s}: {lista}")
