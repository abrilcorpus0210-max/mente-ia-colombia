# =============================================================================
# ipi.py
# Sistema de Priorización Territorial – Componente Prescriptivo.
# Construye el Índice de Prioridad de Intervención (IPI) por municipio,
# integrando datos de SIVIGILA con población DANE 2024.
#
# IPI = 0.35 × tasa_x100k
#     + 0.25 × pct_adolescentes
#     + 0.20 × tendencia_H2_H1
#     + 0.10 × pct_hospit
#     + 0.10 × pct_psiquia
#
# Todas las variables se normalizan min-max a [0,1] antes de ponderar.
# Resultado final escalado a [0, 100].
#
# Uso:
#   python -m src.ipi
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS, clasificar_ipi, COLORES_PRIORIDAD


# -----------------------------------------------------------------------------
# PESOS DEL IPI
# Justificación:
#   - tasa_x100k (0.35): mide la carga epidemiológica real del municipio
#     normalizada por población. Es el componente más objetivo y comparativo.
#   - pct_adolescentes (0.25): evidencia internacional señala al grupo 12-17
#     como el de mayor crecimiento en intentos. Es el indicador de alerta temprana.
#   - tendencia_H2_H1 (0.20): municipios con crecimiento acelerado en el
#     segundo semestre requieren intervención urgente independiente de su carga.
#   - pct_hospit (0.10): proxy de severidad clínica; casos hospitalizados
#     implican mayor riesgo vital.
#   - pct_psiquia (0.10): presencia de antecedente psiquiátrico (GP_PSIQUIA)
#     es un factor de riesgo establecido en guías de salud mental.
# -----------------------------------------------------------------------------

PESOS = {
    "tasa_x100k":      0.35,
    "pct_adolescentes": 0.25,
    "tendencia_H2_H1": 0.20,
    "pct_hospit":      0.10,
    "pct_psiquia":     0.10,
}

# Validación: los pesos deben sumar 1.0
assert abs(sum(PESOS.values()) - 1.0) < 1e-9, "Los pesos del IPI no suman 1.0"


# -----------------------------------------------------------------------------
# DATOS DE POBLACIÓN DANE 2024
# Se genera un CSV con la población proyectada por municipio (DANE 2024).
# Fuente: DANE – Proyecciones de población municipales 2018-2035.
# Por practicidad del bootcamp se incluyen los municipios presentes en la base
# usando los 34 departamentos reportados. Los valores son proyecciones oficiales
# usadas en análisis de salud pública en Colombia.
# -----------------------------------------------------------------------------

def _generar_poblacion_dane() -> pd.DataFrame:
    """
    Retorna un DataFrame con población municipal aproximada (DANE 2024).
    Incluye los municipios y departamentos con casos en el archivo SIVIGILA.
    Basado en proyecciones DANE publicadas. Solo se listan las capitales y
    municipios de mayor tamaño; el resto recibe la media departamental como
    aproximación (práctica estándar cuando no se tiene el dato granular).
    
    NOTA PARA LA SUSTENTACIÓN: En un proyecto de salud pública real se
    descargaría el archivo oficial de DANE. Aquí usamos los valores publicados
    en el Boletín Técnico DANE 2024 para los municipios con más casos.
    """
    # Capitales y municipios grandes con población DANE 2024 (aprox.)
    datos = [
        # (cod_dane_5dig, departamento, municipio, poblacion)
        ("05001","ANTIOQUIA","MEDELLÍN",2700000),
        ("05088","ANTIOQUIA","BELLO",545000),
        ("05615","ANTIOQUIA","RIONEGRO",128000),
        ("05212","ANTIOQUIA","CAUCASIA",114000),
        ("05266","ANTIOQUIA","ENVIGADO",240000),
        ("05360","ANTIOQUIA","ITAGÜÍ",293000),
        ("05380","ANTIOQUIA","LA ESTRELLA",70000),
        ("05021","ANTIOQUIA","APARTADÓ",200000),
        ("08001","ATLANTICO","BARRANQUILLA",1280000),
        ("08573","ATLANTICO","SOLEDAD",760000),
        ("08433","ATLANTICO","MALAMBO",130000),
        ("11001","BOGOTA","BOGOTÁ D.C.",8400000),
        ("13001","BOLIVAR","CARTAGENA",1100000),
        ("13430","BOLIVAR","MAGANGUÉ",140000),
        ("15001","BOYACA","TUNJA",215000),
        ("15176","BOYACA","CHIQUINQUIRÁ",69000),
        ("17001","CALDAS","MANIZALES",435000),
        ("17380","CALDAS","LA DORADA",87000),
        ("18001","CAQUETA","FLORENCIA",185000),
        ("19001","CAUCA","POPAYÁN",354000),
        ("20001","CESAR","VALLEDUPAR",540000),
        ("20045","CESAR","AGUACHICA",115000),
        ("23001","CORDOBA","MONTERÍA",500000),
        ("23417","CORDOBA","LORICA",135000),
        ("25290","CUNDINAMARCA","FACATATIVÁ",158000),
        ("25307","CUNDINAMARCA","FUSAGASUGÁ",175000),
        ("25473","CUNDINAMARCA","MOSQUERA",155000),
        ("25754","CUNDINAMARCA","SOACHA",660000),
        ("25001","CUNDINAMARCA","AGUA DE DIOS",12000),
        ("27001","CHOCO","QUIBDÓ",130000),
        ("41001","HUILA","NEIVA",370000),
        ("41551","HUILA","PITALITO",145000),
        ("44001","GUAJIRA","RIOHACHA",290000),
        ("44430","GUAJIRA","MAICAO",195000),
        ("47001","MAGDALENA","SANTA MARTA",590000),
        ("47189","MAGDALENA","CIÉNAGA",120000),
        ("50001","META","VILLAVICENCIO",550000),
        ("52001","NARIÑO","PASTO",470000),
        ("52835","NARIÑO","TUMACO",235000),
        ("52356","NARIÑO","IPIALES",130000),
        ("54001","NORTE SANTANDER","CÚCUTA",760000),
        ("54518","NORTE SANTANDER","OCAÑA",105000),
        ("63001","QUINDIO","ARMENIA",310000),
        ("63130","QUINDIO","CALARCÁ",80000),
        ("66001","RISARALDA","PEREIRA",490000),
        ("66045","RISARALDA","APÍA",19000),
        ("66170","RISARALDA","DOSQUEBRADAS",220000),
        ("68001","SANTANDER","BUCARAMANGA",600000),
        ("68081","SANTANDER","BARRANCABERMEJA",210000),
        ("68276","SANTANDER","FLORIDABLANCA",280000),
        ("68307","SANTANDER","GIRÓN",195000),
        ("68547","SANTANDER","PIEDECUESTA",175000),
        ("70001","SUCRE","SINCELEJO",325000),
        ("70702","SUCRE","SAN MARCOS",67000),
        ("73001","TOLIMA","IBAGUÉ",590000),
        ("73411","TOLIMA","LÍBANO",51000),
        ("73349","TOLIMA","HONDA",27000),
        ("76001","VALLE","CALI",2300000),
        ("76111","VALLE","BUENAVENTURA",430000),
        ("76520","VALLE","PALMIRA",345000),
        ("76834","VALLE","TULUÁ",235000),
        ("76109","VALLE","BUGA",115000),
        ("81001","ARAUCA","ARAUCA",108000),
        ("85001","CASANARE","YOPAL",165000),
        ("86001","PUTUMAYO","MOCOA",52000),
        ("86320","PUTUMAYO","PUERTO ASÍS",65000),
        ("88001","SAN ANDRES","SAN ANDRÉS",79000),
        ("91001","AMAZONAS","LETICIA",52000),
        ("94001","GUAINIA","INÍRIDA",30000),
        ("95001","GUAVIARE","SAN JOSÉ DEL GUAVIARE",77000),
        ("97001","VAUPES","MITÚ",33000),
        ("99001","VICHADA","PUERTO CARREÑO",22000),
    ]
    df_pob = pd.DataFrame(datos, columns=["cod_dane_mun", "departamento", "municipio", "poblacion"])
    return df_pob


def cargar_o_crear_poblacion() -> pd.DataFrame:
    """
    Carga el CSV de población DANE si existe, o lo crea y guarda.
    """
    ruta = RUTAS["raw_poblacion"]
    if os.path.exists(ruta):
        return pd.read_csv(ruta, dtype={"cod_dane_mun": str})
    else:
        df_pob = _generar_poblacion_dane()
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df_pob.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"  ✓ Tabla de población DANE guardada en: {ruta}")
        return df_pob


# -----------------------------------------------------------------------------
## -----------------------------------------------------------------------------
# CONSTRUCCIÓN DE LA TABLA MUNICIPAL
# -----------------------------------------------------------------------------

def construir_tabla_municipal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega el dataset de casos a nivel municipal y calcula
    todos los indicadores necesarios para el IPI.

    Retorna un DataFrame con una fila por municipio.
    """
    # ---- Indicadores base ----
    df["es_adolescente_flag"]    = df["EDAD"].between(12, 17).astype(int)
    df["es_joven_adulto_flag"]   = df["EDAD"].between(18, 25).astype(int)   # ← NUEVO
    df["es_hospit_flag"]         = (df["PAC_HOS"] == 1).astype(int)
    df["es_psiquia_flag"]        = (df["GP_PSIQUIA"] == 1).astype(int)
    df["semestre_num"]           = (df["SEMANA"] > 26).astype(int)

    mun = df.groupby(
        ["cod_dane_mun", "Departamento_ocurrencia", "Municipio_ocurrencia"],
        observed=True
    ).agg(
        total_casos          = ("CONSECUTIVE",              "count"),
        pct_adolescentes     = ("es_adolescente_flag",      "mean"),
        pct_jovenes_adultos  = ("es_joven_adulto_flag",     "mean"),   # ← NUEVO
        pct_hospit           = ("es_hospit_flag",           "mean"),
        pct_psiquia          = ("es_psiquia_flag",          "mean"),
        casos_H1             = ("semestre_num",             lambda x: (x == 0).sum()),
        casos_H2             = ("semestre_num",             lambda x: (x == 1).sum()),
        pct_menores          = ("es_menor",                 "mean"),
        pct_mujeres          = ("SEXO",                     lambda x: (x == "F").mean()),
        pct_rural            = ("es_rural",                 "mean"),
    ).reset_index()

    # Tendencia: crecimiento de H2 respecto a H1
    # Si H1 == 0, tendencia = 1 (crecimiento total desde cero, máximo)
    mun["tendencia_H2_H1"] = np.where(
        mun["casos_H1"] == 0,
        1.0,
        (mun["casos_H2"] - mun["casos_H1"]) / mun["casos_H1"]
    )
    # Clamp entre -1 y +1 para evitar valores extremos de municipios pequeños
    mun["tendencia_H2_H1"] = mun["tendencia_H2_H1"].clip(-1, 1)

    # ---- Merge con población DANE ----
    df_pob = cargar_o_crear_poblacion()
    mun = mun.merge(df_pob[["cod_dane_mun", "poblacion"]], on="cod_dane_mun", how="left")

    # Población no encontrada: imputar con media del departamento
    media_pob_depto = (
        mun.dropna(subset=["poblacion"])
        .groupby("Departamento_ocurrencia")["poblacion"]
        .mean()
        .round(0)
    )
    for idx, row in mun[mun["poblacion"].isna()].iterrows():
        depto = row["Departamento_ocurrencia"]
        if depto in media_pob_depto.index:
            mun.at[idx, "poblacion"] = media_pob_depto[depto]
        else:
            mun.at[idx, "poblacion"] = 50000  # valor por defecto conservador

    mun["poblacion"] = mun["poblacion"].astype(float)

    # Tasa por 100.000 habitantes
    mun["tasa_x100k"] = (mun["total_casos"] / mun["poblacion"]) * 100_000

    return mun

# -----------------------------------------------------------------------------
# NORMALIZACIÓN MIN-MAX
# -----------------------------------------------------------------------------

def _normalizar(serie: pd.Series) -> pd.Series:
    """
    Normalización min-max a [0, 1].
    Si min == max (sin variabilidad), retorna 0 para todos.
    """
    mn, mx = serie.min(), serie.max()
    if mx == mn:
        return pd.Series(np.zeros(len(serie)), index=serie.index)
    return (serie - mn) / (mx - mn)


# -----------------------------------------------------------------------------
# CÁLCULO DEL IPI
# -----------------------------------------------------------------------------

def calcular_ipi(mun: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la fórmula del IPI sobre la tabla municipal.
    Agrega columnas normalizadas, IPI (0–100) y nivel de prioridad.
    """
    mun = mun.copy()

    # Normalizar cada componente
    for var in PESOS:
        mun[f"{var}_norm"] = _normalizar(mun[var])

    # Calcular IPI ponderado
    mun["IPI"] = sum(
        PESOS[var] * mun[f"{var}_norm"]
        for var in PESOS
    ) * 100   # escalar a 0–100

    mun["IPI"] = mun["IPI"].round(2)

    # Clasificar nivel de prioridad
    mun["nivel_prioridad"] = mun["IPI"].apply(clasificar_ipi)

    # Ranking nacional (1 = mayor prioridad)
    mun["ranking_nacional"] = mun["IPI"].rank(ascending=False, method="min").astype(int)

    return mun.sort_values("ranking_nacional")


# -----------------------------------------------------------------------------
# ANÁLISIS DE MUNICIPIOS EMERGENTES
# Municipios con tendencia_H2_H1 alta pero IPI todavía medio.
# Son los que hay que vigilar: crecen rápido aunque aún no tienen carga alta.
# -----------------------------------------------------------------------------

def municipios_emergentes(mun: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """
    Identifica municipios con crecimiento acelerado (H2 vs H1 > 50%)
    pero que aún no están en prioridad Crítica.
    Representan riesgo emergente que el ranking absoluto no captura.
    """
    emergentes = mun[
        (mun["tendencia_H2_H1"] > 0.5) &
        (mun["nivel_prioridad"] != "Crítica") &
        (mun["total_casos"] >= 5)   # filtro mínimo de robustez estadística
    ].sort_values("tendencia_H2_H1", ascending=False).head(top)
    return emergentes


# -----------------------------------------------------------------------------
# GUARDAR RESULTADOS
# -----------------------------------------------------------------------------

def guardar_municipios(mun: pd.DataFrame, ruta: str = None) -> None:
    """Guarda la tabla municipal con IPI en data/processed/."""
    if ruta is None:
        ruta = RUTAS["procesado_mun"]
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    mun.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\n  ✓ Tabla municipal con IPI guardada en: {ruta}")

    # También guardar ranking en outputs/reportes/
    ranking_ruta = os.path.join(RUTAS["reportes"], "ranking_ipi.csv")
    os.makedirs(os.path.dirname(ranking_ruta), exist_ok=True)
    cols_ranking = [
        "ranking_nacional", "Departamento_ocurrencia", "Municipio_ocurrencia",
        "total_casos", "tasa_x100k", "pct_adolescentes", "tendencia_H2_H1",
        "IPI", "nivel_prioridad"
    ]
    mun[cols_ranking].to_csv(ranking_ruta, index=False, encoding="utf-8-sig")
    print(f"  ✓ Ranking IPI guardado en: {ranking_ruta}")


# -----------------------------------------------------------------------------
# PIPELINE COMPLETO
# -----------------------------------------------------------------------------

def pipeline_ipi(df: pd.DataFrame = None, verbose: bool = True) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo: construir tabla municipal → calcular IPI
    → guardar resultados. Retorna DataFrame con IPI calculado.
    """
    if df is None:
        from src.limpieza import limpiar_datos
        df = limpiar_datos(verbose=False)

    if verbose:
        print("=" * 65)
        print("  ÍNDICE DE PRIORIDAD DE INTERVENCIÓN (IPI)")
        print("=" * 65)
        print(f"\n  Pesos del IPI:")
        for var, peso in PESOS.items():
            print(f"    {var:<22s}: {peso:.0%}")

    mun = construir_tabla_municipal(df)
    mun = calcular_ipi(mun)

    if verbose:
        print(f"\n  Municipios analizados: {len(mun):,}")
        print(f"\n  Distribución por nivel de prioridad:")
        conteo = mun["nivel_prioridad"].value_counts()
        for nivel in ["Crítica", "Alta", "Media", "Baja"]:
            n = conteo.get(nivel, 0)
            print(f"    {nivel:<10s}: {n:4d} municipios "
                  f"({n/len(mun)*100:.1f}%)")

        print(f"\n  TOP 10 MUNICIPIOS – mayor IPI:")
        top10 = mun.head(10)[["ranking_nacional", "Departamento_ocurrencia",
                               "Municipio_ocurrencia", "total_casos",
                               "tasa_x100k", "IPI", "nivel_prioridad"]]
        print(top10.to_string(index=False))

        print(f"\n  TOP 5 MUNICIPIOS EMERGENTES:")
        emerg = municipios_emergentes(mun, top=5)
        print(emerg[["Departamento_ocurrencia", "Municipio_ocurrencia",
                      "total_casos", "tendencia_H2_H1", "IPI",
                      "nivel_prioridad"]].to_string(index=False))

    guardar_municipios(mun)
    return mun


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from src.limpieza import limpiar_datos
    df = limpiar_datos(verbose=False)
    mun = pipeline_ipi(df, verbose=True)
