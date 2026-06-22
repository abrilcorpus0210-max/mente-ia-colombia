# =============================================================================
# limpieza.py
# Fase 2 CRISP-DM: Preparación de los datos.
# Normaliza nombres, convierte fechas, construye el código DANE de 5 dígitos,
# crea variables derivadas, descarta columnas vacías y guarda el dataset limpio.
#
# Uso desde la raíz del proyecto:
#   python -m src.limpieza
#
# -----------------------------------------------------------------------------
# CHANGELOG de esta versión (auditoría sobre suicidios_2024.xlsx, 38.769 filas):
#   1. FIX IMPORTANTE: el filtro de año ahora usa la columna ANO (asignada por
#      SIVIGILA) en vez del año calendario de FEC_NOT. FEC_NOT va de
#      2023-12-31 a 2025-03-28 por efecto de semanas epidemiológicas en los
#      bordes del año; 255 filas con ANO=2024 legítimo tenían FEC_NOT en 2023
#      o 2025 y se estaban EXCLUYENDO por error con el filtro anterior.
#   2. ANO ya no se elimina del dataset (se quitó de COLUMNAS_ELIMINAR): se
#      necesita para el punto 1, y dejarla evita que Tab 6 de app.py tenga
#      que reconstruirla a partir de FEC_NOT (lo cual reintroduciría el mismo
#      problema en cargas futuras).
#   3. Limpieza general de espacios en blanco aplicada a TODAS las columnas
#      de texto (no solo nom_grupo). Varias columnas (estrato, sem_ges,
#      nombre_nacionalidad, nom_grupo) vienen con relleno de ancho fijo y
#      tenían "nulos disfrazados" de strings de solo espacios que df.isna()
#      no detectaba.
#   4. estrato y sem_ges se convierten a numérico (Int64) tras la limpieza
#      de espacios, en vez de quedar como texto con padding.
#   5. Se valida INI_SIN contra FEC_NOT: 347 registros tenían INI_SIN más de
#      un año antes de la notificación (varios con patrón de año mal
#      tecleado, ej. 2004 en vez de 2024). Se anulan los valores implausibles
#      y se deja la bandera ini_sin_corregido para trazabilidad.
#   6. Se agrega flag_edad_inconsistente: marca (sin eliminar ni sobrescribir)
#      los registros donde EDAD no coincide con la edad calculada desde
#      FECHA_NTO con una diferencia mayor a 2 años (9 casos detectados).
#   7. Se agregan TIP_CAS y Estado_final_de_caso a COLUMNAS_ELIMINAR: ambas
#      son constantes en el 100% de las filas (sin variabilidad analítica),
#      igual que CON_FIN y nom_est_f_caso que ya estaban en la lista.
#   8. NUEVO: Normalización de nombres de departamentos para unificar
#      variantes ("BOGOTA D.C.", "BOGOTA", "BOGOTÁ" → "BOGOTÁ D.C.") y
#      eliminar duplicados que inflaban el conteo a 33 departamentos.
#   9. NUEVO: Asignación explícita del código DANE 11001 para todos los
#      registros de Bogotá D.C., independientemente de cómo venga COD_MUN_O.
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
    "TIP_CAS",      # Todos son 4 (mismo tipo de caso); sin variabilidad
    "CON_FIN",      # Todos son 1 (confirmado); sin variabilidad
    "Estado_final_de_caso",  # Todos son 4; sin variabilidad (ver nom_est_f_caso)
    "nom_est_f_caso", # Idem texto
    "AJUSTE",       # Código de ajuste de notificación, no analítico
    "va_sispro",    # Variable interna SIVIGILA
    "FEC_AJU",      # Fecha de ajuste, no analítica
    "FEC_ARC_XL",   # Fecha de archivo, no analítica
    "confirmados",  # Siempre 1
    # NOTA: "ANO" YA NO SE ELIMINA (ver changelog, punto 2). Aunque es
    # constante (2024) en este corte, es la fuente de verdad del año
    # epidemiológico asignado por SIVIGILA y se necesita para el filtro
    # del Paso 4. Eliminarla forzaba a app.py (Tab "Actualizar Datos") a
    # reconstruirla a partir de FEC_NOT, reintroduciendo el mismo error.
]

# -----------------------------------------------------------------------------
# AÑO DE ANÁLISIS Y FILTROS DE CALIDAD
# El proyecto analiza específicamente el corte 2024 del SIVIGILA. Cualquier
# registro fuera de ese alcance se excluye explícitamente y se reporta,
# en vez de dejarlo "flotando" en el dataset y distorsionando los KPIs
# (por ejemplo, inflando el conteo de departamentos o de semanas).
#
# IMPORTANTE: el filtro usa la columna ANO (año epidemiológico asignado por
# SIVIGILA), NO el año calendario de FEC_NOT. FEC_NOT puede caer en los
# bordes del año (ej. 2023-12-31 o 2025-03-28) para casos que de todas
# formas pertenecen al corte 2024 según ANO. Filtrar por FEC_NOT.year
# excluía ~255 casos válidos por este efecto de borde.
# -----------------------------------------------------------------------------

ANIO_ANALISIS = 2024

# Departamentos que no son entidades territoriales colombianas. SIVIGILA
# permite notificar casos de colombianos atendidos en el exterior; estos
# no tienen código DANE real ni población asociada, por lo que se excluyen
# del análisis territorial (no se descartan del conteo total nacional de
# casos hasta el momento del filtro, pero sí del análisis por municipio).
DEPARTAMENTOS_NO_TERRITORIALES = ["EXTERIOR"]

# Umbral (en días) para considerar implausible la fecha de inicio de
# síntomas (INI_SIN) respecto a la fecha de notificación (FEC_NOT). Se
# detectó un patrón sistemático de años mal tecleados (ej. 2004 en vez de
# 2024) que generaba diferencias de 10–23 años. Un año de diferencia ya es
# clínicamente atípico para un evento de intento de suicidio, así que se usa
# como corte conservador para anular el valor sin tocar el resto del registro.
UMBRAL_DIAS_INI_SIN = 365

# Umbral (en años) para marcar como inconsistente la diferencia entre EDAD
# reportada y la edad calculada a partir de FECHA_NTO. Solo se usa para
# generar una bandera de auditoría (flag_edad_inconsistente); no se elimina
# ni se sobrescribe ningún valor, porque no se puede saber con certeza cuál
# de los dos campos (EDAD o FECHA_NTO) tiene el error de captura.
UMBRAL_DIFERENCIA_EDAD_ANIOS = 2


# -----------------------------------------------------------------------------
# NORMALIZACIÓN DE DEPARTAMENTOS
# Unifica variantes del mismo departamento que SIVIGILA usa indistintamente.
# Esto resuelve el problema de que aparecían 33 departamentos en lugar de 32.
# -----------------------------------------------------------------------------

def normalizar_departamentos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica nombres de departamentos para eliminar duplicados.
    
    SIVIGILA usa variantes como "BOGOTA D.C.", "BOGOTA", "BOGOTÁ" para el mismo
    departamento. Esta función las unifica a un nombre canónico.
    
    También unifica "NORTE DE SANTANDER" → "NORTE SANTANDER" y variantes de
    San Andrés.
    """
    df = df.copy()
    
    # Limpiar espacios y convertir a mayúsculas (por si acaso)
    df["Departamento_ocurrencia"] = df["Departamento_ocurrencia"].str.upper().str.strip()
    
    # Mapeo de variantes → nombre canónico
    mapeo = {
        "BOGOTA D.C.": "BOGOTÁ D.C.",
        "BOGOTA": "BOGOTÁ D.C.",
        "BOGOTÁ": "BOGOTÁ D.C.",
        "BOGOTÁ D.C": "BOGOTÁ D.C.",
        "BOGOTA DC": "BOGOTÁ D.C.",
        "BOGOTÁ DC": "BOGOTÁ D.C.",
        "NORTE DE SANTANDER": "NORTE SANTANDER",
        "NORTE SANTANDER": "NORTE SANTANDER",  # ya está bien
        "SAN ANDRES": "SAN ANDRÉS",
        "SAN ANDRES Y PROVIDENCIA": "SAN ANDRÉS",
        "SAN ANDRÉS": "SAN ANDRÉS",  # ya está bien
    }
    
    df["Departamento_ocurrencia"] = df["Departamento_ocurrencia"].replace(mapeo)
    
    return df


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
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip()
    if verbose:
        print("\n[1/14] Nombres de columnas normalizados (espacios eliminados)")

    # ------------------------------------------------------------------
    # PASO 2: Limpieza de espacios en blanco en TODAS las columnas de texto
    # Varias columnas de SIVIGILA vienen con relleno de ancho fijo (texto
    # rellenado con espacios hasta una longitud constante). Esto genera
    # "nulos disfrazados": celdas que son solo espacios y que df.isna() no
    # detecta como faltantes. Se limpia aquí, antes de cualquier filtro o
    # transformación que dependa del contenido textual.
    # ------------------------------------------------------------------
    columnas_texto = df.select_dtypes(include=["object", "string"]).columns
    for col in columnas_texto:
        df[col] = df[col].str.strip()
    if verbose:
        print(f"[2/14] Espacios en blanco eliminados en {len(columnas_texto)} "
              f"columnas de texto")

    # ------------------------------------------------------------------
    # PASO 3: Normalizar departamentos (unificar variantes)
    # Esto resuelve el problema de los 33 departamentos en lugar de 32.
    # ------------------------------------------------------------------
    df = normalizar_departamentos(df)
    deptos_unicos = df["Departamento_ocurrencia"].nunique()
    if verbose:
        print(f"[3/14] Departamentos normalizados | "
              f"Departamentos únicos: {deptos_unicos}")

    # ------------------------------------------------------------------
    # PASO 4: Eliminar duplicados
    # ------------------------------------------------------------------
    antes = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"[4/14] Duplicados eliminados: {antes - len(df)}")

    # ------------------------------------------------------------------
    # PASO 5: Filtrar registros fuera del alcance del análisis
    # (a) Departamentos no territoriales (ej. "EXTERIOR").
    # (b) Año epidemiológico (columna ANO) distinto al año de análisis.
    #     Se usa ANO en vez del año calendario de FEC_NOT: ver nota en la
    #     constante ANIO_ANALISIS más arriba.
    # ------------------------------------------------------------------
    antes_filtro = len(df)

    mask_no_territorial = df["Departamento_ocurrencia"].isin(DEPARTAMENTOS_NO_TERRITORIALES)
    n_exterior = int(mask_no_territorial.sum())

    mask_anio_invalido = df["ANO"] != ANIO_ANALISIS
    n_anio_invalido = int(mask_anio_invalido.sum())

    mask_excluir = mask_no_territorial | mask_anio_invalido
    n_excluidos = int(mask_excluir.sum())

    df = df[~mask_excluir].copy()

    if verbose:
        print(f"[5/14] Filtro de calidad aplicado:")
        print(f"        - Casos en departamentos no territoriales (EXTERIOR): {n_exterior}")
        print(f"        - Casos con ANO fuera de {ANIO_ANALISIS}: {n_anio_invalido}")
        print(f"        - Total excluido (sin doble conteo): {n_excluidos} "
              f"({n_excluidos/antes_filtro*100:.2f}% del dataset)")
        print(f"        - Filas restantes: {len(df):,}")

    # ------------------------------------------------------------------
    # PASO 6: Descartar columnas de alta nulidad o sin variabilidad
    # ------------------------------------------------------------------
    cols_a_eliminar = [c for c in COLUMNAS_ELIMINAR if c in df.columns]
    df = df.drop(columns=cols_a_eliminar)
    if verbose:
        print(f"[6/14] Columnas descartadas: {len(cols_a_eliminar)}")

    # ------------------------------------------------------------------
    # PASO 7: Construir código DANE de 5 dígitos (clave municipal)
    # CRÍTICO: COD_MUN_O solo no es único entre departamentos.
    # El código DANE correcto es DPTO (2 dígitos) + MUN (3 dígitos).
    #
    # IMPORTANTE: Bogotá D.C. tiene código DANE 11001. Algunos registros
    # pueden tener COD_MUN_O = "001" o "000" o valores inconsistentes.
    # Se asigna explícitamente el código correcto para todos los registros
    # de Bogotá D.C.
    # ------------------------------------------------------------------
    df["cod_dane_mun"] = (
        df["COD_DPTO_O"].astype(str).str.zfill(2) +
        df["COD_MUN_O"].astype(str).str.zfill(3)
    )
    
    # Asignación explícita del código DANE 11001 para Bogotá D.C.
    # Esto asegura que todos los registros de Bogotá tengan el mismo código,
    # independientemente de cómo venga COD_MUN_O en el archivo original.
    mask_bogota = df["Departamento_ocurrencia"] == "BOGOTÁ D.C."
    df.loc[mask_bogota, "cod_dane_mun"] = "11001"
    
    if verbose:
        print(f"[7/14] Código DANE 5 dígitos construido | "
              f"Municipios únicos: {df['cod_dane_mun'].nunique()}")
        print(f"        - Registros de Bogotá D.C. con código 11001: {int(mask_bogota.sum()):,}")

    # ------------------------------------------------------------------
    # PASO 8: Convertir fechas (vienen como texto 'YYYY-MM-DD')
    # ------------------------------------------------------------------
    columnas_fecha = ["FEC_NOT", "INI_SIN", "FEC_CON", "FEC_HOS", "FECHA_NTO"]
    for col in columnas_fecha:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")

    if verbose:
        nulos_fecha = df["FEC_NOT"].isna().sum()
        print(f"[8/14] Fechas convertidas | FEC_NOT nulos: {nulos_fecha}")

    # ------------------------------------------------------------------
    # PASO 9: Validar consistencia de INI_SIN (inicio de síntomas) vs FEC_NOT
    # Se detectó un patrón sistemático de años mal tecleados (ej. INI_SIN en
    # 2004 para un caso notificado en 2024 con el mismo día/mes). Un valor
    # de INI_SIN posterior a FEC_NOT, o más de UMBRAL_DIAS_INI_SIN días
    # antes, se considera un error de captura: se anula (NaT) y se marca en
    # ini_sin_corregido para que quede trazabilidad de qué se tocó.
    # ------------------------------------------------------------------
    if "INI_SIN" in df.columns:
        mask_ini_sin_invalido = df["INI_SIN"].notna() & (
            (df["INI_SIN"] > df["FEC_NOT"]) |
            ((df["FEC_NOT"] - df["INI_SIN"]).dt.days > UMBRAL_DIAS_INI_SIN)
        )
        n_ini_sin_invalido = int(mask_ini_sin_invalido.sum())
        df["ini_sin_corregido"] = mask_ini_sin_invalido
        df.loc[mask_ini_sin_invalido, "INI_SIN"] = pd.NaT
    else:
        n_ini_sin_invalido = 0
        df["ini_sin_corregido"] = False

    if verbose:
        print(f"[9/14] INI_SIN validado contra FEC_NOT | "
              f"Valores implausibles anulados: {n_ini_sin_invalido}")

    # ------------------------------------------------------------------
    # PASO 10: Variables temporales derivadas
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
        print("[10/14] Variables temporales derivadas: mes, trimestre, semestre")

    # ------------------------------------------------------------------
    # PASO 11: Variables demográficas derivadas
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

    # Flag de auditoría: EDAD reportada no coincide con la edad calculada a
    # partir de FECHA_NTO (más de UMBRAL_DIFERENCIA_EDAD_ANIOS años de
    # diferencia). No se corrige automáticamente porque no se puede saber
    # cuál de los dos campos tiene el error sin volver a la fuente original;
    # se deja disponible para revisión manual o exclusión opcional en
    # análisis posteriores.
    if "FECHA_NTO" in df.columns:
        edad_calculada = (df["FEC_NOT"] - df["FECHA_NTO"]).dt.days / 365.25
        df["flag_edad_inconsistente"] = (
            df["FECHA_NTO"].notna() &
            ((edad_calculada - df["EDAD"]).abs() > UMBRAL_DIFERENCIA_EDAD_ANIOS)
        )
    else:
        df["flag_edad_inconsistente"] = False
    n_edad_inconsistente = int(df["flag_edad_inconsistente"].sum())

    if verbose:
        print("[11/14] Variables demográficas derivadas: grupo_edad, flags "
              "menor/rural/edad_inconsistente "
              f"({n_edad_inconsistente} registros marcados)")

    # ------------------------------------------------------------------
    # PASO 12: Normalizar campos de texto con relleno de ancho fijo
    # nom_grupo, estrato y sem_ges venían con relleno de espacios hasta una
    # longitud fija (ya recortado en el Paso 2). Aquí se completa el
    # tratamiento específico de cada campo: nom_grupo se deja como texto
    # categórico con valor explícito para "sin dato"; estrato y sem_ges se
    # convierten a numérico, ya que representan escalas/conteos, no texto.
    # ------------------------------------------------------------------
    if "nom_grupo" in df.columns:
        df["nom_grupo"] = df["nom_grupo"].replace("", "Sin especificar")
        df["nom_grupo"] = df["nom_grupo"].fillna("Sin especificar")

    if "estrato" in df.columns:
        df["estrato"] = pd.to_numeric(
            df["estrato"].replace("", np.nan), errors="coerce"
        ).astype("Int64")

    if "sem_ges" in df.columns:
        df["sem_ges"] = pd.to_numeric(
            df["sem_ges"].replace("", np.nan), errors="coerce"
        ).astype("Int64")

    if verbose:
        print("[12/14] Campos nom_grupo, estrato y sem_ges normalizados "
              "(texto → categoría/numérico explícito)")

    # ------------------------------------------------------------------
    # PASO 13: Verificación final de columnas vacías remanentes
    # Auditoría de seguridad: confirma que ninguna columna quedó 100%
    # nula después de todos los filtros y transformaciones anteriores.
    # Si aparece alguna (por ejemplo, al cargar un Excel nuevo distinto
    # al de referencia), se reporta para que sea revisada.
    # ------------------------------------------------------------------
    columnas_vacias_remanentes = [c for c in df.columns if df[c].isna().all()]
    if verbose:
        if columnas_vacias_remanentes:
            print(f"[13/14] ⚠ Columnas 100% vacías detectadas tras la limpieza: "
                  f"{columnas_vacias_remanentes}")
        else:
            print("[13/14] Verificación de columnas vacías: ninguna pendiente ✓")

    # ------------------------------------------------------------------
    # PASO 14: Reporte final
    # ------------------------------------------------------------------
    filas_finales = len(df)
    if verbose:
        print(f"[14/14] Dataset limpio listo")
        print(f"\n  Filas iniciales          : {filas_iniciales:,}")
        print(f"  Filas excluidas          : {n_excluidos:,} "
              f"(EXTERIOR + ANO fuera de {ANIO_ANALISIS})")
        print(f"  Filas finales            : {filas_finales:,}")
        print(f"  Columnas finales         : {df.shape[1]}")
        print(f"  Departamentos únicos     : {df['Departamento_ocurrencia'].nunique()}")
        print(f"  Municipios únicos        : {df['cod_dane_mun'].nunique()}")
        print(f"  INI_SIN corregidos       : {n_ini_sin_invalido:,}")
        print(f"  EDAD marcada inconsistente: {n_edad_inconsistente:,}")
        print(f"  Columnas nuevas          : cod_dane_mun, mes, mes_nombre, "
              f"trimestre, semestre, grupo_edad, sexo_nombre, area_nombre, "
              f"hospitalizado, es_menor, es_adolescente, es_rural, "
              f"ini_sin_corregido, flag_edad_inconsistente")
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
