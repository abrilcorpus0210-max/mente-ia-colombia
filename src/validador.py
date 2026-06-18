# =============================================================================
# validador.py
# Validación de archivos Excel nuevos antes de integrarlos al pipeline.
#
# Permite que cualquier persona con acceso al dashboard suba un archivo
# de actualización (mismo formato de SIVIGILA) y el sistema verifica que
# tenga las columnas y formatos correctos ANTES de reemplazar los datos
# del proyecto. Si algo falla, se reporta en lenguaje claro, sin
# tecnicismos, señalando exactamente qué columna o fila tiene el problema.
#
# Uso desde app.py:
#   from src.validador import validar_excel, ResultadoValidacion
#   resultado = validar_excel(archivo_subido)
#   if resultado.es_valido:
#       ...
#   else:
#       mostrar resultado.errores
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# CONTRATO DE FORMATO
# Columnas obligatorias: sin estas, el pipeline (limpieza -> ipi -> modelo)
# no puede ejecutarse. Columnas opcionales: mejoran el análisis si existen,
# pero su ausencia no bloquea la carga.
# -----------------------------------------------------------------------------

COLUMNAS_OBLIGATORIAS = [
    "CONSECUTIVE", "COD_EVE", "Nombre_evento", "ANO", "SEMANA", "FEC_NOT",
    "EDAD", "UNI_MED", "SEXO", "AREA", "PAC_HOS",
    "COD_DPTO_O", "COD_MUN_O", "Departamento_ocurrencia", "Municipio_ocurrencia",
]

COLUMNAS_OPCIONALES = [
    "GP_PSIQUIA", "GP_MIGRANT", "GP_CARCELA", "GP_GESTAN", "GP_INDIGEN",
    "GP_POBICFB", "GP_DISCAPA", "GP_DESPLAZ", "GP_VIC_VIO", "GP_OTROS",
    "INI_SIN", "FEC_CON", "FEC_HOS", "PER_ETN", "nom_grupo",
]

# Año esperado para el análisis. Si el archivo nuevo corresponde a otro
# año, se actualiza esta constante (o se parametriza en una fase futura).
ANIO_ANALISIS_ESPERADO = 2024

# Valores válidos para columnas categóricas estrictas
VALORES_VALIDOS_SEXO = {"F", "M"}
VALORES_VALIDOS_AREA = {1, 2, 3}
VALORES_VALIDOS_PAC_HOS = {1, 2}

# Tamaño máximo razonable de archivo (filas) para evitar cargas accidentales
# de archivos corruptos o de formato distinto con millones de filas vacías.
MAX_FILAS_RAZONABLE = 500_000


@dataclass
class ResultadoValidacion:
    """Resultado estructurado de la validación de un archivo Excel."""
    es_valido: bool
    errores: list = field(default_factory=list)
    advertencias: list = field(default_factory=list)
    resumen: dict = field(default_factory=dict)
    df: pd.DataFrame = None


def _error(lista, mensaje):
    lista.append(mensaje)


def _advertencia(lista, mensaje):
    lista.append(mensaje)


# -----------------------------------------------------------------------------
# VALIDACIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def validar_excel(archivo, hoja: str = "Hoja1") -> ResultadoValidacion:
    """
    Valida un archivo Excel subido contra el contrato de formato esperado.

    Parámetros
    ----------
    archivo : file-like object (lo que entrega st.file_uploader) o ruta str.
    hoja : nombre de la hoja a leer. Por defecto "Hoja1" (igual al SIVIGILA original).

    Retorna
    -------
    ResultadoValidacion con es_valido, errores, advertencias, resumen y
    el DataFrame leído (si pudo abrirse) para reutilizar en el pipeline.
    """
    errores = []
    advertencias = []
    resumen = {}

    # ---- Paso 1: intentar abrir el archivo ----
    try:
        df = pd.read_excel(
            archivo,
            sheet_name=hoja,
            dtype={
                "COD_DPTO_O": str, "COD_MUN_O": str,
                "COD_DPTO_R": str, "COD_MUN_R": str,
                "COD_DPTO_N": str, "COD_MUN_N": str,
            },
        )
    except ValueError as e:
        if "Worksheet" in str(e) or "sheet" in str(e).lower():
            _error(errores, f"No se encontró una hoja llamada '{hoja}' en el archivo. "
                              f"Verifica que el archivo tenga una hoja con ese nombre exacto.")
        else:
            _error(errores, f"El archivo no se pudo leer como Excel válido. "
                              f"Detalle técnico: {e}")
        return ResultadoValidacion(es_valido=False, errores=errores)
    except Exception as e:
        _error(errores, f"No fue posible abrir el archivo. Verifica que sea un "
                          f"archivo .xlsx válido y no esté dañado o protegido con "
                          f"contraseña. Detalle técnico: {type(e).__name__}: {e}")
        return ResultadoValidacion(es_valido=False, errores=errores)

    df.columns = df.columns.str.strip()
    resumen["filas_totales"] = len(df)
    resumen["columnas_totales"] = df.shape[1]

    # ---- Paso 2: tamaño razonable ----
    if len(df) == 0:
        _error(errores, "El archivo no contiene ninguna fila de datos.")
        return ResultadoValidacion(es_valido=False, errores=errores, resumen=resumen)

    if len(df) > MAX_FILAS_RAZONABLE:
        _advertencia(advertencias, f"El archivo tiene {len(df):,} filas, un volumen "
                       f"inusualmente alto. Verifica que sea el archivo correcto.")

    # ---- Paso 3: columnas obligatorias presentes ----
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        _error(errores, "Faltan columnas obligatorias en el archivo: "
                          + ", ".join(faltantes) + ". "
                          "Revisa que el archivo tenga exactamente estos nombres de columna, "
                          "respetando mayúsculas y guiones bajos.")
        # Si faltan columnas clave, no tiene sentido seguir validando contenido.
        return ResultadoValidacion(es_valido=False, errores=errores,
                                    advertencias=advertencias, resumen=resumen)

    # ---- Paso 4: columnas 100% vacías ----
    columnas_vacias = [c for c in df.columns if df[c].isna().all()]
    columnas_vacias_obligatorias = [c for c in columnas_vacias if c in COLUMNAS_OBLIGATORIAS]
    if columnas_vacias_obligatorias:
        _error(errores, "Las siguientes columnas obligatorias están completamente "
                          "vacías: " + ", ".join(columnas_vacias_obligatorias) + ". "
                          "El archivo no puede procesarse sin estos datos.")
    columnas_vacias_otras = [c for c in columnas_vacias if c not in COLUMNAS_OBLIGATORIAS]
    if columnas_vacias_otras:
        _advertencia(advertencias, f"Se detectaron {len(columnas_vacias_otras)} columnas "
                       f"completamente vacías que serán descartadas automáticamente: "
                       + ", ".join(columnas_vacias_otras[:10])
                       + (" (y más...)" if len(columnas_vacias_otras) > 10 else ""))

    # ---- Paso 5: ANO coincide con el año de análisis ----
    if "ANO" in df.columns:
        anios_presentes = set(pd.to_numeric(df["ANO"], errors="coerce").dropna().astype(int).unique())
        anios_inesperados = anios_presentes - {ANIO_ANALISIS_ESPERADO}
        if anios_inesperados:
            _advertencia(advertencias, f"El archivo contiene registros con año(s) distinto(s) "
                           f"a {ANIO_ANALISIS_ESPERADO}: {sorted(anios_inesperados)}. "
                           f"Estos registros serán excluidos automáticamente del análisis.")

    # ---- Paso 6: fechas de notificación dentro del año esperado ----
    if "FEC_NOT" in df.columns:
        fechas = pd.to_datetime(df["FEC_NOT"], errors="coerce")
        n_fecha_invalida_formato = fechas.isna().sum() - df["FEC_NOT"].isna().sum()
        if n_fecha_invalida_formato > 0:
            _advertencia(advertencias, f"{n_fecha_invalida_formato} registros tienen una fecha "
                           f"de notificación (FEC_NOT) en un formato no reconocible. "
                           f"Verifica que las fechas estén en formato AAAA-MM-DD.")
        fuera_de_anio = fechas.dt.year.dropna()
        n_fuera = int((fuera_de_anio != ANIO_ANALISIS_ESPERADO).sum())
        if n_fuera > 0:
            _advertencia(advertencias, f"{n_fuera} registros tienen una fecha de notificación "
                           f"(FEC_NOT) fuera del año {ANIO_ANALISIS_ESPERADO}. "
                           f"Estos registros serán excluidos automáticamente.")
        resumen["fechas_fuera_de_anio"] = n_fuera

    # ---- Paso 7: SEXO solo admite F o M ----
    if "SEXO" in df.columns:
        valores_sexo = set(df["SEXO"].dropna().astype(str).str.strip().str.upper().unique())
        invalidos_sexo = valores_sexo - VALORES_VALIDOS_SEXO
        if invalidos_sexo:
            n_filas_invalidas = df["SEXO"].astype(str).str.strip().str.upper().isin(invalidos_sexo).sum()
            _error(errores, f"La columna SEXO contiene valores no permitidos: "
                              f"{sorted(invalidos_sexo)} ({n_filas_invalidas} filas). "
                              f"Solo se aceptan los valores 'F' o 'M'.")

    # ---- Paso 8: AREA solo admite 1, 2 o 3 ----
    if "AREA" in df.columns:
        valores_area = set(pd.to_numeric(df["AREA"], errors="coerce").dropna().unique())
        invalidos_area = valores_area - VALORES_VALIDOS_AREA
        if invalidos_area:
            _error(errores, f"La columna AREA contiene valores no permitidos: "
                              f"{sorted(invalidos_area)}. Solo se aceptan los códigos "
                              f"1 (Cabecera municipal), 2 (Centro poblado) o 3 (Rural disperso).")

    # ---- Paso 9: PAC_HOS solo admite 1 o 2 ----
    if "PAC_HOS" in df.columns:
        valores_hos = set(pd.to_numeric(df["PAC_HOS"], errors="coerce").dropna().unique())
        invalidos_hos = valores_hos - VALORES_VALIDOS_PAC_HOS
        if invalidos_hos:
            _error(errores, f"La columna PAC_HOS contiene valores no permitidos: "
                              f"{sorted(invalidos_hos)}. Solo se aceptan los códigos "
                              f"1 (Hospitalizado) o 2 (No hospitalizado).")

    # ---- Paso 10: códigos de departamento/municipio con formato correcto ----
    if "COD_DPTO_O" in df.columns:
        # Si Excel convirtió el código a número y se perdió el cero a la
        # izquierda, los códigos de un solo dígito (ej. "5" en vez de "05")
        # son la señal de alerta.
        codigos_cortos = df["COD_DPTO_O"].astype(str).str.strip()
        n_cortos = (codigos_cortos.str.len() < 2).sum()
        if n_cortos > 0:
            _advertencia(advertencias, f"{n_cortos} registros tienen un código de "
                           f"departamento (COD_DPTO_O) con menos de 2 dígitos. Es posible "
                           f"que Excel haya eliminado ceros a la izquierda (ej. '5' en vez "
                           f"de '05'). El sistema intentará corregirlo automáticamente, "
                           f"pero verifica el archivo original si el resultado final no "
                           f"coincide con tus municipios.")

    if "COD_MUN_O" in df.columns:
        codigos_mun_cortos = df["COD_MUN_O"].astype(str).str.strip()
        n_mun_cortos = (codigos_mun_cortos.str.len() < 3).sum()
        if n_mun_cortos > 0:
            _advertencia(advertencias, f"{n_mun_cortos} registros tienen un código de "
                           f"municipio (COD_MUN_O) con menos de 3 dígitos. El sistema "
                           f"intentará corregirlo automáticamente con ceros a la izquierda.")

    # ---- Paso 11: EDAD en rango razonable ----
    if "EDAD" in df.columns:
        edades = pd.to_numeric(df["EDAD"], errors="coerce")
        n_invalida = edades.isna().sum() - df["EDAD"].isna().sum()
        n_fuera_rango = ((edades < 0) | (edades > 120)).sum()
        if n_invalida > 0:
            _advertencia(advertencias, f"{n_invalida} registros tienen un valor de EDAD "
                           f"no numérico y serán tratados como datos faltantes.")
        if n_fuera_rango > 0:
            _advertencia(advertencias, f"{n_fuera_rango} registros tienen una EDAD fuera "
                           f"de un rango razonable (0-120 años). Verifica esos datos en "
                           f"el archivo original.")

    # ---- Paso 12: COD_EVE corresponde al evento esperado (356) ----
    if "COD_EVE" in df.columns:
        eventos = set(pd.to_numeric(df["COD_EVE"], errors="coerce").dropna().astype(int).unique())
        if eventos != {356}:
            _advertencia(advertencias, f"El archivo contiene código(s) de evento distinto(s) "
                           f"a 356 (Intento de suicidio): {sorted(eventos)}. Verifica que "
                           f"sea el archivo correcto.")

    # ---- Paso 13: columnas opcionales presentes (informativo) ----
    opcionales_presentes = [c for c in COLUMNAS_OPCIONALES if c in df.columns]
    resumen["columnas_opcionales_detectadas"] = len(opcionales_presentes)
    resumen["columnas_opcionales_totales"] = len(COLUMNAS_OPCIONALES)

    # ---- Resultado final ----
    es_valido = len(errores) == 0
    resumen["filas_validas_estimadas"] = len(df) - resumen.get("fechas_fuera_de_anio", 0)

    return ResultadoValidacion(
        es_valido=es_valido,
        errores=errores,
        advertencias=advertencias,
        resumen=resumen,
        df=df if es_valido else None,
    )


# -----------------------------------------------------------------------------
# FORMATO LEGIBLE PARA MOSTRAR EN EL DASHBOARD
# -----------------------------------------------------------------------------

def formato_columnas_requeridas() -> str:
    """
    Retorna un texto en Markdown describiendo el formato exacto esperado,
    para mostrarlo en el dashboard como guía antes de que alguien suba un archivo.
    """
    obligatorias_md = "\n".join(f"- `{c}`" for c in COLUMNAS_OBLIGATORIAS)
    opcionales_md = "\n".join(f"- `{c}`" for c in COLUMNAS_OPCIONALES)
    return f"""
**Columnas obligatorias** (el archivo debe tenerlas todas, con estos nombres exactos):

{obligatorias_md}

**Columnas opcionales** (mejoran el análisis, pero no son indispensables):

{opcionales_md}

**Reglas de formato:**
- El archivo debe ser `.xlsx`, con una hoja llamada `Hoja1`.
- Las fechas (`FEC_NOT`, etc.) deben estar en formato `AAAA-MM-DD`.
- `SEXO` solo admite `F` o `M`.
- `AREA` solo admite `1`, `2` o `3`.
- `PAC_HOS` solo admite `1` o `2`.
- `COD_DPTO_O` debe tener 2 dígitos (ej. `05`) y `COD_MUN_O` 3 dígitos (ej. `001`).
- No debe haber columnas obligatorias completamente vacías.
"""
