# =============================================================================
# utils.py
# Constantes compartidas: paleta visual, etiquetas de dominio y funciones
# auxiliares reutilizadas por todos los módulos del proyecto.
#
# Proyecto: Análisis de Intentos de Suicidio – SIVIGILA Colombia 2024
# Bootcamp: Talento Tech – Inteligencia Artificial
# =============================================================================

# -----------------------------------------------------------------------------
# PALETA VISUAL
# Colores celadon y azul Bondi en tonos sobrios para salud pública.
# Se usan en visualizaciones.py y en el dashboard Streamlit.
# -----------------------------------------------------------------------------

PALETA = {
    "primario":    "#0D7C8F",   # Azul Bondi – elemento principal
    "secundario":  "#A8D5BA",   # Celadon – elemento de apoyo
    "acento":      "#5BA4B0",   # Azul Bondi medio
    "suave":       "#D4EDE1",   # Celadon muy claro – fondos
    "neutro":      "#4A4A4A",   # Gris oscuro – texto sobre fondo claro
    "advertencia": "#E07B54",   # Terracota suave – niveles críticos
    "peligro":     "#C0392B",   # Rojo sobrio – solo para nivel Crítico
    "fondo":       "#F7FBFA",   # Fondo general del dashboard
}

# Secuencia para gráficas con múltiples categorías
SECUENCIA_COLORES = [
    "#0D7C8F", "#A8D5BA", "#5BA4B0", "#7EBFCC",
    "#3A8FA3", "#C8E6D0", "#1A6B7A", "#85C7B4",
]

# Mapa de colores por nivel de prioridad del IPI
COLORES_PRIORIDAD = {
    "Baja":     "#A8D5BA",  # Celadon
    "Media":    "#5BA4B0",  # Azul Bondi medio
    "Alta":     "#E07B54",  # Terracota
    "Crítica":  "#C0392B",  # Rojo sobrio
}

# -----------------------------------------------------------------------------
# ETIQUETAS DE DOMINIO
# Traducen los códigos numéricos de SIVIGILA a texto legible.
# Fuente: Diccionario de datos SIVIGILA – Evento 356.
# -----------------------------------------------------------------------------

# SEXO: F = Femenino, M = Masculino
ETIQUETAS_SEXO = {
    "F": "Femenino",
    "M": "Masculino",
}

# AREA: 1 = Cabecera municipal, 2 = Centro poblado, 3 = Rural disperso
ETIQUETAS_AREA = {
    1: "Cabecera municipal",
    2: "Centro poblado",
    3: "Rural disperso",
}

# PAC_HOS: 1 = Hospitalizado, 2 = No hospitalizado
ETIQUETAS_HOSPITALIZACION = {
    1: "Hospitalizado",
    2: "No hospitalizado",
}

# GP_*: 1 = Pertenece al grupo, 2 = No pertenece
ETIQUETAS_GP = {
    1: "Sí",
    2: "No",
}

# UNI_MED: 1 = Años (toda la base usa años, verificado)
ETIQUETAS_UNI_MED = {
    1: "Años",
}

# Grupos de edad estandarizados para salud pública
# Se aplican sobre la columna EDAD (en años)
BINS_EDAD = [0, 11, 17, 25, 35, 45, 59, 150]
ETIQUETAS_EDAD = [
    "0–11",
    "12–17",
    "18–25",
    "26–35",
    "36–45",
    "46–59",
    "60+",
]

# Trimestres epidemiológicos (semana SIVIGILA 1–52)
BINS_SEMANA = [0, 13, 26, 39, 52]
ETIQUETAS_TRIMESTRE = ["T1 (Sem 1–13)", "T2 (Sem 14–26)", "T3 (Sem 27–39)", "T4 (Sem 40–52)"]

# -----------------------------------------------------------------------------
# UMBRALES DEL IPI (Índice de Prioridad de Intervención)
# Escala 0–100. Definidos en ipi.py y referenciados aquí para consistencia.
# -----------------------------------------------------------------------------

UMBRALES_IPI = {
    "Baja":    (0,  39),
    "Media":   (40, 59),
    "Alta":    (60, 79),
    "Crítica": (80, 100),
}

def clasificar_ipi(valor: float) -> str:
    """
    Clasifica un valor del IPI (0–100) en su nivel de prioridad.
    Función centralizada para que dashboard y módulos usen la misma lógica.
    """
    if valor >= 80:
        return "Crítica"
    elif valor >= 60:
        return "Alta"
    elif valor >= 40:
        return "Media"
    else:
        return "Baja"

# -----------------------------------------------------------------------------
# RUTAS DEL PROYECTO
# Rutas relativas desde la raíz del proyecto para portabilidad.
# Se importan en cada módulo con: from src.utils import RUTAS
# -----------------------------------------------------------------------------

import os

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUTAS = {
    "raw_xlsx":          os.path.join(_RAIZ, "data", "raw",       "suicidios_2024.xlsx"),
    "raw_poblacion":     os.path.join(_RAIZ, "data", "raw",       "poblacion_dane_2024.csv"),
    "raw_acumulado":     os.path.join(_RAIZ, "data", "raw",       "datos_crudos_acumulados.csv"),
    "procesado_casos":   os.path.join(_RAIZ, "data", "processed", "casos_limpios.csv"),
    "procesado_mun":     os.path.join(_RAIZ, "data", "processed", "municipios_riesgo.csv"),
    "graficas":          os.path.join(_RAIZ, "outputs", "graficas"),
    "reportes":          os.path.join(_RAIZ, "outputs", "reportes"),
}

# Referencia de pesos del IPI para el dashboard (espejo de ipi.py)
PESOS_IPI_REF = {
    "tasa_x100k":      0.35,
    "pct_adolescentes": 0.25,
    "tendencia_H2_H1": 0.20,
    "pct_hospit":      0.10,
    "pct_psiquia":     0.10,
}