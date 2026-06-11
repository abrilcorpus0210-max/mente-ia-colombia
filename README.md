# MENTE-IA Colombia
## Sistema de Vigilancia Epidemiológica en Salud Mental
### Análisis de Intentos de Suicidio · SIVIGILA 2024

> Proyecto final · Bootcamp Talento Tech – Inteligencia Artificial  
> Evento SIVIGILA: 356 | Año: 2024 | Metodología: CRISP-DM

---

## Descripción

Plataforma de inteligencia de datos que analiza los intentos de suicidio
registrados por el SIVIGILA en Colombia durante 2024. Combina análisis
exploratorio, modelos de IA (Árbol de Decisión, Random Forest, K-Means)
y un sistema de priorización territorial (IPI) para apoyar la toma de
decisiones en salud pública.

**Enfoque:** preventivo, territorial, ético y orientado a políticas públicas.

---

## Estructura del proyecto

```
proyecto_sivigila_2024/
├── data/
│   ├── raw/
│   │   ├── suicidios_2024.xlsx          ← archivo original SIVIGILA
│   │   └── poblacion_dane_2024.csv      ← generado automáticamente
│   └── processed/
│       ├── casos_limpios.csv
│       └── municipios_riesgo.csv
├── src/
│   ├── utils.py          ← paleta, constantes, rutas
│   ├── carga.py          ← lectura y perfilado
│   ├── limpieza.py       ← pipeline de limpieza CRISP-DM
│   ├── eda.py            ← tablas resumen y análisis descriptivo
│   ├── visualizaciones.py← gráficas estáticas (PNG)
│   ├── ipi.py            ← Índice de Prioridad de Intervención
│   └── modelo.py         ← Árbol, Random Forest, K-Means
├── dashboard/
│   └── app.py            ← Dashboard Streamlit
├── outputs/
│   ├── graficas/         ← PNGs generados
│   └── reportes/         ← CSVs de ranking y clusters
├── requirements.txt
└── README.md
```

---

## Instalación en VS Code

### 1. Clonar o descomprimir el proyecto
```bash
cd C:\Users\TuUsuario\Documents
# descomprimir proyecto_sivigila_2024.zip aquí
cd proyecto_sivigila_2024
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## Ejecución del pipeline completo

Ejecutar en este orden desde la raíz del proyecto:

```bash
# 1. Carga y perfilado
python -m src.carga

# 2. Limpieza → genera data/processed/casos_limpios.csv
python -m src.limpieza

# 3. EDA (tablas en consola)
python -m src.eda

# 4. Visualizaciones → genera outputs/graficas/*.png
python -m src.visualizaciones

# 5. IPI → genera data/processed/municipios_riesgo.csv
python -m src.ipi

# 6. Modelos IA → genera métricas y gráficas de modelos
python -m src.modelo

# 7. Dashboard
streamlit run dashboard/app.py
```

El dashboard abre automáticamente en http://localhost:8501

---

## Índice de Prioridad de Intervención (IPI)

| Variable                 | Peso | Justificación |
|--------------------------|------|---------------|
| Tasa por 100k hab.       | 35%  | Carga epidemiológica relativa (equidad entre municipios) |
| % Adolescentes (12–17)   | 25%  | Grupo de mayor crecimiento según evidencia internacional |
| Tendencia H2 vs H1       | 20%  | Velocidad de crecimiento → urgencia de intervención |
| % Hospitalizados         | 10%  | Proxy de severidad clínica |
| % Antec. psiquiátrico    | 10%  | Factor de riesgo establecido (GP_PSIQUIA) |

Escala: 0–100. Clasificación: Baja (0–39), Media (40–59), Alta (60–79), Crítica (80–100).

---

## Modelos de IA

| Modelo           | Tipo           | Uso principal |
|------------------|----------------|---------------|
| Árbol de Decisión| Supervisado    | Reglas interpretables para salud pública |
| Random Forest    | Supervisado    | Mejor rendimiento + feature importance |
| K-Means          | No supervisado | Perfiles territoriales sin variable objetivo |

**Unidad de análisis:** municipio (una fila = un municipio con sus indicadores agregados).  
**Variable objetivo:** `nivel_prioridad` (Baja / Media / Alta / Crítica).  
**Train/Test split:** 80/20 estratificado.

---

## Datos y ética

- **Fuente:** SIVIGILA – Ministerio de Salud y Protección Social de Colombia.
- **Período:** 2024 | **Evento:** 356 (Intento de suicidio).
- Este análisis es de naturaleza **epidemiológica y preventiva**.
- No se realizan diagnósticos clínicos ni se identifican individuos.
- Todas las recomendaciones son de carácter **poblacional y territorial**.
- El tema se aborda con enfoque ético y de salud pública.

---

## Equipo

Proyecto desarrollado en el marco del Bootcamp Talento Tech – UTB  
Programa: Inteligencia Artificial · Nivel Explorador
