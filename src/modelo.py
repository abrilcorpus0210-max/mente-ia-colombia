# =============================================================================
# modelo.py
# Fase 5 CRISP-DM: Modelado – Inteligencia Artificial.
#
# Tres modelos complementarios sobre la tabla municipal:
#
# A. Árbol de Decisión (interpretabilidad máxima)
#    → Genera reglas legibles: "Si tasa_x100k > X y pct_adolescentes > Y → Crítica"
#    → Ideal para sustentar ante un equipo de salud pública.
#
# B. Random Forest (mejor rendimiento predictivo)
#    → Ensemble de árboles con mayor robustez.
#    → Feature importance para identificar las variables más explicativas del IPI.
#
# C. K-Means (agrupamiento no supervisado)
#    → Agrupa municipios por perfil epidemiológico sin depender del IPI.
#    → Complementa el análisis prescriptivo con perfiles territoriales.
#
# Variable objetivo: nivel_prioridad (Baja / Media / Alta / Crítica)
# Unidad de análisis: municipio (una fila por municipio)
#
# Uso:
#   python -m src.modelo
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import RUTAS, PALETA, COLORES_PRIORIDAD


# -----------------------------------------------------------------------------
# FEATURES USADOS EN EL MODELO
# Solo variables que existen en el dataset municipal y son interpretables.
# Se excluyen las columnas normalizadas (_norm) y el IPI para evitar fuga
# de datos (data leakage): el IPI fue construido con estas mismas variables.
# -----------------------------------------------------------------------------

FEATURES = [
    "tasa_x100k",
    "pct_adolescentes",
    "tendencia_H2_H1",
    "pct_hospit",
    "pct_psiquia",
    "total_casos",
    "pct_menores",
    "pct_mujeres",
    "pct_rural",
]

VARIABLE_OBJETIVO = "nivel_prioridad"

# Orden de clases para matrices y reportes (orden de severidad, NO alfabético).
# IMPORTANTE: este orden es válido para classification_report()/confusion_matrix()
# porque su parámetro `labels=` solo controla qué filas/columnas mostrar y en qué
# orden, sin reasignar datos. NO es válido para plot_tree(class_names=...), que
# exige el orden alfabético real de modelo.classes_ (ver grafica_arbol_decision).
ORDEN_CLASES = ["Baja", "Media", "Alta", "Crítica"]


# -----------------------------------------------------------------------------
# PREPARACIÓN DE DATOS
# -----------------------------------------------------------------------------

def preparar_datos(mun: pd.DataFrame):
    """
    Filtra municipios con suficientes casos (>=5) para robustez estadística,
    codifica la variable objetivo y retorna X, y listos para modelar.
    """
    # Solo municipios con al menos 5 casos para evitar ruido estadístico
    df_modelo = mun[mun["total_casos"] >= 5].copy()

    print(f"  Municipios usados en el modelo: {len(df_modelo)}")
    print(f"  Distribución de clases:")
    print(df_modelo[VARIABLE_OBJETIVO].value_counts().to_string())

    X = df_modelo[FEATURES].fillna(0)
    y = df_modelo[VARIABLE_OBJETIVO]

    return X, y, df_modelo


# -----------------------------------------------------------------------------
# A. ÁRBOL DE DECISIÓN
# -----------------------------------------------------------------------------

def entrenar_arbol(X_train, X_test, y_train, y_test,
                   verbose: bool = True) -> DecisionTreeClassifier:
    """
    Entrena un Árbol de Decisión con profundidad máxima 5 para mantener
    la interpretabilidad. Profundidades mayores dificultan la explicación oral.
    """
    modelo = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",   # compensa el desbalance entre clases
        random_state=42,
        criterion="gini"
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    if verbose:
        print("\n" + "─" * 55)
        print("  ÁRBOL DE DECISIÓN")
        print("─" * 55)
        print(f"  Accuracy (test)    : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Profundidad real   : {modelo.get_depth()}")
        print(f"  Hojas              : {modelo.get_n_leaves()}")
        print("\n  Reporte de clasificación:")
        print(classification_report(
            y_test, y_pred,
            labels=ORDEN_CLASES,
            zero_division=0
        ))
        # Primeras reglas del árbol (para la sustentación oral)
        print("  Primeras reglas del árbol (profundidad 2):")
        reglas = export_text(modelo, feature_names=FEATURES, max_depth=2)
        print(reglas)

    return modelo


def grafica_arbol_decision(modelo: DecisionTreeClassifier) -> plt.Figure:
    """
    Diagrama visual del árbol de decisión.
    """
    import traceback
    import warnings
    import streamlit as st
    warnings.filterwarnings('ignore')
    
    st.write("🔧🔧🔧 grafica_arbol_decision() INICIADA")  # DEBUG
    
    ruta_png = os.path.join(RUTAS["graficas"], "15_arbol_visual.png")
    ruta_error = os.path.join(RUTAS["graficas"], "15_arbol_visual_ERROR.txt")
    
    st.write(f"🔧 Ruta: {ruta_png}")
    st.write(f"🔧 Carpeta graficas existe: {os.path.exists(RUTAS['graficas'])}")
    
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        
        n_hojas = modelo.get_n_leaves()
        profundidad = modelo.get_depth()
        
        st.write(f"🔧 Árbol: {n_hojas} hojas, profundidad {profundidad}")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Intentar plot_tree
        st.write("🔧 Ejecutando plot_tree...")
        plot_tree(
            modelo,
            feature_names=FEATURES,
            class_names=list(modelo.classes_),
            filled=True,
            rounded=True,
            proportion=True,
            impurity=False,
            fontsize=9,
            ax=ax
        )
        st.write("🔧 plot_tree OK")
        
        ax.set_title(f"Árbol de Decisión — {n_hojas} hojas, profundidad {profundidad}")
        
        # Guardar
        st.write(f"🔧 Guardando PNG...")
        fig.savefig(ruta_png, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        # Verificar
        existe = os.path.exists(ruta_png)
        tamano = os.path.getsize(ruta_png) if existe else 0
        st.write(f"🔧 PNG existe: {existe}, tamaño: {tamano}")
        
        if existe and tamano > 100:
            st.write("✅ Árbol generado correctamente")
            if os.path.exists(ruta_error):
                os.remove(ruta_error)
            return fig
        else:
            raise RuntimeError(f"PNG vacío o no creado: {ruta_png}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        st.error(f"🔧 ERROR: {error_msg[:200]}...")  # Mostrar primeros 200 chars
        
        # Escribir error
        try:
            with open(ruta_error, "w", encoding="utf-8") as f:
                f.write(error_msg)
            st.write("🔧 Error escrito en archivo")
        except Exception as e2:
            st.write(f"🔧 No se pudo escribir error: {e2}")
        
        return None

# -----------------------------------------------------------------------------
# B. RANDOM FOREST
# -----------------------------------------------------------------------------

def entrenar_random_forest(X_train, X_test, y_train, y_test,
                            verbose: bool = True) -> RandomForestClassifier:
    """
    Entrena un Random Forest con 100 árboles.
    Retorna el modelo con las métricas impresas.
    """
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1          # usa todos los núcleos disponibles
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    # Validación cruzada (5 folds) para una estimación más robusta.
    # Se ajusta el número de folds según el tamaño de la clase más
    # pequeña en train, para evitar errores cuando hay pocos municipios
    # o niveles de prioridad con muy pocos casos (ej. tras combinar
    # datos nuevos con un perfil de riesgo poco común).
    n_clase_min_train = y_train.value_counts().min() if len(y_train) > 0 else 0
    cv_folds = min(5, n_clase_min_train) if n_clase_min_train >= 2 else 0

    if cv_folds >= 2:
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv_folds,
                                     scoring="accuracy")
    else:
        # Conjunto demasiado pequeño o desbalanceado para validación
        # cruzada confiable; se omite y se reporta solo el accuracy de test.
        cv_scores = None

    if verbose:
        print("\n" + "─" * 55)
        print("  RANDOM FOREST")
        print("─" * 55)
        print(f"  Accuracy (test)         : {acc:.4f}  ({acc*100:.1f}%)")
        if cv_scores is not None:
            print(f"  Accuracy CV (media)     : {cv_scores.mean():.4f}  ({cv_scores.mean()*100:.1f}%)")
            print(f"  Accuracy CV (std)       : ± {cv_scores.std():.4f}")
        else:
            print(f"  Accuracy CV             : no disponible (muestra insuficiente)")
        print("\n  Reporte de clasificación:")
        print(classification_report(
            y_test, y_pred,
            labels=ORDEN_CLASES,
            zero_division=0
        ))
        # Feature importance
        fi = pd.Series(modelo.feature_importances_, index=FEATURES)
        fi = fi.sort_values(ascending=False)
        print("  Feature Importance (Gini):")
        for feat, imp in fi.items():
            barra = "█" * int(imp * 40)
            print(f"    {feat:<22s} {imp:.4f}  {barra}")

    return modelo


def grafica_feature_importance(modelo: RandomForestClassifier) -> plt.Figure:
    """Gráfica de barras horizontales con la importancia de cada variable."""
    fi = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values()

    etiquetas = {
        "tasa_x100k":       "Tasa por 100k hab.",
        "pct_adolescentes": "% Adolescentes (12–17)",
        "tendencia_H2_H1":  "Tendencia H2 vs H1",
        "pct_hospit":       "% Hospitalizados",
        "pct_psiquia":      "% Antec. psiquiátrico",
        "total_casos":      "Total de casos",
        "pct_menores":      "% Menores de 18",
        "pct_mujeres":      "% Femenino",
        "pct_rural":        "% Rural",
    }
    fi.index = [etiquetas.get(i, i) for i in fi.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    colores_bar = [PALETA["primario"] if v > fi.mean() else PALETA["secundario"]
                   for v in fi.values]
    bars = ax.barh(fi.index, fi.values, color=colores_bar, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}",
                va="center", fontsize=8, color=PALETA["neutro"])
    ax.set_title("Importancia de variables – Random Forest\n"
                 "Clasificación de nivel de prioridad municipal")
    ax.set_xlabel("Importancia Gini (normalizada)")
    ax.axvline(fi.mean(), color=PALETA["advertencia"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Media")
    ax.legend()
    fig.tight_layout()
    ruta = os.path.join(RUTAS["graficas"], "11_feature_importance.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=PALETA["fondo"])
    print(f"  ✓ Gráfica guardada: {ruta}")
    return fig


def grafica_matriz_confusion(modelo, X_test, y_test, titulo: str) -> plt.Figure:
    """Matriz de confusión normalizada."""
    y_pred = modelo.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=ORDEN_CLASES, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proporción")

    ax.set_xticks(range(len(ORDEN_CLASES)))
    ax.set_yticks(range(len(ORDEN_CLASES)))
    ax.set_xticklabels(ORDEN_CLASES, rotation=30)
    ax.set_yticklabels(ORDEN_CLASES)

    for i in range(len(ORDEN_CLASES)):
        for j in range(len(ORDEN_CLASES)):
            color = "white" if cm[i, j] > 0.5 else PALETA["neutro"]
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión (normalizada)\n{titulo}")
    fig.tight_layout()

    nombre_archivo = "12_confusion_rf.png" if "Forest" in titulo else "13_confusion_dt.png"
    ruta = os.path.join(RUTAS["graficas"], nombre_archivo)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=PALETA["fondo"])
    print(f"  ✓ Gráfica guardada: {ruta}")
    return fig


# -----------------------------------------------------------------------------
# C. K-MEANS
# -----------------------------------------------------------------------------

def entrenar_kmeans(mun: pd.DataFrame, n_clusters: int = 4,
                    verbose: bool = True) -> tuple:
    """
    Agrupa todos los municipios (sin filtro de mínimo de casos) en
    n_clusters perfiles epidemiológicos usando K-Means.

    Retorna (modelo_kmeans, df_con_cluster, scaler).
    """
    df_km = mun[FEATURES].fillna(0).copy()

    # Escalado estándar (Z-score) obligatorio para K-Means
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(df_km)

    # Método del codo: evaluar inercia para k = 2..8
    inercias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    # Entrenar con k óptimo (por defecto 4, alineado con los 4 niveles del IPI)
    modelo_km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    modelo_km.fit(X_scaled)

    mun_km = mun.copy()
    mun_km["cluster_kmeans"] = modelo_km.labels_

    # Caracterizar cada cluster por su perfil promedio
    perfil = (
        mun_km.groupby("cluster_kmeans")[FEATURES]
        .mean()
        .round(4)
    )

    # Etiquetar clusters por tasa promedio (de menor a mayor = Bajo → Alto)
    orden_clusters = perfil["tasa_x100k"].rank().astype(int) - 1
    etiquetas_cluster = {
        0: "Cluster A – Perfil Rural/Bajo",
        1: "Cluster B – Perfil Urbano Medio",
        2: "Cluster C – Metrópoli Única (Bogotá)",
        3: "Cluster D – Ciudades Grandes/Medianas",
    }
    mun_km["perfil_cluster"] = mun_km["cluster_kmeans"].map(
        {v: etiquetas_cluster.get(k, f"Cluster {k}")
         for k, v in orden_clusters.items()}
    )

    if verbose:
        print("\n" + "─" * 55)
        print("  K-MEANS CLUSTERING")
        print("─" * 55)
        print(f"  Número de clusters: {n_clusters}")
        print(f"  Inercia final     : {modelo_km.inertia_:.2f}")
        print(f"\n  Municipios por cluster:")
        print(mun_km["perfil_cluster"].value_counts().to_string())
        print(f"\n  Perfil promedio por cluster:")
        print(perfil.to_string())

    # Gráfica del codo
    _grafica_codo(inercias, n_clusters)

    return modelo_km, mun_km, scaler


def _grafica_codo(inercias: list, k_optimo: int) -> None:
    """Curva del codo para justificar el número de clusters."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = range(2, 9)
    ax.plot(ks, inercias, marker="o", color=PALETA["primario"], linewidth=2)
    ax.axvline(k_optimo, color=PALETA["advertencia"], linestyle="--",
               linewidth=1.5, label=f"k = {k_optimo} (elegido)")
    ax.set_title("Método del codo – K-Means\nSelección del número óptimo de clusters")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (suma de distancias cuadradas)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(RUTAS["graficas"], "14_kmeans_codo.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=PALETA["fondo"])
    print(f"  ✓ Gráfica guardada: {ruta}")


# -----------------------------------------------------------------------------
# PIPELINE COMPLETO DE MODELADO
# -----------------------------------------------------------------------------

def pipeline_modelos(mun: pd.DataFrame = None, verbose: bool = True) -> dict:
    import streamlit as st  # Añadir esto
    
    st.write("🔧🔧🔧 pipeline_modelos() INICIADO")  # DEBUG VISIBLE
    
    if mun is None:
        st.write("🔧 Cargando mun desde CSV...")  # DEBUG
        mun = pd.read_csv(RUTAS["procesado_mun"], dtype={"cod_dane_mun": str})

    # Preparar datos
    st.write("🔧 Preparando datos...")  # DEBUG
    X, y, df_modelo = preparar_datos(mun)
    st.write(f"🔧 X shape: {X.shape}, y shape: {y.shape}")  # DEBUG

    # Split
    st.write("🔧 Haciendo train_test_split...")  # DEBUG
    # ... (todo el código de split sin cambios) ...
    
    st.write(f"🔧 Train: {len(X_train)}, Test: {len(X_test)}")  # DEBUG

    # A. Árbol de Decisión
    st.write("🔧 Entrenando árbol...")  # DEBUG
    modelo_dt = entrenar_arbol(X_train, X_test, y_train, y_test, verbose=False)
    st.write(f"🔧 Árbol entrenado. ¿None?: {modelo_dt is None}")  # DEBUG
    
    st.write("🔧 Llamando grafica_arbol_decision()...")  # DEBUG CRÍTICO
    fig_arbol = grafica_arbol_decision(modelo_dt)
    st.write(f"🔧 grafica_arbol_decision() retornó: {fig_arbol is not None}")  # DEBUG
    
    st.write("🔧 Llamando grafica_matriz_confusion()...")  # DEBUG
    grafica_matriz_confusion(modelo_dt, X_test, y_test, "Árbol de Decisión")

    # ... resto sin cambios ...
    
    st.write("🔧🔧🔧 pipeline_modelos() TERMINADO")  # DEBUG
    return {...}


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mun = pd.read_csv(RUTAS["procesado_mun"], dtype={"cod_dane_mun": str})
    resultados = pipeline_modelos(mun, verbose=True)
