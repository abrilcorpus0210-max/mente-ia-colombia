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
    Diagrama visual completo del árbol de decisión entrenado (plot_tree).

    Complementa las reglas en texto (export_text, ya impresas en consola)
    con una versión gráfica pensada para insertarse en el dashboard
    (Tab 4 → Árbol de Decisión), donde hasta ahora solo se mostraba la
    matriz de confusión sin ningún diagrama del árbol en sí.

    Notas de implementación:
    - El ancho de la figura se ajusta según el número de hojas reales
      (hasta 32 posibles con max_depth=5) para que las reglas no queden
      amontonadas ni ilegibles, sin importar cuántas hojas resulten del
      entrenamiento real.
    - `class_names` se construye desde `modelo.classes_` (orden alfabético
      real que usa sklearn internamente: Alta, Baja, Crítica, Media) y NO
      desde `ORDEN_CLASES` (orden de severidad). plot_tree() asigna las
      etiquetas de clase por posición, no por nombre: pasar ORDEN_CLASES
      aquí etiquetaría los nodos con la clase incorrecta de forma
      silenciosa (ej. un nodo "Alta" real aparecería rotulado "Baja").
    - `proportion=True` muestra porcentaje de muestras por nodo en vez de
      conteos absolutos, e `impurity=False` oculta el gini interno: ambos
      pensados para una audiencia de salud pública, no técnica.
    """
    n_hojas = modelo.get_n_leaves()
    ancho   = max(16, min(40, n_hojas * 1.8))

    fig, ax = plt.subplots(figsize=(ancho, 10))
    plot_tree(
        modelo,
        feature_names=FEATURES,
        class_names=list(modelo.classes_),  # orden real de sklearn, no ORDEN_CLASES
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        fontsize=9,
        ax=ax
    )
    ax.set_title(
        f"Árbol de Decisión completo — profundidad {modelo.get_depth()}, "
        f"{n_hojas} hojas\nClasificación de nivel de prioridad municipal",
        fontsize=13
    )
    fig.tight_layout()
    ruta = os.path.join(RUTAS["graficas"], "15_arbol_visual.png")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=PALETA.get("fondo", "white"))
    plt.close(fig)
    print(f"  ✓ Gráfica guardada: {ruta}")
    return fig


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
        0: "Cluster A – Carga Baja",
        1: "Cluster B – Carga Media",
        2: "Cluster C – Carga Alta",
        3: "Cluster D – Carga Crítica",
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
    """
    Ejecuta los tres modelos en secuencia y retorna un diccionario
    con todos los artefactos.
    """
    if mun is None:
        mun = pd.read_csv(RUTAS["procesado_mun"], dtype={"cod_dane_mun": str})

    if verbose:
        print("=" * 65)
        print("  MODELADO – INTELIGENCIA ARTIFICIAL")
        print("=" * 65)

    # Preparar datos para clasificación supervisada
    X, y, df_modelo = preparar_datos(mun)

    # ------------------------------------------------------------------
    # Protección robusta para datasets muy pequeños o con clases raras.
    # Esto puede pasar después de combinar datos nuevos (ej. al subir un
    # archivo de actualización) que agreguen un municipio con un nivel
    # de prioridad que antes no existía o que tenga muy pocos casos.
    #
    # sklearn puede fallar con "the resulting train set will be empty"
    # si test_size=0.20 produce 0 muestras de entrenamiento o de prueba
    # para alguna clase, incluso sin usar stratify. Para evitarlo:
    #   1. Si el total de municipios es muy pequeño, se reduce test_size
    #      dinámicamente para garantizar al menos 1 muestra en cada lado.
    #   2. Si aun así no es posible dividir de forma segura (por ejemplo,
    #      menos de 5 municipios en total), se usa todo el conjunto como
    #      train y test (evaluación no es representativa, pero el modelo
    #      no se rompe).
    # ------------------------------------------------------------------
    clase_min = y.value_counts().min()
    usar_stratify = y if clase_min >= 2 else None

    n_total = len(X)
    test_size = 0.20
    # Garantizar al menos 1 muestra en test y suficientes en train.
    n_test_estimado = max(1, round(n_total * test_size))
    n_train_estimado = n_total - n_test_estimado

    if n_total < 5 or n_train_estimado < 1 or n_test_estimado < 1:
        # Conjunto demasiado pequeño para dividir de forma segura.
        if verbose:
            print(f"\n  ⚠ Conjunto de municipios muy pequeño ({n_total}). "
                  f"Se usará el conjunto completo para train y test "
                  f"(la evaluación no será representativa).")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=usar_stratify
            )
        except ValueError:
            # Última red de seguridad: si train_test_split falla por
            # cualquier otra combinación de clases raras, se reintenta
            # sin estratificar y sin riesgo de quedar vacío.
            if verbose:
                print("\n  ⚠ train_test_split falló con la configuración "
                      "estratificada. Reintentando sin estratificar.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=None
            )

    if verbose:
        print(f"\n  Train: {len(X_train)} municipios | Test: {len(X_test)} municipios")

    # A. Árbol de Decisión
    modelo_dt = entrenar_arbol(X_train, X_test, y_train, y_test, verbose)
    grafica_arbol_decision(modelo_dt)
    grafica_matriz_confusion(modelo_dt, X_test, y_test, "Árbol de Decisión")

    # Guardar métricas del DT como CSV para el dashboard
    y_pred_dt = modelo_dt.predict(X_test)
    reporte_dt = classification_report(
        y_test, y_pred_dt, labels=ORDEN_CLASES, zero_division=0, output_dict=True
    )
    metricas_dt_df = (
        pd.DataFrame(reporte_dt).T
        .reset_index()
        .rename(columns={"index": "Clase"})
        .round(3)
    )
    ruta_metricas_dt = os.path.join(RUTAS["reportes"], "metricas_dt.csv")
    os.makedirs(os.path.dirname(ruta_metricas_dt), exist_ok=True)
    metricas_dt_df.to_csv(ruta_metricas_dt, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"\n  ✓ Métricas DT guardadas en: {ruta_metricas_dt}")

    # B. Random Forest
    modelo_rf = entrenar_random_forest(X_train, X_test, y_train, y_test, verbose)
    grafica_feature_importance(modelo_rf)
    grafica_matriz_confusion(modelo_rf, X_test, y_test, "Random Forest")

    # C. K-Means
    modelo_km, mun_km, scaler = entrenar_kmeans(mun, n_clusters=4, verbose=verbose)

    # Guardar tabla municipal enriquecida con clusters
    ruta_km = os.path.join(RUTAS["reportes"], "municipios_clusters.csv")
    os.makedirs(os.path.dirname(ruta_km), exist_ok=True)
    mun_km.to_csv(ruta_km, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"\n  ✓ Municipios con clusters guardados en: {ruta_km}")

    return {
        "arbol_decision": modelo_dt,
        "random_forest":  modelo_rf,
        "kmeans":         modelo_km,
        "scaler":         scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "mun_km":  mun_km,
    }


# -----------------------------------------------------------------------------
# EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mun = pd.read_csv(RUTAS["procesado_mun"], dtype={"cod_dane_mun": str})
    resultados = pipeline_modelos(mun, verbose=True)
