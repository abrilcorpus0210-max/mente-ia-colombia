# =============================================================================
# preparar_datos_deploy.py
# Crea versiones AGREGADAS y SEGURAS de los datos para subir a Streamlit Cloud.
# Filtra municipios con menos de 5 casos (supresión de celdas).
#
# Ejecutar desde la raíz del proyecto:
#   python preparar_datos_deploy.py
#
# Genera en data/processed/:
#   - municipios_riesgo_deploy.csv   (solo municipios con >= 5 casos)
#   - casos_agregados_deploy.csv     (conteos agregados, sin filas individuales)
# =============================================================================

import pandas as pd
import os

UMBRAL = 5
PROC = os.path.join("data", "processed")

print("=" * 60)
print("  PREPARACIÓN DE DATOS SEGUROS PARA DEPLOY")
print("=" * 60)

# ── 1. Municipios con IPI — filtrar < 5 casos ────────────────────────────────
ruta_mun = os.path.join(PROC, "municipios_riesgo.csv")
mun = pd.read_csv(ruta_mun, dtype={"cod_dane_mun": str})
antes = len(mun)
mun_seguro = mun[mun["total_casos"] >= UMBRAL].copy()
despues = len(mun_seguro)

ruta_mun_out = os.path.join(PROC, "municipios_riesgo_deploy.csv")
mun_seguro.to_csv(ruta_mun_out, index=False, encoding="utf-8-sig")
print(f"\n[1] municipios_riesgo_deploy.csv")
print(f"    Municipios originales : {antes}")
print(f"    Municipios publicados : {despues}")
print(f"    Suprimidos (<{UMBRAL} casos) : {antes - despues}")

# ── 2. Casos individuales → agregar a nivel municipio (NO subir filas) ───────
ruta_casos = os.path.join(PROC, "casos_limpios.csv")
casos = pd.read_csv(ruta_casos, dtype={"cod_dane_mun": str})

# Agregaciones seguras que el dashboard necesita, SIN registros individuales
agregado = (
    casos.groupby(["cod_dane_mun", "Departamento_ocurrencia",
                   "Municipio_ocurrencia", "mes", "sexo_nombre",
                   "grupo_edad", "area_nombre"], observed=True)
    .size().reset_index(name="casos")
)
# Suprimir combinaciones con < 5 casos
agregado_seguro = agregado[agregado["casos"] >= UMBRAL].copy()

ruta_casos_out = os.path.join(PROC, "casos_agregados_deploy.csv")
agregado_seguro.to_csv(ruta_casos_out, index=False, encoding="utf-8-sig")
print(f"\n[2] casos_agregados_deploy.csv")
print(f"    Registros individuales originales : {len(casos):,}")
print(f"    Filas agregadas seguras           : {len(agregado_seguro):,}")
print(f"    (no contiene registros persona a persona)")

print("\n" + "=" * 60)
print("  ✓ Listo. Sube SOLO los archivos *_deploy.csv")
print("  ✗ NUNCA subas casos_limpios.csv ni el .xlsx original")
print("=" * 60)