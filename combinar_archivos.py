import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# --- Configuración ---
START_DATE = datetime(2018, 1, 1)
# ---- MODIFICADO AQUÍ ----
END_DATE = datetime(2025, 8, 1) # Detenerse en el mes 8 (Agosto)
# -------------------------
BASE_FILENAME_PREFIX = "dnrpa-robos-recuperos-autos-"
FILE_EXTENSION = ".csv"
OUTPUT_FILENAME = "combined_optimized.csv"

# Encabezado final (las 27 columnas "singularizadas" que pediste)
FINAL_COLUMNS = [
    'tramite_tipo', 'tramite_fecha', 'fecha_inscripcion_inicial',
    'registro_seccional_codigo', 'registro_seccional_descripcion',
    'registro_seccional_provincia', 'automotor_origen',
    'automotor_anio_modelo', 'automotor_tipo_codigo',
    'automotor_tipo_descripcion', 'automotor_marca_codigo',
    'automotor_marca_descripcion', 'automotor_modelo_codigo',
    'automotor_modelo_descripcion', 'automotor_uso_codigo',
    'automotor_uso_descripcion', 'titular_tipo_persona',
    'titular_domicilio_localidad', 'titular_domicilio_provincia',
    'titular_genero', 'titular_anio_nacimiento',
    'titular_pais_nacimiento', 'titular_porcentaje_titularidad',
    'titular_domicilio_provincia_indec_id',
    'titular_pais_nacimiento_indec_id', 'titular_domicilio_provincia_id',
    'titular_pais_nacimiento_id'
]

# Definimos los tipos de datos para columnas con códigos o IDs
dtype_spec = {
    'registro_seccional_codigo': 'object',
    'automotor_anio_modelo': 'object',
    'automotor_tipo_codigo': 'object',
    'automotor_marca_codigo': 'object',
    'automotor_modelo_codigo': 'object',
    'automotor_uso_codigo': 'object',
    'titular_anio_nacimiento': 'object',
    'titular_domicilio_provincia_indec_id': 'object',
    'titular_pais_nacimiento_indec_id': 'object',
    'titular_domicilio_provincia_id': 'object',
    'titular_pais_nacimiento_id': 'object'
}
# --- Fin Configuración ---

all_dataframes = []  # Lista para guardar los datos de cada archivo
current_date = START_DATE
files_found = 0
files_missing = 0
# ---- MODIFICADO AQUÍ ----
total_files_expected = 92 # 7 años * 12 meses + 8 meses de 2025 = 84 + 8 = 92
# -------------------------

print(f"Iniciando proceso para combinar {total_files_expected} archivos...")
print(f"Rango: {START_DATE.strftime('%Y%m')} hasta {END_DATE.strftime('%Y%m')}")
print(f"Archivo de salida: {OUTPUT_FILENAME}")
print("-" * 30)

while current_date <= END_DATE:
    # Formatea el sufijo (ej: "201801")
    date_suffix = current_date.strftime("%Y%m")
    filename = f"{BASE_FILENAME_PREFIX}{date_suffix}{FILE_EXTENSION}"

    if os.path.exists(filename):
        try:
            # Leemos el CSV especificando los dtypes y el separador (coma por defecto)
            df = pd.read_csv(
                filename,
                dtype=dtype_spec,
                low_memory=False # Ayuda a pandas a leer tipos mixtos
            )

            # --- AQUÍ OCURRE LA "SINGULARIZACIÓN" ---
            # Reordena/filtra el DF para que coincida EXACTAMENTE con las columnas pedidas
            df_reindexed = df.reindex(columns=FINAL_COLUMNS)

            all_dataframes.append(df_reindexed)
            files_found += 1
            print(f"  [OK] Cargado y procesado: {filename}")

        except pd.errors.EmptyDataError:
            print(f"  [AVISO] Archivo vacío: {filename}. Omitiendo.")
        except Exception as e:
            print(f"  [ERROR] No se pudo leer {filename}. Error: {e}")
    else:
        print(f"  [AVISO] No se encontró: {filename}. Omitiendo.")
        files_missing += 1

    # Avanzar al próximo mes
    current_date += relativedelta(months=1)

print("-" * 30)
print("\n--- Proceso de carga finalizado ---")
print(f"Archivos procesados (encontrados): {files_found}")
print(f"Archivos omitidos (no encontrados): {files_missing}")
print(f"Total de archivos esperados: {total_files_expected}")

if not all_dataframes:
    print("\n[ERROR] No se cargó ningún dato. No se generará el archivo de salida.")
else:
    print("\nCombinando todos los archivos en memoria (esto puede tardar)...")
    
    # Combinar la lista de DataFrames en uno solo
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Datos combinados. Total de filas: {len(combined_df)}")
    print(f"Guardando en: {OUTPUT_FILENAME}...")

    try:
        # Guardar el resultado final sin el índice de pandas
        combined_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
        print("\n¡ÉXITO! Archivo combinado guardado correctamente.")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el archivo final. Error: {e}")