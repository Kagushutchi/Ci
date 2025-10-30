import pandas as pd
from rapidfuzz import fuzz, process
from collections import Counter
import numpy as np

# --- Parámetros ---
min_frecuencia = 150 # Frecuencia mínima para incluir el cluster
threshold = 90

df_marcas = pd.read_csv("marcas.csv", low_memory=False)
marcas = df_marcas.drop(columns=["FRECUENCIA_TOTAL"])

df_datos = pd.read_csv("combined_optimized.csv", low_memory=False)

# Crear el diccionario de mapeo
mapeo_marcas = {}

# Iterar sobre las filas del DataFrame de marcas
for index, row in df_marcas.iterrows():
    marca_correcta = row['CORRECTO']
    variantes_str = row['VARIANTES']

    # 1. Mapear la marca correcta a sí misma (para asegurar que pase)
    mapeo_marcas[marca_correcta] = marca_correcta

    # 2. Mapear las variantes a la marca correcta
    if isinstance(variantes_str, str) and variantes_str.strip():
        # Separar las variantes por el '|' y limpiar posibles espacios
        variantes_list = [v.strip() for v in variantes_str.split('|')]
        for variante in variantes_list:
            if variante: # Asegurar que no sea una cadena vacía
                mapeo_marcas[variante] = marca_correcta

print(f"Número de entradas únicas en el mapeo: {len(mapeo_marcas)}")


# Columna a limpiar
columna_a_limpiar = 'automotor_marca_descripcion'
columna_limpia = 'marca'


df_datos[columna_limpia] = df_datos[columna_a_limpiar].map(mapeo_marcas, na_action='ignore')


marcas_no_mapeadas = df_datos[columna_limpia].isna().sum()
print(f"Número de marcas que se clasificarán como 'OTROS': {marcas_no_mapeadas}")

# Reemplazar los valores NaN (los no mapeados) con 'OTROS'
df_datos[columna_limpia] = df_datos[columna_limpia].fillna('OTROS')


marcas_originales_otros = df_datos[df_datos[columna_limpia] == 'OTROS'][columna_a_limpiar].unique()

columnas_a_borrar = [
    'automotor_marca_descripcion',
    'automotor_marca_codigo',
    'tramite_tipo',
    'titular_tipo_persona',
    'titular_genero',
    'titular_anio_nacimiento',
    'titular_pais_nacimiento',
    'titular_porcentaje_titularidad',
    'titular_domicilio_provincia_indec_id',
    'titular_pais_nacimiento_indec_id',
    'titular_pais_nacimiento_id'
]

df_datos.drop(columns=[c for c in columnas_a_borrar if c in df_datos.columns], inplace=True)

nombre_archivo_salida = 'datos_robos_limpio.csv'



columna_modelo_sucio = "automotor_modelo_descripcion"
columna_modelo_limpio = "modelo_normalizado"

df_datos[columna_modelo_limpio] = df_datos[columna_modelo_sucio].fillna('').str.upper().str.strip()

df_datos[columna_modelo_limpio] = df_datos[columna_modelo_limpio].str.replace(r'[^A-Z0-9\s]', ' ', regex=True) 
df_datos[columna_modelo_limpio] = df_datos[columna_modelo_limpio].str.replace(r'\s+', ' ', regex=True)
df_datos[columna_modelo_limpio] = df_datos[columna_modelo_limpio].str.strip()


df_datos[columna_modelo_limpio] = df_datos[columna_modelo_limpio].str.split(' ').str[:3].str.join(' ') 

df_datos[columna_modelo_limpio] = df_datos[columna_modelo_limpio].replace('', np.nan) 


df_conteo = df_datos.dropna(subset=['marca', columna_modelo_limpio])
modelo_counts = df_conteo.groupby(['marca', columna_modelo_limpio]).size().reset_index(name='frecuencia')

# Obtener marcas únicas para iterar (
marcas_unicas = modelo_counts['marca'].unique()

output_rows = []

for marca in marcas_unicas:
    # 2.1 Filtrar modelos y conteo solo para la marca actual
    df_marca = modelo_counts[modelo_counts['marca'] == marca]
    modelos_marca = df_marca[columna_modelo_limpio].tolist()
    
    # Crear un diccionario de conteo para la marca
    conteo_modelo_marca = dict(zip(df_marca[columna_modelo_limpio], df_marca['frecuencia']))
    
    visited = set()
    
    # 2.2 Aplicar clustering fuzzy SOLO DENTRO DE ESTA MARCA
    for modelo_referencia in modelos_marca:
        if modelo_referencia in visited:
            continue
            
        # Encontrar modelos similares
        matches = process.extract(modelo_referencia, modelos_marca, scorer=fuzz.token_sort_ratio, limit=None)
        similar_modelos = [m[0] for m in matches if m[1] >= threshold]
        
        # Marcar los modelos similares como visitados
        for match in similar_modelos:
            visited.add(match)
            
       
        
        # El modelo que inicia el clúster será el "CORRECTO"
        modelo_correcto = modelo_referencia 
        
        # Las variantes son todos los demás modelos del clúster
        variantes = [v for v in similar_modelos if v != modelo_correcto]
        total_frecuencia = sum(conteo_modelo_marca[v] for v in similar_modelos)
        
        # Filtrar por frecuencia mínima
        if total_frecuencia >= min_frecuencia:
            output_rows.append({
                "MARCA_CORRECTA": marca,
                "MODELO_CORRECTO": modelo_correcto,
                "VARIANTES": "|".join(variantes) if variantes else "",
                "FRECUENCIA_TOTAL": total_frecuencia
            })

# 3. Guardar el DataFrame de Mapeo de Modelos
df_mapeo_modelos = pd.DataFrame(output_rows)
nombre_archivo_mapeo = "mapeo_modelos.csv"
df_mapeo_modelos.to_csv(nombre_archivo_mapeo, index=False)

print(f"✅ ¡Mapeo de modelos por marca generado con éxito en '{nombre_archivo_mapeo}'!")

# Exportar el DataFrame a CSV
df_datos.to_csv(
    nombre_archivo_salida,
    sep=',',         
    index=False,     
    encoding='utf-8'
)

print(f"✅ ¡Datos exportados con éxito a '{nombre_archivo_salida}'!")