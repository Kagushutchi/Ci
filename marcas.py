import pandas as pd
from rapidfuzz import fuzz, process
from collections import Counter

# Parámetro de frecuencia mínima
min_frecuencia = 150  # Cambiá este valor según lo que quieras filtrar

# Cargar el CSV original
dnrpa_path = "combined_optimized.csv"
df = pd.read_csv(dnrpa_path, low_memory=False)

# Normalizar marcas (mayúsculas y sin espacios extra)
df["marca_normalizada"] = df["automotor_marca_descripcion"].dropna().str.upper().str.strip()

# Contar frecuencia de cada marca
brand_counts = Counter(df["marca_normalizada"])

# Obtener marcas únicas
brands = list(brand_counts.keys())

# Inicializar estructuras
clusters = {}
visited = set()

# Umbral de similitud (ajustable)
threshold = 90

# Agrupar marcas similares
for brand in brands:
    if brand in visited:
        continue
    matches = process.extract(brand, brands, scorer=fuzz.token_sort_ratio, limit=None)
    similar = [m[0] for m in matches if m[1] >= threshold]
    for match in similar:
        visited.add(match)
    clusters[brand] = similar

# Elegir la forma "correcta" y sumar frecuencias
output_rows = []
for correct, variants in clusters.items():
    variants_clean = [v for v in variants if v != correct]
    total_frecuencia = sum(brand_counts[v] for v in variants)

    # Filtrar por frecuencia mínima
    if total_frecuencia >= min_frecuencia:
        output_rows.append({
            "CORRECTO": correct,
            "VARIANTES": "|".join(variants_clean) if variants_clean else "",
            "FRECUENCIA_TOTAL": total_frecuencia
        })

# Guardar a CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv("marcas.csv", index=False)
