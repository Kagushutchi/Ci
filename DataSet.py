import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd



# ==========================
# 1 CARGA DE DATOS
# ==========================
dnrpa_path = "combined_optimized.csv"
dnrpa = pd.read_csv(dnrpa_path, low_memory=False)
print("DNRPA:", dnrpa.shape)

# ==========================
# 2 LIMPIEZA PREVIA
# ==========================
umbral = 0.8
cols_nulas = dnrpa.columns[dnrpa.isnull().mean() > umbral]
dnrpa = dnrpa.drop(columns=cols_nulas)
dnrpa['titular_domicilio_localidad'] = dnrpa['titular_domicilio_localidad'].fillna('SIN_LOCALIDAD')

# ==========================
# 3 FILTRO GEOGRÁFICO
# ==========================
mask_caba = dnrpa.apply(lambda col: col.astype(str).str.contains("CABA|BUENOS AIRES", case=False, na=False))
dnrpa_caba = dnrpa[mask_caba.any(axis=1)].copy()
print("Filas con CABA:", dnrpa_caba.shape)

# ==========================
# 4 LIMPIEZA DE DATOS
# ==========================

# ==========================
# 5 CREACIÓN DE VARIABLES
# ==========================
dnrpa_caba['automotor_anio_modelo'] = pd.to_numeric(dnrpa_caba['automotor_anio_modelo'], errors='coerce')
dnrpa_caba['vehicle_age'] = 2025 - dnrpa_caba['automotor_anio_modelo']

robos_por_loc = dnrpa_caba.groupby('titular_domicilio_localidad').size().reset_index(name='robos_count')
dnrpa_caba = dnrpa_caba.merge(robos_por_loc, on='titular_domicilio_localidad', how='left')

# ==========================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================

# ---- 1 Revisión general ----
print("\n Información general del dataset:")
print(dnrpa_caba.info())
print("\nEstadísticas descriptivas:")
print(dnrpa_caba.describe(include='all').T)

# ---- 3 Localidades con más robos ----
robos_por_loc = (
    dnrpa_caba['titular_domicilio_localidad']
    .value_counts()
    .rename_axis('titular_domicilio_localidad')
    .reset_index(name='robos_count')
)


print("\nTop 10 localidades con más robos:")
print(robos_por_loc.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    data=robos_por_loc.head(20),
    x='robos_count',
    y='titular_domicilio_localidad',
    hue='titular_domicilio_localidad',  # Usamos y como hue
    palette='Reds_r',
    legend=False
)
plt.title(" Top 20 localidades con más robos de autos")
plt.xlabel("Cantidad de robos")
plt.ylabel("Localidad")
plt.tight_layout()
plt.show()

# ---- 4 Marcas más robadas ----
dnrpa_caba['automotor_marca_descripcion_clean'] = (
    dnrpa_caba['automotor_marca_descripcion']
    .str.upper()
    .str.strip()
)

top_marcas = (
    dnrpa_caba['automotor_marca_descripcion_clean']
    .value_counts()
    .head(10)
    .rename_axis('Marca')
    .reset_index(name='Cantidad de robos')
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=top_marcas,
    x='Cantidad de robos',
    y='Marca',
    palette='Oranges_r',
    hue='Marca',
    legend=False
)
plt.title("Top 10 marcas más robadas")
plt.tight_layout()
plt.show()

# ---- 5 Modelos más robados ----
top_modelos = (
    dnrpa_caba['automotor_modelo_descripcion']
    .str.upper()
    .str.strip()
    .value_counts()
    .head(15)
    .rename_axis('Modelo')
    .reset_index(name='Cantidad de robos')
)


plt.figure(figsize=(10,6))
sns.barplot(data=top_modelos, x='Cantidad de robos', y='Modelo', palette='Blues_r')
plt.title("Modelos más robados")
plt.tight_layout()
plt.show()

# ---- 6 Relación año del vehículo vs robos ----
plt.figure(figsize=(10,5))
sns.histplot(
    dnrpa_caba['automotor_anio_modelo'].dropna(),
    bins=30,
    kde=True,
    color='steelblue'
)
plt.title("Distribución del año del vehículo en robos")
plt.xlabel("Año del vehículo")
plt.ylabel("Cantidad de robos")

# Limitar el eje X desde el 2000
plt.xlim(2000, dnrpa_caba['automotor_anio_modelo'].max())

plt.tight_layout()
plt.show()

# ---- 7 Mapa de calor geográfico (usando GeoPandas) ----
# Cargamos un shapefile con los límites de provincias argentinas
try:

    # Cargar mapa de provincias
    provincias = gpd.read_file("https://raw.githubusercontent.com/juaneladio/argentina-geojson/master/argentina.json")
    provincias = provincias.rename(columns={'name': 'provincia'})
    provincias['provincia'] = provincias['provincia'].str.upper()

    # Agrupar robos por provincia
    robos_por_prov = (
        dnrpa_caba['titular_domicilio_provincia']
        .value_counts()
        .reset_index()
        .rename(columns={'index': 'provincia', 'titular_domicilio_provincia': 'robos'})
    )
    robos_por_prov['provincia'] = robos_por_prov['provincia'].str.upper()

    # Normalizar nombres
    robos_por_prov['provincia'] = robos_por_prov['provincia'].replace({
        'BUENOS AIRES': 'PROVINCIA DE BUENOS AIRES',
        'CABA': 'CIUDAD AUTONOMA DE BUENOS AIRES',
    })

    provincias_robos = provincias.merge(robos_por_prov, on='provincia', how='left').fillna(0)
    # Graficar
    plt.figure(figsize=(8,8))
    provincias_robos['robos_log'] = np.log1p(provincias_robos['robos']) 
    provincias_robos.plot(
    column='robos_log',
    cmap='OrRd',  
    linewidth=0.8,
    edgecolor='black',
    legend=True
    )
    plt.title("🗺️ Mapa de calor: robos de autos por provincia")
    plt.axis('off')
    plt.show()

except Exception as e:
    print("⚠️ No se pudo generar el mapa geográfico:", e)

# ==========================
#  MAPA DE CALOR: ROBOS POR MARCA Y LOCALIDAD 
# ==========================

# 1 Filtramos solo localidades de Buenos Aires 
dnrpa_ba = dnrpa_caba[
    dnrpa_caba['titular_domicilio_provincia']
    .str.contains("BUENOS AIRES|CABA", case=False, na=False)
].copy()

# 2 Tomamos las 10 marcas más robadas
top_10_marcas = (
    dnrpa_ba['automotor_marca_descripcion_clean']
    .value_counts()
    .nlargest(10)
    .index
)

# 3 Agrupamos las demás como "OTRAS"
dnrpa_ba['marca_grupo'] = dnrpa_ba['automotor_marca_descripcion_clean'].apply(
    lambda x: x if x in top_10_marcas else 'OTRAS'
)

# 4 Agrupamos por localidad y marca
robos_loc_marca = (
    dnrpa_ba
    .groupby(['titular_domicilio_localidad', 'marca_grupo'])
    .size()
    .reset_index(name='robos')
)

# 5 Creamos una tabla pivote para el mapa de calor
pivot_heatmap = robos_loc_marca.pivot_table(
    index='titular_domicilio_localidad',
    columns='marca_grupo',
    values='robos',
    fill_value=0
)

# 6 Mostramos las 20 localidades con más robos totales
top_localidades = (
    robos_loc_marca
    .groupby('titular_domicilio_localidad')['robos']
    .sum()
    .nlargest(20)
    .index
)
pivot_heatmap = pivot_heatmap.loc[top_localidades]

# 7 Graficamos el mapa de calor
plt.figure(figsize=(12,8))
sns.heatmap(
    pivot_heatmap,
    cmap="Reds",
    linewidths=0.5,
    linecolor='gray',
    annot=True,
    fmt='.0f'
)
plt.title("Mapa de calor de robos de autos por marca y localidad (Buenos Aires)")
plt.xlabel("Marca de auto")
plt.ylabel("Localidad")
plt.tight_layout()
plt.show()

"""
# ==========================
# 6 VARIABLE OBJETIVO
# ==========================
dnrpa_caba['risk_score'] = dnrpa_caba['robos_count'].fillna(0).rank(pct=True)
dnrpa_caba['risk_level'] = pd.qcut(dnrpa_caba['risk_score'], q=3, labels=['bajo', 'medio', 'alto'])
print("Distribución de riesgo:")
print(dnrpa_caba['risk_level'].value_counts())

# ==========================
# 7 PREPARACIÓN DE FEATURES
# ==========================
dnrpa_caba['robos_count'] = dnrpa_caba['robos_count'].fillna(0)
dnrpa_caba['vehicle_age'] = dnrpa_caba['vehicle_age'].fillna(dnrpa_caba['vehicle_age'].median())
dnrpa_caba['automotor_marca_descripcion'] = dnrpa_caba['automotor_marca_descripcion'].fillna('SIN_MARCA')

features = ['vehicle_age', 'automotor_marca_descripcion'] #  CORREGIDO: Usar solo las variables independientes

dnrpa_caba = dnrpa_caba.dropna(subset=features)

top_marcas = dnrpa_caba['automotor_marca_descripcion'].value_counts().nlargest(20).index
# Reemplazar las marcas raras con 'OTROS'
dnrpa_caba['automotor_marca_descripcion_clean'] = dnrpa_caba['automotor_marca_descripcion'].apply(lambda x: x if x in top_marcas else 'OTRAS')

# Cambiar la lista de features para usar la columna limpia
features = ['vehicle_age', 'automotor_marca_descripcion_clean']
dnrpa_caba = dnrpa_caba.dropna(subset=features)

X = dnrpa_caba[features].copy()
y = dnrpa_caba['risk_level']
# Codificar la nueva columna con menos categorías
X = pd.get_dummies(X, columns=['automotor_marca_descripcion_clean'], drop_first=True)

# ==========================
# 8 SPLIT TRAIN/TEST
# ==========================
if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n {name}")
        print(classification_report(y_test, preds))
        results[name] = (y_test, preds)

    # ==========================
    # 9 MATRIZ DE CONFUSIÓN (VISUAL)
    # ==========================
    for name, (yt, yp) in results.items():
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(yt, yp, labels=['bajo', 'medio', 'alto'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['bajo','medio','alto'], yticklabels=['bajo','medio','alto'])
        plt.title(f"Matriz de confusión - {name}")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.show()
else:
    print(" No hay datos suficientes para entrenar modelos.")

"""
# ==========================
# 10 EXPORTAR RESULTADOS
# ==========================
dnrpa_caba.to_csv("autos_caba_riesgo.csv", index=False)
print("\nArchivo exportado: autos_caba_riesgo.csv")
