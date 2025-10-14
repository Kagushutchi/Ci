import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

"""
Head of the file: tramite_tipo,tramite_fecha,fecha_inscripcion_inicial,registro_seccional_codigo,registro_seccional_descripcion,registro_seccional_provincia,automotor_origen,automotor_anio_modelo,automotor_tipo_codigo,automotor_tipo_descripcion,automotor_marca_codigo,automotor_marca_descripcion,automotor_modelo_codigo,automotor_modelo_descripcion,automotor_uso_codigo,automotor_uso_descripcion,titular_tipo_persona,titular_domicilio_localidad,titular_domicilio_provincia,titular_genero,titular_anio_nacimiento,titular_pais_nacimiento,titular_porcentaje_titularidad,titular_domicilio_provincia_indec_id,titular_pais_nacimiento_indec_id,titular_domicilio_provincia_id,titular_pais_nacimiento_id
DENUNCIA DE ROBO O HURTO,2018-01-17,2000-10-05,1029,ESTEBAN ECHEVERRIA Nº 1,Buenos Aires,Nacional,2000.0,,SEDAN,08,CHEVROLET,57,CORSA 4 PUERTAS WIND 1.6 MPFI,1.0,Privado,Física,MONTE GRANDE,BUENOS AIRES,Masculino,1981.0,Argentina,100.0,6.0,ARG,,
DENUNCIA DE ROBO O HURTO,2018-01-03,2007-11-22,1047,LANUS Nº 1,Buenos Aires,Nacional,2007.0,,FURGON 600,34,PEUGEOT,EP,PARTNER FURGON D PLC PRESENCE,1.0,Privado,Física,VALENTIN ALSINA,BUENOS AIRES,Femenino,1990.0,Argentina,100.0,6.0,ARG,,
DENUNCIA DE ROBO O HURTO,2018-01-12,1995-02-01,1059,MAR DEL PLATA Nº 02,Buenos Aires,Nacional,1995.0,,BERLINA 5 PUERTAS,37,RENAULT,AH,19 RN INYECCION BIC.,1.0,Privado,Física,UNIDAD TURISTICA CHAPADMALAL,BUENOS AIRES,Masculino,1986.0,Argentina,100.0,6.0,ARG,,
DENUNCIA DE ROBO O HURTO,2018-01-02,1999-09-28,1066,NECOCHEA Nº 1,Buenos Aires,Nacional,1999.0,,BERLINA 3 PUERTAS,37,RENAULT,CC,CLIO RL DIESEL 3 PUERTAS,1.0,Privado,Física,NECOCHEA BS.AS.,BUENOS AIRES,No identificado,1964.0,No identificado,100.0,6.0,,,
DENUNCIA DE ROBO O HURTO,2018-01-09,2006-09-07,1074,PILAR Nº 1,Buenos Aires,Nacional,2006.0,,FURGON 600,34,PEUGEOT,DM,PARTNER FURGON 1.4 N PRESENCE,1.0,Privado,Física,PTE. DERQUI,BUENOS AIRES,Femenino,1961.0,Argentina,100.0,6.0,ARG,,

"""

# ==========================
# 1️⃣ CARGA DE DATOS
# ==========================
dnrpa_path = "combined_optimized.csv"
dnrpa = pd.read_csv(dnrpa_path, low_memory=False)
print("DNRPA:", dnrpa.shape)

# ==========================
# 2️⃣ LIMPIEZA PREVIA
# ==========================
umbral = 0.8
cols_nulas = dnrpa.columns[dnrpa.isnull().mean() > umbral]
dnrpa = dnrpa.drop(columns=cols_nulas)
dnrpa['titular_domicilio_localidad'] = dnrpa['titular_domicilio_localidad'].fillna('SIN_LOCALIDAD')

# ==========================
# 3️⃣ FILTRO GEOGRÁFICO
# ==========================
mask_caba = dnrpa.apply(lambda col: col.astype(str).str.contains("CABA|BUENOS AIRES", case=False, na=False))
dnrpa_caba = dnrpa[mask_caba.any(axis=1)].copy()
print("Filas con CABA:", dnrpa_caba.shape)

# ==========================
# 4️⃣ LIMPIEZA DE DATOS
# ==========================
cols_nulas = dnrpa_caba.columns[dnrpa_caba.isnull().mean() > umbral]
dnrpa_caba = dnrpa_caba.drop(columns=cols_nulas)
dnrpa_caba = dnrpa_caba.fillna({'titular_domicilio_localidad': 'SIN_LOCALIDAD'})
print("Columnas eliminadas:", list(cols_nulas))

# ==========================
# 5️⃣ CREACIÓN DE VARIABLES
# ==========================
dnrpa_caba['automotor_anio_modelo'] = pd.to_numeric(dnrpa_caba['automotor_anio_modelo'], errors='coerce')
dnrpa_caba['vehicle_age'] = 2025 - dnrpa_caba['automotor_anio_modelo']

robos_por_loc = dnrpa_caba.groupby('titular_domicilio_localidad').size().reset_index(name='robos_count')
dnrpa_caba = dnrpa_caba.merge(robos_por_loc, on='titular_domicilio_localidad', how='left')

# ==========================
# 📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==========================

# ---- 1️⃣ Revisión general ----
print("\n📋 Información general del dataset:")
print(dnrpa_caba.info())
print("\nEstadísticas descriptivas:")
print(dnrpa_caba.describe(include='all').T)

# ---- 2️⃣ Valores nulos ----
cols_with_nulls = dnrpa_caba.columns[dnrpa_caba.isnull().any()]
subset = dnrpa_caba[cols_with_nulls].head(500)  # limitar a primeras 500 filas

plt.figure(figsize=(12,6))
sns.heatmap(subset.isnull(), cbar=False, yticklabels=False, cmap='mako')
plt.title("🧯 Mapa de calor de valores nulos (primeras 500 filas)")
plt.tight_layout()
plt.show()


# ---- 3️⃣ Localidades con más robos ----
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
plt.title("🏙️ Top 20 localidades con más robos de autos")
plt.xlabel("Cantidad de robos")
plt.ylabel("Localidad")
plt.tight_layout()
plt.show()

# ---- 4️⃣ Marcas más robadas ----
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
plt.title("🚗 Top 10 marcas más robadas")
plt.tight_layout()
plt.show()

# ---- 5️⃣ Modelos más robados ----
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
plt.title("🚙 Modelos más robados")
plt.tight_layout()
plt.show()

# ---- 6️⃣ Relación año del vehículo vs robos ----
plt.figure(figsize=(10,5))
sns.histplot(dnrpa_caba['automotor_anio_modelo'].dropna(), bins=30, kde=True, color='steelblue')
plt.title("📅 Distribución del año del vehículo en robos")
plt.xlabel("Año del vehículo")
plt.ylabel("Cantidad de robos")
plt.tight_layout()
plt.show()

# ---- 7️⃣ 🔥 Mapa de calor geográfico (usando GeoPandas) ----
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
    provincias_robos['robos_log'] = np.log1p(provincias_robos['robos'])  # log(robos + 1)
    provincias_robos.plot(
    column='robos_log',
    cmap='OrRd',  # Más contraste que 'Reds'
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
# 6️⃣ VARIABLE OBJETIVO
# ==========================
dnrpa_caba['risk_score'] = dnrpa_caba['robos_count'].fillna(0).rank(pct=True)
dnrpa_caba['risk_level'] = pd.qcut(dnrpa_caba['risk_score'], q=3, labels=['bajo', 'medio', 'alto'])
print("Distribución de riesgo:")
print(dnrpa_caba['risk_level'].value_counts())

# ==========================
# 7️⃣ PREPARACIÓN DE FEATURES
# ==========================
dnrpa_caba['robos_count'] = dnrpa_caba['robos_count'].fillna(0)
dnrpa_caba['vehicle_age'] = dnrpa_caba['vehicle_age'].fillna(dnrpa_caba['vehicle_age'].median())
dnrpa_caba['automotor_marca_descripcion'] = dnrpa_caba['automotor_marca_descripcion'].fillna('SIN_MARCA')

features = ['vehicle_age', 'automotor_marca_descripcion'] # ✅ CORREGIDO: Usar solo las variables independientes

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
# 8️⃣ SPLIT TRAIN/TEST
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
        print(f"\n🔹 {name}")
        print(classification_report(y_test, preds))
        results[name] = (y_test, preds)

    # ==========================
    # 9️⃣ MATRIZ DE CONFUSIÓN (VISUAL)
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
    print("⚠️ No hay datos suficientes para entrenar modelos.")

# ==========================
# 🔟 EXPORTAR RESULTADOS
# ==========================
dnrpa_caba.to_csv("autos_caba_riesgo.csv", index=False)
print("\nArchivo exportado: autos_caba_riesgo.csv")
