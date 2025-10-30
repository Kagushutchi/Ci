from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ==========================
# 1 CARGA DE DATOS
# ==========================
dnrpa_path = "datos_robos_limpio.csv"
dnrpa = pd.read_csv(dnrpa_path, low_memory=False)
print("DNRPA:", dnrpa.shape)

# ==========================
# 2 LIMPIEZA PREVIA
# ==========================
umbral = 0.8
cols_nulas = dnrpa.columns[dnrpa.isnull().mean() > umbral]
dnrpa = dnrpa.drop(columns=cols_nulas)
dnrpa['titular_domicilio_localidad'] = dnrpa['titular_domicilio_localidad'].fillna(
    'SIN_LOCALIDAD')

# ==========================
# 3 FILTRO GEOGRÁFICO
# ==========================
# 1. Normalizamos la columna de domicilio del titular para el filtro
dnrpa['titular_domicilio_provincia_clean_filtro'] = dnrpa['titular_domicilio_provincia'].str.upper(
).str.strip().fillna('')

# 2. Definimos las variantes de CABA
caba_variantes = ['CIUDAD AUTONOMA DE BUENOS AIRES', 'C.AUTONOMA DE BS.AS',
                  'C.AUTONOMA BS.AS.', 'CABA', 'C.A.B.A', 'CAPITAL FEDERAL']

# 3. Creamos una máscara estricta para la Provincia de Buenos Aires, excluyendo CABA
mask_gba = (
    (dnrpa['titular_domicilio_provincia_clean_filtro'].str.contains("BUENOS AIRES|BS.AS", case=False, na=False)) &
    (~dnrpa['titular_domicilio_provincia_clean_filtro'].isin(caba_variantes)) &
    (~dnrpa['titular_domicilio_provincia_clean_filtro'].str.contains(
        "CABA|AUTONOMA|FEDERAL", case=False, na=False))
)

# Filtramos y usamos 'dnrpa_gba'
dnrpa_gba = dnrpa[mask_gba].copy()
print("Filas con Provincia de Buenos Aires (excluyendo CABA):", dnrpa_gba.shape)
# ==========================
# 4 LIMPIEZA DE DATOS
# ==========================

# ==========================
# 5 CREACIÓN DE VARIABLES
# ==========================
dnrpa_gba['automotor_anio_modelo'] = pd.to_numeric(
    dnrpa_gba['automotor_anio_modelo'], errors='coerce')
# Usaremos 'tramite_anio' para calcular la edad más adelante
dnrpa_gba['tramite_fecha'] = pd.to_datetime(
    dnrpa_gba['tramite_fecha'], errors='coerce')
mediana_anio_tramite = dnrpa_gba['tramite_fecha'].dt.year.median()
dnrpa_gba['tramite_anio'] = dnrpa_gba['tramite_fecha'].dt.year.fillna(
    mediana_anio_tramite).astype(int)

mediana_anio_modelo = dnrpa_gba['automotor_anio_modelo'].median()
dnrpa_gba['automotor_anio_modelo'] = dnrpa_gba['automotor_anio_modelo'].fillna(
    mediana_anio_modelo)

dnrpa_gba['vehicle_age'] = dnrpa_gba['tramite_anio'] - \
    dnrpa_gba['automotor_anio_modelo']
dnrpa_gba['vehicle_age'] = dnrpa_gba['vehicle_age'].apply(lambda x: max(0, x))
dnrpa_gba['vehicle_age'] = dnrpa_gba['vehicle_age'].fillna(
    dnrpa_gba['vehicle_age'].median())


robos_por_loc = dnrpa_gba.groupby(
    'titular_domicilio_localidad').size().reset_index(name='robos_count')
dnrpa_gba = dnrpa_gba.merge(
    robos_por_loc, on='titular_domicilio_localidad', how='left')
# ==========================
#  ANÁLISIS EXPLORATORIO
# ==========================

# ---- Localidades con más robos ----
top_localidades = (
    dnrpa_gba['titular_domicilio_localidad']
    .value_counts()
    .rename_axis('titular_domicilio_localidad')
    .reset_index(name='robos_count')
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_localidades.head(20),
    x='robos_count',
    y='titular_domicilio_localidad',
    palette='Reds_r',
    legend=False
)
plt.title(" Top 20 localidades con más robos de autos")
plt.xlabel("Cantidad de robos")
plt.ylabel("Localidad")
plt.tight_layout()
plt.show()

# ---- Marcas más robadas ----
top_marcas = (
    dnrpa_gba['marca']
    .str.upper()
    .str.strip()
    .value_counts()
    .head(10)
    .rename_axis('Marca')
    .reset_index(name='Cantidad de robos')
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_marcas,
    x='Cantidad de robos',
    y='Marca',
    palette='Oranges_r',
    legend=False
)
plt.title(" Top 10 marcas más robadas")
plt.tight_layout()
plt.show()

# ---- Modelos más robados ----
top_modelos = (
    dnrpa_gba['automotor_modelo_descripcion']
    .str.upper()
    .str.strip()
    .value_counts()
    .head(15)
    .rename_axis('Modelo')
    .reset_index(name='Cantidad de robos')
)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_modelos, x='Cantidad de robos',
            y='Modelo', palette='Blues_r')
plt.title(" Modelos más robados")
plt.tight_layout()
plt.show()

# ---- Tipos de vehículo más robados ----
top_tipos = (
    dnrpa_gba['automotor_tipo_descripcion']
    .str.upper()
    .str.strip()
    .value_counts()
    .head(10)
    .rename_axis('Tipo')
    .reset_index(name='Cantidad de robos')
)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_tipos, x='Cantidad de robos',
            y='Tipo', palette='Purples_r')
plt.title(" Tipos de vehículo más robados")
plt.tight_layout()
plt.show()

# ---- Distribución por año de modelo ----
plt.figure(figsize=(10, 5))
sns.histplot(dnrpa_gba['automotor_anio_modelo'].dropna(),
             bins=30, kde=True, color='steelblue')
plt.title(" Distribución del año del vehículo en robos")
plt.xlabel("Año del vehículo")
plt.ylabel("Cantidad de robos")
plt.tight_layout()
plt.show()

# ==========================
#  MAPA DE CALOR: ROBOS POR MARCA Y LOCALIDAD
# ==========================
top_10_marcas = (
    dnrpa_gba['marca']
    .value_counts()
    .nlargest(10)
    .index
)

dnrpa_gba['marca_grupo'] = dnrpa_gba['marca'].apply(
    lambda x: x if x in top_10_marcas else 'OTRAS'
)

robos_loc_marca = (
    dnrpa_gba
    .groupby(['titular_domicilio_localidad', 'marca_grupo'])
    .size()
    .reset_index(name='robos')
)

pivot_heatmap = robos_loc_marca.pivot_table(
    index='titular_domicilio_localidad',
    columns='marca_grupo',
    values='robos',
    fill_value=0
)

top_localidades_heatmap = (
    robos_loc_marca
    .groupby('titular_domicilio_localidad')['robos']
    .sum()
    .nlargest(20)
    .index
)
pivot_heatmap = pivot_heatmap.loc[top_localidades_heatmap]

plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_heatmap,
    cmap="Reds",
    linewidths=0.5,
    linecolor='gray',
    annot=True,
    fmt='.0f'
)
plt.title("Mapa de calor de robos por marca y localidad (Buenos Aires)")
plt.xlabel("Marca")
plt.ylabel("Localidad")
plt.tight_layout()
plt.show()
dnrpa_gba['risk_score'] = dnrpa_gba['robos_count'].fillna(0).rank(pct=True)
dnrpa_gba['risk_level'] = pd.qcut(
    dnrpa_gba['risk_score'], q=3, labels=['bajo', 'medio', 'alto'])
print("Distribución de riesgo:")
print(dnrpa_gba['risk_level'].value_counts())

# ==========================
# 6 FUNCIÓN DE AGRUPACIÓN Y ENCODING
# ==========================


def map_top_n(series, n):
    top_n = series.value_counts().nlargest(n).index
    return series.apply(lambda x: x if x in top_n else 'OTROS')


def frequency_encoding(df, col, new_col_name=None):
    counts = df[col].value_counts(dropna=False)
    freq = counts / len(df)
    mapping = freq.to_dict()
    if new_col_name is None:
        new_col_name = f'{col}_freq'
    df[new_col_name] = df[col].map(mapping).fillna(0)
    return df

# ==========================
# 7 OPTIMIZACIÓN DE FEATURES (AJUSTADA)
# ==========================


# --- Limpieza básica
dnrpa_gba['automotor_marca_clean'] = dnrpa_gba['marca'].str.upper(
).str.strip().fillna('SIN_MARCA')

dnrpa_gba['automotor_modelo_clean'] = dnrpa_gba['modelo_normalizado'].str.upper(
).str.strip().fillna('SIN_MODELO')

# Limpieza de la nueva feature de ubicación
dnrpa_gba['titular_domicilio_localidad_clean'] = dnrpa_gba['titular_domicilio_localidad'].str.upper(
).str.strip().fillna('SIN_LOCALIDAD')

# --- Top-N + OTROS para categóricas
dnrpa_gba['automotor_marca_desc_top'] = map_top_n(
    dnrpa_gba['automotor_marca_clean'], 20)

# === NUEVA FEATURE DE UBICACIÓN (Localidad del titular) ===
dnrpa_gba['titular_domicilio_localidad_top'] = map_top_n(
    dnrpa_gba['titular_domicilio_localidad_clean'], 50)


# --- Frequency encoding para modelo (Usando modelo normalizado)
dnrpa_gba = frequency_encoding(
    dnrpa_gba, 'automotor_modelo_clean', 'modelo_freq')


# --- Fecha
dnrpa_gba['tramite_mes'] = dnrpa_gba['tramite_fecha'].dt.month.fillna(
    0).astype(int)  # 0 para nulos


# --- Variables finales  ---
features = [
    'vehicle_age',
    'automotor_marca_desc_top',
    'titular_domicilio_localidad_top',
    'modelo_freq',
    'tramite_mes',
    'tramite_anio'
]


# Eliminamos filas donde CUALQUIERA de nuestras features clave tenga nulos
dnrpa_gba = dnrpa_gba.dropna(subset=features)

# --- One-hot encoding para las categóricas finales
X = dnrpa_gba[features].copy()
X = pd.get_dummies(X, drop_first=True)

# --- Target
y = dnrpa_gba['risk_level']

# ==========================
# 8 SPLIT TRAIN/TEST
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# ==========================
#       RANDOM FOREST
# ==========================
importances = None

if len(X_train) > 0:
    rf_model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Crear DataFrame con importancias originales
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # ==========================
    # 🌲 IMPORTANCIA DE FEATURES 
    # ==========================

# ========= Agrupar por categoría base  =========
    def agrupar_feature(nombre):
        if 'automotor_marca_desc_top' in nombre:
            return 'Marca'
        elif 'modelo_freq' in nombre:
            return 'Modelo (Freq)'
        # === GRUPO ACTUALIZADO ===
        elif 'titular_domicilio_localidad_top' in nombre:
            return 'Domicilio Localidad'
        elif 'vehicle_age' in nombre:
            return 'Antigüedad Vehículo'
        else:
            return nombre  # Tramite mes/anio

    importances['Grupo'] = importances['Feature'].apply(agrupar_feature)


    grouped_importances = (
        importances.groupby('Grupo', as_index=False)
        .agg({'Importance': 'sum'})
        .sort_values(by='Importance', ascending=False)
        )

    print("\n Importancia agrupada de las variables (Random Forest):")
    print(grouped_importances.head(15))

  
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_importances.head(15),
                x='Importance', y='Grupo', palette='viridis')
    plt.title(" Importancia de las principales variables (agrupadas)")
    plt.tight_layout()
    plt.show()

else:
    print(" No hay datos suficientes para entrenar modelos.")
# ==========================
# 9 EVALUACIÓN DEL MODELO 
# ==========================

if 'rf_model' in locals():
    # 1. Generar predicciones en el conjunto de prueba
    y_pred = rf_model.predict(X_test)

    print("\n Métricas de Evaluación (Random Forest) en el conjunto de prueba:")

    # 2. Classification Report (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 3. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred, labels=['bajo', 'medio', 'alto']) # Aseguramos el orden de las etiquetas
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['bajo', 'medio', 'alto'],
        yticklabels=['bajo', 'medio', 'alto']
    )
    plt.title(' Matriz de Confusión (Random Forest)')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Verdadero')
    plt.show()

    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nExactitud (Accuracy): {accuracy:.4f}")

else:
    print("El modelo 'rf_model' no pudo ser entrenado o no existe. Revise la sección 8.")
# ==========================
# 10 EXPORTAR RESULTADOS
# ==========================
dnrpa_gba.to_csv("autos_gba_riesgo.csv", index=False)
print("\nArchivo exportado: autos_gba_riesgo.csv")


# ==========================
# PREDICCIÓN CON NUEVOS CASOS (AÑADIDO)
# ==========================

# 1. Crear un DataFrame con datos de un nuevo caso (variables originales)
# NOTA: Los valores de las categóricas deben coincidir con los valores únicos
# en el training set (incluyendo 'OTROS' y marcas/localidades en el top N)

nuevos_datos_originales = pd.DataFrame({
    'vehicle_age': [3.0, 15.0, 8.0],  # Edad del vehículo
    'automotor_marca_desc_top': [' PEUGEOT', 'VOLKSWAGEN', 'OTROS'], # Marca (usando una del top o 'OTROS')
    'titular_domicilio_localidad_top': ['LA PLATA', 'ISIDRO CASANOVA', 'SIN_LOCALIDAD'], # Localidad (usando una del top o 'OTROS')
    'modelo_freq': [0.005, 0.0001, 0.05], # Frecuencia del modelo (simulada)
    'tramite_mes': [10, 5, 1], # Mes del trámite
    'tramite_anio': [2023, 2024, 2022] # Año del trámite
})

# 2. Preparar los datos nuevos con One-Hot Encoding
# La clave es usar las COLUMNAS DE X_train para asegurar que los dummies
# de los nuevos datos sean idénticos al formato de entrenamiento.

if 'rf_model' in locals():
    # Asignamos las variables que el modelo espera
    X_new_case = nuevos_datos_originales.copy()
    
    # Aplicar One-Hot Encoding
    X_new_case_encoded = pd.get_dummies(X_new_case, drop_first=True)

    # Alinear las columnas con las de entrenamiento (X_train)
    # Rellenamos con 0 las columnas que faltan en X_new_case_encoded y
    # eliminamos las columnas sobrantes.
    
    # Crear un DataFrame vacío con las columnas de X_train
    X_new_ready = pd.DataFrame(columns=X_train.columns)
    
    # Llenar con los datos codificados, rellenando con 0 lo que falte
    for col in X_train.columns:
        if col in X_new_case_encoded.columns:
            X_new_ready[col] = X_new_case_encoded[col]
        else:
            X_new_ready[col] = 0.0

    # Asegurar el orden de las filas si se usa un índice diferente (aunque aquí no es un problema)
    X_new_ready = X_new_ready.fillna(0.0) # Rellenar cualquier NaN potencial con 0

    print("\n🧪 Predicciones para Nuevos Casos:")
    
    # 3. Realizar la predicción
    new_predictions = rf_model.predict(X_new_ready)

    # 4. Mostrar el resultado
    resultados = nuevos_datos_originales.copy()
    resultados['Riesgo Predicho'] = new_predictions
    print(resultados)

else:
    print("El modelo 'rf_model' no pudo ser entrenado para realizar la predicción en casos nuevos.")
    