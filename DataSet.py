

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# 1️⃣ CARGA DE DATOS
# ==========================

 
dnrpa_path = "dnrpa-robos-recuperos-autos-202508.csv"
snic_path = "snic-provincias.csv"


dnrpa = pd.read_csv(dnrpa_path)
snic = pd.read_csv(snic_path)


print("DNRPA:", dnrpa.shape)
print("SNIC:", snic.shape)

# ==========================
# 2️⃣ LIMPIEZA PREVIA
# ==========================
snic['codigo_delito_snic_id'] = pd.to_numeric(snic['codigo_delito_snic_id'], errors='coerce')

umbral = 0.8
cols_nulas = dnrpa.columns[dnrpa.isnull().mean() > umbral]
dnrpa = dnrpa.drop(columns=cols_nulas)
dnrpa['titular_domicilio_localidad'] = dnrpa['titular_domicilio_localidad'].fillna('SIN_LOCALIDAD')
# ==========================
# 3️⃣ FILTROS SEGÚN NEGOCIO
# ==========================

mask_caba = dnrpa.apply(lambda col: col.astype(str).str.contains("CABA|BUENOS AIRES", case=False, na=False))
dnrpa_caba = dnrpa[mask_caba.any(axis=1)].copy()
print("Filas con CABA:", dnrpa_caba.shape)

snic_filtrado = snic[(snic['codigo_delito_snic_id'] >= 15) & (snic['codigo_delito_snic_id'] <= 21)]
print("Delitos 15–21:", snic_filtrado.shape)

# ==========================
# 4️⃣ LIMPIEZA DE DATOS
# ==========================


umbral = 0.8
cols_nulas = dnrpa_caba.columns[dnrpa_caba.isnull().mean() > umbral]
dnrpa_caba = dnrpa_caba.drop(columns=cols_nulas)
print("Columnas eliminadas:", list(cols_nulas))


dnrpa_caba = dnrpa_caba.fillna({'titular_domicilio_localidad': 'SIN_LOCALIDAD'})

# ==========================
# 5️⃣ CREACIÓN DE VARIABLES
# ==========================


dnrpa_caba['automotor_anio_modelo'] = pd.to_numeric(dnrpa_caba['automotor_anio_modelo'], errors='coerce')
dnrpa_caba['vehicle_age'] = 2025 - dnrpa_caba['automotor_anio_modelo']

robos_por_loc = dnrpa_caba.groupby('titular_domicilio_localidad').size().reset_index(name='robos_count')
dnrpa_caba = dnrpa_caba.merge(robos_por_loc, on='titular_domicilio_localidad', how='left')

delitos_caba = snic_filtrado[snic_filtrado['provincia_nombre'].str.contains('BUENOS AIRES', case=False, na=False)]
delitos_sum = delitos_caba.groupby('provincia_nombre')['cantidad_hechos'].sum().reset_index()
delitos_sum.rename(columns={'cantidad_hechos': 'crime_count'}, inplace=True)
dnrpa_caba = dnrpa_caba.merge(delitos_sum, left_on='titular_domicilio_provincia', right_on='provincia_nombre', how='left')

# ==========================
# 6️⃣ VARIABLE OBJETIVO
# ==========================

dnrpa_caba['risk_score'] = (
    dnrpa_caba['robos_count'].fillna(0).rank(pct=True) * 0.6 +
    dnrpa_caba['crime_count'].fillna(0).rank(pct=True) * 0.4
)
dnrpa_caba['risk_level'] = pd.qcut(dnrpa_caba['risk_score'], q=3, labels=['bajo', 'medio', 'alto'])

print("Distribución de riesgo:")
print(dnrpa_caba['risk_level'].value_counts())

# ==========================
# 7️⃣ PREPARACIÓN DE FEATURES
# ==========================

dnrpa_caba['crime_count'] = dnrpa_caba['crime_count'].fillna(0)
dnrpa_caba['robos_count'] = dnrpa_caba['robos_count'].fillna(0)
dnrpa_caba['vehicle_age'] = dnrpa_caba['vehicle_age'].fillna(dnrpa_caba['vehicle_age'].median())
dnrpa_caba['automotor_marca_descripcion'] = dnrpa_caba['automotor_marca_descripcion'].fillna('SIN_MARCA')

features = ['vehicle_age', 'robos_count', 'crime_count', 'automotor_marca_descripcion']
dnrpa_caba = dnrpa_caba.dropna(subset=features)

X = dnrpa_caba[features].copy()
y = dnrpa_caba['risk_level']
X = pd.get_dummies(X, columns=['automotor_marca_descripcion'], drop_first=True)

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
