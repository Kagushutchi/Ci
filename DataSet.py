import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
