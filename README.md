# Clasificación supervisada de riesgo en pólizas para Provincia de Buenos Aires

__Este proyecto aplica técnicas de ciencia de datos para analizar y predecir el nivel de riesgo asociado a pólizas de seguros en la Provincia de Buenos Aires.__ A partir de un dataset del __DNRPA__ (Dirección Nacional de los Registros Nacionales de la Propiedad del Automotor y Créditos Prendarios), se realizó un análisis exploratorio y descriptivo para comprender la distribución de variables relevantes, detectar patrones y preparar los datos para el modelado. 
Posteriormente, desarrollamos un algoritmo de clasificación supervisada que permite categorizar cada póliza como de riesgo __BAJO, MEDIO o ALTO__. 

El objetivo principal es ofrecer una herramienta que ayude a la toma de decisiones en la gestión de clientes y administración de pólizas. El trabajo incluye limpieza de datos, visualizaciones, selección de variables, entrenamiento del modelo y evaluación de desempeño. Todo el código está documentado y organizado para facilitar su comprensión y reutilización.

## Datos

El dataset que usamos en este proyecto proviene del portal de datos abiertos del gobierno argentino, específicamente de la __DNRPA__. El recurso se titula Robos y recuperos de autos - 2025 y contiene información sobre denuncias de robo o hurto de vehículos, así como las comunicaciones de recupero realizadas durante el año 2025 en la República Argentina.

Los datos están organizados en formato __CSV__ y se actualizan mensualmente. Incluyen variables como fecha del hecho, tipo de denuncia, tipo de recupero, ubicación geográfica, y características del vehículo involucrado. Estas variables las utilizamos para realizar un análisis exploratorio y construir un modelo de clasificación supervisada que predice el nivel de riesgo de una póliza de seguro __(BAJO, MEDIO o ALTO)__.

Este conjunto de datos es clave para entender el comportamiento delictivo relacionado con vehículos en la región, y permite generar insights relevantes para la industria aseguradora en términos de prevención, segmentación de clientes y clasificación de pólizas.

Cabe aclarar que los __CSV__ están divididos por mes, en el repo incluimos un script [combinar_archivos.py](combinar_archivos.py) que nos permite combinar múltiples CSV para poder abarcar más datos.

__Referencias__

Esta lista contiene links a distintos recursos relacionados al proyecto:

* **[DNRPA Robos y recuperos de autos - 2025](https://datos.gob.ar/dataset/justicia-robos-recuperos-autos/archivo/justicia_c5eb0b07-c6e6-49f1-8ea0-165c957f5f94)** - Fuente oficial de datos abiertos sobre denuncias y recuperos de vehículos en la república Argentina.
* **[Looker dashboard](https://lookerstudio.google.com/reporting/fb3bf161-2054-445e-a9ff-10c2d54496bb)** -  Visualización interactiva del análisis exploratorio y descriptivo realizado sobre el dataset.

## Librerías utilizadas

- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)

## Instalación

1. Clonar el repositorio.

```bash
git clone https://github.com/Kagushutchi/Ci
```

2. Cambiar el directorio al del proyecto.

```bash
cd Ci
```

3. Crear entorno virtual (opcional pero recomendado).

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

4. Instalar los requerimientos.

```bash
pip install -r requirements.txt
```

## Inicio rápido

Una guía paso a paso de como ejecutar el proyecto.

1. Descomprimir el archivo rar o descargar manualmente los CSV desde [DNRPA Robos y recuperos de autos - 2025](https://datos.gob.ar/dataset/justicia-robos-recuperos-autos/archivo/justicia_c5eb0b07-c6e6-49f1-8ea0-165c957f5f94).

2. Ejecutar el script [combinar_archivos.py](combinar_archivos.py) en la misma carpeta donde extrajo los CSV. Esto retornara [combined_optimized.csv](combined_optimized.csv).

3. Ejecutar el script [marcas.py](marcas.py) en la misma carpeta donde se encuentra [combined_optimized.csv](combined_optimized.csv). Esto retornara [marcas.csv](marcas.csv) que un CSV filtrado con [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) que consta de tres columnas CORRECTO, VARIANTES y FRECUENCIA_TOTAL. **Precaución porque esto sobrescribirá el [marcas.csv](marcas.csv) ya generado, este posee algunos cambios manuales que el algoritmo no abarco, recomendamos temporalmente cambiarle el nombre al [marcas.csv](marcas.csv) original con tal de conservarlo.**

4. Ejecutar el script [strip_marcas.py](strip_marcas.py) en la misma carpeta donde se encuentran [combined_optimized.csv](combined_optimized.csv) y [marcas.csv](marcas.csv). Este retornara el CSV con las marcas corregidas y eliminara las columnas no necesarias, tendrá el nombre de [datos_robos_limpio.csv]().

5. Ejecutar el script [DataSet.py](DataSet.py) en la misma carpeta donde se encuentra [datos_robos_limpio.csv](). Este script contiene nuestro análisis exploratorio que consta de múltiples gráficos, posteriormente ejecuta un RandomForest para identificar las features más importantes para poder entrenar el modelo de clasificación supervisada.

## 👨‍💻 Autores
- **[Flac222](https://github.com/Flac222)**
- **[Kagushutchi](https://github.com/Kagushutchi)**
- **[Ramadion](https://github.com/Ramadion)**

## Licencia

- Este proyecto no cuenta con una licencia de uso. Su contenido es exclusivamente académico y fue desarrollado como parte de una materia universitaria. No está destinado a distribución ni reutilización comercial.

## Reconocimientos

* Este README fue hecho en base a esta [plantilla](https://gist.github.com/danielecook/94272f387d3366070d2546e2eadefe57) hecha por [Daniel E. Cook](https://gist.github.com/danielecook).
