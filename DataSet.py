import pandas as pd
import numpy as np

archivo = "https://cloud-snic.minseg.gob.ar/Bases/SNIC/snic-provincias.csv"
df=pd.read_csv(archivo)
print(df.head())
print(df.describe())
print(df.info())