import pandas as pd
import numpy as np
from models import Autoencoder
from utils import get_device, crear_datasets_proporcionales, estandarizar_columnas

df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# "Binarizamos" los datos, eliminando registros de pacientes con prediabetes
df = df[df["Diabetes_012"] != 1]
df["Diabetes_012"] = df["Diabetes_012"].replace(2, 1)

# Estandarizamos las columnas no binarias
df = estandarizar_columnas(df=df, cols_estandarizar=["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income"])
list_x_train, list_y_train, list_x_test, list_y_test, resumen_df = crear_datasets_proporcionales(df, "Diabetes_012")
device = get_device()

EPOCAS = 100

for i, x in enumerate(list_x_train):

    modelo = Autoencoder([x.shape[1], 32, 16, 8, 4, 2]).to(device)
    if i == 0: modelo.summary()

    modelo.fit(x_data=x, device=device, lr=0.0001, batch_size=512, epocas=EPOCAS, verbose=1)

    modelo.save(path="models/autoencoder", i=i)