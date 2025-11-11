import pandas as pd
import torch
from utils_legacy import Autoencoder, Autoencoder2, entrenar_autoencoder, crear_datasets_proporcionales, cargar_modelo
import numpy as np
import json
import os
from datetime import datetime

from autoencoder import Autoencoder
from utils import get_device

# --- Carga de datos y configuración del dispositivo ---
#df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# Estandarizamos las columnas no binarias
cols = ["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()

# Creamos los conjuntos para el entrenamiento
#list_x_train, list_y_train, list_x_test, list_y_test, resumen_df = crear_datasets_proporcionales(df, "Diabetes_binary")
list_x_train, list_y_train, list_x_test, list_y_test, resumen_df = crear_datasets_proporcionales(df, "Diabetes_012")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo usado: {device}")
if device.type == "cuda":
    print(f"Nombre GPU: {torch.cuda.get_device_name(0)}")

# --- Cargar los mejores parámetros desde el archivo JSON ---
with open('best_parameters.json', 'r') as f:
    mejores_parametros = json.load(f)

# --- Entrenamiento ---
print("\n--- Entrenando modelo final con los mejores parámetros ---")
print(mejores_parametros)

EPOCAS = 1000

for i, x in enumerate(list_x_train):
    # Entrada de 21
    """
    modelo = Autoencoder(
        entrada= x.shape[1],
        oculta=mejores_parametros['hidden_size'],
        latente= mejores_parametros['latent_size']
    ).to(device)
    """
    modelo = Autoencoder2(
        entrada= x.shape[1]
    ).to(device)
    #"""
    #ruta_modelo = f"models/autoencoder{i}.pth"
    #modelo = cargar_modelo(ruta_modelo)

    modelo, _ = entrenar_autoencoder(
        modelo=modelo,
        x_data=x,
        device=device,
        lr= 0.001, #mejores_parametros['lr'], 0.0001,
        batch_size=512,
        epocas=EPOCAS,
        verbose=1
    )

    # --- Guardar el modelo ---
    fecha = datetime.now().strftime("%H-%M_%d-%m-%y")
    path = f"models/autoencoder_{fecha}_{i}.pth"
    torch.save(modelo, path)
    print(f"\nModelo guardado en '{path}'")