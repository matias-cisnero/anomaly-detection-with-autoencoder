import pandas as pd
import torch
from utils import Autoencoder, entrenar_autoencoder, crear_datasets_proporcionales, cargar_modelo
import numpy as np
import json
import os

# --- Carga de datos y configuración del dispositivo ---
conjuntos, _ = crear_datasets_proporcionales("data/diabetes_binary_health_indicators_BRFSS2015.csv", "Diabetes_binary")

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

for i, x in enumerate(conjuntos):
    """
    modelo = Autoencoder(
        entrada= x.shape[1],
        oculta=mejores_parametros['hidden_size'],
        latente=mejores_parametros['latent_size']
    ).to(device)
    """
    ruta_modelo = f"models/autoencoder{i}.pth"
    modelo = cargar_modelo(ruta_modelo)

    modelo, _ = entrenar_autoencoder(
        modelo=modelo,
        x_data=x,
        device=device,
        lr= 0.0001, #mejores_parametros['lr'],
        batch_size=512,
        epocas=1000,
        verbose=True
    )

    # --- Guardar el modelo ---
    path = f"models/autoencoder{i}.pth"
    torch.save(modelo, path)
    print(f"\nModelo guardado en '{path}'")