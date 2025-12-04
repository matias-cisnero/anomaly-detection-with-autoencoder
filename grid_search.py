import pandas as pd
import time
import numpy as np
import json
import os
from datetime import datetime
from models import Autoencoder
from utils import estandarizar_columnas_no_binarias, get_device, save_grid_search_results, crear_conjuntos_proporcionales

# =================== CARGA Y PREPROCESAMIENTO ===================
df = pd.read_csv("data/breast-cancer-wisconsin.csv")

# Quitamos atributos no necesarios
df = df.drop(columns=["id", "Unnamed: 32"])

# Reemplazamos los valores en diagnosis por 0 y 1
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Estandarizamos las columnas no binarias
df = estandarizar_columnas_no_binarias(df)

# Dividimos nuestros conjuntos de datos
conjuntos = crear_conjuntos_proporcionales(df, "diagnosis", 0.2)
x_train, x_test, y_train, y_test = conjuntos[0] # <--- Tomo el conjunto que tiene 0% de anómalos

device = get_device()

# =================== HIPERPARÁMETROS ===================
tasas_de_aprendizaje = [0.01, 0.001, 0.0001]
capas_posibles = [
    [x_train.shape[1], 32, 16, 8, 4],
    [x_train.shape[1], 32, 16, 8, 4, 3],
    [x_train.shape[1], 32, 16, 8, 4, 2],
    
    [x_train.shape[1], 64, 32, 16, 8, 4],
    [x_train.shape[1], 64, 32, 16, 8, 4, 3],
    [x_train.shape[1], 64, 32, 16, 8, 4, 2],

    [x_train.shape[1], 128, 64, 32, 16, 8, 4],
    [x_train.shape[1], 128, 64, 32, 16, 8, 4, 3],
    [x_train.shape[1], 128, 64, 32, 16, 8, 4, 2],

    [x_train.shape[1], 256, 128, 64, 32, 16, 8, 4],
    [x_train.shape[1], 256, 128, 64, 32, 16, 8, 4, 3],
    [x_train.shape[1], 256, 128, 64, 32, 16, 8, 4, 2]
]

BATCH_SIZE = 16
EPOCHS = 1000
USE_LR_SCHEDULER = False
PATIENCE_EARLY_STOPPING = 50

historial = []

print("INICIANDO GRID SEARCH")

for lr in tasas_de_aprendizaje:
    print("\n=============================")
    print(f"  Entrenando con LR = {lr}")
    print("=============================\n")
    for capas in capas_posibles:
        
        modelo = Autoencoder(capas).to(device)

        errores = modelo.fit(x_data=x_train, device=device, lr=lr, batch_size=BATCH_SIZE, num_epochs=EPOCHS, verbose=0,
                             use_lr_scheduler=USE_LR_SCHEDULER, patience_early_stopping=PATIENCE_EARLY_STOPPING)

        error_final = errores[-1]
        
        print(f"Error final = {error_final:.6f}")

        historial.append({
            "lr": lr,
            "capas": capas,
            "error": round(float(error_final), 3),
        })

# =================== MEJOR CONFIGURACIÓN ===================
mejor = min(historial, key=lambda h: h["error"])

print("\nMEJOR CONFIGURACIÓN ENCONTRADA:")
print(f"    Capas: {mejor['capas']}")
print(f"    lr: {mejor['lr']}")
print(f"    Error: {mejor['error']:.6f}")

# =================== GUARDADO ===================

save_grid_search_results(historial, verbose=True)

print("Grid Search finalizado con éxito")