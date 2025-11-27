import pandas as pd
import time
import numpy as np
import json
from models import Autoencoder
from utils import estandarizar_columnas_no_binarias, crear_datasets_proporcionales, get_device

# =================== CARGA Y PREPROCESAMIENTO ===================
df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# "Binarizamos" los datos, eliminando registros de pacientes con prediabetes
df = df[df["Diabetes_012"] != 1]
df["Diabetes_012"] = df["Diabetes_012"].replace(2, 1)

# Estandarizamos las columnas no binarias
df = estandarizar_columnas_no_binarias(df)
list_x_train, list_y_train, list_x_test, list_y_test, resumen_df = crear_datasets_proporcionales(df, "Diabetes_012")
device = get_device()

x = list_x_train[0] # <--- Tomo el conjunto de x que tiene 0% de personas con diabetes

# =================== HIPERPARÁMETROS ===================
tasas_de_aprendizaje = [0.001]
capas_posibles = [
    [x.shape[1], 32, 16, 8, 4],
    [x.shape[1], 32, 16, 8, 4, 3],
    [x.shape[1], 32, 16, 8, 4, 2],
    [x.shape[1], 16, 8, 4],
    [x.shape[1], 16, 8, 4, 3],
    [x.shape[1], 16, 8, 4, 2]
]

BATCH_SIZE = 512
EPOCHS = 100

historial = []

print("INICIANDO GRID SEARCH")

for lr in tasas_de_aprendizaje:
    for capas in capas_posibles:
        
        modelo = Autoencoder(capas).to(device)

        start_time = time.time()
        errores = modelo.fit(x_data=x, device=device, lr=lr, batch_size=BATCH_SIZE, num_epochs=EPOCHS, verbose=0)
        end_time = time.time()

        error_final = errores[-1]
        duracion = end_time - start_time
        
        print(f"Error final = {error_final:.6f} | Tiempo = {duracion:.2f}s")

        historial.append({
            "lr": lr,
            "capas": capas,
            "error": round(float(error_final), 3),
            "tiempo": round(duracion, 3) 
        })

# =================== MEJOR CONFIGURACIÓN ===================
mejor = min(historial, key=lambda h: h["error"])

print("\nMEJOR CONFIGURACIÓN ENCONTRADA:")
print(f"    Capas: {mejor['capas']}")
print(f"    lr: {mejor['lr']}")
print(f"    Error: {mejor['error']:.6f}")
print(f"    Tiempo: {mejor['tiempo']:.2f}s")

# =================== GUARDADO ===================
with open("grid_search_results.json", "w") as f:
    json.dump(historial, f, indent=4, separators=(',', ': '))

print("\nResultados guardados en 'grid_search_results.json'")
print("Grid Search finalizado con éxito")