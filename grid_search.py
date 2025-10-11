import pandas as pd
import torch
import time
from model import Autoencoder, entrenar_autoencoder
import numpy as np
import json

# --- Carga de datos y configuración del dispositivo ---
df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
atributos = df.drop(columns=["Diabetes_binary"])
x = atributos.to_numpy().astype("float32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo usado: {device}")
if device.type == "cuda":
    print(f"Nombre GPU: {torch.cuda.get_device_name(0)}")

# --- Definición del Grid Search y variables de seguimiento ---
tasas_de_aprendizaje = [0.001, 0.0001, 0.00001]
tamanos_oculta = [16, 12, 8]
tamanos_latente = [2, 4, 6]

mejores_parametros = {}
menor_error = float('inf')

print("INICIANDO GRID SEARCH")

# --- Bucle del Grid Search ---
for lr in tasas_de_aprendizaje:
    for oculta in tamanos_oculta:
        for latente in tamanos_latente:
            print(f"\n--- Probando: lr={lr}, oculta={oculta}, latente={latente}")
            start_time = time.time()

            modelo_prueba = Autoencoder(
                entrada=x.shape[1],
                oculta=oculta,
                latente=latente
            ).to(device)

            _, errores = entrenar_autoencoder(
                modelo=modelo_prueba,
                x_data=x,
                device=device,
                lr=lr,
                batch_size=512,
                epocas=100,
                verbose=False
            )
            
            end_time = time.time()
            error_final = errores[-1]
            print(f"    Error final: {error_final:.6f} | Tiempo: {end_time - start_time:.2f}s")

            if error_final < menor_error:
                menor_error = error_final
                mejores_parametros = {'lr': lr, 'hidden_size': oculta, 'latent_size': latente}
                print(f"¡Nuevo mejor resultado encontrado!")

print("\n========================================")
print(f"Grid Search finalizado")
print(f"Mejor error encontrado: {menor_error:.6f}")
print(f"Mejores parámetros: {mejores_parametros}")

with open('best_parameters.json', 'w') as f:
    json.dump(mejores_parametros, f, indent=4)

print("\nMejores parámetros guardados en 'best_parameters.json'")