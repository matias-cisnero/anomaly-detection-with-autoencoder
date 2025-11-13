import pandas as pd
import time
import numpy as np
import json
from models import Autoencoder
from utils import estandarizar_columnas, crear_datasets_proporcionales, get_device

df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2015.csv")

# "Binarizamos" los datos, eliminando registros de pacientes con prediabetes
df = df[df["Diabetes_012"] != 1]
df["Diabetes_012"] = df["Diabetes_012"].replace(2, 1)

# Estandarizamos las columnas no binarias
df = estandarizar_columnas(df=df, cols_estandarizar=["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income"])
list_x_train, list_y_train, list_x_test, list_y_test, resumen_df = crear_datasets_proporcionales(df, "Diabetes_012")
device = get_device()

x = list_x_train[0] # <--- Tomo el conjunto de x que tiene 0% de personas con diabetes

tasas_de_aprendizaje =  [0.001] #[0.001, 0.0001, 0.00001]
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

            modelo = Autoencoder([x.shape[1], oculta, latente]).to(device)

            errores = modelo.fit(x_data=x, device=device, lr=lr, batch_size=512, epocas=100, verbose=1)
            
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