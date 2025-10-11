import pandas as pd
import torch
from model import Autoencoder, entrenar_autoencoder
import numpy as np
import json
import os

# --- Carga de datos y configuración del dispositivo ---
df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
atributos = df.drop(columns=["Diabetes_binary"])
x = atributos.to_numpy().astype("float32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo usado: {device}")
if device.type == "cuda":
    print(f"Nombre GPU: {torch.cuda.get_device_name(0)}")

# --- Cargar los mejores parámetros desde el archivo JSON ---
with open('best_parameters.json', 'r') as f:
    mejores_parametros = json.load(f)

# --- Cargar modelo si existe ---
MODEL_PATH = "autoencoder.pth"

modelo = Autoencoder(
    entrada= x.shape[1],
    oculta=mejores_parametros['hidden_size'],
    latente=mejores_parametros['latent_size']
).to(device)

if os.path.exists(MODEL_PATH):
    print(f"\n--- Cargando modelo existente desde '{MODEL_PATH}' ---")
    modelo.load_state_dict(torch.load(MODEL_PATH))
else:
    print("\n--- No se encontró un modelo guardado, se entrenará desde cero ---")

# --- Entrenamiento ---
print("\n--- Entrenando modelo final con los mejores parámetros ---")
print(mejores_parametros)
modelo, _ = entrenar_autoencoder(
    modelo=modelo,
    x_data=x,
    device=device,
    lr=mejores_parametros['lr'],
    batch_size=512,
    epocas=500,
    verbose=True
)

# --- Guardar el estado del modelo actualizado ---
torch.save(modelo.state_dict(), MODEL_PATH)
print(f"\nModelo guardado en '{MODEL_PATH}'")

# --- Predicción del modelo (para pruebas) ---
print("\n--- Ejemplo de Predicción con el Modelo Final ---")
ejemplo = torch.tensor(x[:5], dtype=torch.float32).to(device)
modelo.eval()
with torch.no_grad():
    reconstruido = modelo(ejemplo)

print("\nEntrada Original:")
print(ejemplo.cpu().numpy())
print("\nSalida Reconstruida:")
with np.printoptions(precision=3, suppress=True):
    print(reconstruido.cpu().numpy())
    print("\nSalida Reconstruida Redondeada:")
print(np.round(reconstruido.cpu().numpy()))