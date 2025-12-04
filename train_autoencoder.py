import pandas as pd
import numpy as np
from models import Autoencoder
from utils import get_device, crear_conjuntos_proporcionales, mostrar_resumen, estandarizar_columnas_no_binarias

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

LR = 0.001
BATCH_SIZE = 16
EPOCHS = 1000
USE_LR_SCHEDULER = False
PATIENCE_EARLY_STOPPING = 50
SAVE_MODEL = True

modelo = Autoencoder([x_train.shape[1], 128, 64, 32, 16, 8, 4, 2]).to(device) 
modelo.summary()
modelo.fit(x_data=x_train, device=device, lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS, verbose=1, use_lr_scheduler=USE_LR_SCHEDULER)
if SAVE_MODEL:
    modelo.save(path="models/autoencoder", set_id=0, lr=LR)
