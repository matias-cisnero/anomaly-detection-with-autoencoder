import pandas as pd
import numpy as np
from models import Autoencoder
from utils import get_device, crear_conjuntos_proporcionales_estandarizados

df = pd.read_csv("data/breast-cancer-wisconsin.csv")

# Quitamos atributos no necesarios
df = df.drop(columns=["id", "Unnamed: 32"])

# Reemplazamos los valores en diagnosis por 0 y 1
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Dividimos nuestro conjuntos de datos y lo estandarizamos respecto a la división de entrenamiento
conjuntos = crear_conjuntos_proporcionales_estandarizados(df, "diagnosis", 0.2)
etiquetas = ["A", "B", "C"]

device = get_device()

LR = 0.001
BATCH_SIZE = 16
EPOCHS = 2000
USE_LR_SCHEDULER = False
PATIENCE_EARLY_STOPPING = 50
SAVE_MODEL = True

for i, conjunto in enumerate(conjuntos):
    x_train, x_test, y_train, y_test = conjunto

    modelo = Autoencoder([x_train.shape[1], 64, 32, 16, 8, 4, 2]).to(device) 
    if i == 0: modelo.summary()

    modelo.fit(x_data=x_train, device=device, lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS, verbose=2, use_lr_scheduler=USE_LR_SCHEDULER)
    if SAVE_MODEL:
        modelo.save(path="models/autoencoder", set_id=etiquetas[i], lr=LR)