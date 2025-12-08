import pandas as pd
import numpy as np
from models import Autoencoder, SAE, CAE
from utils import get_device, set_seed, crear_conjuntos_con_validacion_estandarizados

df = pd.read_csv("data/breast-cancer-wisconsin.csv")

# Quitamos atributos no necesarios
df = df.drop(columns=["id", "Unnamed: 32"])

# Reemplazamos los valores en diagnosis por 0 y 1
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Dividimos nuestro conjuntos de datos y lo estandarizamos respecto a la división de entrenamiento
conjuntos = crear_conjuntos_con_validacion_estandarizados(df, "diagnosis")
etiquetas = ["A", "B", "C"]

x_test, y_test, x_val, y_val, x_val_norm, y_val_norm, conjuntos_train = conjuntos 

device = get_device()

LR = 0.01
BATCH_SIZE = 16
EPOCHS = 1000
USE_LR_SCHEDULER = False
PATIENCE_EARLY_STOPPING = 50
SAVE_MODEL = True
MODEL = Autoencoder

for i, conjunto_train in enumerate(conjuntos_train):
    x_train, y_train = conjunto_train
    set_seed(11)
    
    modelo = MODEL([x_train.shape[1], 32, 16, 8, 4, 2]).to(device)  
    if i == 0: modelo.summary()

    modelo.fit(x_train=x_train, x_val_norm=x_val_norm, device=device, lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS, verbose=2, use_lr_scheduler=USE_LR_SCHEDULER, patience_early_stopping=PATIENCE_EARLY_STOPPING)

    print(modelo.evaluate(x_train, x_test[y_test==0], x_test[y_test==1], device, tipo_epsilon=2, tipo_norma="L2"))

    if SAVE_MODEL:
        modelo.save(path=f"models/{MODEL.__name__}", set_id=etiquetas[i], lr=LR)