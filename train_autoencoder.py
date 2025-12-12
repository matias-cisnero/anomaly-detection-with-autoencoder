import pandas as pd
import numpy as np
from models import Autoencoder, SAE, CAE
from utils import (get_device, train_val_test_split_scaled, calculate_reconstruction_error, evaluate,
                   calculate_mean_std_threshold, calculate_percentil_n_threshold, calculate_youden_threshold)

# =================== CARGA Y PREPROCESAMIENTO ===================
df = pd.read_csv("data/breast-cancer-wisconsin.csv")

# Quitamos atributos no necesarios
df = df.drop(columns=["id", "Unnamed: 32"])

# Reemplazamos los valores en diagnosis por 0 y 1
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Dividimos nuestro conjuntos de datos y lo estandarizamos respecto a la división de entrenamiento
x_test, y_test, x_val, y_val, train_sets = train_val_test_split_scaled(df, "diagnosis")
etiquetas = ["A", "B", "C"]
x_val_norm = x_val[y_val==0]

device = get_device()

LR = 0.001
BATCH_SIZE = 16
EPOCHS = 200
VERBOSE = 2
PATIENCE_EARLY_STOPPING = 20
SAVE_MODEL = True
METRIC_TYPE = "MSE"
MODEL = CAE

for i, train_set in enumerate(train_sets):
    x_train, y_train = train_set
    
    modelo = MODEL([x_train.shape[1], 32, 16]).to(device)  
    if i == 0: modelo.summary()

    modelo.fit(x_train=x_train, x_val=x_val_norm, device=device, lr=LR, batch_size=BATCH_SIZE, num_epochs=EPOCHS,
               verbose=VERBOSE, patience_early_stopping=PATIENCE_EARLY_STOPPING)
    
    norm_errors = calculate_reconstruction_error(modelo, x_test[y_test==0], device, metric_type=METRIC_TYPE)
    anom_errors = calculate_reconstruction_error(modelo, x_test[y_test==1], device, metric_type=METRIC_TYPE)
    val_errors = calculate_reconstruction_error(modelo, x_val, device, metric_type=METRIC_TYPE)

    eps_mean = calculate_mean_std_threshold(val_errors, y_val, 0)
    eps_mean_2std = calculate_mean_std_threshold(val_errors, y_val, 2)
    eps_mean_3std = calculate_mean_std_threshold(val_errors, y_val, 3)
    eps_p95 = calculate_percentil_n_threshold(val_errors, y_val, 95)
    eps_youden = calculate_youden_threshold(val_errors, y_val)

    eval = evaluate(norm_errors, anom_errors, eps_youden)
    print(eval)

    if SAVE_MODEL:
        modelo.save(path=f"models/{MODEL.__name__}", set_id=etiquetas[i], lr=LR)