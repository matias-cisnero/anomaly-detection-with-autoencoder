import pandas as pd
import numpy as np
from models import Autoencoder, SAE, CAE
from utils import (get_device ,save_grid_search_results, train_val_test_split_scaled, calculate_reconstruction_error, evaluate,
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
x_train, y_train = train_sets[0]
input_size = x_train.shape[1]

# =================== HIPERPARÁMETROS ===================

modelos = [Autoencoder, SAE, CAE]
tasas_de_aprendizaje = [0.001, 0.0001]
capas_posibles = [
    [input_size, 32, 16, 8, 4, 2],
    [input_size, 64, 32, 16, 8, 4, 2],

    [input_size, 32, 16, 8, 4],
    [input_size, 64, 32, 16, 8, 4],

    [input_size, 32, 16, 8],
    [input_size, 64, 32, 16, 8],

    [input_size, 32, 16],
    [input_size, 64, 32, 16],

    [input_size, 16],
    [input_size, 32],
    [input_size, 64]
]

BATCH_SIZE = 16
EPOCHS = 200
PATIENCE_EARLY_STOPPING = 50

historial = []
threshold_names = ["MeanStd", "P95", "Youden"]

print("INICIANDO GRID SEARCH")
print(f"Probando [{len(modelos) * len(tasas_de_aprendizaje) * len(capas_posibles)}] configuraciones")

for ModeloClass in modelos:
    nombre_modelo = ModeloClass.__name__
    print(f"\n==================== MODELO {nombre_modelo} ====================")

    for lr in tasas_de_aprendizaje:
        print(f"\n-- LR = {lr} --")
        
        for capas in capas_posibles:

            model = ModeloClass(capas).to(device)

            errores, _ = model.fit(x_train=x_train, x_val=x_val_norm, device=device, lr=lr, batch_size=BATCH_SIZE, num_epochs=EPOCHS,
                                    verbose=0, patience_early_stopping=PATIENCE_EARLY_STOPPING)

            val_errors = calculate_reconstruction_error(model, x_val, device)

            mean_std_threshold = calculate_mean_std_threshold(val_errors, y_val, 2)
            percentil_95_threshold = calculate_percentil_n_threshold(val_errors, y_val, 95)
            youden_threshold = calculate_youden_threshold(val_errors, y_val)

            evaluaciones = [
                evaluate(val_errors[y_val==0], val_errors[y_val==1], mean_std_threshold),
                evaluate(val_errors[y_val==0], val_errors[y_val==1], percentil_95_threshold),
                evaluate(val_errors[y_val==0], val_errors[y_val==1], youden_threshold) 
            ]
            error_final = errores[-1]

            combined_metrics = {}
            for name, result in zip(threshold_names, evaluaciones):
                for k, v in result.items():
                    combined_metrics[f"{name}_{k}"] = v

            auc_value = evaluaciones[0]['auc'] 
            
            historial.append({
                "modelo": nombre_modelo,
                "epocas": int(len(errores)),
                "capas": str(capas),
                "lr": lr,
                "error": round(float(error_final), 4),
                "auc": auc_value,
                **combined_metrics
            })
            
            print(
                f"Error: {error_final:.6f} | AUC: {auc_value:.4f} "
                f"| MeanStd [Eps: {combined_metrics['MeanStd_epsilon']:.4f}, F1: {combined_metrics['MeanStd_f1_score']:.4f}, Rec: {combined_metrics['MeanStd_recall']:.4f}] "
                f"| P95 [Eps: {combined_metrics['P95_epsilon']:.4f}, F1: {combined_metrics['P95_f1_score']:.4f}, Rec: {combined_metrics['P95_recall']:.4f}] "
                f"| Youden [Eps: {combined_metrics['Youden_epsilon']:.4f}, F1: {combined_metrics['Youden_f1_score']:.4f}, Rec: {combined_metrics['Youden_recall']:.4f}]"
            )

# =================== GUARDADO ===================

save_grid_search_results(historial, verbose=True)
print("Grid Search finalizado con éxito")