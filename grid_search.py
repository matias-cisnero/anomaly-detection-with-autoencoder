import pandas as pd
import numpy as np
from models import Autoencoder, SAE, CAE, VAE
from utils import get_device, set_seed ,save_grid_search_results, crear_conjuntos_con_validacion_estandarizados

# =================== CARGA Y PREPROCESAMIENTO ===================
df = pd.read_csv("data/breast-cancer-wisconsin.csv")

# Quitamos atributos no necesarios
df = df.drop(columns=["id", "Unnamed: 32"])

# Reemplazamos los valores en diagnosis por 0 y 1
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Dividimos nuestros conjuntos de datos
conjuntos = crear_conjuntos_con_validacion_estandarizados(df, "diagnosis")
input_size = 30

x_test, y_test, x_val, y_val, x_val_norm, y_val_norm, conjuntos_train = conjuntos
x_train, y_train = conjuntos_train[0]

device = get_device()

# =================== HIPERPARÁMETROS ===================

modelos = [Autoencoder, SAE, CAE]
tasas_de_aprendizaje = [0.001, 0.0001]
capas_posibles = [
    [input_size, 32, 16, 8, 4, 2],
    [input_size, 64, 32, 16, 8, 4, 2],
    [input_size, 128, 64, 32, 16, 8, 4, 2],
    [input_size, 256, 128, 64, 32, 16, 8, 4, 2],

    [input_size, 64, 32, 16, 8],
    [input_size, 128, 64, 32, 16],
    [input_size, 256, 128, 64, 32],

    [input_size, 64, 32],
    [input_size, 64],
    [input_size, 32],

    [input_size, 128, 32],
    [input_size, 64, 16]
]

BATCH_SIZE = 16
EPOCHS = 1000
USE_LR_SCHEDULER = False
PATIENCE_EARLY_STOPPING = 50

historial = []

print("INICIANDO GRID SEARCH")
print(f"Probando [{len(modelos) * len(tasas_de_aprendizaje) * len(capas_posibles)}] configuraciones")

for ModeloClass in modelos:
    nombre_modelo = ModeloClass.__name__
    print(f"\n==================== MODELO {nombre_modelo} ====================")

    for lr in tasas_de_aprendizaje:
        print(f"\n-- LR = {lr} --")
        
        for capas in capas_posibles:
            set_seed(11)

            modelo = ModeloClass(capas).to(device)

            errores, _ = modelo.fit(x_train=x_train, x_val_norm=x_val_norm, device=device, lr=lr, batch_size=BATCH_SIZE, num_epochs=EPOCHS,
                                    verbose=0, use_lr_scheduler=USE_LR_SCHEDULER, patience_early_stopping=PATIENCE_EARLY_STOPPING)

            evaluacion = modelo.evaluate(x_train, x_val[y_val==0], x_val[y_val==1], device, tipo_epsilon=2, tipo_norma="L2")
            error_final = errores[-1]

            historial.append({
                    "modelo": nombre_modelo,
                    "conjunto": "A",
                    "capas": str(capas), # una sola línea
                    "lr": lr,
                    "error": round(error_final, 4),
                    **evaluacion
            })
            print(f"error = {error_final:.6f} | conf_matrix = {evaluacion['conf_matrix']} | accuracy = {evaluacion['accuracy']} | recall = {evaluacion['recall']} | f1_score = {evaluacion['f1_score']} | auc = {evaluacion['auc']}")

# =================== MEJORES MÉTRICAS ===================
best_recall = max(historial, key=lambda h: h.get("recall", -1)).get("recall", None)
best_precision = max(historial, key=lambda h: h.get("precision", -1)).get("precision", None)
best_f1 = max(historial, key=lambda h: h.get("f1_score", -1)).get("f1_score", None)

print("\nMEJORES MÉTRICAS ENCONTRADAS:")
print(f"    Recall: {best_recall}")
print(f"    Precision: {best_precision}")
print(f"    F1-score: {best_f1}")

# =================== GUARDADO ===================

save_grid_search_results(historial, verbose=True)

print("Grid Search finalizado con éxito")