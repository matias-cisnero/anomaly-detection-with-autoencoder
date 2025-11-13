import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}{f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}\n")
    return device

def estandarizar_columnas(df: pd.DataFrame, cols_estandarizar: List[str]) -> pd.DataFrame:
    df[cols_estandarizar] = (df[cols_estandarizar] - df[cols_estandarizar].mean()) / df[cols_estandarizar].std()
    return df

def crear_datasets_proporcionales(
    df: pd.DataFrame,
    concepto: str,
    proporciones: List[float] = [0.0, 0.10, 0.25, 0.50]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    
    df = df.copy()

    # Separar clases
    df_positivos = df[df[concepto] == 1.0]
    df_negativos = df[df[concepto] == 0.0]
    total_positivos = len(df_positivos)

    # Listas de salida
    list_x_train, list_y_train = [], []
    list_x_test, list_y_test = [], []
    resumen_frecuencias = []

    for p in proporciones:
        tam_total = total_positivos * 2
        n_positivos = int(p * tam_total)
        n_negativos = tam_total - n_positivos

        # Muestras para entrenamiento
        sample_pos = df_positivos.sample(n=n_positivos, random_state=42)
        sample_neg = df_negativos.sample(n=n_negativos, random_state=42)

        df_train = pd.concat([sample_pos, sample_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Filas restantes (test)
        usados_idx = set(sample_pos.index).union(sample_neg.index)
        df_test = df.loc[~df.index.isin(usados_idx)].reset_index(drop=True)

        # Frecuencias del conjunto de entrenamiento
        conteo = df_train[concepto].value_counts()
        total = len(df_train)
        freq_pos = conteo.get(1.0, 0)
        freq_neg = conteo.get(0.0, 0)

        resumen_frecuencias.append({
            "Proporción": p,
            "Positivos": freq_pos,
            "Negativos": freq_neg,
            "Total": total,
            "% Positivos": round((freq_pos / total) * 100, 2),
            "% Negativos": round((freq_neg / total) * 100, 2)
        })

        # Convertir a arrays NumPy (X e y)
        x_train = df_train.drop(columns=[concepto]).to_numpy()
        y_train = df_train[concepto].to_numpy()

        x_test = df_test.drop(columns=[concepto]).to_numpy()
        y_test = df_test[concepto].to_numpy()

        list_x_train.append(x_train)
        list_y_train.append(y_train)
        list_x_test.append(x_test)
        list_y_test.append(y_test)

    resumen_df = pd.DataFrame(resumen_frecuencias)
    return list_x_train, list_y_train, list_x_test, list_y_test, resumen_df

def evaluar_reconstruccion(modelo, original, device, redondear=True):
    reconstruccion = modelo.predict(x=original, device=device)

    if redondear:
        reconstruccion = np.round(np.abs(reconstruccion))
    
    diferencias = np.linalg.norm(original - reconstruccion, axis=1)
    return original, reconstruccion, diferencias

def evaluar_anomalias(modelo, x, y, device, tipo_epsilon=1, redondear=True):
    """
    Calcula FP, FN, TP y TN evaluando la reconstrucción del autoencoder.
    tipo_epsilon:
        0 → max(diferencias normales)
        1 → media(diferencias normales)
        2 → media + 2*std (por defecto)
    """
    x_norm = x[y == 0]
    x_anom = x[y == 1]

    _, _, dif_norm = evaluar_reconstruccion(modelo, x_norm, device, redondear)
    _, _, dif_anom = evaluar_reconstruccion(modelo, x_anom, device, redondear)

    if tipo_epsilon == 0:
        epsilon = np.max(dif_norm)
    elif tipo_epsilon == 1:
        epsilon = np.mean(dif_norm)
    else:
        epsilon = np.mean(dif_norm) + 2 * np.std(dif_norm)

    FP = np.sum(dif_norm > epsilon)
    TN = np.sum(dif_norm <= epsilon)

    TP = np.sum(dif_anom > epsilon)
    FN = np.sum(dif_anom <= epsilon)

    return {"TP": int(TP), "FN": int(FN), "TN": int(TN), "FP": int(FP)}

def obtener_metricas(TP: int, FN: int, TN: int, FP: int):

    total = TP + TN + FP + FN
    matriz_confusion = np.array([[TN, FP],
                                 [FN, TP]], dtype=float)
    matriz_confusion_pct = np.round(100 * matriz_confusion / total, 2)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP )
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "matriz_confusion": matriz_confusion.astype(int),
        "matriz_confusion_pct": matriz_confusion_pct,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    }