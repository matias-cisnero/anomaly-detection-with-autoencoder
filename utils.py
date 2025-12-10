import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}{f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}\n")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Para reproducibilidad en CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_zscore_scaling(df: pd.DataFrame, target_col_name: str, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    x_scaled = df.drop(columns=[target_col_name])
    y = df[target_col_name]

    x_scaled = (x_scaled - means) / stds
    df_scaled = pd.concat([x_scaled, y], axis=1)
    return df_scaled

def train_val_test_split_scaled(
    df: pd.DataFrame, target_col_name: str, test_size: float = 0.2, val_size: float = 0.2, proportions: list = [0.0, 0.10, 0.25], random_state = 22):

    val_size = val_size / (1 - test_size)

    df_train_val, df_test = train_test_split(df, test_size=test_size, stratify=df[target_col_name], random_state=random_state)
    df_train, df_val = train_test_split(df_train_val, test_size=val_size, stratify=df_train_val[target_col_name], random_state=random_state)

    train_means = df_train.drop(columns=[target_col_name]).mean()
    train_stds = df_train.drop(columns=[target_col_name]).std()

    df_train_scaled = apply_zscore_scaling(df_train, target_col_name, train_means, train_stds)
    df_val_scaled  = apply_zscore_scaling(df_val, target_col_name, train_means, train_stds)
    df_test_scaled = apply_zscore_scaling(df_test, target_col_name, train_means, train_stds)

    x_test = df_test_scaled.drop(columns=[target_col_name]).to_numpy()
    y_test = df_test_scaled[target_col_name].to_numpy()

    x_val  = df_val_scaled.drop(columns=[target_col_name]).to_numpy()
    y_val  = df_val_scaled[target_col_name].to_numpy()

    df0 = df_train_scaled[df_train_scaled[target_col_name] == 0]
    df1 = df_train_scaled[df_train_scaled[target_col_name] == 1]

    train_sets = []

    for p in proportions:
        n1 = int(len(df0) * p)
        n0 = len(df0) - n1

        # Para evitar pedir más de lo disponible en el dataset
        n1 = min(n1, len(df1))
        n0 = min(n0, len(df0))

        df_mix = pd.concat([
            df0.sample(n0, random_state=random_state),
            df1.sample(n1, random_state=random_state)
        ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        x_train = df_mix.drop(columns=[target_col_name]).to_numpy()
        y_train = df_mix[target_col_name].to_numpy()

        train_sets.append((x_train, y_train))

    return [x_test, y_test, x_val, y_val, train_sets]

def print_summary_split(sets, proportions=[0.0, 0.10, 0.25]):
    x_test, y_test, x_val, y_val, train_sets = sets

    summary_rows = []

    summary_rows.append({
        "conjunto": "test",
        "proporcion": None,
        "normales": int((y_test == 0).sum()),
        "anomalías": int((y_test == 1).sum()),
        "total": len(y_test),
    })

    summary_rows.append({
        "conjunto": "validacion",
        "proporcion": None,
        "normales": int((y_val == 0).sum()),
        "anomalías": int((y_val == 1).sum()),
        "total": len(y_val),
    })

    for (x_train, y_train), p in zip(train_sets, proportions):
        summary_rows.append({
            "conjunto": "train",
            "proporcion": p,
            "normales": int((y_train == 0).sum()),
            "anomalías": int((y_train == 1).sum()),
            "total": len(y_train),
        })

    df = pd.DataFrame(summary_rows)
    print(df)

def evaluar_reconstruccion(modelo, original, device, tipo_norma="L2"):
    reconstruccion = modelo.predict(input=original, device=device)
    
    diferencias = calcular_error_reconstruccion(original, reconstruccion, tipo_norma=tipo_norma)
    return reconstruccion, diferencias

def calcular_error_reconstruccion(original, reconstruccion, tipo_norma="L2"):
    diff = original - reconstruccion

    if tipo_norma == "L1":
        return np.linalg.norm(diff, ord=1, axis=1)
    elif tipo_norma == "L2":
        return np.linalg.norm(diff, ord=2, axis=1)
    elif tipo_norma == "Linf":
        return np.linalg.norm(diff, ord=np.inf, axis=1)
    elif tipo_norma == "MSE":
        return np.mean(diff**2, axis=1)
    else:
        raise ValueError("tipo norma debe ser: L1, L2, Linf, MSE")

def obtener_epsilon(modelo, x_train, device, tipo_epsilon=1, tipo_norma="L2"):
    _, dif_norm = evaluar_reconstruccion(modelo, x_train, device, tipo_norma)

    # 0: Máximo (umbral más permisivo)
    if tipo_epsilon == 0:
        return np.max(dif_norm)

    # 1: Media
    elif tipo_epsilon == 1:
        return np.mean(dif_norm)

    # 2: Media + 2*STD
    elif tipo_epsilon == 2:
        return np.mean(dif_norm) + 2 * np.std(dif_norm)

    # 3: Percentil 95
    elif tipo_epsilon == 3:
        return np.percentile(dif_norm, 95)

    # 4: Mediana + IQR
    elif tipo_epsilon == 4:
        q1 = np.percentile(dif_norm, 25)
        q3 = np.percentile(dif_norm, 75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr

    # 5: Mediana + MAD
    elif tipo_epsilon == 5:
        med = np.median(dif_norm)
        mad = np.median(np.abs(dif_norm - med))
        return med + 3 * mad

    # 6: Media + 3*STD
    elif tipo_epsilon == 6:
        return np.mean(dif_norm) + 3 * np.std(dif_norm)

    else:
        return 0

def obtener_matriz_confusion(dif_norm, dif_anom, epsilon):
    # Normales
    FP = np.sum(dif_norm > epsilon)
    TN = np.sum(dif_norm <= epsilon)

    # Anómalas
    TP = np.sum(dif_anom > epsilon)
    FN = np.sum(dif_anom <= epsilon)

    return int(TP), int(FN), int(TN), int(FP)

def obtener_metricas(TP: int, FN: int, TN: int, FP: int):
    total = TP + FN + TN + FP

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    f1_score = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "f1_score": round(float(f1_score), 4)
    }

def calcular_auc(dif_norm, dif_anom):
    scores = np.concatenate([dif_norm, dif_anom])
    labels = np.concatenate([np.zeros(len(dif_norm)), np.ones(len(dif_anom))])

    auc = roc_auc_score(labels, scores)
    return auc

def save_grid_search_results(data, folder="results", verbose=False):
    fecha = datetime.now().strftime("%Y-%m-%dT%H.%M")
    path = os.path.join(folder, f"grid_search_results_{fecha}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    if verbose: print(f"Resultados de grid search guardados en: {path}")