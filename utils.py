import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict
import pandas as pd
import os
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}{f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}\n")
    return device

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

def calculate_reconstruction_error(modelo, original, device, metric_type: str = "MSE"):
    reconstruction = modelo.predict(input=original, device=device)
    error = original - reconstruction

    if metric_type == "MSE":
        return np.mean(error**2, axis=1)
    elif metric_type == "L1":
        return np.linalg.norm(error, ord=1, axis=1)
    elif metric_type == "L2":
        return np.linalg.norm(error, ord=2, axis=1)
    elif metric_type == "Linf":
        return np.linalg.norm(error, ord=np.inf, axis=1)
    else:
        raise ValueError("tipo norma debe ser: L1, L2, Linf, MSE")

def get_confusion_matrix(error_norm: np.ndarray, error_anom: np.ndarray, epsilon: float):
    # Normales
    FP = np.sum(error_norm > epsilon)
    TN = np.sum(error_norm <= epsilon)
    # Anómalas
    TP = np.sum(error_anom > epsilon)
    FN = np.sum(error_anom <= epsilon)

    return int(TP), int(FN), int(TN), int(FP)

def calculate_metrics(TP: int, FN: int, TN: int, FP: int) -> dict:
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

def calculate_auc(error_norm: np.ndarray, error_anom: np.ndarray) -> float:
    scores = np.concatenate([error_norm, error_anom])
    labels = np.concatenate([np.zeros(len(error_norm)), np.ones(len(error_anom))])

    auc = roc_auc_score(labels, scores)
    return float(auc)

def evaluate(error_norm: np.ndarray, error_anom: np.ndarray, epsilon: float) -> dict:

    TP, FN, TN, FP = get_confusion_matrix(error_norm, error_anom, epsilon)

    metrics = calculate_metrics(TP, FN, TN, FP)
    auc = calculate_auc(error_norm, error_anom)

    return {
        "epsilon": round(float(epsilon), 4),
        "conf_matrix": str({"TP": TP, "FN": FN, "FP": FP, "TN": TN}),
        **metrics,
        "auc": round(float(auc), 4)
    }

def calculate_mean_std_threshold(errors: np.ndarray, labels: np.ndarray, factor: int = 2) -> float:
    benign_errors = errors[labels == 0]
    return np.mean(benign_errors) + (factor * np.std(benign_errors))

def calculate_percentil_n_threshold(errors: np.ndarray, labels: np.ndarray, n: int = 95) -> float:
    benign_errors = errors[labels == 0]
    return np.percentile(benign_errors, n)

def calculate_youden_threshold(errors: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, errors)
    youden_idx = np.argmax(tpr - fpr)

    youden_threshold = thresholds[youden_idx]
    return youden_threshold

def save_grid_search_results(data, folder="results", verbose=False):
    fecha = datetime.now().strftime("%Y-%m-%dT%H.%M")
    path = os.path.join(folder, f"grid_search_results_{fecha}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    if verbose: print(f"Resultados de grid search guardados en: {path}")