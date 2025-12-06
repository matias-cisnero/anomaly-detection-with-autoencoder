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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}{f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''}\n")
    return device

def estandarizar_columnas_no_binarias(df: pd.DataFrame, mostrar_cols: bool = False) -> pd.DataFrame:

    cols_estandarizar = [col for col in df.columns if df[col].nunique() > 2]
    if mostrar_cols: print("Columnas estandarizadas:", cols_estandarizar)

    df[cols_estandarizar] = (df[cols_estandarizar] - df[cols_estandarizar].mean()) / df[cols_estandarizar].std()
    return df

def estandarizar_columnas_no_binarias_train(df_train: pd.DataFrame, df_test: pd.DataFrame, mostrar_cols: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    cols_estandarizar = [col for col in df_train.columns if df_train[col].nunique() > 2]
    if mostrar_cols: print("Columnas estandarizadas:", cols_estandarizar)

    media = df_train[cols_estandarizar].mean()
    std = df_train[cols_estandarizar].std()

    df_train[cols_estandarizar] = (df_train[cols_estandarizar] - media) / std
    df_test[cols_estandarizar] = (df_test[cols_estandarizar] - media) / std
    
    return df_train, df_test

def crear_conjuntos_proporcionales(df, concepto, test_size=0.2, proporciones=[0.0, 0.10, 0.25]):
    df_train_global, df_test_global = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[concepto]
    )

    x_test = df_test_global.drop(columns=[concepto]).to_numpy()
    y_test = df_test_global[concepto].to_numpy()

    df0 = df_train_global[df_train_global[concepto] == 0]
    df1 = df_train_global[df_train_global[concepto] == 1]

    conjuntos = []

    for p in proporciones:
        n1 = int(len(df0) * p)
        n0 = len(df0) - n1

        # Para evitar pedir más de lo disponible en el dataset
        n1 = min(n1, len(df1))
        n0 = min(n0, len(df0))

        df_mix = pd.concat([
            df0.sample(n0, random_state=42),
            df1.sample(n1, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos.append((x_train, x_test, y_train, y_test))

    return conjuntos

def crear_conjuntos_proporcionales_estandarizados(df, concepto, test_size=0.2, proporciones=[0.0, 0.10, 0.25]):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[concepto]
    )
    df_train_std, df_test_std = estandarizar_columnas_no_binarias_train(df_train, df_test)
    
    x_test = df_test_std.drop(columns=[concepto]).to_numpy()
    y_test = df_test_std[concepto].to_numpy()

    df0 = df_train_std[df_train_std[concepto] == 0]
    df1 = df_train_std[df_train_std[concepto] == 1] 
    
    conjuntos = []

    for p in proporciones:
        n1 = int(len(df0) * p)
        n0 = len(df0) - n1

        # Para evitar pedir más de lo disponible en el dataset
        n1 = min(n1, len(df1))
        n0 = min(n0, len(df0))

        df_mix = pd.concat([
            df0.sample(n0, random_state=42),
            df1.sample(n1, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos.append((x_train, x_test, y_train, y_test))

    return conjuntos

def mostrar_resumen(conjuntos, proporciones=[0.0, 0.10, 0.25]):
    filas = []

    for (X_train, x_test, y_train, y_test), p in zip(conjuntos, proporciones):
        freq_pos = (y_train == 1).sum()
        freq_neg = (y_train == 0).sum()
        total = len(y_train)

        filas.append({
            "Proporción": p,
            "Positivos": freq_pos,
            "Negativos": freq_neg,
            "Total": total,
            "% Positivos": round(freq_pos / total * 100, 2),
            "% Negativos": round(freq_neg / total * 100, 2),
        })

    resumen_df = pd.DataFrame(filas)
    print(resumen_df)

def evaluar_reconstruccion(modelo, original, device, idx_cols_binarias = None):
    reconstruccion = modelo.predict(x=original, device=device)

    if idx_cols_binarias is not None:
        reconstruccion = reconstruccion.copy()
        reconstruccion[:, idx_cols_binarias] = np.round(np.abs(reconstruccion[:, idx_cols_binarias]))
    
    diferencias = np.linalg.norm(original - reconstruccion, axis=1)
    return original, reconstruccion, diferencias

def obtener_epsilon(dif_norm, tipo_epsilon=1):
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

    total = TP + TN + FP + FN

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

def graficar_matriz_confusion(TP, FN, FP, TN):
    matriz = np.array([
        [TP, FN],
        [FP, TN]
    ])

    matriz_norm = matriz / matriz.sum()

    display_labels = ["Anómalo", "Normal"]

    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_norm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    
    # mover etiquetas arriba
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Etiqueta predicha")
    
    plt.show()  

def save_grid_search_results(data, folder="results", verbose=False):
    fecha = datetime.now().strftime("%Y-%m-%dT%H.%M")
    path = os.path.join(folder, f"grid_search_results_{fecha}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, separators=(',', ': '))

    if verbose: print(f"Resultados de grid search guardados en: {path}")

def graficar_espacio_latente(z: np.ndarray, y_labels: np.ndarray = None, target_names: List[str] = ["Clase 0", "Clase 1"]):

    latent_dim = z.shape[1]
    if latent_dim != 2:
        raise ValueError("Solo se puede graficar la capa latente si tiene dimensión 2.")
    
    df_plot = pd.DataFrame({'x': z[:, 0], 'y': z[:, 1]})
    
    if y_labels.size > 0:
        df_plot['clase'] = y_labels.astype(str)
        hue_param = 'clase'
        legend_param = "full"
    else:
        hue_param = None
        legend_param = False

    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        x='x', 
        y='y', 
        data=df_plot, 
        hue=hue_param, 
        palette="viridis",
        legend=legend_param
    )

    if y_labels.size > 0 and target_names:
        plt.legend(title="Clase", labels=target_names)

    plt.xlabel("Dimensión Latente 1")
    plt.ylabel("Dimensión Latente 2")
    plt.grid(True, alpha=0.5)
    plt.show()

def graficar_histograma_errores_reconstruccion(dif_norm, dif_anom, epsilon = None, kde = False, bins = 'auto'):
    plt.figure(figsize=(8,5))

    sns.histplot(dif_norm, bins=bins, kde=kde, color='blue', alpha=0.4, label="Error de reconstrucción (normales)", stat="density")
    sns.histplot(dif_anom, bins=bins, kde=kde, color='red', alpha=0.4, label="Error de reconstrucción (anómalos)", stat="density")

    ax = plt.gca()
    for patch in ax.patches:
        patch.set_edgecolor("none")

    if epsilon is not None:
        plt.axvline(epsilon, color='red', linestyle='--', linewidth=2, label="Umbral (epsilon)")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Error de reconstrucción")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.show()