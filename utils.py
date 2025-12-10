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
        random_state=11,
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
            df0.sample(n0, random_state=11),
            df1.sample(n1, random_state=11)
        ]).sample(frac=1, random_state=11).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos.append((x_train, x_test, y_train, y_test))

    return conjuntos

def crear_conjuntos_proporcionales_estandarizados(df: pd.DataFrame, concepto: str, test_size: float = 0.2, proporciones: list = [0.0, 0.10, 0.25]):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=11,
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
            df0.sample(n0, random_state=11),
            df1.sample(n1, random_state=11)
        ]).sample(frac=1, random_state=11).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos.append((x_train, x_test, y_train, y_test))

    return conjuntos

def crear_conjuntos_train_val_test(x, y, test_size: float = 0.2, val_size: float = 0.2, random_state= 42):

    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random_state)

    x_train_benign = x_train[y_train == 0]

    scaler = StandardScaler().fit(x_train_benign)

    x_train_scaled = scaler.transform(x_train_benign)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test, scaler

def crear_conjuntos_con_validacion_estandarizados(
    df: pd.DataFrame, concepto: str, test_size: float = 0.2, val_size: float = 0.2, proporciones: list = [0.0, 0.10, 0.25]):
    seed = 11
    df_train, df_temp = train_test_split(
        df,
        test_size=0.40,         # 40% → se divide en val y test
        random_state=seed,
        stratify=df[concepto]
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,         # 50% de 40% → 20%
        random_state=seed,
        stratify=df_temp[concepto]
    )

    df_train_std, df_val_std  = estandarizar_columnas_no_binarias_train(df_train, df_val)
    _, df_test_std = estandarizar_columnas_no_binarias_train(df_train_std, df_test)

    x_test = df_test_std.drop(columns=[concepto]).to_numpy()
    y_test = df_test_std[concepto].to_numpy()

    x_val  = df_val_std.drop(columns=[concepto]).to_numpy()
    y_val  = df_val_std[concepto].to_numpy()

    # Validación solo normales (early stopping)
    df_val_norm = df_val_std[df_val_std[concepto] == 0]
    x_val_norm = df_val_norm.drop(columns=[concepto]).to_numpy()
    y_val_norm = df_val_norm[concepto].to_numpy()

    df0 = df_train_std[df_train_std[concepto] == 0]
    df1 = df_train_std[df_train_std[concepto] == 1]

    conjuntos_train = []

    for p in proporciones:
        n1 = int(len(df0) * p)
        n0 = len(df0) - n1

        # Para evitar pedir más de lo disponible en el dataset
        n1 = min(n1, len(df1))
        n0 = min(n0, len(df0))

        df_mix = pd.concat([
            df0.sample(n0, random_state=seed),
            df1.sample(n1, random_state=seed)
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos_train.append((x_train, y_train))

    conjuntos = [x_test, y_test, x_val, y_val, x_val_norm, y_val_norm, conjuntos_train]

    return conjuntos 

def crear_conjuntos_con_validacion(
    df: pd.DataFrame, concepto: str, test_size: float = 0.2, val_size: float = 0.2, proporciones: list = [0.0, 0.10, 0.25]):
    seed = 11
    df_train, df_temp = train_test_split(
        df,
        test_size=0.40,         # 40% → se divide en val y test
        random_state=seed,
        stratify=df[concepto]
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,         # 50% de 40% → 20%
        random_state=seed,
        stratify=df_temp[concepto]
    )

    df_train_std, df_val_std  = estandarizar_columnas_no_binarias_train(df_train, df_val)
    _, df_test_std = estandarizar_columnas_no_binarias_train(df_train_std, df_test)

    x_test = df_test_std.drop(columns=[concepto]).to_numpy()
    y_test = df_test_std[concepto].to_numpy()

    x_val  = df_val_std.drop(columns=[concepto]).to_numpy()
    y_val  = df_val_std[concepto].to_numpy()

    # Validación solo normales (early stopping)
    df_val_norm = df_val_std[df_val_std[concepto] == 0]
    x_val_norm = df_val_norm.drop(columns=[concepto]).to_numpy()
    y_val_norm = df_val_norm[concepto].to_numpy()

    df0 = df_train_std[df_train_std[concepto] == 0]
    df1 = df_train_std[df_train_std[concepto] == 1]

    conjuntos_train = []

    for p in proporciones:
        n1 = int(len(df0) * p)
        n0 = len(df0) - n1

        # Para evitar pedir más de lo disponible en el dataset
        n1 = min(n1, len(df1))
        n0 = min(n0, len(df0))

        df_mix = pd.concat([
            df0.sample(n0, random_state=seed),
            df1.sample(n1, random_state=seed)
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)

        x_train = df_mix.drop(columns=[concepto]).to_numpy()
        y_train = df_mix[concepto].to_numpy()

        conjuntos_train.append((x_train, y_train))

    conjuntos = [x_test, y_test, x_val, y_val, x_val_norm, y_val_norm, conjuntos_train]

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

def mostrar_resumen_validacion(conjuntos, proporciones=[0.0, 0.10, 0.25]):
    x_test, y_test, x_val, y_val, x_val_norm, y_val_norm, conjuntos_train = conjuntos

    filas = []

    filas.append({
        "conjunto": "test",
        "proporcion": None,
        "normales": int((y_test == 0).sum()),
        "anomalías": int((y_test == 1).sum()),
        "total": len(y_test),
    })

    filas.append({
        "conjunto": "validacion",
        "proporcion": None,
        "normales": int((y_val == 0).sum()),
        "anomalías": int((y_val == 1).sum()),
        "total": len(y_val),
    })

    filas.append({
        "conjunto": "validacion_norm",
        "proporcion": None,
        "normales": int((y_val_norm == 0).sum()),
        "anomalías": int((y_val_norm == 1).sum()),
        "total": len(y_val_norm),
    })

    for (x_train, y_train), p in zip(conjuntos_train, proporciones):
        filas.append({
            "conjunto": "train",
            "proporcion": p,
            "normales": int((y_train == 0).sum()),
            "anomalías": int((y_train == 1).sum()),
            "total": len(y_train),
        })

    df = pd.DataFrame(filas)
    print(df)
    return df

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

def graficar_matriz_confusion(TP: int, FN: int, TN: int, FP: int, norm: bool = False):
    
    matriz = np.array([[TP, FN], [FP, TN]])
    if norm: matriz = matriz / matriz.sum()
    
    sns.set_theme(style="white", context="notebook", font_scale=1.1)
    plt.figure(figsize=(6, 6))
    
    # Heatmap configurado en una sola línea
    ax = sns.heatmap(matriz, annot=True, fmt=".1%" if norm else "d", cmap="Blues", cbar=False, square=True, linewidths=1.5, linecolor='white', xticklabels=["Anómalo", "Normal"], yticklabels=["Anómalo", "Normal"], annot_kws={"size": 14})
    
    # Mover etiquetas al tope
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Etiquetas de ejes limpias
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Etiqueta Predicha")
    
    plt.tight_layout()
    plt.show()

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

def graficar_histograma_errores_reconstruccion(dif_norm, dif_anom, epsilon=None, kde=True, bins=50, stat="density"):
    
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    plt.figure(figsize=(8, 5))
    
    # Colores
    c_norm, c_anom, c_thresh = '#2E86C1', '#E74C3C', "#FF0000"
    
    # Histogramas en una sola línea
    sns.histplot(dif_norm, bins=bins, kde=kde, color=c_norm, alpha=0.5, label="Normales", stat=stat, linewidth=0, kde_kws={'linewidth': 2})
    sns.histplot(dif_anom, bins=bins, kde=kde, color=c_anom, alpha=0.5, label="Anómalos", stat=stat, linewidth=0, kde_kws={'linewidth': 2})
    
    if epsilon is not None:
        plt.axvline(epsilon, color=c_thresh, linestyle='--', linewidth=2.5, label=f"Umbral ($\epsilon = {epsilon:.2f}$)")
    
    # Etiquetas sin negrita y sin título
    plt.xlabel("Error de Reconstrucción (MSE)")
    plt.ylabel("Densidad" if stat == "density" else "Frecuencia")
    
    plt.legend(frameon=True, fancybox=True, framealpha=0.95, loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.show()