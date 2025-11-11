import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd

class Autoencoder(nn.Module):
    def __init__(self, entrada: int, oculta: int = 16, latente: int = 2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(entrada, oculta),
            nn.GELU(),
            nn.Linear(oculta, latente),
            #nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latente, oculta),
            nn.GELU(),
            nn.Linear(oculta, entrada),
            #nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder2(nn.Module): # oculta1=44
    def __init__(self, entrada: int, oculta1: int = 32, oculta2: int = 16, oculta3: int = 8, oculta4: int = 4, latente: int = 2):
        super(Autoencoder2, self).__init__()

        # Encoder: entrada → oculta1 → oculta2 → latente
        self.encoder = nn.Sequential(
            nn.Linear(entrada, oculta1),
            nn.GELU(),
            nn.Linear(oculta1, oculta2),
            nn.GELU(),
            nn.Linear(oculta2, oculta3),
            nn.GELU(),
            nn.Linear(oculta3, oculta4),
            nn.GELU(),
            nn.Linear(oculta4, latente)
        )

        # Decoder: latente → oculta2 → oculta1 → salida
        self.decoder = nn.Sequential(
            nn.Linear(latente, oculta4),
            nn.GELU(),
            nn.Linear(oculta4, oculta3),
            nn.GELU(),
            nn.Linear(oculta3, oculta2),
            nn.GELU(),
            nn.Linear(oculta2, oculta1),
            nn.GELU(),
            nn.Linear(oculta1, entrada)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def entrenar_autoencoder(
    modelo: nn.Module,
    x_data: np.ndarray,
    device: torch.device,
    lr: float,
    batch_size: int,
    epocas: int,
    verbose: bool = True
) -> Tuple[nn.Module, List[float]]:

    dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterio = nn.MSELoss()
    optimizador = optim.Adam(modelo.parameters(), lr=lr)
    errores = []

    for epoca in range(epocas):
        epoca_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            optimizador.zero_grad()
            salida = modelo(batch_x)
            perdida = criterio(salida, batch_x)
            perdida.backward()
            optimizador.step()
            epoca_loss += perdida.item() * batch_x.size(0)

        epoca_loss /= len(dataset)
        errores.append(epoca_loss)
        
        if verbose in (1, 2) and (epoca + 1) % 25 == 0:
            print(f"    Época {epoca+1}/{epocas}, Error medio: {epoca_loss:.6f}")
    
    if verbose == 2:
      plt.plot(errores)
      #plt.title(f"Curva de Aprendizaje (lr={lr})")
      plt.xlabel("Época")
      plt.ylabel("Error Medio")
      plt.grid(True)
      plt.show()

    return modelo, errores

def cargar_modelo(ruta: str) -> torch.nn.Module:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = torch.load(ruta, map_location=dispositivo)
    modelo.to(dispositivo)
    modelo.eval()

    print(f"Modelo cargado correctamente en {dispositivo}.")
    return modelo

def estandarizar_columnas(df: pd.DataFrame, cols_estandarizar: List[str]) -> pd.DataFrame:
    df[cols_estandarizar] = (df[cols_estandarizar] - df[cols_estandarizar].mean()) / df[cols_estandarizar].std()
    return df

def crear_datasets_proporcionales(
    df: pd.DataFrame,
    concepto: str,
    proporciones: List[float] = [0.0, 0.10, 0.25, 0.50],
    cols_estandarizar: List[str] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], pd.DataFrame]:
    
    df = df.copy()

    if cols_estandarizar != None:
        df = estandarizar_columnas(df, cols_estandarizar)

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

def obtener_error(x_test: np.ndarray) -> float:
    pass

def evaluar_reconstruccion(modelo, x_numpy, device, redondear=True):

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    modelo.eval()
    with torch.no_grad():
        reconstruido = modelo(x_tensor)

    original = x_tensor.cpu().numpy()
    prediccion = np.round(np.abs(reconstruido.cpu().numpy())) if redondear else reconstruido.cpu().numpy()
    diferencias = np.linalg.norm(original - prediccion, axis=1)

    return original, prediccion, diferencias