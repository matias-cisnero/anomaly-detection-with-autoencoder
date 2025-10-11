import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple

class Autoencoder(nn.Module):
    def __init__(self, entrada: int, oculta: int = 16, latente: int = 2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(entrada, oculta),
            nn.GELU(),
            nn.Linear(oculta, latente),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latente, oculta),
            nn.GELU(),
            nn.Linear(oculta, entrada),
            nn.GELU()
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
        
        if verbose and (epoca + 1) % 25 == 0:
            print(f"    Época {epoca+1}/{epocas}, Error medio: {epoca_loss:.6f}")
    
    if verbose:
      plt.plot(errores)
      #plt.title(f"Curva de Aprendizaje (lr={lr})")
      plt.xlabel("Época")
      plt.ylabel("Error Medio")
      plt.grid(True)
      plt.show()

    return modelo, errores