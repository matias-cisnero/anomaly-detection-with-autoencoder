import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Autoencoder(nn.Module):
    def __init__(self, capas: list[int], activacion=nn.GELU):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        for i in range(len(capas) - 1):
            encoder_layers.append(nn.Linear(capas[i], capas[i + 1]))
            if i < len(capas) - 2:
                encoder_layers.append(activacion())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(capas) - 1, 0, -1):
            decoder_layers.append(nn.Linear(capas[i], capas[i - 1]))
            if i > 1:
                decoder_layers.append(activacion())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def predict(self, x: np.ndarray, device="cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(device)
            return self.forward(t).cpu().numpy()
        
    def fit(self, x_data: np.ndarray, device, lr: float, batch_size: int, epocas: int, verbose=1):
        dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterio = nn.MSELoss()
        optimizador = optim.Adam(self.parameters(), lr=lr)
        errores = []

        self.to(device)
        self.train()

        for epoca in range(epocas):
            epoca_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizador.zero_grad()
                salida = self(batch_x)
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
            plt.xlabel("Época")
            plt.ylabel("Error medio")
            plt.grid(True)
            plt.show()

        return errores
    
    def save(self, path: str, i: int = -1):
        fecha = datetime.now().strftime("%H-%M_%d-%m-%y")
        final_path = f"{path}_{fecha}_{i}.pth"
        torch.save(self, final_path)
        print(f"\nModelo guardado correctamente en '{final_path}'")

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        modelo = torch.load(path, map_location=device)
        modelo.to(device)
        modelo.eval()
        print(f"\nModelo cargado correctamente en '{path}'")
        return modelo