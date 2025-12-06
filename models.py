import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from utils import obtener_matriz_confusion, obtener_epsilon, obtener_metricas, evaluar_reconstruccion

class BaseAutoencoder(nn.Module):
    def __init__(self, dims: list[int], activation=nn.GELU):
        super(BaseAutoencoder, self).__init__()

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(activation())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(dims[i], dims[i - 1]))
            if i > 1:
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def propagate(self, module_or_function, input: np.ndarray, device="cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(input, dtype=torch.float32).to(device)
            output = module_or_function(t)

            if isinstance(output, tuple):
                return output[0].cpu().numpy() 
            else:
                return output.cpu().numpy()
            
    def forward(self, input):
        raise NotImplementedError("El método 'forward' debe ser implementado por la subclase.")

    def predict(self, input: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'predict' debe ser implementado por la subclase.")
    
    def encode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'encode' debe ser implementado por la subclase.")

    def decode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'decode' debe ser implementado por la subclase.")
        
    def compute_loss(self, batch_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("El método 'compute_loss' debe ser implementado por la subclase.")

    def fit(self, x_train: np.ndarray, device, lr: float, batch_size: int, num_epochs: int, verbose = 1,
            use_lr_scheduler: bool = False, lr_decay_factor=0.5, lr_patience: int = 2, patience_early_stopping: int = sys.maxsize):
        
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        scheduler = None
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_decay_factor, patience=lr_patience)

        loss_history = []
        self.to(device)
        self.train()

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for (batch_input,) in loader:

                batch_input = batch_input.to(device)
                optimizer.zero_grad()

                output = self(batch_input)
                loss = self.compute_loss(batch_input, output)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_input.size(0)

            epoch_loss /= len(dataset)
            loss_history.append(epoch_loss)
            
            if scheduler is not None:
                scheduler.step(epoch_loss)

            if verbose in (1, 2) and (epoch + 1) % 25 == 0:
                delta = epoch_loss - loss_history[-2] if len(loss_history) > 1 else 0
                signo = "↓" if delta < 0 else "↑"
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:>3}/{num_epochs:<3} │ Loss: {epoch_loss:.6f} {signo} │ lr: {current_lr:.6f}")

            # Early stopping clásico
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience_early_stopping:
                if verbose >= 1:
                    print(f"Early stopping activado por no haber mejora en {patience_early_stopping} épocas de la perdida")
                break

            # Early stopping de scheduler
            if use_lr_scheduler and optimizer.param_groups[0]['lr'] < 1e-6:
                if verbose >= 1:
                    print(f"Early stopping activado por lr mínima en época {epoch+1}")
                break

        if verbose == 2:
            plt.plot(loss_history)
            plt.xlabel("Época")
            plt.ylabel("Error medio")
            plt.grid(True)
            plt.show()

        return loss_history

    def evaluate(self, x_train, x_test_norm, x_test_anom, device, tipo_epsilon=1, tipo_norma="L2") -> dict:

        epsilon = obtener_epsilon(self, x_train, device, tipo_epsilon=tipo_epsilon, tipo_norma=tipo_norma)

        _, dif_norm = evaluar_reconstruccion(self, x_test_norm, device, tipo_norma=tipo_norma)
        _, dif_anom = evaluar_reconstruccion(self, x_test_anom, device, tipo_norma=tipo_norma)

        TP, FN, TN, FP = obtener_matriz_confusion(dif_norm, dif_anom, epsilon)

        metricas = obtener_metricas(TP, FN, TN, FP)

        resultado = {
        "epsilon": np.round(float(epsilon), 4),
        "conf_matrix": [
            [f"TP:{TP}", f"FN:{FN}"],
            [f"FP:{FP}", f"TN:{TN}"]
        ],
        **metricas
        }

        return resultado

    def save(self, path: str, set_id: str = "-1", lr: float = -1):
        fecha = datetime.now().strftime("%Y-%m-%dT%H.%M")
        final_path = f"{path}_{fecha}_lr={lr}_set={set_id}.pth"
        torch.save(self, final_path)
        print(f"\nModelo guardado correctamente en '{final_path}'")

    @classmethod
    def load(cls, path: str, device: str = "cpu", verbose = False):
        model = torch.load(path, map_location=device)
        model.to(device)
        model.eval()
        if verbose: print(f"\nModelo cargado correctamente de '{path}'")
        return model
    
    def summary(self):
        caps_encoder, caps_decoder = [], []

        for i in range(0, len(self.encoder), 2):
            caps_encoder.append(str(self.encoder[i].in_features))
        latent_dim = self.encoder[-1].out_features

        for i in range(0, len(self.decoder), 2):
            caps_decoder.append(str(self.decoder[i].out_features))

        print(f"<Autoencoder: In {' → '.join(caps_encoder)} → [{latent_dim}] → {' → '.join(caps_decoder)} Out>")
    
    def __repr__(self):
        lines = ["\nResumen del Autoencoder", "-" * 60, "Codificador:"]
        for i in range(0, len(self.encoder), 2):
            capa = self.encoder[i]
            act = self.encoder[i + 1] if i + 1 < len(self.encoder) else None
            nombre_act = act.__class__.__name__ if act else "—"
            lines.append(f"  {capa.in_features:>3} → {capa.out_features:<3}  ({nombre_act})")

        lines.append("Decodificador:")
        for i in range(0, len(self.decoder), 2):
            capa = self.decoder[i]
            act = self.decoder[i + 1] if i + 1 < len(self.decoder) else None
            nombre_act = act.__class__.__name__ if act else "—"
            lines.append(f"  {capa.in_features:>3} → {capa.out_features:<3}  ({nombre_act})")

        lines.append("-" * 60)
        return "\n".join(lines)
    
class Autoencoder(BaseAutoencoder):
    def __init__(self, dims: list[int], activation=nn.GELU):
        super(Autoencoder, self).__init__(dims, activation)

    def forward(self, input):
        return self.decoder(self.encoder(input))
    
    def predict(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.forward, input, device)
    
    def encode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.encoder, input, device)

    def decode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.decoder, input, device)
    
    def compute_loss(self, batch_input, output):
        return F.mse_loss(output, input, reduction="mean")