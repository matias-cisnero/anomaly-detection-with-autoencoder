import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

class BaseAutoencoder(nn.Module):
    def __init__(self, capas: list[int], activacion=nn.GELU):
        super(BaseAutoencoder, self).__init__()

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

    def propagate(self, module_or_function, x: np.ndarray, device="cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).to(device)
            output = module_or_function(t)

            if isinstance(output, tuple):
                return output[0].cpu().numpy() 
            else:
                return output.cpu().numpy()
            
    def forward(self, x):
        raise NotImplementedError("El método 'forward' debe ser implementado por la subclase.")

    def predict(self, x: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'predict' debe ser implementado por la subclase.")
    
    def encode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'encode' debe ser implementado por la subclase.")

    def decode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        raise NotImplementedError("El método 'decode' debe ser implementado por la subclase.")
        
    def compute_loss(self, batch_x, output) -> torch.Tensor:
        raise NotImplementedError("El método 'compute_loss' debe ser implementado por la subclase.")

    def fit(self, x_data: np.ndarray, device, lr: float, batch_size: int, num_epochs: int, verbose = 1,
            use_lr_scheduler: bool = False, lr_decay_factor=0.5, lr_patience: int = 2, patience_early_stopping: int = sys.maxsize):
        
        dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32))
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
            for (inputs,) in loader:

                inputs = inputs.to(device)
                optimizer.zero_grad()

                output = self(inputs)
                loss = self.compute_loss(inputs, output)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

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
    
    def save(self, path: str, set_id: int = -1, lr: float = -1):
        fecha = datetime.now().strftime("%Y-%m-%dT%H.%M")
        final_path = f"{path}_{fecha}_lr={lr}_set={set_id}.pth"
        torch.save(self, final_path)
        print(f"\nModelo guardado correctamente en '{final_path}'")

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        model = torch.load(path, map_location=device)
        model.to(device)
        model.eval()
        print(f"\nModelo cargado correctamente de '{path}'")
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
    def __init__(self, capas: list[int], activacion=nn.GELU):
        super(Autoencoder, self).__init__(capas, activacion)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def predict(self, x: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.forward, x, device)
    
    def encode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.encoder, x, device)

    def decode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.decoder, x, device)
    
    def compute_loss(self, batch_x, output):
        return F.mse_loss(output, batch_x, reduction="mean")
    
class HybridLossAutoencoder(BaseAutoencoder):
    def __init__(self, capas: list[int], columnas_binarias: np.ndarray, activacion=nn.GELU):
        super(HybridLossAutoencoder, self).__init__(capas, activacion)

        self.register_buffer("mask_bin", torch.tensor(columnas_binarias, dtype=torch.bool))
        self.register_buffer("mask_cont", ~self.mask_bin)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def predict(self, x: np.ndarray, device="cpu") -> np.ndarray:
        out = self.propagate(self.forward, x, device)

        # Aplicamos sigmoide SOLO a las columnas binarias
        out[:, self.mask_bin.cpu().numpy()] = 1 / (1 + np.exp(-out[:, self.mask_bin.cpu().numpy()]))
        return out

    def encode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.encoder, x, device)

    def decode(self, x: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.decoder, x, device)

    def compute_loss(self, batch_x, output):
        x_bin = batch_x[:, self.mask_bin]
        x_cont = batch_x[:, self.mask_cont]

        out_bin = output[:, self.mask_bin]
        out_cont = output[:, self.mask_cont]

        bce_loss = F.binary_cross_entropy_with_logits(out_bin, x_bin)
        mse_loss = F.mse_loss(out_cont, x_cont)

        w_bce = 1.0 / x_bin.size(1)
        w_mse = 1.0 / x_cont.size(1)

        return w_bce * bce_loss + w_mse * mse_loss
    
# DAE (Denoising Autoencoder)

# CAE (Contractive Autoencoder)

# SAE (Sparse Autoencoder)