import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from utils import obtener_matriz_confusion, obtener_epsilon, obtener_metricas, evaluar_reconstruccion, calcular_auc

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

    def fit(self, x_train: np.ndarray, x_val_norm: np.ndarray, device, lr: float, batch_size: int, num_epochs: int, verbose = 1,
            use_lr_scheduler: bool = False, lr_decay_factor=0.5, lr_patience: int = 2, patience_early_stopping: int = 100000):
        
        dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        scheduler = None
        if use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_decay_factor, patience=lr_patience)

        loss_history = []
        val_loss_history = []

        self.to(device)
        self.train()

        best_val_loss = float('inf')
        epochs_no_improve = 0

        val_tensor = torch.tensor(x_val_norm, dtype=torch.float32).to(device)

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

            # Para validación
            self.eval()
            with torch.no_grad():
                val_output = self(val_tensor)
                val_loss = self.compute_loss(val_tensor, val_output).item()
            self.train()

            val_loss_history.append(val_loss)
            
            if scheduler is not None:
                scheduler.step(epoch_loss)

            if verbose in (1, 2) and (epoch + 1) % 25 == 0:
                delta = epoch_loss - loss_history[-2] if len(loss_history) > 1 else 0
                signo = "↓" if delta < 0 else "↑"
                current_lr = optimizer.param_groups[0]['lr']
                print(f"epoch {epoch+1:>3}/{num_epochs:<3} │ loss_train: {epoch_loss:.6f} {signo} │ loss_val={val_loss:.6f} │ lr: {current_lr:.6f}")

            # Early stopping clásico
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience_early_stopping:
                if verbose >= 1:
                    print(f"Early stopping: sin mejora en val_norm en época [{epoch+1}]")
                break

            # Early stopping de scheduler
            if use_lr_scheduler and optimizer.param_groups[0]['lr'] < 1e-6:
                if verbose >= 1:
                    print(f"Early stopping: LR mínimo alcanzado en época [{epoch+1}]")
                break

        if verbose == 2:
            plt.figure(figsize=(8, 5))

            plt.plot(loss_history, label="Pérdida de entrenamiento")
            plt.plot(val_loss_history, label="Pérdida de validación")

            plt.xlabel("Época")
            plt.ylabel("Error medio (MSE)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return loss_history, val_loss_history

    def evaluate(self, x_train, x_test_norm, x_test_anom, device, tipo_epsilon=1, tipo_norma="L2") -> dict:

        epsilon = obtener_epsilon(self, x_train, device, tipo_epsilon=tipo_epsilon, tipo_norma=tipo_norma)

        _, dif_norm = evaluar_reconstruccion(self, x_test_norm, device, tipo_norma=tipo_norma)
        _, dif_anom = evaluar_reconstruccion(self, x_test_anom, device, tipo_norma=tipo_norma)

        TP, FN, TN, FP = obtener_matriz_confusion(dif_norm, dif_anom, epsilon)

        metricas = obtener_metricas(TP, FN, TN, FP)

        auc = calcular_auc(dif_norm, dif_anom)

        resultado = {
        "epsilon": round(float(epsilon), 4),
        "conf_matrix": str([
            [f"TP:{TP}", f"FN:{FN}"],
            [f"FP:{FP}", f"TN:{TN}"]
        ]),
        **metricas,
        "auc": round(float(auc), 4)
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

        latent_dim = None
        for layer in reversed(self.encoder):
            if isinstance(layer, nn.Linear):
                latent_dim = layer.out_features
                break

        for i in range(0, len(self.decoder), 2):
            caps_decoder.append(str(self.decoder[i].out_features))

        latent_display = str(latent_dim) if latent_dim is not None else '?'
        print(f"<Autoencoder: In {' → '.join(caps_encoder)} → [{latent_display}] → {' → '.join(caps_decoder)} Out>")
    
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
    
    def compute_loss(self, batch_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, batch_input, reduction="mean")
    
class SAE(BaseAutoencoder):
    def __init__(self, dims: list[int], activation=nn.GELU, rho: float = 0.05, lambda_sparse: float = 0.1):
        nn.Module.__init__(self) 
        self.rho = rho 
        self.lambda_sparse = lambda_sparse

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(activation())

        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(dims[i], dims[i - 1]))
            if i > 1:
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def sparsity_penalty_kl(self, z: torch.Tensor) -> torch.Tensor:
        rho_hat = torch.mean(z, dim=0)

        epsilon = 1e-6 # Pequeño valor para evitar log(0)
        
        # D_KL(rho || rho_hat) = rho * log(rho / rho_hat) + (1-rho) * log((1-rho) / (1-rho_hat))
        term1 = self.rho * torch.log(self.rho / (rho_hat + epsilon))
        term2 = (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat + epsilon))
        
        kl_divergence = torch.sum(term1 + term2)
        return kl_divergence

    def forward(self, input):
        z = self.encoder(input)
        return self.decoder(z), z

    def predict(self, input: np.ndarray, device="cpu") -> np.ndarray:
        def func(t):
            recon, _ = self.forward(t)
            return recon
        return self.propagate(func, input, device)

    def encode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.encoder, input, device)

    def decode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.decoder, input, device)

    def compute_loss(self, batch_input, output_tuple):
        recon, z = output_tuple
        
        mse = F.mse_loss(recon, batch_input, reduction="mean")
        sparse_pen = self.sparsity_penalty_kl(z)

        return mse + self.lambda_sparse * sparse_pen

class CAE(BaseAutoencoder):
    def __init__(self, dims: list[int], activation=nn.GELU, lambda_c=1e-4):
        super(CAE, self).__init__(dims, activation)
        self.lambda_c = lambda_c

    def _contractive_penalty(self, z, input_with_grad):
        """Calcula la penalización del Jacobiano usando autograd."""
        jacobian_norm = 0
        
        # Iterar sobre las dimensiones latentes
        for i in range(z.size(1)):
            grad_output = torch.zeros_like(z)
            grad_output[:, i] = 1.0 
            
            # Cálculo de la derivada: d(z_i)/d(input_with_grad)
            jacobian_i, = torch.autograd.grad(z, input_with_grad, grad_outputs=grad_output, 
                                              retain_graph=True, create_graph=True)
            
            jacobian_norm += torch.sum(jacobian_i**2)
            
        return jacobian_norm / input_with_grad.size(0)

    def forward(self, input):
        z = self.encoder(input)
        return self.decoder(z), z
    
    def predict(self, input: np.ndarray, device="cpu") -> np.ndarray:
        def func(t):
            recon, _ = self.forward(t)
            return recon
        return self.propagate(func, input, device)

    def compute_loss(self, batch_input, output_tuple):
        # Aquí, 'target' es el batch_x original, 'output_tuple' es (recon, z)
        
        # 1. Pérdida de Reconstrucción
        recon, z = output_tuple
        mse = F.mse_loss(recon, batch_input, reduction="mean")

        # 2. Penalización Contractiva
        # Si el input no tiene gradiente (predicción), el Jacobiano será 0
        if batch_input.requires_grad: 
            # target debe ser la misma variable con requires_grad=True
            contractive_pen = self._contractive_penalty(z, batch_input)
        else:
            contractive_pen = torch.tensor(0.0, device=batch_input.device)

        return mse + self.lambda_c * contractive_pen

class VAE(BaseAutoencoder):
    def __init__(self, dims: list[int], activation=nn.GELU):
        nn.Module.__init__(self)
        
        latent_dim = dims[-1]
        penultima_dim = dims[-2] if len(dims) > 1 else dims[0]

        encoder_layers = []
        for i in range(len(dims) - 2):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(activation())
        # Añadimos la última capa lineal que alimenta mu/logvar
        encoder_layers.append(nn.Linear(dims[-2], penultima_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_layer = nn.Linear(penultima_dim, latent_dim)
        self.logvar_layer = nn.Linear(penultima_dim, latent_dim)

        decoder_layers = []
        for i in range(len(dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(dims[i], dims[i - 1]))
            if i > 1:
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode_raw(self, input):
        h = self.encoder(input)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        mu, logvar = self.encode_raw(input)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        def func(t):
            mu, _ = self.encode_raw(t)
            return mu
        return self.propagate(func, input, device)

    def decode(self, input: np.ndarray, device="cpu") -> np.ndarray:
        return self.propagate(self.decoder, input, device)

    def predict(self, input: np.ndarray, device="cpu") -> np.ndarray:
        def func(t):
            recon, _, _ = self.forward(t)
            return recon
        return self.propagate(func, input, device)

    def compute_loss(self, batch_input, output_tuple):
        recon, mu, logvar = output_tuple

        mse = F.mse_loss(recon, batch_input, reduction="mean")

        # Divergencia KL (promedio por batch)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return mse + kl