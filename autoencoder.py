# Librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Datos

df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

atributos = df.drop(columns=["Diabetes_binary"])
concepto = df[["Diabetes_binary"]]

x = atributos.to_numpy()
print(f"Shape de x: {x.shape}")
y = concepto.to_numpy()
print(f"Shape de y: {y.shape}")

# División de datos

x_train = x.copy()
y_train = y.copy()

# Funciones de activación

def signo(h):
  return 1 if h >= 0 else -1

def sigmoide(h, β=1):
  return 1 / (1 + np.exp(-2 * β * h))
def sigmoide_derivada(h, β=1):
  return 2 * β * sigmoide(h, β) * (1 - sigmoide(h, β))

def tanh(h, β=1): # Usarlo
  return np.tanh(β * h)
def tanh_derivada(h, β=1):
  return β * (1 - tanh(h, β) ** 2)

def gelu(h):
  return 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
def gelu_derivada(h):
  c = np.sqrt(2 / np.pi)
  a = h + 0.044715 * h**3
  t = tanh(c * a)
  dt = tanh_derivada(c * a) * c * (1 + 3 * 0.044715 * h**2)
  return 0.5 * (1 + t) + 0.5 * h * dt

# Calculos de error

def binary_crossentropy(y, ypred):
  ε = 1e-8  # evitar log(0)
  return -np.mean(y * np.log(ypred + ε) + (1 - y) * np.log(1 - ypred + ε))

def MSE(y, ypred):
  return 1/2 * np.sum((y - ypred)**2)

# Autoencoder