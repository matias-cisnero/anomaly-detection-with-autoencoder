import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo usado: {device}")
    if device.type == "cuda":
        print(f"Nombre GPU: {torch.cuda.get_device_name(0)}")
    return device

