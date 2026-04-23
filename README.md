# 🧠 Detección de Anomalías Médicas con Autoencoder en PyTorch

Proyecto de detección de anomalías en datos médicos usando autoencoders en PyTorch. El modelo aprende el comportamiento “normal” y detecta casos anómalos mediante el error de reconstrucción.

![Texto alternativo](arquitectura.png)

## 📊 Conjunto de Datos Utilizados

- **[Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)**  

## ⚙️ Tecnologías

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib

## 📈 Resultados 
* SAE (A): mejor Precision y F1-score
* CAE (A): mejor Recall
* El rendimiento cae al aumentar anomalías en el entrenamiento