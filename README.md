# 🧠 Detección de Anomalías Médicas con Autoencoder en PyTorch

Este proyecto implementa un autoencoder utilizando PyTorch para detectar anomalías en datos médicos. El modelo se entrena para identificar patrones inusuales en conjuntos de datos relacionados con la salud, lo que puede ayudar en la detección temprana de enfermedades o condiciones anómalas.

## 🧬 ¿Qué hace este proyecto?

- **Entrena un autoencoder** en datos médicos para aprender representaciones comprimidas de las características.
- **Detecta anomalías** al identificar instancias con altos errores de reconstrucción.
- **Utiliza PyTorch**, una biblioteca popular para el aprendizaje automático y redes neuronales profundas.

## 📊 Conjuntos de Datos Utilizados

- **[Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)**  
  Información sobre características de tumores mamarios, útil para clasificar entre benignos y malignos.

- **[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)**  
  Contiene indicadores de salud relacionados con la diabetes, como niveles de glucosa y presión arterial.

- **[Cancer Data](https://www.kaggle.com/datasets/erdemtaha/cancer-data)**  
  Datos sobre diferentes tipos de cáncer, incluyendo características clínicas y demográficas.

## ⚙️ Tecnologías Utilizadas

- <a href="https://www.python.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/640px-Python_logo_and_wordmark.svg.png" alt="Logo Python" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Lenguaje de programación utilizado para todo el proyecto</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://pytorch.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/PyTorch_logo_white.svg/640px-PyTorch_logo_white.svg.png" alt="Logo PyTorch" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Framework para construcción y entrenamiento de modelos DL</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://numpy.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/640px-NumPy_logo_2020.svg.png" alt="Logo NumPy" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Biblioteca para operaciones numéricas en Python</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://matplotlib.org/" target="_blank">
    <img src="https://matplotlib.org/stable/_static/logo_light.svg" alt="Logo Matplotlib" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Biblioteca para la creación de gráficos y visualizaciones</span>
  <div style="margin-bottom: 10px;"></div>

- <a href="https://pandas.pydata.org/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/640px-Pandas_logo.svg.png" alt="Logo Pandas" width="80" style="vertical-align: middle;"/>
  </a>
  <span style="vertical-align: middle;">– Herramienta para la manipulación y análisis de datos</span>
  <div style="margin-bottom: 10px;"></div>


## 🚀 ¿Cómo Ejecutarlo?

1. Clona este repositorio:

   ```bash
   git clone https://github.com/matias-cisnero/anomaly-detection-with-autoencoder.git
   cd anomaly-detection-with-autoencoder

2. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt

3. Ejecuta el script principal para entrenar el autoencoder:

    ```bash
    python train_autoencoder.py
Para detectar anomalías en un nuevo conjunto de datos:

## 📈 Resultados Esperados
* Visualización de anomalías: Gráficos que muestran las instancias con mayores errores de reconstrucción.

* Métricas de rendimiento: Evaluación del modelo utilizando métricas como precisión, recall y F1-score.