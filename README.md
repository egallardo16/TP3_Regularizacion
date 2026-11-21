# TP3 – Regularización de un MLP en 20 Newsgroups

Trabajo práctico de Redes Neuronales. Parte de una red que **sobreajusta** el problema de clasificación de texto *20 Newsgroups* (~18.000 posts de Usenet, 20 categorías, vectores TF-IDF) y estudia cómo cambian su comportamiento y su generalización al incorporar **L1, L2, L1 + L2, Dropout y Batch Normalization**. Primero se mide cada técnica de forma aislada (un hiperparámetro por vez, con varias semillas) y después se las explora en conjunto con una búsqueda de Optuna con pruning.

> Este trabajo continúa el TP2 (MLP): la configuración de partida se lee de `overfit_config.json`, que genera la sección 7 de ese notebook. Ver [Relación con el TP2](#relación-con-el-tp2).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP3_RegularizacionN.ipynb` | Notebook con todo el trabajo (código, experimentos, análisis y conclusiones), ejecutado de punta a punta. |
| `overfit_config.json` | Configuración sobreajustada de partida (arquitectura, activación, `lr`, `batch_size`, preprocesamiento y vectorización). Viene del TP2 y **debe estar en la raíz**, junto al notebook. |
| `requirements.txt` | Dependencias de Python. |
| `README.md` | Este archivo. |
| `.gitignore` | Excluye archivos que no deben versionarse (`.DS_Store`, checkpoints de Jupyter, logs descomprimidos, etc.). |
| `results_reg/` | Resultados en CSV (experimentos de un factor por vez, poda por magnitud, trials y pruning de la búsqueda, resúmenes por hiperparámetro, importancia, resultados finales por semilla, comparaciones pareadas y diagnóstico de sesgo y varianza) y la base de los estudios de Optuna (`optuna_reg.db`). |

## Artefactos de seguimiento (`artefactos_regularizacion.zip`)

Los logs de TensorBoard, la base de MLflow y los modelos registrados son numerosos y pesados, por lo que se distribuyen comprimidos en `artefactos_regularizacion.zip`. Este archivo lo genera la sección 11 del notebook y contiene:

| Contenido del zip | Descripción |
|---|---|
| `runs_reg/` | Logs de TensorBoard de cada entrenamiento (un directorio por experimento, trial y entrenamiento final). |
| `mlflow_reg.db` | Base SQLite de MLflow con el experimento `regularizacion_20newsgroups`: hiperparámetros, métricas por época y tiempos de cada run, anidados bajo cada búsqueda. |
| `mlruns/` | Artefactos registrados por MLflow. |
| `results_reg/` | Copia de las tablas CSV y de la base de Optuna. |

Para explorarlos, descomprimir el zip en la raíz del repositorio y ejecutar:

```bash
unzip artefactos_regularizacion.zip

# Curvas de aprendizaje de todos los entrenamientos
tensorboard --logdir runs_reg

# Parámetros, métricas y tiempos de cada run
mlflow ui --backend-store-uri sqlite:///mlflow_reg.db
```

## Entorno de ejecución

- El notebook se ejecutó en **Google Colab con una GPU NVIDIA T4**.
- Todos los tiempos reportados (segundos por época, tiempo hasta converger, tiempo de búsqueda) se midieron con `time.perf_counter()` en esa GPU; en otro hardware los valores absolutos cambiarán, aunque no las comparaciones relativas.
- Los datos quedan residentes en GPU como tensores densos. Funciona también en CPU, pero es **bastante más lento**.
- Referencia de tiempo en la T4: la búsqueda completa (8 estudios) tomó unos 1.540 s: unos 510 s la red de 3×64 y unos 1.030 s la de 3×1024.

## Cómo ejecutarlo

### Opción A: Google Colab

1. Subir `TP3_RegularizacionN.ipynb` y `overfit_config.json` a Colab (o abrirlos desde Drive/GitHub). El JSON tiene que estar en el directorio de trabajo, porque el notebook lo lee con la ruta relativa `overfit_config.json`.
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas en orden. La primera celda instala `optuna` y `mlflow`; el resto de las librerías ya viene en Colab.
4. Al final (sección 11) se genera `artefactos_regularizacion.zip` y Colab lo descarga automáticamente.

### Opción B: ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar PyTorch según el sistema/CUDA (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

jupyter notebook TP3_RegularizacionN.ipynb
```

La primera celda (`!pip install -q optuna mlflow`) puede omitirse si ya se instalaron los requirements.

### Notas

- El notebook descarga automáticamente el dataset (`fetch_20newsgroups`) y los recursos de NLTK (`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`), por lo que se necesita conexión a internet la primera vez.
- Las semillas están fijadas (`SEED = 42`; `SWEEP_SEEDS = [42, 123, 2024]` para los experimentos controlados y `FINAL_SEEDS = [42, 123, 2024, 7, 99]` para la evaluación final), pero los resultados pueden variar levemente entre ejecuciones por el no determinismo de la GPU.
- Los estudios de Optuna se guardan en `results_reg/optuna_reg.db` y **se reanudan** si la sesión se interrumpe. Para repetir la búsqueda desde cero hay que borrar ese archivo.
- El notebook pesa unos 16 MB porque incluye las figuras de salida.

## Contenido del notebook

| Sección | Qué hace |
|---|---|
| **0. Setup** | Instalación de dependencias, importaciones, semillas, detección de GPU y backend de MLflow. Tabla de herramientas utilizadas y por qué se eligió cada una. |
| **1. Datos y configuración** | Carga de las particiones oficiales train/test (sin encabezados, firmas ni citas) y lectura de `overfit_config.json`. El test no se usa en ninguna decisión previa a la evaluación final. |
| **2. Preprocesamiento y vectorización** | Tokenización y TF-IDF ajustado solo con train, partición de búsqueda (80 % train / 20 % validación, estratificada) y partición final (train completo contra test). Tensores residentes en GPU. |
| **3. Modelo** | `MLP` con `dropout` y `batch_norm` opcionales. La penalidad L1/L2 se suma a la cross-entropy sobre los pesos de las capas lineales, con un factor de crecimiento por capa. Incluye medidas de pesos y poda por magnitud. |
| **4. Entrenamiento** | Bucle con Adam; métricas por época, tiempos, época de convergencia, ruido de las curvas, early stopping y pruning. |
| **5. Fundamentos** | Tabla con lo que predice la teoría para cada técnica, para contrastarlo después con lo observado. |
| **6. Experimentos controlados** | Un hiperparámetro por vez desde la configuración sobreajustada (3 semillas, 40 épocas, sin early stopping): referencia, L2, L1 y poda por magnitud, L1 + L2, λ por capa, dropout y Batch Normalization. |
| **7. Búsqueda con Optuna** | Dos arquitecturas (la sobreajustada y 3×1024) × cuatro tipos de regularización (`none`, `l2`, `l1`, `l1_l2`): 8 estudios de 30 trials, muestreo TPE y `MedianPruner`. Espacio: λ₁, λ₂, dropout, Batch Norm, `lr` y `batch_size`. |
| **8. Análisis de la búsqueda** | Pruning y tiempo ahorrado, mejores trials por estudio, resúmenes por hiperparámetro y por `lr`, efecto sobre los pesos, chequeo de bordes del espacio e importancia de hiperparámetros (fANOVA). |
| **9. Evaluación final** | Tres condiciones por arquitectura (sobreajustada, solo early stopping y regularizada) con 5 semillas, entrenadas con todo el train y evaluadas en el test real. Diferencias pareadas (test t) y diagnóstico de sesgo y varianza. |
| **10. Conclusiones** | Resultados, fenómenos observados y su fundamento, tiempos, búsqueda y limitaciones. |
| **11. Exportación** | Guarda las tablas finales y genera `artefactos_regularizacion.zip`. |
| **12. Nota sobre el uso de IA** | Detalle de en qué aspectos del trabajo se usaron modelos de IA. |

## Resultados principales

Evaluación en el test oficial, media ± desvío entre 5 semillas:

| Arquitectura | Condición | Épocas | Test acc | Macro-F1 | Gap train − test |
|---|---|---|---|---|---|
| 3×64 | sobreajustada | 20 | 0.5481 ± 0.0100 | 0.5411 ± 0.0108 | 0.4241 |
| 3×64 | solo early stopping | 5 | 0.5652 ± 0.0047 | 0.5614 ± 0.0034 | 0.3982 |
| 3×64 | **regularizada** | 4 | **0.6483 ± 0.0028** | **0.6363 ± 0.0054** | **0.3007** |
| 3×1024 | sobreajustada | 20 | 0.6099 ± 0.0128 | 0.6051 ± 0.0114 | 0.3592 |
| 3×1024 | solo early stopping | 5 | 0.6187 ± 0.0098 | 0.6149 ± 0.0087 | 0.3525 |
| 3×1024 | **regularizada** | 5 | **0.6721 ± 0.0040** | **0.6660 ± 0.0036** | **0.2951** |

- La regularización mejora de forma consistente en las 5 semillas: +10,0 puntos en 3×64 y +6,2 en 3×1024 frente a la red sobreajustada. Detenerse antes por sí solo aporta poco (1,7 y 0,9 puntos) y reduce el tiempo de entrenamiento unas 4 veces.
- **Batch Normalization** fue la técnica más efectiva: con `lr = 1e-3` sube la `val_acc` de 0.6372 a 0.7085 y todos los mejores trials de la búsqueda la usan. **Dropout** mejora la generalización (hasta 0.6836 con p = 0.5) sin reducir la memorización, y exige más épocas.
- **L1 y L2** contraen y dispersan los pesos (más L1 que L2) y L1 resiste mejor la poda, pero no mejoraron la accuracy: el tipo de regularización L1/L2 no se distingue entre los mejores trials.
- Ninguna técnica elimina la varianza: las seis redes finales siguen siendo de **alta varianza** (gap de 0.30 a 0.42).
- El pruning cortó 76 de los 239 trials (32 %). Un trial de 3×1024 con L1 + L2 no quedó registrado como terminado, por eso son 239 y no 240.
- Las redes regularizadas de tres capas quedan por debajo de las del TP2 (MLP de una capa: 0.7033; softmax: 0.6837) en accuracy y en varianza.
- Más detalle (efecto de cada técnica, tiempos, importancia de hiperparámetros y limitaciones) en la sección 10 del notebook.

## Relación con el TP2

El TP2 (`TP2_MLP.ipynb`) busca la configuración más chica que sobreajusta el problema y la guarda en `overfit_config.json`: una red de 3 capas × 64 neuronas con ReLU, Adam, `lr = 1e-3` y `batch_size = 64`, sin dropout ni weight decay. Este trabajo parte de ese archivo, que fija la arquitectura, el preprocesamiento y la vectorización de todos los experimentos de la sección 6.
