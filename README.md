# TP3 – Regularización de un MLP sobreajustado en 20 Newsgroups

Trabajo práctico de Redes Neuronales. Parte de un MLP que **sobreajusta** el problema de clasificación de texto de *20 Newsgroups* (~18.000 posts, 20 categorías, representados con TF-IDF) y estudia cómo cambian su comportamiento y su generalización al incorporar tres familias de regularización:

- **L1, L2 y L1 + L2** sobre los pesos (incluyendo un λ distinto por capa).
- **Dropout.**
- **Batch Normalization.**

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `TP3_Regularizacion.ipynb` | Notebook con todo el trabajo (código, experimentos, análisis y conclusiones). |
| `overfit_config.json` | Configuración de la red sobreajustada de partida (ver más abajo). **Es necesario para ejecutar el notebook.** |
| `requirements.txt` | Dependencias de Python. |
| `README.md` | Este archivo. |
| `results_reg/` | Resultados del notebook en CSV: barridos de los experimentos controlados (`sweep_*.csv`, `poda_por_magnitud.csv`, `resumen_por_tecnica.csv`), trials y análisis de la búsqueda con Optuna (`trials_regularizacion.csv`, `pruning_regularizacion.csv`, `mejor_por_estudio.csv`, `resumen_por_hiperparametro_reg.csv`, `resumen_por_lr_reg.csv`, `chequeo_de_bordes_reg.csv`, `importancia_reg.csv`), resultados de la evaluación final (`final_por_semilla.csv`, `final_media_desvio.csv`, `final_comparaciones_pareadas.csv`) y la base de los estudios de Optuna (`optuna_reg.db`). |
| `runs_reg/` | Logs de TensorBoard de cada entrenamiento (un directorio por corrida y semilla, con loss, accuracy, tiempo por época y peso medio de la capa 1). Se visualizan con `tensorboard --logdir runs_reg`. |
| `mlflow_reg.db` | Base SQLite de MLflow con el experimento `regularizacion_20newsgroups`: hiperparámetros, métricas por época y tiempos de todas las corridas. Se explora con `mlflow ui --backend-store-uri sqlite:///mlflow_reg.db`. |

## Entorno de ejecución

- Todos los experimentos y tiempos del notebook se corrieron en **Google Colab con una GPU NVIDIA T4** (`Tesla T4`).
- Todos los tiempos reportados (segundos por época, tiempo hasta converger, tiempo de búsqueda) corresponden a esa GPU; en otro hardware los valores absolutos cambiarán, aunque no las comparaciones relativas entre técnicas.
- El notebook funciona también en CPU, pero es **mucho más lento**: los datos se cargan como tensores densos residentes en GPU y se entrenan miles de redes.
- El tiempo total depende de la sesión: solo la búsqueda con Optuna (sección 7) tomó unos 1.540 s en la T4 (509 s para la red 3×64 y 1.033 s para la 3×1024), a lo que se suman los experimentos controlados y la evaluación final.

## Cómo ejecutarlo

### Opción A: Google Colab

1. Subir `TP3_Regularizacion.ipynb` a Colab (o abrirlo desde Drive/GitHub).
2. Activar la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Subir **`overfit_config.json`** al directorio de trabajo de la sesión (panel de archivos de la izquierda, en la misma carpeta que el notebook, sin renombrarlo).
4. Ejecutar todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*). La primera celda instala `optuna` y `mlflow`; el resto de las librerías ya viene en Colab.
5. Al final (sección 11) se genera `artefactos_regularizacion.zip` y Colab lo descarga automáticamente.

### Opción B: ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# Instalar PyTorch según el sistema/CUDA (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt

jupyter notebook TP3_Regularizacion.ipynb
```

Colocar `overfit_config.json` en la misma carpeta que el notebook. La primera celda (`!pip install -q optuna mlflow`) puede omitirse si ya se instalaron los requirements.

### Notas

- El notebook descarga automáticamente el dataset (`fetch_20newsgroups`) y los recursos de NLTK (`stopwords`, `wordnet`, `omw-1.4`, `punkt`, `punkt_tab`), por lo que se necesita conexión a internet la primera vez.
- Las semillas están fijadas (`SEED = 42`, `SWEEP_SEEDS = [42, 123, 2024]`, `FINAL_SEEDS = [42, 123, 2024, 7, 99]`), pero los resultados pueden variar levemente entre ejecuciones por el no determinismo de la GPU.
- Los estudios de Optuna se guardan en `results_reg/optuna_reg.db` y **se reanudan** si la sesión se interrumpe (por ejemplo, por una desconexión de Colab). Para repetir la búsqueda desde cero hay que borrar ese archivo.
- El notebook está dividido en celdas y secciones pensadas para minimizar el trabajo perdido ante desconexiones de la GPU.

## El archivo `overfit_config.json`

Contiene la configuración de la red que se toma como punto de partida y que **ya sobreajusta**. Se lee en la sección 1 del notebook y de ahí salen la arquitectura, el optimizador, el preprocesamiento y la vectorización que usan todos los experimentos; así, las diferencias entre variantes se deben únicamente a la regularización, al `lr` y al `batch_size`.

**Origen:** el archivo se obtuvo de una **nueva corrida del TP2**, aplicando las recomendaciones y correcciones generales que se hicieron en clase (como probar varias semillas y comparar los resultados, medir tiempos y épocas de convergencia, etc.). El TP2 mejorado se encuentra en https://github.com/egallardo16/TP2_MLP_mejorado.


Valores de la configuración utilizada en este trabajo:

```json
{
  "hidden_units": [64, 64, 64],
  "activation": "relu",
  "optimizer": "adam",
  "lr": 0.001,
  "batch_size": 64,
  "epochs": 20,
  "dropout": 0.0,
  "weight_decay": 0.0,
  "preprocessing": {
    "remove_stopwords": true,
    "normalization": "lemmatize",
    "ngram_range": "1gram",
    "max_features": 20000,
    "min_df": 1
  },
  "train_acc": 0.9722,
  "val_acc": 0.5930,
  "n_params": 1289684
}
```

Claves que usa el notebook: `hidden_units`, `activation`, `lr`, `batch_size`, `epochs` (épocas de la condición "sobreajustada" en la evaluación final) y `preprocessing`. El resto (`train_acc`, `val_acc`, `n_params`, etc.) es informativo.

## Contenido del notebook

| Sección | Qué hace |
|---|---|
| **0. Setup** | Importaciones, semillas, directorios de salida (`results_reg/`, `runs_reg/`), MLflow y detección de GPU. |
| **1. Datos y configuración** | Carga las particiones oficiales train/test de 20 Newsgroups (sin encabezados, firmas ni citas) y lee `overfit_config.json`. El test se reserva para la evaluación final. |
| **2. Preprocesamiento y vectorización** | Tokenización, stopwords y lematización según la config; TF-IDF ajustado solo con train; partición de búsqueda (80 % train / 20 % validación, estratificada) y partición final; tensores residentes en GPU. |
| **3. Modelo** | `MLP` con opciones de dropout y BatchNorm; penalidad L1/L2 explícita en la función de costo (con λ por capa); utilidades para medir pesos, esparsidad y poda por magnitud. |
| **4. Entrenamiento** | `train_model` con Adam, medición de tiempos con sincronización de GPU, época de convergencia, early stopping, pruning de Optuna e índice de ruido de las curvas. |
| **5. Fundamentos** | Tabla de hipótesis: qué predice la teoría para cada técnica (L1, L2, dropout, BatchNorm, learning rate, early stopping). |
| **6. Experimentos controlados** | Un hiperparámetro por vez, con 3 semillas: referencia sin regularización, L2, L1 y poda por magnitud, L1 + L2, λ por capa, dropout (con dos `lr`), BatchNorm (con tres `lr` y combinación con dropout) y resumen por técnica. |
| **7. Búsqueda con Optuna** | 8 estudios (2 arquitecturas × 4 tipos de regularización: `none`, `l2`, `l1`, `l1_l2`), 30 trials cada uno, con `MedianPruner` y early stopping. Las arquitecturas son la de la config (3×64) y una más grande (3×1024). |
| **8. Análisis de la búsqueda** | Pruning y tiempo ahorrado, mejor trial por estudio, tablas por hiperparámetro, learning rate, chequeo de bordes del espacio de búsqueda, importancia de hiperparámetros (fANOVA) e historia de optimización. |
| **9. Evaluación final** | Compara "sobreajustada", "solo early stopping" y "regularizada" con 5 semillas sobre el test real: accuracy, macro-F1, gap, pesos, ruido, tiempos y tests t pareados. |
| **10. Conclusiones** | Resultados, fenómenos observados con su fundamento teórico, tiempos, análisis de la búsqueda y limitaciones. |
| **11. Exportación** | Genera `artefactos_regularizacion.zip` con las tablas, logs de TensorBoard y base de MLflow. |
| **12. Nota sobre el uso de IA** | Detalle de las herramientas de IA utilizadas en el trabajo. |

### Resultado principal

Con 5 semillas y evaluación sobre el test oficial, la mejor red regularizada supera tanto a la red sobreajustada como a la que solo usa early stopping:

| Arquitectura | Sobreajustada | Solo early stopping | Regularizada |
|---|---|---|---|
| 3×64 | 0.548 | 0.565 | **0.648** |
| 3×1024 | 0.610 | 0.619 | **0.672** |

(accuracy en test, media entre semillas). El detalle, los tests pareados y las limitaciones están en la sección 10 del notebook.

## Salidas generadas

- `results_reg/`: tablas en CSV (barridos, poda, resumen por técnica, trials de Optuna, importancia de hiperparámetros, resultados finales, etc.) y la base `optuna_reg.db`.
- `runs_reg/`: logs de TensorBoard (se visualizan en el notebook con `%tensorboard --logdir runs_reg`).
- `mlflow_reg.db` y `mlruns/`: seguimiento de experimentos con MLflow.
- `artefactos_regularizacion.zip`: todo lo anterior empaquetado.
