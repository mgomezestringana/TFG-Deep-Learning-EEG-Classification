# Clasificación de EEG mediante Matrices de Conectividad Funcional con GCN y CNN

Trabajo de Fin de Grado (TFG) en el que se evalúa el rendimiento de arquitecturas de aprendizaje profundo (Graph Convolutional Networks y Convolutional Neural Networks) sobre matrices de conectividad funcional EEG (coherencia cuadrada en banda alpha) para tareas de clasificación en seis datasets de distinta naturaleza.

**Autor:** Manuel Gómez Estríngana
**Curso:** 2025/2026

---

## 📊 Datasets evaluados

| Dataset | Tarea de clasificación | Sujetos | Canales originales |
|---|---|---|---|
| **PhysioNet Motor Imagery** | Reposo vs movimiento | 109 | 64 |
| **DS003766 v2 Decision Making** | Reposo vs tarea | 31 | 64 |
| **ds003490 Parkinson** | CTL vs PD | 50 | 63 |
| **Schizophrenia (Kaggle)** | HC vs SZ | 40 | 64 |
| **TUH EEG Seizure Corpus** | Seizure vs background | 43 pacientes | 19 |
| **HBN Healthy Brain Network** | Attention CBCL (low vs high) | 136 | 128 |

Todos los datasets se evalúan con configuraciones de **8, 16, 32 y 64 canales** (excepto TUH EEG, que al tener solo 19 canales originales se evalúa únicamente con 8 y 16).

---

## 🧠 Metodología

1. **Cálculo de matrices de conectividad funcional** mediante coherencia cuadrada en banda alpha (8-12 Hz), con ventanas temporales de 2 segundos.
2. **Construcción de grafos** a partir de las matrices, usando un threshold de 0.3 para las aristas (conexiones débiles descartadas).
3. **Entrenamiento de modelos GCN y CNN** con división cross-subject (los sujetos de test no aparecen en train).
4. **Evaluación con accuracy en test**, comparando configuraciones de 8, 16, 32 y 64 canales.

### Arquitecturas

- **GCN**: `GCNConv(N, 64) → GCNConv(64, 64) → FC(64, 32) → FC(32, 2)` con dropout 0.4.
- **CNN**: `Conv2d + BN + Pool` (×2) → `FC(64)` → `FC(2)` con dropout 0.5.

### Hiperparámetros comunes

- Optimizador: Adam (lr=0.001, weight_decay=0.001)
- Pérdida: CrossEntropyLoss
- Épocas máximas: 50
- Early stopping: paciencia de 10 épocas
- División train/val: 80/20
- Semilla aleatoria: 42

---

## 📁 Estructura del repositorio

```
.
├── data_preparation/                  # Notebooks para generar matrices de conectividad
│   ├── MI.ipynb                       # PhysioNet Motor Imagery
│   ├── DM.ipynb                       # DS003766_v2 Decision Making
│   ├── PARKINSON.ipynb                # ds003490 Parkinson
│   ├── SCHIZOPHRENIA.ipynb            # Schizophrenia (Kaggle)
│   ├── TUH.ipynb                      # TUH EEG Seizure
│   └── HBN.ipynb                      # HBN Healthy Brain Network
│
├── classification/                    # Notebook de clasificación
│   └── train_models.ipynb             # Entrenamiento con GCN y CNN (una celda por modelo)
│
├── docs/                              # Documentación
│   └── memoria_TFG.pdf                # (pendiente)
│
├── requirements.txt                   # Dependencias Python
├── .gitignore
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.10 o superior
- Las dependencias están listadas en `requirements.txt`

### Instalación recomendada

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/TFG-EEG-Connectivity-GCN-CNN.git
cd TFG-EEG-Connectivity-GCN-CNN

# Crear entorno virtual
python -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🚀 Cómo ejecutar

### 1. Descargar los datasets crudos

Los datasets son públicos y se pueden conseguir en:

- **PhysioNet Motor Imagery**: https://physionet.org/content/eegmmidb/
- **DS003766**: https://openneuro.org/datasets/ds003766
- **ds003490**: https://openneuro.org/datasets/ds003490
- **Schizophrenia (Kaggle)**: enlace pendiente
- **TUH EEG**: https://isip.piconepress.com/projects/tuh_eeg/
- **HBN**: http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/

> ⚠️ **Importante**: los datasets crudos NO están incluidos en este repositorio. Algunos requieren registro o acuerdo de uso (TUH EEG, HBN). Consultar las fuentes oficiales.

### 2. Generar las matrices de conectividad

Ejecutar los notebooks de `data_preparation/` (uno por dataset). Cada uno genera las matrices de conectividad y las guarda en una carpeta `ConnectivityMatrices_<dataset>/`.

### 3. Entrenar los modelos

Abrir el notebook `classification/train_models.ipynb`. Contiene dos celdas:
una para el modelo GCN y otra para el modelo CNN. En cada una, ajustar las
variables al inicio de la sección de configuración:

```python
DATASET = "PhysioNet"   # o "DS003766_v2", "ds003490", "Schizophrenia", "TUH", "HBN"
channels = "8"           # o "16", "32", "64"
```

Y modificar la ruta a la carpeta de matrices si es necesario.

---

## 📈 Resumen de resultados

Mejor test accuracy por dataset y arquitectura:

| Dataset | GCN | CNN | Mejor global |
|---|---|---|---|
| PhysioNet | 56.94% (16 ch) | **59.38%** (64 ch) | CNN |
| DS003766 | 97.67% (16 ch) | **99.39%** (64 ch) | CNN |
| ds003490 | 61.18% (16 ch) | **62.20%** (16 ch) | CNN |
| Schizophrenia | 59.35% (8 ch) | **64.18%** (8 ch) | CNN |
| TUH EEG | **46.31%** (16 ch) | 44.88% (16 ch) | GCN |
| HBN | 50.00% (16 ch) | **50.95%** (64 ch) | CNN |

**Observaciones generales:**

- CNN supera a GCN en 5 de 6 datasets en su mejor configuración.
- En TUH EEG y HBN ambos modelos rinden cerca o por debajo del azar (50%) → la metodología no funciona cross-subject para estas tareas (seizures y attention CBCL).
- DS003766 es un problema fácilmente separable (>97% en todas las configuraciones).

Para el análisis detallado, ver la memoria del TFG en `docs/memoria_TFG.pdf`.

---

