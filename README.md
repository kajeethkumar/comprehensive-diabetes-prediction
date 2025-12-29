# 🩺 Convolution-Based Feature Learning Framework for Comprehensive Diabetes Diagnosis Using Clinical Health Indicators

This project presents a unified machine learning framework for diabetes diagnosis and risk score prediction by integrating feature selection techniques with convolution-based ROCKET-family models. Filter, Wrapper, and Embedded feature selection methods are employed to reduce dimensionality and enhance discriminative learning on high-dimensional clinical data. The framework evaluates binary classification, multiclass classification, and regression tasks using Arsenal, Rocket, MiniRocket, and MultiRocket models.

The project supports:

- ✅ **Binary classification** (Diabetes: Yes / No)
- ✅ **Multiclass classification** (Diabetes stages)
- ✅ **Regression** (Diabetes risk score)
- ✅ **Filter-based, Wrapper-based, and Embedded feature selection**
- ✅ **AEON ROCKET-family models + sklearn baselines**
- ✅ **YAML-driven experiment configuration**

This repository is designed for **research, thesis work, and reproducible experiments**.

---

## 📌 Features

- Unified preprocessing pipeline
- Feature selection methods:
  - Filter (SelectKBest – ANOVA)
  - Wrapper (RFE + Random Forest)
  - Embedded (Random Forest importance)
- Models:
  - ROCKET
  - MiniROCKET
  - MultiROCKET
  - Arsenal
  - Rotation Forest (regression)
- YAML configuration for experiments
- Clean result logging to CSV

---

## 🐍 Python Version

**Python 3.9 or newer** is required.

Check your version:
```
python --version
```

## Installation 

### Clone this repository
```
git clone https://github.com/your-username/diabetes-ml-pipeline.git
cd diabetes-ml-pipeline
```

### Create a virtual environment
```
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Install dependencies
```
pip install -r requirements.txt
```
### Recommended AEON installation
```
pip install numpy scipy scikit-learn
pip install aeon --no-deps
```

Install the editable mode
```
pip install -e .
```

```
diabetes-ml-pipeline/
│
├── data/
│   └── diabetes_dataset.csv        # Dataset
│
├── configs/                         # YAML experiment configs
│   ├── binary.yaml
│   ├── multiclass.yaml
│   └── regression.yaml
│
├── src/
│   ├── data/
│   │   ├── loader.py                # Dataset loading
│   │   └── preprocessing.py         # Encoding + scaling
│   │
│   ├── feature_selection/
│   │   ├── base.py                  # Base interface
│   │   ├── filter.py                # SelectKBest
│   │   ├── wrapper.py               # RFE
│   │   └── embedded.py              # RF importance
│   │
│   ├── models/
│   │   ├── classifiers.py           # AEON classifiers
│   │   └── regressors.py            # AEON regressors
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Metrics
│   │   └── evaluator.py             # Model evaluation
│   │
│   ├── utils/
│       ├── aeon_utils.py             # 3D reshaping
│       ├── io.py                     # File utilities
│       ├── logging.py                # Logging
|       └── config.py                 # YAML loader
│       ├── run_binary.py
│       ├── run_multiclass.py
│       ├── run_regression.py
│       └── runner.py                 # Core experiment runner
├── requirements.txt
├── setup.py
└── README.md
```

## configuration

Experiments are controlled using YAML files inside ```configs/```.

Example: ```configs/binary.yaml```
```
task: binary
target: diagnosed_diabetes
csv_path: data/diabetes_dataset.csv

feature_selection:
  methods: [filter, wrapper, embedded]
  k_values: [10, 15, 20, 25]

split:
  test_size: 0.2
  random_state: 42
  stratify: true

output_dir: results/binary
```

## How to run experiments

### Binary Classification
```
!python -m src.experiments.run_binary

```
### Multi-Classification
```
!python -m src.experiments.run_multiclass.py
```

### Regression
```
!python -m src.experiments.run_regression.py
```


## Summary

Feature selection enhances convolution-based diabetes prediction by improving accuracy, and efficiency. MultiRocket consistently outperforms Rocket, MiniRocket, and Arsenal across classification and regression tasks, with embedded feature selection delivering the most robust and compact performance.