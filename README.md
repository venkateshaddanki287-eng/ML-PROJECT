# 🚦 Machine Learning Traffic Signal Optimization System

A machine learning-driven traffic signal optimization system that dynamically controls traffic lights at a 4-way intersection based on real-time vehicle queue features, queue growth rates, and arrival trends.

---

## 1. System Overview

The system extends a rule-based 4-way intersection traffic simulator into a data-driven Machine Learning pipeline:
* **Dataset Generation**: Simulates traffic state transitions across diverse arrival scenarios.
* **Feature Engineering**: Extracts 14 structured traffic features (`north_queue`, `south_queue`, `east_queue`, `west_queue`, arrival rates, timers, queue growth, traffic density, phase).
* **Supervised Learning**: Trains Linear models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression) and Tree Ensembles (DecisionTree, RandomForest, GradientBoosting, LightGBM/XGBoost).
* **Unsupervised Learning**: Discovers latent traffic regimes via K-Means, Agglomerative Clustering, DBSCAN, 2D PCA, t-SNE/UMAP, and detects anomalies using Isolation Forest.
* **Model Evaluation & Tuning**: Performs 5-fold Stratified Cross-Validation, probability calibration (Brier score), Grid Search, Random Search, and Sequential Bayesian Optimization.
* **REST API Deployment**: Serves low-latency JSON predictions via FastAPI (`/health`, `/predict`, `/model-info`, `/metrics`).
* **Simulator Integration**: Connects the trained ML model back into the traffic signal controller to adaptively switch lights while maintaining hard safety clearance rules.
* **Monitoring & Drift**: Logs live predictions and tracks Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics for data drift detection.

---

## 2. Project Architecture

```text
              🚗 TRAFFIC SIMULATOR (simulation/intersection.py)
                                     │
                                     ↓
                          Traffic State Features
                                     │
                                     ↓
                    Feature Engineering Pipeline (features/)
                                     │
                                     ↓
              ┌──────────────────────┴──────────────────────┐
              │                                             │
      Training Pipeline                             REST API Serving (FastAPI)
   (models/ & training/)                              (deployment/api.py)
              │                                             │
              ↓                                             ↓
   Trained Model Pipeline                         Signal Action Prediction
(models_saved/best_model.joblib)             (KEEP / SWITCH_TO_NS / SWITCH_TO_EW)
                                                            │
                                                            ↓
                                                Traffic Signal Controller
                                             (simulation/controller.py)
                                                            │
                                                            ↓
                                               🚦 INTERSECTION SIGNAL SWITCH
                                                            │
                                                            ↓
                                               Live Monitoring & Drift Log
                                                 (monitoring/monitor.py)
```

---

## 3. How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Generate Dataset
```bash
python generate_dataset.py
```

### Step 2: Train Models & Package Best Pipeline
```bash
python train.py
```

### Step 3: Run Standalone Evaluation Suite
```bash
python evaluate.py
```

### Step 4: Run REST API Server
```bash
uvicorn deployment.api:app --reload
```
View FastAPI interactive documentation at `http://127.0.0.1:8000/docs`.

### Step 5: Run Traffic Simulator
```bash
# Baseline Rule-Based Controller
python main.py --mode rule_based --scenario heavy_traffic

# ML-Based Learned Controller
python main.py --mode ml_based --scenario heavy_traffic

# Terminal signal visualization with ML-based controller
python main.py --mode visual --scenario heavy_traffic

# Terminal visualization with rule-based controller
python main.py --mode visual --controller rule_based --scenario heavy_traffic
```

### Visual Traffic Signal Simulation

Run `python main.py --mode visual` to display traffic lights in the terminal. The ML controller uses the saved model by default; pass `--controller rule_based` for the baseline. Lights mirror the live intersection state and controller-driven yellow clearance. Use Space to pause/resume, S to step while paused, R to reset, and Q to quit.

ML prediction → controller decision → intersection signal state → terminal lights

### Step 6: Run Unit Tests
```bash
pytest
```

---

## 4. API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health and model load status |
| `POST` | `/predict` | Serves real-time signal action predictions (`KEEP`, `SWITCH_TO_NS`, `SWITCH_TO_EW`) |
| `GET` | `/model-info` | Returns trained model metadata and accuracy metrics |
| `GET` | `/metrics` | Returns live monitoring stats, prediction distribution, and PSI data drift analysis |

---

## 5. College Presentation Mapping
For college evaluation reference mapping requirements to exact source files and code classes, view [`IMPLEMENTATION_MAP.md`](file:///c:/Users/nagaraju/Downloads/ML---Project-main/IMPLEMENTATION_MAP.md).
