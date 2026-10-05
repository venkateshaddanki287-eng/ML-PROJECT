# IMPLEMENTATION MAP (COLLEGE PRESENTATION REFERENCE)

> **Note**: This document maps course requirements and outcomes to exact source files, classes, methods, and verified runtime evidence.

---

## 1. End-to-End ML Lifecycle (CO1)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Data Simulation & Collection** | `generate_dataset.py` | `generate_dataset()` | `python generate_dataset.py` | Generates 5,000 traffic state snapshots across 5 arrival scenarios to `data/processed/traffic_dataset.csv`. |
| **Feature Transformation Pipeline** | `features/feature_engineering.py` | `build_preprocessor()` | `python train.py` | `ColumnTransformer` with `StandardScaler` scaling numeric features without training-serving skew. |
| **Model Packaging & Storage** | `training/train.py` | `run_training_pipeline()` | `python train.py` | Saves trained pipeline to `models_saved/best_model.joblib` and metadata to `models_saved/model_info.json`. |
| **REST API Serving** | `deployment/api.py` | `@app.post("/predict")` | `uvicorn deployment.api:app --reload` | FastAPI endpoint serving live JSON predictions with latency calculation and trace logging. |
| **Prediction Tracing & Logging** | `monitoring/monitor.py` | `log_prediction()` | `python main.py --mode ml_based` | Logs raw inputs, scaled features, model predictions, confidence scores, and latencies to `monitoring/prediction_logs.jsonl`. |

---

## 2. Linear Supervised Learning (CO2)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | `models/linear_models.py` | `LinearRegression` in `get_regression_models()` | `python train.py` | RMSE: 4248.07, MAE: 3463.98 (Predicting `future_waiting_time`). |
| **Ridge ($L_2$ Regularization)** | `models/linear_models.py` | `Ridge(alpha=1.0)` | `python train.py` | Shrinks regression weights smoothly to handle multicollinearity. |
| **Lasso ($L_1$ Regularization)** | `models/linear_models.py` | `Lasso(alpha=0.1)` | `python train.py` | Encourages weight sparsity for feature selection. |
| **ElasticNet ($L_1+L_2$)** | `models/linear_models.py` | `ElasticNet(alpha=0.1, l1_ratio=0.5)` | `python train.py` | Combines $L_1$ absolute and $L_2$ squared coefficient penalties. |
| **Logistic Regression** | `models/linear_models.py` | `LogisticRegression()` | `python train.py` | Test Accuracy: 79.4%, Weighted F1: 0.7441 (Predicting `signal_action`). |
| **Multinomial Logistic Regression** | `models/linear_models.py` | `LogisticRegression(multi_class='multinomial')` | `python train.py` | Evaluated across multiclass signal target (`KEEP`, `SWITCH_TO_NS`, `SWITCH_TO_EW`). |
| **Feature Scaling** | `models/linear_models.py` | `StandardScaler()` in Pipeline | `python train.py` | Scaler fitted strictly on `X_train` to prevent data leakage. |

---

## 3. Tree-Based Learning (CO3)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree** | `models/tree_models.py` | `DecisionTreeClassifier(max_depth=6)` | `python train.py` | Test Accuracy: 86.6%, Weighted F1: 0.8534. |
| **Random Forest** | `models/tree_models.py` | `RandomForestClassifier(n_estimators=100)` | `python train.py` | Test Accuracy: 86.5%, Weighted F1: 0.8506. |
| **Gradient Boosting** | `models/tree_models.py` | `GradientBoostingClassifier()` | `python train.py` | Test Accuracy: 86.5%, Weighted F1: 0.8526. |
| **LightGBM / XGBoost** | `models/tree_models.py` | `LGBMClassifier()` / `XGBClassifier()` | `python train.py` | Test Accuracy: 84.2%, Weighted F1: 0.8334. |
| **Feature Importance** | `models/tree_models.py` | `feature_importances_` extraction | `python evaluate.py` | Identified `current_green_time` (46.5%), `current_phase` (15.4%), `south_queue` (5.2%) as top drivers. |
| **Ensemble Variance Reduction** | `models/tree_models.py` | `demonstrate_variance_reduction()` | `python train.py` | Proves RF variance (Std 0.0039) is 1.72x smoother than single Decision Tree (Std 0.0067). |

---

## 4. Unsupervised Learning (CO4)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **K-Means Clustering** | `models/clustering.py` | `fit_kmeans(n_clusters=3)` | `python train.py` | Discovered 3 traffic regimes: Low (355 cars avg), Medium (4144 cars avg), Heavy (4938 cars avg). |
| **Hierarchical Clustering** | `models/clustering.py` | `fit_hierarchical(n_clusters=3)` | `python train.py` | `AgglomerativeClustering` structure over scaled feature vectors. |
| **DBSCAN** | `models/clustering.py` | `fit_dbscan(eps=1.2)` | `python train.py` | Identified 24 outlier noise points (0.48% of dataset). |
| **PCA** | `models/clustering.py` | `run_pca(n_components=2)` | `python train.py` | 2D projection capturing 70.67% cumulative explained variance. |
| **t-SNE & UMAP** | `models/clustering.py` | `run_tsne()`, `run_umap()` | `python train.py` | Non-linear 2D manifold coordinate reduction. |
| **Anomaly Detection** | `models/anomaly_detection.py` | `IsolationForest(contamination=0.05)` | `python train.py` | Detected 250 unusual traffic states (5.0%) representing sudden gridlocks. |

---

## 5. Model Performance & Evaluation (CO5)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Cross-Validation** | `training/evaluate.py` | `run_cross_validation()` | `python evaluate.py` | 5-Fold Stratified CV Mean Score: `0.8456 +/- 0.0129`. |
| **Classification Metrics** | `training/evaluate.py` | `evaluate_classification()` | `python evaluate.py` | Accuracy: `0.865`, Precision: `0.855`, Recall: `0.655`, F1: `0.851`, ROC-AUC: `0.931`. |
| **Confusion Matrix** | `training/evaluate.py` | `confusion_matrix()` | `python evaluate.py` | Matrix output: `[[752, 9, 11], [64, 49, 0], [51, 0, 64]]`. |
| **Probability Calibration** | `training/evaluate.py` | `evaluate_calibration()` | `python evaluate.py` | Brier Score: `0.0659` (Low Brier score proves well-calibrated probabilities). |
| **Grid Search Tuning** | `training/tune.py` | `tune_grid_search()` | `python train.py` | `GridSearchCV` selected `max_depth=10`, `n_estimators=100` (Best CV: 0.8473). |
| **Random Search Tuning** | `training/tune.py` | `tune_random_search()` | `python train.py` | `RandomizedSearchCV` sampled hyperparameter distributions. |
| **Bayesian Optimization** | `training/tune.py` | `tune_bayesian_lightweight()` | `python train.py` | Sequential Gaussian Process surrogate optimization selected `n_estimators=80`, `max_depth=8` (Best CV: 0.8405). |

---

## 6. ML Engineering, Deployment & Simulator Integration (CO6)

| Requirement | Source File | Class / Function | Execution Command | Verified Output / Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Training-Serving Skew Prevention** | `features/feature_engineering.py` | `Pipeline` with `ColumnTransformer` | `python train.py` | Identical `StandardScaler` transformations used in training and REST API serving. |
| **Model Packaging** | `training/train.py` | `joblib.dump()` | `python train.py` | Saved to `models_saved/best_model.joblib` & `models_saved/model_info.json`. |
| **REST API (`/health`)** | `deployment/api.py` | `health_check()` | `uvicorn deployment.api:app --reload` | Returns `{"status": "healthy", "model_loaded": true}`. |
| **REST API (`/predict`)** | `deployment/api.py` | `predict_signal()` | `uvicorn deployment.api:app --reload` | Accepts JSON features, returns `{"prediction": "SWITCH_TO_EW", "confidence": 0.50, "latency_ms": 4.1}`. |
| **Simulator ML Connection** | `simulation/controller.py` | `TrafficController._evaluate_ml()` | `python main.py --mode ml_based` | Reads intersection queues/timers, calls ML model/API, and toggles signals dynamically while maintaining yellow clearance rules. |
| **Rule-Based Baseline Comparison** | `simulation/controller.py` | `TrafficController._evaluate_rule_based()` | `python main.py --mode rule_based` | Preserved original heuristic rules for comparative benchmark evaluation. |
| **Data Drift Detection** | `monitoring/monitor.py` | `detect_drift()`, `calculate_psi()` | `python evaluate.py` | Calculates Population Stability Index (PSI) and Kolmogorov-Smirnov test statistics against reference dataset. |
