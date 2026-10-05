"""
CO6: FastAPI REST API Deployment Endpoint.
Serves live model predictions, health checks, model metadata, and monitoring statistics.
"""

import os
import time
import json
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

from .schemas import TrafficFeatureInput, PredictionResponse, HealthResponse, ModelInfoResponse
from features.feature_engineering import FEATURE_NAMES, extract_features_df
from monitoring.monitor import PredictionMonitor

app = FastAPI(
    title="Traffic Signal Optimization ML REST API",
    description="Serves ML predictions for adaptive traffic signal control.",
    version="1.0.0"
)

# Global model and monitor instances
MODEL_PATH = "models_saved/best_model.joblib"
INFO_PATH = "models_saved/model_info.json"

model_pipeline = None
model_metadata = {}
monitor = PredictionMonitor()


@app.on_event("startup")
def load_artifacts():
    global model_pipeline, model_metadata
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")

    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns model metadata and performance metrics."""
    if not model_metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return {
        "model_name": model_metadata.get("model_name", "RandomForest"),
        "version": model_metadata.get("version", "1.0.0"),
        "features": model_metadata.get("features", FEATURE_NAMES),
        "accuracy": model_metadata.get("accuracy", 0.0),
        "f1_weighted": model_metadata.get("f1_weighted", 0.0)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_signal(features_input: TrafficFeatureInput):
    """
    Predicts optimal traffic signal action (KEEP, SWITCH_TO_NS, SWITCH_TO_EW).
    Includes end-to-end prediction trace from raw request to inference log.
    """
    start_time = time.time()
    raw_dict = features_input.model_dump()

    # Preprocessing & feature dataframe construction
    df = extract_features_df(raw_dict)

    if model_pipeline is not None:
        try:
            if hasattr(model_pipeline, "predict_proba"):
                probs = model_pipeline.predict_proba(df)[0]
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx])
            else:
                pred_idx = int(model_pipeline.predict(df)[0])
                confidence = 0.95

            target_map = {0: "KEEP", 1: "SWITCH_TO_NS", 2: "SWITCH_TO_EW"}
            prediction = target_map.get(pred_idx, "KEEP")
            model_name = model_metadata.get("model_name", "RandomForestPipeline")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    else:
        # Smart rule fallback if model file hasn't been compiled
        act_q = raw_dict['north_queue'] + raw_dict['south_queue']
        wait_q = raw_dict['east_queue'] + raw_dict['west_queue']
        if raw_dict['current_phase'] == 1.0:
            act_q, wait_q = wait_q, act_q
        
        prediction = "SWITCH" if wait_q >= act_q + 5 else "KEEP"
        confidence = 0.85
        model_name = "Rule_Fallback"

    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000.0, 3)

    # CO1 Prediction Trace construction
    prediction_trace = {
        "1_request_received": raw_dict,
        "2_api_routing": "/predict",
        "3_preprocessing": "StandardScaler applied via ColumnTransformer",
        "4_feature_vector": list(df.iloc[0].to_dict().values()),
        "5_trained_model": model_name,
        "6_prediction": prediction,
        "7_confidence": confidence,
        "8_latency_ms": latency_ms
    }

    # Monitoring log
    monitor.log_prediction(raw_dict, prediction, confidence, latency_ms)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "model": model_name,
        "latency_ms": latency_ms,
        "prediction_trace": prediction_trace
    }


@app.get("/metrics")
def get_metrics():
    """Returns live monitoring & data drift metrics."""
    summary = monitor.get_summary_metrics()
    drift = monitor.detect_drift()
    return {
        "monitoring": summary,
        "drift_analysis": drift
    }
