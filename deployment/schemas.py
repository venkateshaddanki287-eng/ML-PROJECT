"""
Pydantic Schemas for REST API.
Defines strict request/response data validation models.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class TrafficFeatureInput(BaseModel):
    north_queue: float = Field(..., ge=0, description="Cars waiting in North lane")
    south_queue: float = Field(..., ge=0, description="Cars waiting in South lane")
    east_queue: float = Field(..., ge=0, description="Cars waiting in East lane")
    west_queue: float = Field(..., ge=0, description="Cars waiting in West lane")
    north_arrival_rate: float = Field(default=1.0, ge=0)
    south_arrival_rate: float = Field(default=1.0, ge=0)
    east_arrival_rate: float = Field(default=1.0, ge=0)
    west_arrival_rate: float = Field(default=1.0, ge=0)
    current_green_time: float = Field(default=0.0, ge=0, description="Duration current light has been green")
    previous_queue: float = Field(default=0.0, ge=0)
    queue_growth: float = Field(default=0.0)
    waiting_time: float = Field(default=0.0, ge=0)
    traffic_density: float = Field(default=0.0, ge=0, le=1.0)
    current_phase: float = Field(default=0.0, description="0=North/South, 1=East/West")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model: str
    latency_ms: float
    prediction_trace: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    features: list
    accuracy: float
    f1_weighted: float
