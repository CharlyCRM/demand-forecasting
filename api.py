from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from forecast import FEATURES, ROOT
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

app = FastAPI(title="Demand Forecasting", version="1.0.0")


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hr: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    mnth: int = Field(ge=1, le=12)
    workingday: int = Field(ge=0, le=1)
    holiday: int = Field(ge=0, le=1)
    season: int = Field(ge=1, le=4)
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)
    lag24: float = Field(ge=0)
    lag168: float = Field(ge=0)


@lru_cache
def model():
    path = ROOT / "artifacts/model.joblib"
    if not path.exists():
        raise HTTPException(503, "Run python forecast.py before inference.")
    return joblib.load(path)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": (ROOT / "artifacts/model.joblib").exists()}


@app.post("/predict")
def predict(observation: Observation):
    with threadpool_limits(limits=2):
        value = float(
            model().predict(pd.DataFrame([observation.model_dump()])[FEATURES])[0]
        )
    return {
        "demand": max(0, value),
        "unit": "rentals/hour",
        "protocol": "rolling_one_hour_ahead",
    }
