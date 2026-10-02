"""Entrena y evalúa la predicción de alquileres respetando el orden temporal."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "hr",
    "weekday",
    "mnth",
    "workingday",
    "holiday",
    "season",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "lag24",
    "lag168",
]
SOURCE = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def load_data():
    target = ROOT / "data/hour.csv"
    if not target.exists():
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(archive.read("hour.csv"))
    return pd.read_csv(target)


def prepare(raw):
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame.dteday) + pd.to_timedelta(
        frame.hr, unit="h"
    )
    frame = frame.sort_values("timestamp").set_index("timestamp")
    # Buscamos la hora exacta: una lectura ausente no debe desplazar los retardos.
    lookup = frame["cnt"]
    for hours in (24, 168):
        frame[f"lag{hours}"] = lookup.reindex(
            frame.index - pd.Timedelta(hours=hours)
        ).to_numpy()
    return frame.dropna(subset=["lag24", "lag168"]).reset_index()


def metrics(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(root_mean_squared_error(y, prediction)),
    }


def train():
    frame = prepare(load_data())
    n = len(frame)
    train_end, validation_end = int(n * 0.6), int(n * 0.8)
    train_frame, validation, test = (
        frame.iloc[:train_end],
        frame.iloc[train_end:validation_end],
        frame.iloc[validation_end:],
    )
    scores = []
    with threadpool_limits(limits=2):
        for leaves in (15, 31):
            model = HistGradientBoostingRegressor(
                max_leaf_nodes=leaves,
                max_iter=180,
                learning_rate=0.07,
                l2_regularization=1,
                early_stopping=False,
                random_state=42,
            )
            model.fit(train_frame[FEATURES], train_frame.cnt)
            scores.append(
                (
                    mean_absolute_error(
                        validation.cnt, model.predict(validation[FEATURES])
                    ),
                    leaves,
                )
            )
        _, leaves = min(scores)
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=leaves,
            max_iter=180,
            learning_rate=0.07,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        )
        model.fit(
            frame.iloc[:validation_end][FEATURES], frame.iloc[:validation_end].cnt
        )
        prediction = np.maximum(0, model.predict(test[FEATURES]))
    report = {
        "dataset": "UCI Bike Sharing / hour.csv",
        "source": SOURCE,
        "sha256": hashlib.sha256((ROOT / "data/hour.csv").read_bytes()).hexdigest(),
        "protocol": "Rolling one-hour-ahead prediction; exact 24h/168h lags known at each origin. Weather covariates are assumed available, not independently forecast.",
        "split": {
            "train": len(train_frame),
            "validation": len(validation),
            "test": len(test),
            "test_start": str(test.timestamp.iloc[0]),
            "test_end": str(test.timestamp.iloc[-1]),
        },
        "selected_max_leaf_nodes": leaves,
        "baseline_seasonal_168h": metrics(test.cnt, test.lag168),
        "gradient_boosting": metrics(test.cnt, prediction),
        "limitations": [
            "One historical bike-sharing system; no production validation.",
            "Not a multi-step batch forecast: observed history updates at each prediction origin.",
            "Weather values are assumed known; results may be optimistic relative to actual weather forecasts.",
            "Temporal slices have different seasonal distributions.",
        ],
    }
    directory = ROOT / "artifacts"
    directory.mkdir(exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    (directory / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(
        {
            "timestamp": test.timestamp.astype(str),
            "actual": test.cnt,
            "prediction": prediction,
            "baseline": test.lag168,
        }
    ).to_csv(directory / "predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
