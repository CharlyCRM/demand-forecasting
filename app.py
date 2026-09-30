import json

import pandas as pd
import streamlit as st
from api import Observation, predict
from forecast import FEATURES, ROOT, load_data, prepare

st.set_page_config(page_title="Demand Forecasting", layout="wide")
st.title("Anticipar la demanda")
st.caption(
    "Predicción de alquileres de bicicletas · evaluación temporal · Carlos Ramírez Martín"
)
if not (ROOT / "artifacts/metrics.json").exists():
    st.info("Ejecuta python forecast.py para entrenar y evaluar.")
    st.stop()
report = json.loads((ROOT / "artifacts/metrics.json").read_text())
cols = st.columns(3)
cols[0].metric("MAE · modelo", f"{report['gradient_boosting']['mae']:.2f}")
cols[1].metric("MAE · baseline", f"{report['baseline_seasonal_168h']['mae']:.2f}")
cols[2].metric("Observaciones test", report["split"]["test"])
predictions = pd.read_csv(ROOT / "artifacts/predictions.csv")
window = st.slider("Horas de test a visualizar", 24, 336, 168)
st.line_chart(
    predictions.head(window).set_index("timestamp")[
        ["actual", "prediction", "baseline"]
    ]
)
with st.expander("Predicción sobre una observación del conjunto de test"):
    frame = prepare(load_data())
    offset = st.slider("Observación", 0, report["split"]["test"] - 1, 0)
    row = frame.iloc[-report["split"]["test"] + offset]
    result = predict(
        Observation(
            **{
                k: float(row[k])
                if k
                not in [
                    "hr",
                    "weekday",
                    "mnth",
                    "workingday",
                    "holiday",
                    "season",
                    "weathersit",
                ]
                else int(row[k])
                for k in FEATURES
            }
        )
    )
    st.write(
        {
            "timestamp": str(row.timestamp),
            "predicted": result["demand"],
            "actual": int(row.cnt),
        }
    )
st.warning(
    "Evaluación histórica con covariables meteorológicas asumidas conocidas. No constituye una validación de producción ni una predicción multihorizonte."
)
st.json(report)
