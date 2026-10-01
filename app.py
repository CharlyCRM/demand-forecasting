import json
import pandas as pd
import streamlit as st
from forecast import ROOT, load_data, prepare
from scenarios import weather_scenario, WEATHER
from ui import init, tr, hero, require_files, line_chart

init("demand-forecasting", "Demand Forecasting")
hero("04 / DEMAND FORECASTING", tr("Anticipar. Comparar. Decidir.", "Anticipate. Compare. Decide."),
     tr("Explora una evaluación temporal y cambia la meteorología de una observación para descubrir cómo responde el modelo.",
        "Explore a temporal evaluation and change an observation’s weather to see how the model responds."))
require_files(ROOT, ["artifacts/metrics.json", "artifacts/predictions.csv", "artifacts/model.joblib", "data/hour.csv"],
              "python forecast.py")
report = json.loads((ROOT / "artifacts/metrics.json").read_text())
predictions = pd.read_csv(ROOT / "artifacts/predictions.csv", parse_dates=["timestamp"])
cols = st.columns(3)
cols[0].metric("MAE / MODEL", f"{report['gradient_boosting']['mae']:.2f}")
cols[1].metric("MAE / BASELINE", f"{report['baseline_seasonal_168h']['mae']:.2f}")
cols[2].metric("TEST / HOURS", report["split"]["test"])

with st.sidebar:
    st.markdown("### " + tr("Ventana histórica", "Historical window"))
    window = st.slider("Horas / Hours", 24, min(336, len(predictions)), 168, key="window")
    offset = st.slider("Inicio / Start", 0, max(0, len(predictions) - window), 0, key="offset")

with st.container(border=True):
    st.subheader(tr("01 / Predicción frente a realidad", "01 / Prediction versus reality"))
    selected = predictions.iloc[offset:offset + window].copy()
    selected = selected.rename(columns={"actual": "Observed", "prediction": "Model", "baseline": "Baseline"})
    line_chart(selected, ["Observed", "Model", "Baseline"], tr("Alquileres / hora", "Rentals / hour"))
    st.caption(tr("Test reservado. El baseline utiliza la demanda de la misma hora de la semana anterior.",
                  "Held-out test. The baseline uses demand at the same hour of the previous week."))
    st.download_button("CSV / " + tr("Comparación histórica", "Historical comparison"),
                       selected.to_csv(index=False).encode(), "historical-comparison.csv", "text/csv", key="history_export")

with st.container(border=True):
    st.subheader(tr("02 / ¿Y si cambiara el tiempo?", "02 / What if the weather changed?"))
    st.caption(tr("Selecciona una observación: el calendario y la demanda previa permanecen constantes.",
                  "Select an observation: calendar and prior demand stay fixed."))
    prepared = prepare(load_data())
    test_frame = prepared.iloc[-report["split"]["test"]:].reset_index(drop=True)
    observation = st.slider("Observación / Observation", 0, len(test_frame) - 1, 0, key="observation")
    row = test_frame.iloc[observation]
    if st.session_state.get("weather_reference") != observation:
        for field in WEATHER:
            st.session_state[field] = float(row[field])
        st.session_state.weather_reference = observation
    st.caption(f"{row.timestamp} · " + tr("Observado", "Observed") + f": {int(row.cnt)}")
    labels = {"temp": "Temperatura / Temperature", "atemp": "Sensación térmica / Feels like",
              "hum": "Humedad / Humidity", "windspeed": "Viento / Wind"}
    controls = st.columns(2)
    changes = {field: controls[i % 2].slider(labels[field], 0.0, 1.0, step=.01, key=field)
               for i, field in enumerate(WEATHER)}
    result = weather_scenario(row, changes)
    metrics = st.columns(3)
    metrics[0].metric(tr("Predicción original", "Original prediction"), f"{result['original_prediction']:.1f}")
    metrics[1].metric(tr("Escenario hipotético", "Hypothetical scenario"), f"{result['scenario_prediction']:.1f}")
    metrics[2].metric(tr("Diferencia / alquileres", "Difference / rentals"), f"{result['difference']:+.1f}")
    st.download_button("CSV / " + tr("Escenario", "Scenario"),
                       pd.DataFrame([{**result, "timestamp": str(row.timestamp), "observed_reference": int(row.cnt)}]).to_csv(index=False).encode(),
                       "weather-scenario.csv", "text/csv", key="scenario_export")
    st.warning(tr("Escenario hipotético, no efecto causal. Variables meteorológicas normalizadas de 0 a 1; combinaciones poco realistas pueden producir extrapolaciones. El valor observado solo pertenece al caso original.",
                  "Hypothetical scenario, not a causal effect. Weather variables are normalized to 0–1; unrealistic combinations may produce extrapolations. The observed count belongs only to the original case."))

st.info(tr("Predicción sucesiva a una hora con historia observada y meteorología asumida conocida. No es previsión multihorizonte ni validación de producción. Los escenarios no cambian las métricas del test.",
           "Rolling one-hour-ahead prediction with observed history and weather assumed known. Not a multi-step forecast or production validation. Scenarios do not change test metrics."))
with st.expander(tr("Protocolo y resultados completos", "Protocol and full results")):
    st.json(report)
