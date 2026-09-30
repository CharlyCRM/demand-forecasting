# Anticipar la demanda / Demand Forecasting

Proyecto de portfolio de Carlos Ramírez Martín. Predicción horaria de alquileres con **evaluación temporal**, baseline estacional, modelo de gradient boosting, API y demo.

## Ejecutar (Python 3.11)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python forecast.py
pytest -q
streamlit run app.py --server.address 127.0.0.1
# API, en otro terminal
uvicorn api:app --host 127.0.0.1 --port 8001
```

El entrenamiento descarga el dataset, conserva su SHA256 y genera `artifacts/metrics.json`, `predictions.csv` y el modelo local. El modelo no se sube a GitHub; puede reconstruirse con el script.

## Evaluación

60% inicial para entrenamiento, 20% para selección entre dos configuraciones y 20% final como test. Después de seleccionar hiperparámetros se reentrena con train+validación. Se comparan MAE y RMSE con la demanda de la misma hora de la semana anterior. No se utilizan `casual` o `registered`, que revelan el objetivo.

Los lags se unen por hora exacta; los huecos del dataset no desplazan la referencia temporal. La evaluación simula predicciones sucesivas a una hora con historia observada actualizada. **No equivale a predecir una semana sin conocer sus valores intermedios.** La meteorología se asume disponible; el experimento no mide el error adicional de predecirla. No se promete un porcentaje de mejora ni eficacia en otros sistemas.

Resultados ejecutados: ver [artifacts/metrics.json](artifacts/metrics.json). La demo permite visualizar el periodo de test e inferir sobre una observación.

## Datos y licencia

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), Hadi Fanaee-T, 2013. Dataset bajo CC BY 4.0 según UCI. Se descarga en `data/`, excluido del repositorio. Código bajo MIT.
