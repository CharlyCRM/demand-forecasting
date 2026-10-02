# Anticipar la demanda

¿Cuántas bicicletas se alquilarán la próxima hora? En este proyecto trabajo con un histórico de alquileres para comparar una predicción basada en machine learning con una referencia sencilla: los alquileres de la misma hora de la semana anterior.

La aplicación permite recorrer el histórico, contrastar las predicciones con lo que ocurrió y probar cómo responde el modelo cuando cambian las condiciones meteorológicas. Es una forma de explorar tanto sus aciertos como los momentos en los que se equivoca.

[Abrir la aplicación](https://demand-forecasting-production-4989.up.railway.app) · [Ver el proyecto en mi portfolio](https://carlos-ramirez-martin.up.railway.app/es/projects/demand-forecasting/)

## Qué puedes probar

- Elegir un tramo del histórico y comparar alquileres reales, predicción y referencia semanal.
- Partir de una observación y modificar temperatura, sensación térmica, humedad o viento para ver cómo cambia la predicción.
- Descargar el escenario en CSV y consultar el informe de evaluación.

Los controles meteorológicos usan valores normalizados entre 0 y 1, no grados ni velocidades en unidades físicas. Cambiar de observación recupera sus condiciones originales; cambiar de idioma conserva la selección.

La demo está en español e inglés. Se suspende cuando no se utiliza, por lo que la primera apertura puede tardar. Solo está publicada la interfaz de Streamlit; la API se ejecuta por separado en local.

## Cómo lo he planteado

Uso `HistGradientBoostingRegressor` con calendario, meteorología y alquileres registrados 24 y 168 horas antes. Esas referencias se buscan por fecha y hora exactas: si falta una lectura, no tomo por error otra fila como si correspondiera al mismo momento.

Divido los datos por orden temporal: el primer 60 % sirve para entrenar, el siguiente 20 % para elegir entre dos configuraciones y el último 20 % queda reservado para evaluar. Una vez elegida la configuración, vuelvo a entrenar con los dos primeros bloques y comparo MAE y RMSE con la referencia semanal. Excluyo `casual` y `registered`, porque su suma revela el objetivo.

Los resultados, las fechas de cada bloque y el SHA-256 de los datos están en [artifacts/metrics.json](artifacts/metrics.json).

## Hasta dónde llega el experimento

La evaluación simula predicciones sucesivas a una hora, incorporando el histórico observado en cada momento. No es una previsión de toda una semana hecha de una sola vez. Además, supone que conocemos la meteorología de la hora que se va a predecir; no incluye el error de un pronóstico meteorológico.

El escenario mantiene fijos el calendario y el histórico. Cambiar la temperatura muestra la respuesta del modelo, no demuestra que ese cambio provoque más o menos alquileres. Algunas combinaciones pueden quedar fuera de las condiciones habituales del dataset. No asigno un resultado real a esos escenarios ni los uso para recalcular las métricas.

Los datos pertenecen a un único sistema de bicicletas compartidas. Los resultados no garantizan el mismo comportamiento en otro servicio.

## Ejecutarlo en local

Necesitas Python 3.11. Desde la carpeta del repositorio:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python forecast.py
pytest -q
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Abre http://localhost:8501/; añade `/?lang=en` para entrar en inglés.

`forecast.py` descarga el dataset si falta y genera el modelo, `artifacts/metrics.json` y `artifacts/predictions.csv`. El modelo queda fuera de Git y se reconstruye con ese script. La aplicación carga estos archivos; si faltan, muestra instrucciones sin entrenar ni descargar por su cuenta.

Para probar la API, abre otro terminal con el mismo entorno activado:

```sh
uvicorn api:app --host 127.0.0.1 --port 8001
```

Su documentación está en http://127.0.0.1:8001/docs. `POST /predict` recibe las variables definidas en `Observation` y devuelve alquileres por hora. `GET /health` indica si el modelo está disponible.

La configuración local limita Streamlit a tu equipo y desactiva su telemetría. `PORTFOLIO_URL` cambia el enlace de regreso; por defecto usa `http://localhost:4322`.

## Datos y licencia

Utilizo [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), de Hadi Fanaee-T (2013), bajo CC BY 4.0 según UCI. Los datos descargados quedan fuera de Git. El código propio tiene licencia MIT.

Carlos Ramírez Martín
