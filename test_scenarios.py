import pytest
from scenarios import weather_scenario
from forecast import FEATURES


def row():
    return {**dict.fromkeys(FEATURES, 0), "mnth": 1, "season": 1, "weathersit": 1,
            "temp": .3, "atemp": .3, "hum": .4, "windspeed": .2, "lag24": 10, "lag168": 20}


def test_weather_changes_preserve_history_and_source():
    reference = row()
    captured = []
    def predictor(observation):
        captured.append(observation.model_dump())
        return {"demand": observation.temp * 100}
    result = weather_scenario(reference, {"temp": .8}, predictor)
    assert reference == row()
    assert result["original_prediction"] == 30
    assert result["scenario_prediction"] == 80
    assert result["difference"] == 50
    assert captured[1]["lag24"] == captured[0]["lag24"] == 10
    assert captured[1]["hr"] == captured[0]["hr"]


@pytest.mark.parametrize("changes", [{"lag24": 99}, {"temp": 2}, {"hum": float("nan")}])
def test_invalid_changes_are_rejected(changes):
    with pytest.raises(ValueError):
        weather_scenario(row(), changes)


def test_missing_artifacts_render_instructions_without_training(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import forecast
    monkeypatch.setattr(forecast, "ROOT", tmp_path)
    app = AppTest.from_file("app.py").run(timeout=20)
    assert not app.exception
    assert "artefactos" in app.warning[0].value
    assert "forecast.py" in app.code[0].value


def test_language_preserves_weather_scenario():
    from streamlit.testing.v1 import AppTest
    from forecast import ROOT
    if not (ROOT / "artifacts/model.joblib").exists():
        pytest.skip("Local prepared demo artifacts are required.")
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    app.slider(key="temp").set_value(.15).run(timeout=30)
    prediction = app.session_state["temp"]
    app.radio(key="language").set_value("en").run(timeout=30)
    assert not app.exception
    assert app.slider(key="temp").value == prediction == .15
