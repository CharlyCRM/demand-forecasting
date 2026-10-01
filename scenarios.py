"""Hypothetical weather edits, preserving observed calendar and demand history."""
import numpy as np
from api import Observation, predict
from forecast import FEATURES

WEATHER = ("temp", "atemp", "hum", "windspeed")


def weather_scenario(row, changes, predictor=predict):
    if set(changes) - set(WEATHER):
        raise ValueError("Only weather fields may be changed.")
    integer = {"hr", "weekday", "mnth", "workingday", "holiday", "season", "weathersit"}
    original = {key: int(row[key]) if key in integer else float(row[key]) for key in FEATURES}
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in changes.values()):
        raise ValueError("Weather values must be finite and normalized to [0,1].")
    scenario = {**original, **changes}
    reference = predictor(Observation(**original))["demand"]
    modified = predictor(Observation(**scenario))["demand"]
    return {"original_prediction": reference, "scenario_prediction": modified,
            "difference": modified - reference,
            **{f"original_{field}": original[field] for field in WEATHER}, **scenario}
