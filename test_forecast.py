import numpy as np
import pandas as pd
from api import app
from fastapi.testclient import TestClient
from forecast import FEATURES, prepare


def test_seasonal_lags_use_clock_time_and_do_not_include_target():
    time = pd.date_range("2024-01-01", periods=250, freq="h").delete(20)
    raw = pd.DataFrame(
        {
            "dteday": time.strftime("%Y-%m-%d"),
            "hr": time.hour,
            "cnt": np.arange(len(time)),
        }
    )
    frame = prepare(raw)
    original = pd.Series(raw.cnt.to_numpy(), index=time)
    for row in frame.itertuples():
        assert row.lag168 == original.loc[row.timestamp - pd.Timedelta(hours=168)]
        assert row.lag24 == original.loc[row.timestamp - pd.Timedelta(hours=24)]
    assert not {"cnt", "casual", "registered"}.intersection(FEATURES)


def test_api_rejects_invalid_or_missing_features():
    client = TestClient(app)
    assert client.post("/predict", json={"hr": 24}).status_code == 422
    assert client.get("/health").status_code == 200
