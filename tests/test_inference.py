from collections import deque

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from src.inference import init_buffers, predict_recursive
from src.api import app  # for integration test

AEST = timezone(timedelta(hours=10))


@pytest.fixture
def buffers_setup(tmp_path):
    # create a tiny recent snapshot
    dates = pd.date_range("2025-04-10", periods=168, freq="h", tz=AEST)
    df = pd.DataFrame({
        "Sensor_Name": ["X"] * 168,
        "Sensing_Date": dates,
        "Total_of_Directions": range(168)
    })
    init_buffers(df)
    return df


def test_predict_recursive_length(buffers_setup):
    now = buffers_setup["Sensing_Date"].max().astimezone(AEST)
    start = now + timedelta(hours=1)

    buffers = {
        "X": {
            "lag": deque(buffers_setup["Total_of_Directions"].values, maxlen=168),
            "roll": deque(buffers_setup["Total_of_Directions"].values[-3:], maxlen=3),
        }
    }

    preds = predict_recursive("X", start, 5, buffers=buffers)

    for ts, y in preds:
        print(f"{ts=}, {type(y)=}, {y=}")
        assert isinstance(y, (int, float)), f"bad type: {type(y)}"
        assert y >= 0, f"negative prediction: {y}"


def test_api_predict():
    client = TestClient(app)
    resp = client.get("/predict", params={"sensor": "X", "hours": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and len(data) == 3
    for item in data:
        assert "timestamp" in item and "count" in item
        assert isinstance(item["timestamp"], str)
        assert isinstance(item["count"], int)
