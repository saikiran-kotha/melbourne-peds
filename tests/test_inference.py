from collections import deque
from unittest.mock import AsyncMock

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from src.inference import init_buffers
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


# def test_predict_recursive_length(buffers_setup):
#     now = buffers_setup["Sensing_Date"].max().astimezone(AEST)
#     start = now + timedelta(hours=1)
#
#     buffers = {
#         "X": {
#             "lag": deque(buffers_setup["Total_of_Directions"].values, maxlen=168),
#             "roll": deque(buffers_setup["Total_of_Directions"].values[-3:], maxlen=3),
#         }
#     }
#
#     preds = predict_recursive("X", start, 5, buffers=buffers)
#
#     for ts, y in preds:
#         print(f"{ts=}, {type(y)=}, {y=}")
#         assert isinstance(y, (int, float)), f"bad type: {type(y)}"
#         assert y >= 0, f"negative prediction: {y}"


def test_api_predict(buffers_setup, mocker):
    # Mock the external API call
    mock_fetch_lags = mocker.patch('src.api.fetch_live_lags_from_external_api', new_callable=AsyncMock)
    mock_fetch_lags.return_value = {"lag_24h": 100, "lag_168h": 150}

    client = TestClient(app)
    resp = client.get("/predict", params={"sensor": "X"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict) 
    assert "timestamp" in data and "predicted_count" in data and "sensor_name" in data
    assert data["sensor_name"] == "X"
    assert isinstance(data["timestamp"], str)
    assert isinstance(data["predicted_count"], int)
