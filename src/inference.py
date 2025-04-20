"""
inference.py
============
Loads the trained bundle ONCE and provides:

* init_buffers(df_hist)     -> dict   (call at cold‑start)
* predict_one(sensor, ts)   -> int    (1‑hour forecast)
* predict_recursive(sensor, ts_start, n_hours) -> list[(ts, int)]
"""

from collections import deque
from datetime import timedelta
from pathlib import Path
import joblib, numpy as np, pandas as pd
from src.features import vic_holidays, LOCK_WINDOWS, AEST  # reuse existing objects

ARTIFACT_DIR = Path("src/artifacts")

# Load bundle
BUNDLE = joblib.load(ARTIFACT_DIR / "ped_model.joblib")
PRE, MODEL = BUNDLE["pre"], BUNDLE["model"]

# Buffers: one deque per sensor (populated at app start)
BUFFERS = {}  # populated by init_buffers() below


def init_buffers(df_hist: pd.DataFrame):
    """
    df_hist: DataFrame with >=168 real hours per sensor.
    """
    global BUFFERS
    BUFFERS.clear()
    for sensor, grp in (
            df_hist.sort_values("Sensing_Date")
                    .groupby("Sensor_Name")
    ):
        counts = grp["Total_of_Directions"].tail(168).values
        BUFFERS[sensor] = {
            "lag": deque(counts, maxlen=168),
            "roll": deque(counts[-3:], maxlen=3),
        }


# Internal helper to build a single feature row
def _build_row(sensor: str, ts, buf):
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=AEST)

    return pd.DataFrame({
        "Sensor_Name": [sensor],
        "HourDay": [ts.hour],
        "lag_1h": [buf["lag"][-1]],
        "lag_24h": [buf["lag"][-24]],
        "lag_168h": [buf["lag"][0]],
        "roll3h": [np.mean(buf["roll"])],
        "is_holiday": [int(ts.date() in vic_holidays)],
        "is_lockdown": [int(any(s <= ts <= e for s, e in LOCK_WINDOWS))],
        "day_of_week": [ts.strftime("%A")],
    })


# Single‑hour prediction (updates buffers)
def predict_one(sensor, ts, buf):
    row = _build_row(sensor, ts, buf)
    y_hat = MODEL.predict(PRE.transform(row))[0]
    y_hat = max(y_hat, 0)
    y_hat = float(y_hat)
    # build a new buffer dict
    new_buf = {
        "lag": deque(buf["lag"], maxlen=168),
        "roll": deque(buf["roll"], maxlen=3)
    }
    new_buf["lag"].append(y_hat)
    new_buf["roll"].append(y_hat)
    return y_hat, new_buf


# Recursive multi‑hour forecast
def predict_recursive(sensor, ts_start, n_hours, buffers):
    buf = buffers[sensor]
    predictions = []
    ts = ts_start
    for _ in range(n_hours):
        y_hat, buf = predict_one(sensor, ts, buf)
        predictions.append((ts, y_hat))
        ts += timedelta(hours=1)
    return predictions
