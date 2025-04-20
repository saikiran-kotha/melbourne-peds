# api.py
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Query
import pandas as pd
from src.inference import init_buffers, predict_recursive, BUFFERS
from collections import deque

AEST = timezone(timedelta(hours=10))

# Prepare initial buffers from a snapshot (first app start)
_hist = pd.read_parquet("data/interim/pedestrian_recent.parquet")
init_buffers(_hist)

app = FastAPI()


# Route: /predict?sensor=...&hours=24
@app.get("/predict")
def forecast(sensor: str, hours: int = 24):
    # Determine prediction window
    now = datetime.now(AEST).replace(minute=0, second=0, microsecond=0)
    start = now + timedelta(hours=1)

    # Clone only this sensor’s buffers
    buf_clone = {
        "lag": deque(BUFFERS[sensor]["lag"], maxlen=168),
        "roll": deque(BUFFERS[sensor]["roll"], maxlen=3),
    }
    buffers_tmp = {sensor: buf_clone}

    # Run recursive forecast on the clone
    predictions = predict_recursive(sensor, start, hours, buffers=buffers_tmp)

    # Return results
    return [{"timestamp": ts, "count": int(y)} for ts, y in predictions]

