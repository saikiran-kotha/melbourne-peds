# api.py
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
from src.inference import init_buffers, BUFFERS, predict_current_hour_with_live_lags
from collections import deque
import requests
from datetime import datetime
import asyncio

AEST = timezone(timedelta(hours=10))

# Prepare initial buffers from a snapshot (first app start)
_hist = pd.read_parquet("data/interim/pedestrian_recent.parquet")
init_buffers(_hist)

app = FastAPI()

# Ensure this is your current v2.1 records endpoint
CITY_API_URL = (
    "https://melbournetestbed.opendatasoft.com/"
    "api/explore/v2.1/catalog/"
    "datasets/pedestrian-counting-system-monthly-counts-per-hour/records"
)

async def fetch_live_lags_from_external_api(sensor: str, current_ts: datetime) -> dict:
    def _get_lag(hours_back: int) -> float:
        target = current_ts - timedelta(hours=hours_back)

        api_params = {
            "refine.sensing_date": target.strftime("%Y-%m-%d"),
            "refine.hourday": str(target.hour),
            "refine.sensor_name": sensor,
            "limit": 1,
        }

        try:
            # Using CITY_API_URL which already includes the dataset
            resp = requests.get(CITY_API_URL, params=api_params, timeout=10)
            resp.raise_for_status()
        except requests.HTTPError as e: # Catch HTTPError for more specific details
            error_detail = f"External API error: Status {e.response.status_code} - {e.response.text}"
            # Attempt to parse JSON error from response if possible
            try:
                error_content = e.response.json()
                error_detail = f"External API error: {error_content}"
            except ValueError: # response is not JSON
                pass # use the text version
            raise HTTPException(
                status_code=e.response.status_code if e.response is not None else 500,
                detail=error_detail
            )
        except requests.RequestException as e:
            raise HTTPException(
                status_code=503, # Service Unavailable
                detail=f"Failed to connect to external API: {e}"
            )

        try:
            body = resp.json()
            results = body.get("results", []) # v2.1 uses "results"
            if not results:
                # It's possible the API returns 200 OK with empty results if no data matches
                raise HTTPException(
                    status_code=404, # Not Found, as no data matches the query
                    detail=f"No data returned from external API for {hours_back}h lag. Sensor: {sensor}, Target time: {target.isoformat()}"
                )
            return float(results[0]["pedestriancount"])
        except (ValueError, KeyError, IndexError) as e: # Added IndexError
            raise HTTPException(
                status_code=502, # Bad Gateway
                detail=f"Unexpected response format from external API: {e}. Response: {resp.text[:200]}" # Log part of response
            )

    lag_24h = await asyncio.to_thread(_get_lag, 24)
    lag_168h = await asyncio.to_thread(_get_lag, 168)
    
    return {
        "lag_24h": lag_24h,
        "lag_168h": lag_168h,
    }

# Route: /predict?sensor=...&hours=24
@app.get("/predict")
async def forecast_current_hour(sensor: str = Query(..., examples=["Your_Sensor_Name"])):
    """
    Predicts the pedestrian count for the specified sensor for the current hour
    using live 24-hour and 168-hour lag data fetched from an external API.
    """
    current_ts = datetime.now(AEST).replace(minute=0, second=0, microsecond=0)

    # Fetch live lag data
    try:
        # USER ACTION: Replace with your actual call to the implemented live lag fetching function
        live_lags = await fetch_live_lags_from_external_api(sensor, current_ts)
        live_lag_24h = live_lags["lag_24h"]
        live_lag_168h = live_lags["lag_168h"]
    except HTTPException as e:
        raise e # Re-raise HTTPException from the fetch function
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error fetching live lag data: {str(e)}")


    # Run prediction for the current hour
    try:
        prediction_count = predict_current_hour_with_live_lags(
            sensor=sensor,
            ts=current_ts,
            live_lag_24h=live_lag_24h,
            live_lag_168h=live_lag_168h
        )
    except Exception as e:
        # Catch errors from the prediction function itself (e.g., model expecting a feature not provided)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    return {
        "sensor_name": sensor,
        "timestamp": current_ts.isoformat(),
        "predicted_count": int(round(prediction_count))
    }