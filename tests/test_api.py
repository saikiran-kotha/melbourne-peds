# tests/test_api.py
import pytest
import requests
from fastapi import HTTPException
from datetime import datetime
from src.api import fetch_live_lags_from_external_api, CITY_API_URL


@pytest.mark.asyncio
async def test_fetch_live_lags_success(monkeypatch):
    # Fake a 200 OK response with the expected JSON payload
    class FakeResponse:
        status_code = 200
        text = "OK"
        def raise_for_status(self):
            pass
        def json(self):
            return {"lag_24h": 12.5, "lag_168h": 100.0}

    def fake_get(url, params, timeout):
        # Optionally assert that the URL and params are what you expect
        assert url == CITY_API_URL
        assert params["sensing_date"] == "2024-01-01"
        assert params["hourday"] == 0
        assert params["sensor_name"] == "sensorX"
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    ts = datetime(2024, 1, 1)  # no tzinfo needed for formatting YYYY-MM-DD / hour
    result = await fetch_live_lags_from_external_api("sensorX", ts)
    assert result == {"lag_24h": 12.5, "lag_168h": 100.0}

@pytest.mark.asyncio
async def test_fetch_live_lags_http_error(monkeypatch):
    # Fake a downstream API returning a 4xx/5xx
    class BadResponse:
        status_code = 404
        text = "Not Found"
        def raise_for_status(self):
            raise requests.HTTPError("404 Client Error")
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: BadResponse())

    with pytest.raises(HTTPException) as exc:
        await fetch_live_lags_from_external_api("sensorY", datetime(2024, 1, 2))
    assert exc.value.status_code == 404
    assert "External API error" in exc.value.detail

@pytest.mark.asyncio
async def test_fetch_live_lags_network_error(monkeypatch):
    # Fake a network-level failure (timeout, DNS, etc.)
    def fake_get(url, params, timeout):
        raise requests.RequestException("Network down")
    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(HTTPException) as exc:
        await fetch_live_lags_from_external_api("sensorZ", datetime(2024, 1, 3))
    assert exc.value.status_code == 503
    assert "Failed to connect" in exc.value.detail