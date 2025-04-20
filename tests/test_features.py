import pandas as pd
import numpy as np
import pytest
from src.features import (
    add_is_holiday,
    add_lockdown_flag,
    add_day_of_week,
    add_lags,
    add_roll3h,
    vic_holidays,
    LOCK_WINDOWS
)
from tests.test_inference import AEST

# sample date ranges
DATES = pd.to_datetime([
    "2025-01-01 12:00",  # New Year's Day (holiday)
    "2020-04-01 12:00",  # in lockdown window
    "2025-04-18 15:00",  # normal day
])


@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "Sensor_Name": ["X"] * 3,
        "Sensing_Date": DATES,
        "HourDay": [12, 13, 14],
        "Total_of_Directions": [100, 200, 300]
    })


def test_add_is_holiday(toy_df):
    out = add_is_holiday(toy_df)
    for i, row in out.iterrows():
        expected = int(row["Sensing_Date"].date() in vic_holidays)
        assert row["is_holiday"] == expected


def test_add_lockdown_flag(toy_df):
    out = add_lockdown_flag(toy_df)
    # second row falls in LOCK_WINDOWS
    assert out.loc[1, "is_lockdown"] == 1
    assert out.loc[2, "is_lockdown"] == 0


def test_add_day_of_week(toy_df):
    out = add_day_of_week(toy_df)
    # day names match the dates
    expected = [d.day_name() for d in DATES]
    assert out["day_of_week"].tolist() == expected


def test_add_lags():
    df = pd.DataFrame({
        "Sensor_Name": ["X"] * 25,
        "Sensing_Date": pd.date_range("2025-01-01", periods=25, freq="h"),
        "HourDay": list(range(25)),
        "Total_of_Directions": list(range(25))
    })
    df = add_lags(df, lags=(1, 24))

    assert df.loc[1, "lag_1h"] == 0
    assert df.loc[24, "lag_1h"] == 23
    assert df.loc[24, "lag_24h"] == 0


def test_add_roll3h(toy_df):
    df = toy_df.copy()
    # make sure rolling works for 3 rows
    out = add_roll3h(df)
    # first row: window=[100] → mean=100, then shifted -> NaN
    assert np.isnan(out.loc[0, "roll3h"])
    # second row: window=[100,200] → mean=150, shifted -> first row’s value? actually index 1: window→mean of rows0→100
    # to be safe: check that after row 3 we get mean of first three values
    df2 = pd.DataFrame({
        "Sensor_Name": ["X"] * 4,
        "Sensing_Date": pd.date_range("2025-01-01", periods=4, freq="h"),
        "HourDay": [0, 1, 2, 3],
        "Total_of_Directions": [10, 20, 30, 40]
    })
    out2 = add_roll3h(df2)
    # at index 3: roll of [10,20,30] → mean=20, then shifted by 1 → out2.loc[3,"roll3h"]==20
    assert out2.loc[3, "roll3h"] == 20
