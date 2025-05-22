from datetime import timezone, timedelta
import pandas as pd
import numpy as np
import holidays

vic_holidays = holidays.country_holidays(country="Australia", subdiv="Victoria", years=range(2021, 2099), observed=True)
LOCKDOWNS = [
    ("2020-03-31", "2020-05-31"),
    ("2020-07-01", "2020-07-08"),
    ("2020-07-09", "2020-10-27"),
    ("2021-02-13", "2021-02-17"),
    ("2021-05-28", "2021-06-10"),
    ("2021-07-16", "2021-07-27"),
    ("2021-08-05", "2021-10-21"),
]
AEST = timezone(timedelta(hours=10))
# convert once to Timestamp bound pairs
LOCK_WINDOWS = [(pd.Timestamp(s, tz=AEST), pd.Timestamp(e,tz=AEST)) for s, e in LOCKDOWNS]


def add_is_holiday(df: pd.DataFrame,
                   date_col: str = "Sensing_Date") -> pd.DataFrame:
    """
    Add a binary `is_holiday` column (1 = Vic public holiday, else 0).

    Parameters
    ----------
    df : DataFrame with a datetime column
    date_col : name of that datetime column

    Returns
    -------
    Same DataFrame with a new int8 column `is_holiday`.
    """
    out = df.copy()
    out["is_holiday"] = out[date_col].dt.date.isin(vic_holidays)

    return out


def add_lags(df: pd.DataFrame,
             lags: tuple[int, ...] = (24, 168), # Default updated
             target_col: str = "Total_of_Directions") -> pd.DataFrame:
    """
    Append lag feature columns for each sensor.

    Parameters
    ----------
    df : DataFrame sorted by Sensor_Name & time
    lags : tuple of integers (hours)
    target_col : the column to lag

    Returns
    -------
    DataFrame with new columns: f"lag_{h}h" for each h in `lags`
    """
    out = df.copy()
    for hour in lags:
        out[f"lag_{hour}h"] = (out.groupby("Sensor_Name")[target_col].shift(hour))

    return out


def add_lockdown_flag(df: pd.DataFrame,
                      date_col: str = "Sensing_Date") -> pd.DataFrame:
    """
    Add binary `is_lockdown` based on metropolitan‑Melbourne stay‑at‑home orders.
    """
    df = df.copy()
    dts = pd.to_datetime(df[date_col]).dt.tz_localize("Australia/Melbourne")

    # vectorized test: OR‑reduce across all windows
    mask = pd.Series(False, index=df.index)
    for start, end in LOCK_WINDOWS:
        mask |= (dts >= start) & (dts <= end)

    df["is_lockdown"] = mask.astype("int8")
    return df


# def add_roll3h(df, target_col="Total_of_Directions"): # Removed for now, as it's not used in the model.'
#     """
#     Add a 3‑hour rolling mean per sensor, shifted by 1hour to avoid leakage.
#     """
#     df = df.copy()
#
#     # Ensure rows are sorted — already true in your cleaner, but be explicit
#     df = df.sort_values(["Sensor_Name", "Sensing_Date", "HourDay"],
#                         ignore_index=True)
#
#     # Compute rolling mean for each sensor
#     roll = (
#         df.groupby("Sensor_Name")[target_col]
#           .rolling(window=3, min_periods=1)
#           .mean()
#           .shift(1)                      # <-- lag by one hour
#           .reset_index(level=0, drop=True)
#     )
#
#     df["roll3h"] = roll
#     return df


def add_day_of_week(df):
    df = df.copy()
    df['day_of_week'] = df['Sensing_Date'].dt.day_name()  # "Monday"..."Sunday"
    return df