from pathlib import Path
import joblib
from src.load import clean_data
from src.features import add_is_holiday, add_lags, add_lockdown_flag, add_day_of_week
import pandas as pd
from collections import deque
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

ARTIFACT_DIR = Path("src/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CUT_DATE_VAL = pd.Timestamp(year=2024, month=12, day=1)
LAGS = (24, 168)

FEATURES_NUM = ["HourDay", "lag_24h", "lag_168h"]
FEATURES_CAT = ["Sensor_Name", "is_holiday", "is_lockdown", "day_of_week"]

ped, sensors = clean_data()

ped = (
    ped.pipe(add_is_holiday)
    .pipe(add_lockdown_flag)
    .pipe(add_day_of_week)
    .pipe(add_lags, lags=LAGS)
    # .pipe(add_roll3h)
    # Drop rows that still have NaNs in lag columns
    .dropna(subset=["lag_24h", "lag_168h"])
)

X = ped[FEATURES_NUM + FEATURES_CAT]
y = ped["Total_of_Directions"]


train_mask = ped["Sensing_Date"] < CUT_DATE_VAL

X_train, X_val = X[train_mask], X[~train_mask]
y_train, y_val = y[train_mask], y[~train_mask]

print("Mean hourly count :", y_train.mean())
print("Std hourly count :", y_train.std())

# naive_pred = X_val['lag_1h'].values
# naive_mae = mean_absolute_error(y_val, naive_pred)
# print("Naive MAE (lag_1h):", naive_mae)

naive_week = X_val['lag_168h'].values
print("Naive 168h MAE:", mean_absolute_error(y_val, naive_week))

demo = ped[ped['Sensor_Name'] == ped['Sensor_Name'].iloc[0]].head(6)
print("Demo data (first 6 rows, one sensor):")
print(demo[['HourDay', 'Total_of_Directions', 'lag_24h', 'lag_168h']])

t0 = pd.Timestamp("2024-11-25 00:00")
buffers = {}  # {sensor_name: {'lag': deque, 'roll': deque}}
for sensor in ped['Sensor_Name'].unique():
    hist = ped.loc[
        (ped['Sensor_Name'] == sensor) &
        (ped['Sensing_Date'] <= t0)  # inclusive of t0
        ].sort_values('Sensing_Date').tail(168)  # last 168 rows

    counts = hist['Total_of_Directions'].values

    buffers[sensor] = {
        'lag': deque(counts, maxlen=168),
        # 'roll': deque(counts[-3:], maxlen=3)
    }


pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT)
    ]
)

X_train_enc = pre.fit_transform(X_train)   # fits on train
X_val_enc = pre.transform(X_val)         # transform only

model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=24,
    eval_metric="mae",
    early_stopping_rounds=50
)

model.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],
    verbose=False,
)
print("Best iteration:", model.best_iteration)

mae = mean_absolute_error(y_val, model.predict(X_val_enc))
print("Validation MAE:", mae)

# Save the model
joblib.dump({"pre": pre, "model": model}, ARTIFACT_DIR / "ped_model.joblib")