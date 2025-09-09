import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#  helpers 
def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    denom = (np.abs(y_true) + np.abs(y_pred))
    sm = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(sm) * 100.0)

def _pct_within(y_true, y_pred, pct=10.0):
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    ape = np.where(y_true == 0, 0.0, np.abs((y_true - y_pred) / y_true) * 100.0)
    return float(np.mean(ape <= pct) * 100.0)

def _load_series():
    """Load + prepare hourly series consistent with your LSTM pipeline."""
    df = pd.read_csv(
        "data/household_power_consumption.txt",
        sep=';',
        parse_dates=[[0, 1]],
        infer_datetime_format=True,
        low_memory=False,
        na_values=['?']
    )
    df.rename(columns={"Date_Time": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)

    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["Global_active_power"])
    df = df.resample("H").mean()
    df = df.rename(columns={"Global_active_power": "gap"})  # align naming with other model
    return df
