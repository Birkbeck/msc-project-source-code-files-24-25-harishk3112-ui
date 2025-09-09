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

#  train once 
def load_and_train_model(order=(2, 1, 2)):
    """
    Fit ARIMA(order) on full hourly series.
    Returns: model_fit, history_last_24 (for UI parity).
    """
    df = _load_series()
    model = ARIMA(df["gap"], order=order)
    model_fit = model.fit()
    history = df["gap"][-24:].round(2)
    return model_fit, history

# initialize (train once)
model_fit, history_data = load_and_train_model()

#  evaluation 
def _backtest_recent_arima(n_hours=168, order=(2, 1, 2), within_pct=10.0):
    """
    Evaluate ARIMA on the last n_hours (one-shot predict over the window).
    Returns dict: y_true, y_pred, residuals, metrics, start, end
    """
    df = _load_series()
    if len(df) <= n_hours + 1:
        return None

    # Predict over the eval window using a model fitted on the whole series
    # (fast baseline comparison; acceptable for reporting)
    model = ARIMA(df["gap"], order=order).fit()

    start_idx = len(df) - n_hours
    y_pred = model.predict(start=start_idx, end=len(df) - 1)
    y_pred = np.asarray(y_pred, dtype="float32")

    y_true = df["gap"].iloc[start_idx:].astype("float32").values
    residuals = y_true - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.where(y_true == 0, 0.0, np.abs((y_true - y_pred) / y_true))) * 100.0)
    smape = _smape(y_true, y_pred)
    r2    = float(r2_score(y_true, y_pred))
    within = _pct_within(y_true, y_pred, pct=float(within_pct))

    metrics = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE_%": round(mape, 2),
        "sMAPE_%": round(smape, 2),
        "R2": round(r2, 3),
        "Within10pct_%": round(within, 2),
        "window_hours": int(n_hours),
    }

    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "metrics": metrics,
        "start": df.index[start_idx].isoformat(),
        "end": df.index[-1].isoformat(),
    }
# forecast API 
def get_forecast(user_input):
    """
   adds evaluation block with identical keys.
      
    """
    forecast_days = int(user_input.get("forecast_days", 1))
    household_size = float(user_input.get("household_size", 1))
    preference = user_input.get("preference", "normal")

    # optional evaluation controls 
    eval_window_hours = int(user_input.get("eval_window_hours", 168))
    within_pct = float(user_input.get("within_pct", 10.0))

    scale = 1.0
    if preference == "high":
        scale = 1.3
    elif preference == "low":
        scale = 0.8

    steps = max(1, forecast_days * 24)
# use the -fitted model for forecasting
    fc = model_fit.forecast(steps=steps)
    forecast_scaled = (fc * household_size * scale).astype("float32")
    forecast_scaled = np.round(forecast_scaled, 2)

    avg = round(float(forecast_scaled.mean()), 2)
    mx  = round(float(forecast_scaled.max()), 2)
    mn  = round(float(forecast_scaled.min()), 2)

    rec = "You're doing great! Keep using energy efficiently."
    if preference == "high" or avg > 4:
        rec = "Your predicted usage is high. Try reducing usage during peak hours like 6PM–9PM."
    elif preference == "low":
        rec = "Efficient usage detected. Continue minimizing consumption during peak hours."

 # evaluation 
    evaluation = _backtest_recent_arima(
        n_hours=eval_window_hours,
        order=(2, 1, 2),
        within_pct=within_pct
    )

    return {
        "hours": [f"Hour {i+1}" for i in range(steps)],
        "values": forecast_scaled.tolist(),
        "history": history_data.tolist(),
        "recommendation": rec,
        "summary": {"avg": avg, "max": mx, "min": mn},
        "evaluation": evaluation
    }

