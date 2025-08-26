# LSTM with calendar features + residuals/metrics/backtest
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------- constants / globals -----------------------
SEQ_LEN = 72
MODEL_PATH = Path("lstm_model.keras")
TRAIN_INFO_PATH = Path("train_info.json")

_model = None
x_scaler = StandardScaler()
y_scaler = StandardScaler()

_series = None          # raw target series (float32)
_df = None              # features + target dataframe
FEATURES = None         # list[str]
_history_last_24 = None # last 24 target values (float32)

# cache for backtest/evaluation (residuals & metrics)
_eval_cache = None

# data loading & features 
def _load_series():
    df = pd.read_csv(
        "data/household_power_consumption.txt",
        sep=";",
        low_memory=False,
        na_values=["?"],
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )
    df = df.drop(columns=["Date", "Time"]).dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # target
    df["gap"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # hourly average
    df = df.resample("h").mean().dropna(subset=["gap"])

    # time features
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)

    # rolling stats on target
    df["roll_mean_24"] = df["gap"].rolling(24, min_periods=1).mean()
    df["roll_std_24"]  = df["gap"].rolling(24, min_periods=1).std().fillna(0.0)

    feats = [
        "hour_sin","hour_cos",
        "dow_sin","dow_cos",
        "roll_mean_24","roll_std_24",
        "gap"
    ]
    df = df[feats].copy()

    global _series, _df, _history_last_24, FEATURES
    _series = df[["gap"]].astype("float32").values
    _history_last_24 = df["gap"].astype("float32")[-24:].round(2)
    FEATURES = ["hour_sin","hour_cos","dow_sin","dow_cos","roll_mean_24","roll_std_24"]
    _df = df.astype("float32")

def _make_supervised():
    X, y = [], []
    vals = _df.copy()
    X_all = x_scaler.fit_transform(vals[FEATURES].values)
    y_all = y_scaler.fit_transform(vals[["gap"]].values).ravel()

    for i in range(SEQ_LEN, len(vals)):
        X.append(X_all[i-SEQ_LEN:i, :])
        y.append(y_all[i])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

#  model 
def _build_model(input_features: int):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, input_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

#  metrics helpers
def _smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    denom = (np.abs(y_true) + np.abs(y_pred))
    sm = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(sm) * 100.0)

def _pct_within(y_true, y_pred, pct=10):
    """Percent of points whose absolute percentage error is within ±pct%."""
    y_true = np.asarray(y_true, dtype="float32")
    y_pred = np.asarray(y_pred, dtype="float32")
    ape = np.where(y_true == 0, 0.0, np.abs((y_true - y_pred) / y_true) * 100.0)
    return float(np.mean(ape <= pct) * 100.0)

#  backtest (evaluation) 
def _future_feature_row(ts, last_24_vals):
    hour = ts.hour
    dow  = ts.dayofweek
    row = {
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin":  np.sin(2*np.pi*dow/7),
        "dow_cos":  np.cos(2*np.pi*dow/7),
        "roll_mean_24": np.mean(last_24_vals) if len(last_24_vals)>0 else float(_series[-1,0]),
        "roll_std_24":  float(np.std(last_24_vals)) if len(last_24_vals)>1 else 0.0,
    }
    return np.array([row[f] for f in FEATURES], dtype="float32")

def _backtest_recent(n_hours=168):
    """
    One-step-ahead rolling evaluation over the last n_hours.
    Returns y_true, y_pred, residuals and metrics.
    """
    if _model is None or len(_df) <= SEQ_LEN + n_hours + 1:
        return None

    feat_scaled = x_scaler.transform(_df[FEATURES].values)
    start_idx = len(_df) - n_hours
    window = deque(feat_scaled[start_idx-SEQ_LEN:start_idx], maxlen=SEQ_LEN)

    y_true, y_pred = [], []
    tail = deque(_df["gap"].values[start_idx-24:start_idx].tolist(), maxlen=24)
    current_time = _df.index[start_idx-1]

    for i in range(n_hours):
        x = np.expand_dims(np.array(window, dtype="float32"), axis=0)
        yhat_scaled = _model.predict(x, verbose=0)[0, 0]
        yhat = float(y_scaler.inverse_transform([[yhat_scaled]])[0, 0])

        current_time = current_time + pd.Timedelta(hours=1)
        y_true_val = float(_df["gap"].iloc[start_idx + i])
        y_true.append(y_true_val)
        y_pred.append(yhat)

        # strict one-step: append truth
        tail.append(y_true_val)
        f_next = _future_feature_row(current_time, np.array(tail, dtype="float32"))
        f_next_scaled = x_scaler.transform(f_next.reshape(1, -1))[0]
        window.append(f_next_scaled)

    y_true = np.array(y_true, dtype="float32")
    y_pred = np.array(y_pred, dtype="float32")
    residuals = y_true - y_pred

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.where(y_true == 0, 0.0, np.abs((y_true - y_pred) / y_true))) * 100.0)
    smape = _smape(y_true, y_pred)
    r2   = float(r2_score(y_true, y_pred))
    within_10 = _pct_within(y_true, y_pred, pct=10)

    metrics = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MAPE_%": round(mape, 2),
        "sMAPE_%": round(smape, 2),
        "R2": round(r2, 3),
        "Within10pct_%": round(within_10, 2),
        "window_hours": int(n_hours),
    }
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "metrics": metrics,
        "start": _df.index[len(_df) - n_hours].isoformat(),
        "end": _df.index[-1].isoformat(),
    }

#  training info I/O 
def _save_train_info(info: dict):
    try:
        TRAIN_INFO_PATH.write_text(json.dumps(info, indent=2))
    except Exception:
        pass

def _load_train_info():
    if TRAIN_INFO_PATH.exists():
        try:
            return json.loads(TRAIN_INFO_PATH.read_text())
        except Exception:
            return None
    return None

#  training / init
def load_and_train_model(n_runs: int = 1):
    """
    Optionally train multiple runs and store average/variation of val loss.
    """
    _load_series()
    if len(_series) <= SEQ_LEN + 1:
        return None, _history_last_24

    X, y = _make_supervised()

    global _model, _eval_cache
    train_info_runs = []

    if MODEL_PATH.exists():
        _model = load_model(MODEL_PATH)
        # compute and cache evaluation once
        _eval_cache = _backtest_recent(n_hours=168)
        return _model, _history_last_24

    # Train from scratch; record validation loss histories
    for run in range(n_runs):
        m = _build_model(X.shape[2])
        cb = [
            EarlyStopping(patience=4, restore_best_weights=True),
            ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
        ]
        hist = m.fit(
            X, y,
            epochs=14,
            batch_size=64,
            validation_split=0.1,
            shuffle=False,
            callbacks=cb,
            verbose=0
        )
        val_losses = [float(v) for v in hist.history.get("val_loss", [])]
        train_info_runs.append({"run": run+1, "val_loss": val_losses})
        _model = m  # keep last (best is checkpointed)

    _model.save(MODEL_PATH)

    # summarize & persist
    try:
        finals = [r["val_loss"][-1] for r in train_info_runs if r["val_loss"]]
        train_summary = {
            "runs": len(train_info_runs),
            "avg_final_val_loss": float(np.mean(finals)) if finals else None,
            "std_final_val_loss": float(np.std(finals)) if finals else None,
            "val_histories": train_info_runs,
        }
        _save_train_info(train_summary)
    except Exception:
        pass

    # cache evaluation window
    _eval_cache = _backtest_recent(n_hours=168)

    return _model, _history_last_24

# initialize (optionally set n_runs>1)
model_fit, history_data = load_and_train_model(n_runs=1)

# forecasting 
def _multi_step_forecast(steps: int) -> np.ndarray:
    if _model is None or len(_df) <= SEQ_LEN:
        last = float(_series[-1, 0])
        return np.full(steps, last, dtype="float32")

    feat_scaled = x_scaler.transform(_df[FEATURES].values)
    window = deque(feat_scaled[-SEQ_LEN:], maxlen=SEQ_LEN)

    tail = deque(_df["gap"].values[-24:].tolist(), maxlen=24)
    current_time = _df.index[-1]

    preds = []
    for _ in range(steps):
        x = np.expand_dims(np.array(window, dtype="float32"), axis=0)
        yhat_scaled = _model.predict(x, verbose=0)[0, 0]
        yhat = float(y_scaler.inverse_transform([[yhat_scaled]])[0, 0])

        preds.append(yhat)
        tail.append(yhat)

        current_time = current_time + pd.Timedelta(hours=1)
        f_next = _future_feature_row(current_time, np.array(tail, dtype="float32"))
        f_next_scaled = x_scaler.transform(f_next.reshape(1, -1))[0]
        window.append(f_next_scaled)

    preds = np.array(preds, dtype="float32")
    preds = np.maximum(preds, 0.0)
    return preds

def get_forecast(user_input):
    forecast_days = int(user_input.get("forecast_days", 1))
    household_size = float(user_input.get("household_size", 1))
    preference = user_input.get("preference", "normal")

    scale = 1.0
    if preference == "high":
        scale = 1.3
    elif preference == "low":
        scale = 0.8

    steps = max(1, forecast_days * 24)
    base = _multi_step_forecast(steps)

    values = np.round(np.maximum(base * household_size * scale, 0.0), 2)
    avg, mx, mn = map(lambda v: round(float(v), 2),
                      (values.mean(), values.max(), values.min()))
    rec = "You're doing great! Keep using energy efficiently."
    if preference == "high" or avg > 4:
        rec = "Your predicted usage is high. Try reducing usage during peak hours like 6PM–9PM."
    elif preference == "low":
        rec = "Efficient usage detected. Continue minimizing consumption during peak hours."

    # evaluation data (residuals + metrics on last 7 days = 168 hours)
    global _eval_cache
    
    # Check if user wants a different evaluation window
    eval_window_hours = user_input.get("eval_window_hours", 168)
    
    # If window changed, recalculate evaluation
    if _eval_cache is None or _eval_cache["metrics"]["window_hours"] != eval_window_hours:
        _eval_cache = _backtest_recent(n_hours=eval_window_hours)

    train_info = _load_train_info()

    return {
        "hours": [f"Hour {i+1}" for i in range(steps)],
        "values": values.tolist(),
        "history": history_data.tolist(),
        "recommendation": rec,
        "summary": {"avg": avg, "max": mx, "min": mn},
        "evaluation": _eval_cache,   # residuals, y_true, y_pred, metrics
        "training_info": train_info, # avg/std val loss across runs, if available
    }
