# LSTM with calendar features (model)

import os
# Reduce TensorFlow logs and disable some CPU optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import required libraries 
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

# Set sequence length for LSTM input and define path to save or load the model
SEQ_LEN = 72            
MODEL_PATH = Path("lstm_model.keras")

# Setup empty model and scalers for features and target
_model = None
x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Assigning variables for storing series data, dataframe, features, and last 24-hour history
_series = None          
_df = None              
FEATURES = None         
_history_last_24 = None


def _load_series():
    """
Loads household power consumption data, parses datetime,
creates time-based cyclical features, rolling statistics,
and prepares features and target series for modeling.
"""

    df = pd.read_csv(
        "data/household_power_consumption.txt",
        sep=";",
        low_memory=False,
        na_values=["?"],
    )
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["Date", "Time"]).dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    df["gap"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.resample("h").mean().dropna(subset=["gap"])        # use 'h'

    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)

    df["roll_mean_24"] = df["gap"].rolling(24, min_periods=1).mean()
    df["roll_std_24"]  = df["gap"].rolling(24, min_periods=1).std().fillna(0.0)

    feats = ["hour_sin","hour_cos","dow_sin","dow_cos","roll_mean_24","roll_std_24","gap"]
    df = df[feats].copy()

    global _series, _df, _history_last_24, FEATURES
    _series = df[["gap"]].astype("float32").values
    _history_last_24 = df["gap"].astype("float32")[-24:].round(2)
    FEATURES = ["hour_sin","hour_cos","dow_sin","dow_cos","roll_mean_24","roll_std_24"]
    _df = df.astype("float32")


def _make_supervised():
      """
    Converts the time series into supervised learning format by
    creating input sequences (X) of length SEQ_LEN and matching
    them with the next target value (y).
    """
    X, y = [], []
    vals = _df.copy()
    X_all = x_scaler.fit_transform(vals[FEATURES].values)
    y_all = y_scaler.fit_transform(vals[["gap"]].values).ravel()

    for i in range(SEQ_LEN, len(vals)):
        X.append(X_all[i-SEQ_LEN:i, :])
        y.append(y_all[i])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

def _build_model(input_features):
      """
    Creates an LSTM model with two layers, dropout to avoid overfitting,
    and a dense layer to predict energy use.
    """
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, input_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def load_and_train_model():
    """
Loads data, prepares it for training, and trains the LSTM model 
if no saved model exists. Saves the best model for future use.
"""
    _load_series()
    if len(_series) <= SEQ_LEN + 1:
        return None, _history_last_24

    X, y = _make_supervised()

    global _model
    if MODEL_PATH.exists():
        _model = load_model(MODEL_PATH)
        return _model, _history_last_24

    _model = _build_model(X.shape[2])
    cb = [
        EarlyStopping(patience=4, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True),
    ]
    _model.fit(X, y, epochs=14, batch_size=64, validation_split=0.1, shuffle=False, callbacks=cb, verbose=0)
    _model.save(MODEL_PATH)
    return _model, _history_last_24

model_fit, history_data = load_and_train_model()

def _future_feature_row(ts, last_24_vals):
        """
    Creates future prediction features from a timestamp and
    the last 24 hours of power usage.
    """
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


def _multi_step_forecast(steps:int)->np.ndarray:
     """
    Predicts energy use for several future hours.
    Starts with recent data, predicts the next value,
    adds it to the data, and repeats until all steps are done.
    """
    if _model is None or len(_df) <= SEQ_LEN:
        last = float(_series[-1,0])
        return np.full(steps, last, dtype="float32")

    feat_scaled = x_scaler.transform(_df[FEATURES].values)
    window = deque(feat_scaled[-SEQ_LEN:], maxlen=SEQ_LEN)

    tail = deque(_df["gap"].values[-24:].tolist(), maxlen=24)
    current_time = _df.index[-1]

    preds = []
    for _ in range(steps):
        x = np.expand_dims(np.array(window, dtype="float32"), axis=0)
        yhat_scaled = _model.predict(x, verbose=0)[0,0]
        yhat = float(y_scaler.inverse_transform([[yhat_scaled]])[0,0])

        preds.append(yhat)
        tail.append(yhat)

        current_time = current_time + pd.Timedelta(hours=1)
        f_next = _future_feature_row(current_time, np.array(tail, dtype="float32"))
        f_next_scaled = x_scaler.transform(f_next.reshape(1,-1))[0]
        window.append(f_next_scaled)

    preds = np.array(preds, dtype="float32")
    preds = np.maximum(preds, 0.0)
    return preds

    