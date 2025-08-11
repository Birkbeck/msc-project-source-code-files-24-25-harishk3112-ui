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