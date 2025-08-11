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