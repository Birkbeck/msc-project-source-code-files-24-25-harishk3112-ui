# LSTM with calendar features (model)

import os
# Reduce TensorFlow logs and disable some CPU optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"