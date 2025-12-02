import joblib 
import pandas as pd 
import numpy as np 
from pathlib import Path

# 1. Load the standard scaler that was fitted on the training data
SCALER_PATH = Path("models/scaler.pkl")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}, download it from Google Drive.")

scaler = joblib.load(SCALER_PATH)
print("Standard scaler loaded successfully!")

# 2. List of final features - in the order the model expects 
FEATURE_COLS = [f'V{i}' for i in range(1,29)] + ['Amount_log', 'Hour']

# 3. Main Preprocessing function 
def preprocess_transaction(raw_data: dict) -> np.ndarray:
    """
    Takes a raw transaction dictionary and returns scaled features in the order the model expects 
    """
    # step 1: convert dict to dataframe 
    df = pd.DataFrame([raw_data])

    # step 2: Feature engineering 
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = df['Time'] % (24 * 3600) / 3600.0      # converts seconds to hours

    # step 3: Select and order features 
    X = df[FEATURE_COLS].values

    # step 4: Scale features
    X_scaled = scaler.transform(X)

    return X_scaled