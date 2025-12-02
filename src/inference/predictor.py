import joblib  
from pathlib import Path 
import xgboost as xgb 
from src.preprocessing.pipeline import preprocess_transaction 

# Load the XGBoost model 
MODEL_PATH = Path("models/BEST_XGBOOST_MODEL.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}, download it from Google Drive.")

model_package = joblib.load(MODEL_PATH)
booster = model_package['model']

print("XGBoost model loaded successfully!!")
print(f"Model info: {model_package.get('model_info', 'N/A')}")
print(f"Saved AUC: {model_package.get('auc', 'N/A')}")

def predict_fraud_probability(raw_transaction: dict) -> dict:
    # Preprocess --> get scaled features 
    X_scaled = preprocess_transaction(raw_transaction)
    # Convert to DMatrix
    dmatrix = xgb.DMatrix(X_scaled)
    # Predict probability
    proba = float(booster.predict(dmatrix)[0])

    return {
        'fraud_probability': round(proba, 5),
        'is_fraud': bool(proba >= 0.5),
        'confidence': 'HIGH' if proba > 0.8 or proba < 0.2 else 'MEDIUM'
    }

    