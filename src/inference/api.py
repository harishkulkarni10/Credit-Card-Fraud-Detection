from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from typing import Dict, Any
from src.inference.predictor import predict_fraud_probability

# FastAPI 
app = FastAPI(
    title = 'Credit Card Fraud Detection API',
    description = 'Real time fraud detection using XGBoost',
    version = '1.0.0'
)

# Request schema 
class TransactionRequest(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# Routes 
@app.get("/")
def health_check():
    return {"status": "API is live", "model": "XGBoost", "auc": 0.9770 }

@app.post("/predict")
def predict(transaction: TransactionRequest) -> Dict[str, Any]:
    """
    Predict if a transaction is fraudelent
    """
    try:
        raw_data = transaction.dict()
        result = predict_fraud_probability(raw_data)

        return {
            "transaction_id": None,
            "fraud_probability": result["fraud_probability"],
            "is_fraud": result["is_fraud"],
            "confidence": result["confidence"],
            "model_version": "xgboost_1",
            "message": "Fraudulent transaction detected" if result["is_fraud"] else "Transaction appears appropriate"
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction error: {str(e)}")

