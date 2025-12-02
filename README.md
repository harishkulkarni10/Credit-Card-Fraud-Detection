# Credit Card Fraud Detection

A complete, end-to-end credit card fraud detection system built on the European cardholders dataset (284,807 transactions, 492 frauds). The final model achieves **0.977 AUC**, **80% recall** and **73% precision** on fraud cases while keeping false positives manageable.

## Key Features

- XGBoost classifier trained with scale_pos_weight and early stopping
- StandardScaler + log(Amount) + Hour-of-day feature engineering (identical in training and inference)
- Fully modular Python package under `src/`
- Single-transaction inference in < 10 ms
- FastAPI service with automatic OpenAPI/Swagger documentation
- Docker container ready for local or cloud deployment
- All notebooks (EDA → preprocessing → experiments → final inference) preserved under `notebooks/`

## Project Structure

```
.
├── data/                     # raw creditcard.csv (not tracked)
├── models/
│   ├── scaler.pkl
│   └── BEST_XGBOOST_MODEL.pkl
├── notebooks/                # 01_eda.ipynb … 04_best_model_finalize.ipynb
├── src/
│   ├── preprocessing/pipeline.py
│   └── inference/
│       ├── predictor.py
│       └── api.py
├── Dockerfile
├── requirements.txt
└── run_api.py
```

## Quick Start (local)

```bash
# 1. Clone and enter repo
git clone https://github.com/harishkulkarni10/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# 2. Place scaler.pkl and BEST_XGBOOST_MODEL.pkl into models/ folder
#    (downloaded from Google Drive)

# 3. Run with Docker (recommended)
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector

# or run directly with Python
pip install -r requirements.txt
python run_api.py
```


## API Usage Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Time":50000.0,"Amount":0.0,"V1":-10.0,...,"V28":1.5}'
```

Response:

```json
{
  "fraud_probability": 0.9967,
  "is_fraud": true,
  "confidence": "HIGH",
  "message": "Fraudulent transaction detected!"
}
```

## Model Performance (validation set)

| Model               | AUC       | Precision | Recall    | F1        |
| ------------------- | --------- | --------- | --------- | --------- | ----- |
| Logistic Regression | 0.937     |           | 0.052     | 0.880     | 0.098 |
| XGBoost (final)     | **0.977** | **0.738** | **0.800** | **0.768** |
| MLP                 | 0.956     | 0.62      | 0.71      | 0.66      |

## Notes

- The repository follows standard production conventions used in industry ML teams.
- All random states and preprocessing steps are fixed for full reproducibility.
- Model and scaler versions are persisted with joblib; no retraining required for inference.

Project built and maintained by Harish Kulkarni.  
Open for contributions and improvements.
