"""Quick test script for the Fraud Detection API"""
import requests
import json

API_URL = "http://127.0.0.1:8000"

# Test normal transaction
normal_transaction = {
    "Time": 3600.0,
    "Amount": 89.0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
}

# Test fraud transaction
fraud_transaction = {
    "Time": 50000.0,
    "Amount": 0.0,
    "V1": -10.0, "V2": 8.5, "V3": -15.0, "V4": 7.0,
    "V5": -8.0, "V6": -5.0, "V7": -10.0, "V8": 5.0,
    "V9": -7.0, "V10": -10.0, "V11": 5.0, "V12": -12.0,
    "V13": -3.0, "V14": -15.0, "V15": -2.0, "V16": -8.0,
    "V17": -12.0, "V18": -6.0, "V19": -4.0, "V20": 2.0,
    "V21": 2.5, "V22": -1.0, "V23": -2.0, "V24": 0.1,
    "V25": -1.0, "V26": -0.5, "V27": 3.0, "V28": 1.5
}

print("=" * 60)
print("Testing Credit Card Fraud Detection API")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing health check endpoint...")
try:
    response = requests.get(f"{API_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Normal transaction
print("\n2. Testing normal transaction (should NOT be fraud)...")
try:
    response = requests.post(f"{API_URL}/predict", json=normal_transaction)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error response: {response.text}")
    else:
        result = response.json()
        print(f"   Fraud Probability: {result['fraud_probability']}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Message: {result['message']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Fraud transaction
print("\n3. Testing fraud transaction (should BE fraud)...")
try:
    response = requests.post(f"{API_URL}/predict", json=fraud_transaction)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error response: {response.text}")
    else:
        result = response.json()
        print(f"   Fraud Probability: {result['fraud_probability']}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Message: {result['message']}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)

