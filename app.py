from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# ===============================
# LOAD MODEL + FEATURES
# ===============================

model = joblib.load("fraud_detection_model.pkl")

model_features = joblib.load(
    "model_features.pkl"
)

hour_fraud_prob = joblib.load(
    "hour_fraud_prob.pkl"
)

# ===============================
# INPUT SCHEMA
# ===============================

class FraudInput(BaseModel):

    Time: float
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
    Amount: float

# ===============================
# FEATURE ENGINEERING
# ===============================

def prepare_features(data):

    df = pd.DataFrame([data])

    # --------------------------------
    # TIME FEATURE
    # --------------------------------

    df['Hour_24'] = (
        df['Time'] / 3600
    ).astype(int) % 24

    # --------------------------------
    # TARGET ENCODING FEATURE
    # --------------------------------

    global_fraud_prob = (
        sum(hour_fraud_prob.values())
        / len(hour_fraud_prob)
    )

    df['Hour_Fraud_Prob'] = (
        df['Hour_24'].map(hour_fraud_prob)
    )

    df['Hour_Fraud_Prob'] = (
        df['Hour_Fraud_Prob']
        .fillna(global_fraud_prob)
    )

    # --------------------------------
    # CUSTOM FEATURES
    # --------------------------------

    df['Hour_Fraud_Proba'] = (
        df['Hour_Fraud_Prob'] * df['V14']
    )

    df['Risk_Weighted_Amount'] = (
        df['Amount'] * df['Hour_Fraud_Prob']
    )

    # --------------------------------
    # MATCH TRAINING COLUMN ORDER
    # --------------------------------

    df = df.reindex(
        columns=model_features,
        fill_value=0
    )

    return df

# ===============================
# PREDICTION ENDPOINT
# ===============================

@app.post("/predict")

def predict_fraud(data: FraudInput):

    input_data = prepare_features(data.dict())

    prediction = model.predict(input_data)[0]

    probability = (
        model.predict_proba(input_data)[0][1]
    )

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }