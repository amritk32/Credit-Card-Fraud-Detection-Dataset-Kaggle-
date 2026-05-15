Credit Card Fraud Detection using XGBoost
A production-style Machine Learning project focused on detecting fraudulent credit card transactions using advanced feature engineering and gradient boosting techniques.
This project demonstrates:
End-to-End ML Pipeline
Feature Engineering
Imbalanced Classification Handling
Model Serialization with Joblib
FastAPI Deployment
Production-ready API Inference Workflow

Dataset Source - https://www.kaggle.com/datasets/gzdekzlkaya/credit-card-fraud-detection-dataset

🚀 Project Overview
Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a tiny fraction of total transactions.
This project uses:
XGBoost Classifier
Custom fraud-risk feature engineering
Time-based fraud probability encoding
FastAPI deployment pipeline
The goal is to maximize fraud detection performance while maintaining high precision and recall.


📊 Model Performance
Performance Metrics
F1-Score - 0.8852
Precision - 0.9529
Recall - 0.8265
Accuracy - High

Key Insight - The model achieves very high precision while maintaining strong recall,
making it effective for identifying fraudulent transactions with fewer false positives.

⚙️ Tech Stack
Python
Pandas
Scikit-Learn
XGBoost
FastAPI
Joblib

🔥 API Deployment
The trained model is deployed using FastAPI.
Start API Server
Bash
uvicorn app:app --reload
📌 API Endpoint
POST /predict
Predict whether a transaction is fraudulent.
Example Request
JSON
{
  "Time": 10000,
  "V3": -1.5,
  "V4": 2.1,
  "V5": -0.3,
  "V6": 1.2,
  "V7": -0.7,
  "V8": 0.1,
  "V9": -1.1,
  "V10": 0.5,
  "V11": 1.8,
  "V12": -0.9,
  "V13": 0.2,
  "V14": -2.4,
  "V15": 0.3,
  "V16": -0.5,
  "V17": 0.7,
  "V18": -0.2,
  "V19": 0.1,
  "V20": 0.4,
  "V21": 0.2,
  "V22": -0.1,
  "V23": 0.3,
  "V24": 0.4,
  "V25": -0.6,
  "V26": 0.1,
  "V27": -0.3,
  "V28": 0.2,
  "Amount": 250
}
📥 Example Response
JSON
{
  "fraud_prediction": 1,
  "fraud_probability": 0.9734
}
