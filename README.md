🚨 Credit Card Fraud Detection using XGBoost

A production-style Machine Learning project focused on detecting fraudulent credit card transactions using advanced feature engineering and gradient boosting techniques.

This project demonstrates:

- ✅ End-to-End ML Pipeline
- ✅ Feature Engineering
- ✅ Imbalanced Classification Handling
- ✅ XGBoost Optimization
- ✅ Model Serialization with Joblib
- ✅ FastAPI Deployment
- ✅ Production-ready API Inference Workflow

---

📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small fraction of total transactions.

This project was built to simulate a real-world fraud detection pipeline using:

- ⚡ XGBoost Classifier
- ⚡ Fraud-risk feature engineering
- ⚡ Time-based fraud probability encoding
- ⚡ FastAPI deployment architecture
- ⚡ Serialized inference pipeline

The primary goal is to maximize fraud detection capability while minimizing false positives.

---

📂 Dataset Source

Dataset used from Kaggle:

https://www.kaggle.com/datasets/gzdekzlkaya/credit-card-fraud-detection-dataset

---

🧠 Machine Learning Workflow

🔹 Data Preprocessing

- Missing value handling
- Feature scaling
- Outlier-aware preprocessing
- Imbalanced data handling

🔹 Feature Engineering

Custom engineered fraud-risk indicators including:

- Transaction amount behavior
- Time-based transaction encoding
- Fraud probability pattern extraction

🔹 Model Training

Models experimented with:

- Logistic Regression
- Random Forest
- XGBoost

Final selected model:

- ✅ XGBoost Classifier

Reason:

- Superior performance on highly imbalanced tabular datasets
- Better handling of non-linear fraud patterns
- Strong precision-recall tradeoff

---

📊 Model Performance

Metric| Score
🎯 F1-Score| 0.8852
🎯 Precision| 0.9529
🎯 Recall| 0.8265
🎯 Accuracy| High

---

🔍 Key Insight

The model achieves:

- Very high precision
- Strong recall
- Lower false positive rate

This makes the system effective for identifying fraudulent transactions while reducing unnecessary fraud alerts.

---

⚙️ Tech Stack

- 🐍 Python
- 📊 Pandas
- 🤖 Scikit-Learn
- ⚡ XGBoost
- 🚀 FastAPI
- 💾 Joblib

---

🏗️ Project Structure

Credit-Card-Fraud-Detection/
│
├── train.py
├── app.py
├── fraud_model.pkl
├── requirements.txt
├── README.md
└── dataset/

---

💾 Model Serialization

The trained model is serialized using Joblib for production inference.

joblib.dump(model, "fraud_model.pkl")

---

🚀 FastAPI Deployment

The trained fraud detection model is deployed using FastAPI for real-time predictions.

▶️ Start API Server

uvicorn app:app --reload

---

📌 API Endpoint

POST "/predict"

Predict whether a transaction is fraudulent.

---

📥 Example Request

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

---

📤 Example Response

{
  "fraud_prediction": 1,
  "fraud_probability": 0.9734
}

---

🧪 Future Improvements

Planned upgrades for future versions:

- 🔥 Real-time streaming fraud detection
- 🔥 Explainable AI using SHAP
- 🔥 Hybrid anomaly + classification system
- 🔥 Advanced ensemble techniques
- 🔥 GenAI-powered fraud investigation assistant
- 🔥 Dashboard visualization system

---

📈 Key Learning Outcomes

This project helped strengthen understanding of:

- Imbalanced classification problems
- Production ML workflows
- API deployment pipelines
- Gradient boosting optimization
- Feature engineering strategies
- Real-world fraud detection systems

---

👨‍💻 Author

Built by Amrit Gorai as part of hands-on Machine Learning and MLOps practice focused on production-ready AI systems.

---

⭐ Future Vision

This project is part of a larger goal of building scalable AI systems combining:

- Classical Machine Learning
- MLOps
- GenAI
- Real-time inference systems
- Intelligent fraud analysis pipelines