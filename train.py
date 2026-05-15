import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import joblib


class FraudDetectionModels:
    def __init__(self):
        self.models = {
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',colsample_bytree=0.8, random_state=42, n_estimators=523, learning_rate=0.05, max_depth=4,subsample=0.8)
        }
        self.results = {}
        return None
    
    @staticmethod
    def feature_engineering(df):
        df = df.drop('V1',axis = 1)
        df = df.drop('V2',axis = 1)
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        X_train['Hour_24'] = (X_train['Time'] / 3600).astype(int) % 24
        X_test['Hour_24'] = (X_test['Time'] / 3600).astype(int) % 24

        temp_train = X_train.copy()
        temp_train['Class'] = y_train
        hour_fraud_prob = temp_train.groupby('Hour_24')['Class'].mean().to_dict()

        X_train['Hour_Fraud_Prob'] = X_train['Hour_24'].map(hour_fraud_prob)
        X_test['Hour_Fraud_Prob'] = X_test['Hour_24'].map(hour_fraud_prob)

        X_train['Hour_Fraud_Proba'] = X_train['Hour_Fraud_Prob']*X_train['V14']
        X_test['Hour_Fraud_Proba'] = X_test['Hour_Fraud_Prob']*X_test['V14']

        global_fraud_prob = y_train.mean()
        X_test['Hour_Fraud_Prob'] = X_test['Hour_Fraud_Prob'].fillna(global_fraud_prob)
        X_train['Risk_Weighted_Amount'] = X_train['Amount'] * X_train['Hour_Fraud_Prob']
        X_test['Risk_Weighted_Amount'] = X_test['Amount'] * X_test['Hour_Fraud_Prob']

        return X_train, X_test, y_train, y_test, hour_fraud_prob

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        print("\n🚀 Model Training Pipeline Started\n")
        print("-" * 50)
        
        for name, model in self.models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            time_taken = time.time() - start_time
            
            self.results[name] = {
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Accuracy': acc,
                'Time Taken (s)': time_taken
            }
            
            print(f"✅ {name} trained in {time_taken:.2f} seconds.")
            print(f"   ➔ F1-Score: {f1:.4f} |  Precision - {precision}  | Recall: {recall:.4f}")
            print("-" * 50)
            
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values(by='F1-Score', ascending=False)
        return results_df

    def get_detailed_report(self, model_name, X_test, y_test):
        if model_name in self.models:
            model = self.models[model_name]
            y_pred = model.predict(X_test)
            print(f"\n📊 Detailed Report for {model_name}:")
            print(classification_report(y_test, y_pred))
        else:
            print("❌ Check Model Name!")



if __name__ == "__main__":
    df = pd.read_csv("/home/amritkg9009/Downloads/creditcard.csv")
    X_train , X_test, y_train, y_test , hour_fraud_prob= FraudDetectionModels.feature_engineering(df)
    pipeline = FraudDetectionModels()
    leaderboard = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    xgb_model = pipeline.models['XGBoost']
    
    # Save Model
    joblib.dump(X_train.columns.tolist(), "model_features.pkl")
    joblib.dump(hour_fraud_prob, "hour_fraud_prob.pkl")
    joblib.dump(xgb_model, "fraud_detection_model.pkl")
    print("✅ Model saved successfully!")
