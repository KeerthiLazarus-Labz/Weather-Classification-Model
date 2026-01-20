#Import Library
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from pathlib import Path
import joblib

#Dataset Path
path = Path.cwd() /"data"/"processed_weather_dataset.csv" 
df = pd.read_csv(path)

X = df.drop(columns=["condition_severity"]) #features
y = df["condition_severity"] #target

#Train&Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Apply Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#LR - Model
model = LogisticRegression(
    max_iter=1000,
    multi_class="ovr" 
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

#Validation Metrics
print("Logistic Regression Metrics")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob, multi_class="ovr"))
print("Precision :", precision_score(y_test, y_pred, average="weighted"))
print("Recall :", recall_score(y_test, y_pred, average="weighted"))
print("F1 :", f1_score(y_test, y_pred, average="weighted"))
print("MCC :", matthews_corrcoef(y_test, y_pred))

#Extraction of Model(PKL)
modelpath = Path.cwd() /"model"/"LogisticRegression.pkl"
joblib.dump(model,modelpath)
print("Model saved!!!")