#Import Library
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef)
from pathlib import Path

#Dataset Path
path = Path.cwd() /"data"/"processed_weather_dataset.csv" 
df = pd.read_csv(path)

X = df.drop(columns=["condition_severity"]) #features
y = df["condition_severity"] #target

# Number of classes
num_classes = y.nunique()
print("Number of classes:", y.nunique())

#Train&Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#XGB - Model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

#Validation Metrics
print("XGBoost Metrics")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, y_prob, multi_class="ovr"))
print("Precision :", precision_score(y_test, y_pred, average="weighted"))
print("Recall :", recall_score(y_test, y_pred, average="weighted"))
print("F1 :", f1_score(y_test, y_pred, average="weighted"))
print("MCC :", matthews_corrcoef(y_test, y_pred))
