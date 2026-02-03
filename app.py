import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# PAGE CONFIG
st.set_page_config(page_title="Weather Classification App", layout="wide")

st.markdown(
    "<h2 style='text-align:center;'>🌦️ Weather Classification App</h2>",
    unsafe_allow_html=True
)

# LOAD MODELS
BASE_DIR = Path(__file__).resolve().parent

MODELS = {
    "Logistic Regression": joblib.load(BASE_DIR / "model" / "LogisticRegression.pkl"),
    "KNN": joblib.load(BASE_DIR / "model" / "KNN.pkl"),
    "Naive Bayes": joblib.load(BASE_DIR / "model" / "NaiveBayes.pkl"),
    "Decision Tree": joblib.load(BASE_DIR / "model" / "DecisionTree.pkl"),
    "Random Forest": joblib.load(BASE_DIR / "model" / "RandomForest.pkl"),
    "XGBoost": joblib.load(BASE_DIR / "model" / "XGBoost.pkl"),
}

scaler = joblib.load(BASE_DIR / "model" / "Scaler.pkl")

# SIDEBAR – MODEL SELECTION ONLY
with st.sidebar:
    st.markdown("## 🔍 Select Models")

    selected_models = st.multiselect(
        "Choose one or more models",
        list(MODELS.keys()),
        placeholder="Select models",
        width=350,
        help="Select the machine learning models you want to evaluate on the uploaded test dataset."
    )

# TOP BAR – DATASET TOOLS
st.markdown("### Test Dataset")

top_left, top_right = st.columns([1, 2])

with top_left:
    st.markdown("**⬇️ Download Test Datasets**")

    TEST_FILES = {
        "Whole Dataset": "processed_weather_dataset.csv",
        "Test Dataset": "test_main.csv",
        "Sample Dataset 1": "test_sample_1.csv",
        "Sample Dataset 2": "test_sample_2.csv",
        "Sample Dataset 3": "test_sample_3.csv",
        "Sample Dataset 4": "test_sample_4.csv",
        "Sample Dataset 5": "test_sample_5.csv",
    }

    for name, file_name in TEST_FILES.items():
        st.markdown(
            f"[📥 {name}](https://raw.githubusercontent.com/KeerthiLazarus-Labz/Weather-Classification-Model/main/data/{file_name})"
        )

with top_right:
    st.markdown("**📤 Upload Test CSV**")
    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# PROCESS DATA
if uploaded_file and selected_models:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset")
    if st.checkbox("👈 Show Uploaded Data  (Head & Tail)", key="preview"):
        st.dataframe(pd.concat([df.head(5), df.tail(5)], axis=0))
        st.write(f"**Dataset Shape:** {df.shape}")

    y_true = df["condition_severity"]
    X_df = df.drop(columns=["condition_severity"], errors="ignore")

    metrics_rows = []
    auc_scores_dict = {}
    f1_scores_dict = {}

    # CALCULATE METRICS FOR ALL MODELS
    for model_name in selected_models:
        model = MODELS[model_name]

        if model_name in ["Logistic Regression", "KNN"]:
            X_input = scaler.transform(X_df)
        else:
            X_input = X_df.values

        y_pred = model.predict(X_input)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        metrics_rows.append([
            model_name, acc, prec, rec, f1, mcc
        ])

        # ---------- AUC ----------
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_input)
            classes = sorted(y_true.unique())
            y_true_bin = label_binarize(y_true, classes=classes)

            auc_list = []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc_list.append(auc(fpr, tpr))

            auc_scores_dict[model_name] = sum(auc_list) / len(auc_list)

        f1_scores_dict[model_name] = f1

    # METRICS TABLE (RANKED BY F1)
    st.markdown("## 📊 Model Evaluation Metrics")

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    ).sort_values("F1 Score", ascending=False)

    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC": "{:.4f}"
    }), width="stretch")

    st.divider()

    # RANKING CHARTS
    if len(selected_models) > 1:

        col1, col2 = st.columns(2)

        # ---------- AUC RANKING ----------
        with col1:
            st.markdown("### 🏆 Ranking by Average AUC")

            auc_df = (
                pd.DataFrame(auc_scores_dict.items(), columns=["Model", "Avg AUC"])
                .sort_values("Avg AUC", ascending=False)
            )

            fig_auc, ax_auc = plt.subplots(figsize=(4, 5))
            sns.barplot(y="Model", x="Avg AUC", data=auc_df, orient="h", ax=ax_auc)
            ax_auc.set_title("AUC Ranking")
            st.pyplot(fig_auc)

        # ---------- F1 RANKING ----------
        with col2:
            st.markdown("### 🏆 Ranking by F1 Score")

            f1_df = (
                pd.DataFrame(f1_scores_dict.items(), columns=["Model", "F1 Score"])
                .sort_values("F1 Score", ascending=False)
            )

            fig_f1, ax_f1 = plt.subplots(figsize=(4, 5))
            sns.barplot(y="Model", x="F1 Score", data=f1_df, orient="h", ax=ax_f1)
            ax_f1.set_title("F1 Score Ranking")
            st.pyplot(fig_f1)

    st.divider()

    # PER-MODEL CONFUSION MATRIX + ROC
    for model_name in selected_models:
        model = MODELS[model_name]

        if model_name in ["Logistic Regression", "KNN"]:
            X_input = scaler.transform(X_df)
        else:
            X_input = X_df.values

        y_pred = model.predict(X_input)

        st.markdown(f"## 🔹 {model_name}")

        colA, colB = st.columns(2)

        with colA:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted Class")
            ax_cm.set_ylabel("Actual Class")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

        with colB:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_input)
                classes = sorted(y_true.unique())
                y_true_bin = label_binarize(y_true, classes=classes)

                fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax_roc.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(fontsize=6)
                st.pyplot(fig_roc)