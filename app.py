
import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Adult Income Classification App",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Adult Income Classification using Machine Learning")

st.write(
    "This Streamlit application predicts whether annual income is above 50K using the Adult Income Dataset."
)

model_options = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "kNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(model_options.keys())
)

uploaded_file = st.file_uploader(
    "Upload test_data.csv file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")
else:
    if os.path.exists("test_data.csv"):
        data = pd.read_csv("test_data.csv")
        st.info("Using default test_data.csv from repository")
    else:
        st.warning("Please upload test_data.csv")
        st.stop()

st.write("Dataset Preview")
st.dataframe(
    data.head(),
    use_container_width=True
)

if "income" not in data.columns:
    st.error("The uploaded file must contain the target column named income")
    st.stop()

X_test = data.drop(
    "income",
    axis=1
)

y_test = data["income"]

model_path = model_options[selected_model]

with open(
    model_path,
    "rb"
) as file:
    model = pickle.load(file)

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

st.write("Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Accuracy", round(accuracy, 4))
col2.metric("AUC", round(auc, 4))
col3.metric("Precision", round(precision, 4))
col4.metric("Recall", round(recall, 4))
col5.metric("F1 Score", round(f1, 4))
col6.metric("MCC Score", round(mcc, 4))

st.write("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

st.write(cm)

st.write("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(
    pd.DataFrame(report).transpose(),
    use_container_width=True
)

prediction_output = data.copy()
prediction_output["Predicted Income Class"] = y_pred

st.write("Prediction Output")
st.dataframe(
    prediction_output.head(20),
    use_container_width=True
)

csv_output = prediction_output.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Predictions",
    data=csv_output,
    file_name="adult_income_predictions.csv",
    mime="text/csv"
)
