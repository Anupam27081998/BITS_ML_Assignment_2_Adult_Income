import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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
    "This Streamlit application predicts whether a person's income is above 50K using the Adult Income Dataset."
)

uploaded_file = st.file_uploader(
    "Upload Adult Income CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")
else:
    df = pd.read_csv("adult_income_dataset.csv")
    st.info("Using default adult_income_dataset.csv from repository")

st.write("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

if "income" not in df.columns:
    st.error("The dataset must contain the target column named income")
    st.stop()

df = df.dropna()

if df["income"].dtype == "object":
    df["income"] = df["income"].map({
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1
    })

if "race" in df.columns:
    df = df.drop(columns=["race"])

if "sex" in df.columns:
    df = df.drop(columns=["sex"])

X = df.drop("income", axis=1)
y = df["income"]

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            encoder,
            categorical_columns
        )
    ]
)

model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ]
)

if model_name == "Logistic Regression":
    classifier = LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    )

elif model_name == "Decision Tree":
    classifier = DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    )

elif model_name == "kNN":
    classifier = KNeighborsClassifier(
        n_neighbors=7
    )

elif model_name == "Naive Bayes":
    classifier = GaussianNB()

else:
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)

model.fit(
    X_train,
    y_train
)

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

prediction_output = X_test.copy()
prediction_output["Actual Income Class"] = y_test.values
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