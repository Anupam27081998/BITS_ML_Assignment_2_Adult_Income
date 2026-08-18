# Machine Learning Assignment 2

## Problem Statement

The objective of this project is to predict whether a person's annual income is above 50K using machine learning classification techniques.

## Dataset Description

| Item | Description |
|---|---|
| Dataset Name | Adult Income Dataset |
| Source | UCI Machine Learning Repository |
| Dataset Type | Binary Classification |
| Number of Instances | 48,842 |
| Number of Features | 14 |
| Target Variable | income |

## GitHub Repository Link

(Add your repository URL here)

## Live Streamlit App Link

(Add your Streamlit URL here)

## Models Used

1. Logistic Regression
2. Decision Tree
3. kNN
4. Naive Bayes
5. Random Forest

## Model Comparison Table

Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8446	0.9013	0.7333	0.5861	0.6515	0.5588
Decision Tree	0.8482	0.8987	0.782	0.5375	0.6371	0.5605
kNN	0.8329	0.8749	0.6847	0.6044	0.642	0.5354
Naive Bayes	0.6035	0.845	0.3781	0.93	0.5376	0.377
Random Forest	0.8493	0.9029	0.731	0.6204	0.6712	0.5775



## Model Performance Observations

### Logistic Regression
Provided a strong baseline model.

### Decision Tree
Captured non-linear relationships.

### kNN
Performed well after preprocessing.

### Naive Bayes
Fastest model to train.

### Random Forest
Achieved the strongest overall performance.

### Overall Winner
Random Forest

## Streamlit Application Features

- Dataset Upload
- Model Selection
- Evaluation Metrics
- Confusion Matrix
- Classification Report

## Project Structure

```text
adult_income_assignment.ipynb
app.py
requirements.txt
README.md
adult_income_dataset.csv
test_data.csv
model_results.csv
```
