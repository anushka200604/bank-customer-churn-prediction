https://bank-customer-churn-prediction-plzeqz485vbre8yvszboek.streamlit.app/

# AI Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a banking customer is likely to churn and provides risk analysis with recommendations.

## Problem Statement

Banks lose customers due to poor engagement and lack of retention strategies. This project predicts customer churn probability and helps identify high-risk customers.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit

## ML Workflow

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Comparison
- Model Deployment

## Models Compared

- Logistic Regression
- Random Forest
- XGBoost

XGBoost achieved the best performance and was selected for deployment.

## Features

- Customer churn probability prediction
- Risk classification
- Business recommendations
- Explainable AI using SHAP

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## Output

The system provides:
- Churn probability
- Customer risk category
- Factors influencing prediction
