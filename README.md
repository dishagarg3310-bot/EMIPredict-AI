# 💰 EMIPredict AI — Intelligent Financial Risk Assessment Platform

A machine learning powered financial risk assessment platform that predicts **EMI eligibility** (classification) and **maximum safe monthly EMI amount** (regression) for loan applicants, with full experiment tracking via MLflow and an interactive Streamlit web application.

## 🔍 Problem Statement

People often struggle to manage EMIs due to poor financial planning and inadequate risk assessment. EMIPredict AI solves this by providing data-driven insights for better loan decisions — automating eligibility checks and recommending safe loan amounts based on a comprehensive financial profile.

## 🚀 Live App

Add your Streamlit Cloud link here once deployed.

## 📊 Dataset

- **400,000+** financial records across **5 EMI scenarios**: E-commerce Shopping, Home Appliances, Vehicle, Personal Loan, and Education EMIs
- **22 input features**: demographics, employment, income, expenses, credit history, and loan details
- **2 target variables**:
  - `emi_eligibility` (classification): Eligible / High_Risk / Not_Eligible
  - `max_monthly_emi` (regression): maximum safe monthly EMI amount

## 🧠 Models

**Classification (EMI Eligibility)**
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 91.3% | 89.4% |
| Random Forest | 94.9% | 93.2% |
| **XGBoost (best)** | **97.8%** | **97.7%** |

**Regression (Max Monthly EMI)**
| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 4116.77 | 0.72 |
| Random Forest | 937.18 | 0.985 |
| **XGBoost (best)** | **666.87** | **0.993** |

All experiments, parameters, and metrics are tracked using **MLflow**, with the best-performing models registered in the MLflow Model Registry.

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** scikit-learn, XGBoost, pandas, numpy
- **Experiment Tracking:** MLflow
- **Web App:** Streamlit
- **Deployment:** Streamlit Cloud

## 📁 Project Structure

```
EMIPredict-AI/
├── app.py                       # Streamlit web application
├── EMIPredict_0X...ipynb        # Data preprocessing, EDA, feature engineering, model training & MLflow logging
├── best_classifier.pkl          # Trained XGBoost classifier
├── best_regressor.pkl           # Trained XGBoost regressor
├── scaler.pkl                   # StandardScaler for numeric features
├── categorical_encoders.pkl     # LabelEncoders for categorical features
├── label_encoder.pkl            # Target label encoder
├── feature_columns.pkl          # Feature column order used at training time
├── train.csv                    # Training data (used for in-app data insights)
└── README.md
```

## ⚙️ How It Works

1. **Data Preprocessing** — cleaning, missing value imputation, validity checks, stratified train/val/test split
2. **EDA** — distribution analysis, correlation heatmaps, feature relationships
3. **Feature Engineering** — derived financial ratios (debt-to-income, expense-to-income, affordability ratio, savings ratio, disposable income)
4. **Model Training** — 3 classification + 3 regression models trained and evaluated
5. **MLflow Tracking** — all runs logged with parameters and metrics; best models registered
6. **Streamlit App** — multi-page interface for real-time predictions, model performance comparison, and data insights

## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📈 App Features

- **Predict** — enter applicant details and get instant EMI eligibility + safe EMI amount
- **Model Performance** — compare all trained models side by side
- **Data Insights** — explore dataset distributions and patterns
- **About** — project and tech stack overview

## 👩‍💻 Author

Disha Garg
