import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="EMIPredict AI", layout="wide", page_icon="💰")

# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: white; margin: 0; }
.main-header p { color: #d0d9f0; margin: 0.3rem 0 0 0; }

.result-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}
.result-eligible { background: linear-gradient(135deg, #11998e, #38ef7d); }
.result-highrisk { background: linear-gradient(135deg, #f7971e, #ffd200); }
.result-noteligible { background: linear-gradient(135deg, #eb3349, #f45c43); }
.result-card h2 { color: white; margin: 0; }
.result-card p { color: white; font-size: 0.9rem; margin: 0.3rem 0 0 0; }

[data-testid="stMetric"] {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 12px;
}
[data-testid="stMetric"] label, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #1e1e1e !important;
}
[data-testid="stMetricValue"] {
    color: #1e3c72 !important;
    font-weight: 700 !important;
}

.section-card {
    background: #f8f9fb;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load models
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("best_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("best_regressor.pkl", "rb") as f:
        regressor = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le_target = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("categorical_encoders.pkl", "rb") as f:
        cat_encoders = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return classifier, regressor, le_target, scaler, cat_encoders, feature_cols

classifier, regressor, le_target, scaler, cat_encoders, feature_cols = load_models()

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("train.csv")
        return df.sample(min(20000, len(df)), random_state=42)
    except FileNotFoundError:
        return None

# Hardcoded model comparison results (from MLflow experiment runs)
MODEL_RESULTS = {
    "classification": pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.9130, 0.9491, 0.9780],
        "Precision": [0.8863, 0.9374, 0.9768],
        "Recall": [0.9130, 0.9491, 0.9780],
        "F1 Score": [0.8938, 0.9317, 0.9767],
    }),
    "regression": pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "RMSE": [4116.77, 937.18, 666.87],
        "MAE": [2942.71, 244.52, 206.38],
        "R2 Score": [0.7199, 0.9855, 0.9926],
    }),
}

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="main-header">
<h1>💰 EMIPredict AI</h1>
<p>Intelligent Financial Risk Assessment Platform — check EMI eligibility and safe loan amounts instantly</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

page = st.sidebar.radio("Navigate", ["Predict", "Model Performance", "Data Insights", "About"])

# ---------------------------------------------------------------
# PREDICT PAGE
# ---------------------------------------------------------------
if page == "Predict":
    st.header("Applicant Details")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal**")
            age = st.number_input("Age", 18, 70, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            company_type = st.selectbox("Company Type", ["Startup", "MidSize", "MNC", "PSU", "Unknown"])
            years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0)
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])

        with col2:
            st.markdown("**Income & Expenses**")
            monthly_salary = st.number_input("Monthly Salary", 15000, 200000, 50000)
            monthly_rent = st.number_input("Monthly Rent", 0, 40000, 0)
            family_size = st.number_input("Family Size", 1, 10, 3)
            dependents = st.number_input("Dependents", 0, 10, 1)
            school_fees = st.number_input("School Fees", 0, 10000, 0)
            college_fees = st.number_input("College Fees", 0, 20000, 0)
            travel_expenses = st.number_input("Travel Expenses", 0, 15000, 2000)
            groceries_utilities = st.number_input("Groceries/Utilities", 0, 25000, 6000)

        with col3:
            st.markdown("**Financial & Loan**")
            other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 10000, 1000)
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount", 0, 30000, 0)
            credit_score = st.number_input("Credit Score", 300, 850, 700)
            bank_balance = st.number_input("Bank Balance", 0, 1000000, 50000)
            emergency_fund = st.number_input("Emergency Fund", 0, 500000, 20000)
            emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
            requested_amount = st.number_input("Requested Amount", 10000, 1500000, 100000)
            requested_tenure = st.number_input("Requested Tenure (months)", 3, 84, 12)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        total_expenses = (monthly_rent + school_fees + college_fees + travel_expenses +
                           groceries_utilities + other_monthly_expenses + current_emi_amount)

        input_dict = {
            "age": age, "gender": gender, "marital_status": marital_status, "education": education,
            "monthly_salary": monthly_salary, "employment_type": employment_type,
            "years_of_employment": years_of_employment, "company_type": company_type,
            "house_type": house_type, "monthly_rent": monthly_rent,
            "family_size": family_size, "dependents": dependents,
            "school_fees": school_fees, "college_fees": college_fees,
            "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
            "credit_score": credit_score, "bank_balance": bank_balance,
            "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
            "requested_amount": requested_amount, "requested_tenure": requested_tenure,
        }
        input_df = pd.DataFrame([input_dict])

        input_df["total_expenses"] = total_expenses
        input_df["debt_to_income"] = current_emi_amount / monthly_salary
        input_df["expense_to_income"] = total_expenses / monthly_salary
        input_df["disposable_income"] = monthly_salary - total_expenses
        input_df["affordability_ratio"] = (requested_amount / requested_tenure) / monthly_salary
        input_df["savings_ratio"] = bank_balance / monthly_salary

        for col, enc in cat_encoders.items():
            input_df[col] = enc.transform(input_df[col])

        numeric_cols = ["age", "monthly_salary", "years_of_employment", "monthly_rent",
                        "family_size", "dependents", "school_fees", "college_fees",
                        "travel_expenses", "groceries_utilities", "other_monthly_expenses",
                        "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
                        "requested_amount", "requested_tenure", "total_expenses",
                        "debt_to_income", "expense_to_income", "disposable_income",
                        "affordability_ratio", "savings_ratio"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        input_df = input_df[feature_cols]

        pred_class = classifier.predict(input_df)[0]
        pred_label = le_target.inverse_transform([pred_class])[0]
        pred_proba = classifier.predict_proba(input_df)[0]
        pred_emi = max(regressor.predict(input_df)[0], 0)

        st.subheader("Prediction Results")
        c1, c2 = st.columns([1, 1])

        css_class = {"Eligible": "result-eligible", "High_Risk": "result-highrisk",
                     "Not_Eligible": "result-noteligible"}[pred_label]
        emoji = {"Eligible": "✅", "High_Risk": "⚠️", "Not_Eligible": "❌"}[pred_label]

        with c1:
            st.markdown(f"""
            <div class="result-card {css_class}">
                <h2>{emoji} {pred_label.replace('_', ' ')}</h2>
                <p>EMI Eligibility Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.metric("Max Safe Monthly EMI", f"₹{pred_emi:,.0f}")
            est_emi = requested_amount / requested_tenure
            st.metric("Estimated EMI for Requested Loan", f"₹{est_emi:,.0f}")

        st.markdown("**Class Probability Breakdown**")
        proba_df = pd.DataFrame({
            "Class": le_target.classes_,
            "Probability": pred_proba
        }).sort_values("Probability", ascending=True)
        st.bar_chart(proba_df.set_index("Class"))

        st.session_state.history.insert(0, {
            "Salary": monthly_salary, "Credit Score": credit_score,
            "Scenario": emi_scenario, "Eligibility": pred_label,
            "Max EMI": f"₹{pred_emi:,.0f}"
        })
        st.session_state.history = st.session_state.history[:5]

    if st.session_state.history:
        st.subheader("Recent Predictions (this session)")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# MODEL PERFORMANCE PAGE
# ---------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance & Comparison")
    st.caption("Tracked and compared using MLflow experiment tracking across 6 trained models.")

    st.markdown("### Classification Models — EMI Eligibility")
    cls_df = MODEL_RESULTS["classification"]
    st.dataframe(cls_df.style.highlight_max(subset=["Accuracy", "F1 Score"], color="#d2f4dd"),
                 use_container_width=True, hide_index=True)
    st.bar_chart(cls_df.set_index("Model")[["Accuracy", "F1 Score"]])
    st.success("Best Model: **XGBoost Classifier** — 97.8% accuracy, selected for production deployment.")

    st.markdown("### Regression Models — Max Monthly EMI")
    reg_df = MODEL_RESULTS["regression"]
    st.dataframe(reg_df.style.highlight_max(subset=["R2 Score"], color="#d2f4dd"),
                 use_container_width=True, hide_index=True)
    st.bar_chart(reg_df.set_index("Model")[["R2 Score"]])
    st.success("Best Model: **XGBoost Regressor** — R² 0.993, RMSE ₹667, selected for production deployment.")

    st.info("All experiments, parameters, and metrics for every model variant are logged in MLflow under the 'EMIPredict_AI' experiment, with the best models registered in the MLflow Model Registry.")

# ---------------------------------------------------------------
# DATA INSIGHTS PAGE
# ---------------------------------------------------------------
elif page == "Data Insights":
    st.header("Dataset & EDA Insights")
    df = load_sample_data()

    if df is None:
        st.warning("train.csv not found in the app folder — insights unavailable in this deployment.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**EMI Eligibility Distribution**")
            st.bar_chart(df["emi_eligibility"].value_counts())
        with c2:
            st.markdown("**Records by EMI Scenario**")
            st.bar_chart(df["emi_scenario"].value_counts())

        st.markdown("**Credit Score Distribution by Eligibility**")
        pivot = df.groupby("emi_eligibility")["credit_score"].mean()
        st.bar_chart(pivot)

        st.markdown("**Monthly Salary vs Max Monthly EMI (sample)**")
        st.scatter_chart(df.sample(min(2000, len(df))), x="monthly_salary", y="max_monthly_emi", color="emi_eligibility")

        st.caption(f"Insights generated from a sample of {len(df):,} training records.")

# ---------------------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------------------
else:
    st.header("About EMIPredict AI")
    st.markdown("""
    <div class="section-card">
    <b>EMIPredict AI</b> is a financial risk assessment platform that combines machine learning
    with MLflow experiment tracking to help predict EMI eligibility and safe loan amounts.

    **Tech Stack:** Python, scikit-learn, XGBoost, MLflow, Streamlit<br>
    **Dataset:** 400,000+ financial records across 5 EMI scenarios, 22 input features<br>
    **Models:** 3 classification + 3 regression models, best ones selected via MLflow comparison
    </div>
    """, unsafe_allow_html=True)