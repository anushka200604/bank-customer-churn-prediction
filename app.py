import streamlit as st
import joblib
import pandas as pd
import shap

# Load model
model = joblib.load("customer_churn_xgboost.pkl")

# Load features
features = joblib.load("model_features.pkl")
# SHAP explainer
explainer = shap.TreeExplainer(model)

st.sidebar.header("Customer Information")
st.caption("Built using XGBoost Machine Learning")
st.title("AI Customer Churn Prediction System")
st.caption(
    "Machine Learning based customer retention system using XGBoost"
)

st.write(
    "Predict whether a bank customer is likely to leave."
)


st.header("Customer Details")


age = st.number_input("Age", min_value=18, max_value=100)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
salary = st.number_input("Estimated Salary")

tenure = st.number_input("Tenure")
products = st.number_input("Number of Products")

active = st.selectbox("Active Member", [0,1])
card = st.selectbox("Has Credit Card", [0,1])

geography = st.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


if st.button("Predict Churn"):

    customer = {

        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active,
        "EstimatedSalary": salary,

        "CustomerValue": (balance + salary) / 2,

        "BalanceSalaryRatio": balance / salary if salary != 0 else 0,

        "ProductEngagement": products + active,

        "RiskScore": 0,

        "Geography_Germany": 1 if geography == "Germany" else 0,

        "Geography_Spain": 1 if geography == "Spain" else 0,

        "Gender_Male": 1 if gender == "Male" else 0
    }


    input_data = pd.DataFrame([customer])


    # arrange columns exactly like training
    input_data = input_data[features]


    prediction = model.predict_proba(input_data)


    churn_probability = prediction[0][1]


    st.subheader("Prediction Result")


    st.metric(
        "Churn Probability",
        f"{churn_probability*100:.2f}%"
        )

    st.progress(float(churn_probability))
    st.subheader("Why this prediction?")


    shap_values = explainer(input_data)


    importance = pd.DataFrame({
    "Feature": input_data.columns,
    "Impact": shap_values.values[0]
    })


    importance["Absolute Impact"] = importance["Impact"].abs()


    importance = importance.sort_values(
        by="Absolute Impact",
        ascending=False
    )


    st.write(
        importance[["Feature", "Impact"]].head(5)
    )
    st.progress(float(churn_probability))


    if churn_probability > 0.7:

        st.error("High Risk: Customer may leave")
        st.write("Recommendation: Offer retention benefits, discounts, or personalized support.")


    elif churn_probability > 0.4:

        st.warning("Medium Risk: Customer needs attention")
        st.write("Recommendation: Engage customer with offers and check satisfaction.")


    else:

        st.success("Low Risk: Customer likely to stay")
        st.write("Recommendation: No action needed.")