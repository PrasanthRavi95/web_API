import streamlit as st
import requests

# Streamlit App


def main():
    st.title("Loan Approval Predictor")

    # User Input
    income = st.slider("Income", 10000, 100000, 50000)
    credit_score = st.slider("Credit Score", 500, 850, 700)
    loan_amount = st.slider("Loan Amount", 1000, 10000, 5000)

    # Make Prediction Button
    if st.button("Predict Loan Approval"):
        # Prepare input features
        input_data = {
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount
        }

        # Make a request to your Flask API
        api_url = "http://127.0.0.1:5000/predict_loan_approval"
        response = requests.post(api_url, json=input_data)

        # Display Prediction
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error making prediction. Please try again.")


if __name__ == "__main__":
    main()
