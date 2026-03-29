import streamlit as st
import pandas as pd
from pipelines.deployment_pipeline import prediction_service_loader

st.title("E-Commerce Delivery Predictor ZENML")

st.sidebar.header("Enter Order Details")

payment_value = st.sidebar.number_input("Payment Value", value=100.0)
payment_installments = st.sidebar.number_input("Installments", value=1)
price = st.sidebar.number_input("Product Price", value=50.0)
freight_value = st.sidebar.number_input("Freight Value", value=10.0)
product_weight = st.sidebar.number_input("Product Weight (g)", value=500.0)

# Create dataframe
input_data = pd.DataFrame({
    "payment_value": [payment_value],
    "payment_installments": [payment_installments],
    "price": [price],
    "freight_value": [freight_value],
    "product_weight_g": [product_weight],
})

if st.button("Predict"):
    try:
        service = prediction_service_loader(
            pipeline_name="deployment_pipeline",
            step_name="mlflow_model_deployer_step",
            running=True,
        )

        service.start(timeout=10)

        prediction = service.predict(input_data)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")