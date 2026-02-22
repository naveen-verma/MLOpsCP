import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="nv185001/pred-model", filename="best_engine_failure_predictor_model.joblib")
# Load the model
model = joblib.load(model_path)

# Streamlit UI for Engine Failure Prediction
st.title("Engine Failure Prediction App")
st.write("The Engine Failure Prediction App is an internal tool to predict whether engine would fail due to current vital parameters.")
st.write("Kindly enter different parameters of engine to check whether they are likely to fail or not")

Engine_rpm = st.number_input("Engine RPM", min_value=0 )
Lub_Oil_Pressure = st.number_input("Lub Oil Pressure", min_value=0)
Fuel_Pressure = st.number_input("Fuel Pressure", min_value=0)
Coolant_Pressure = st.number_input("Coolant Pressure", min_value=0
Lub_Oil_Temperature = st.number_input("Lub Oil Temperature", min_value=0)
Coolant_Temperature = st.number_input("Coolant Temperature", min_value=0)


input_data = pd.DataFrame([{
    'Engine_rpm': Engine_rpm,
    'Lub_Oil_Pressure': Lub_Oil_Pressure,
    'Fuel_Pressure': Fuel_Pressure,
    'Coolant_Pressure': Coolant_Pressure,
    'Lub_Oil_Temperature': Lub_Oil_Temperature,
    'Coolant_Temperature': Coolant_Temperature
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "to buy the package" if prediction == 1 else "that will not buy the package"
    st.write(f"Based on the information provided, the customer is likely {result}.")

