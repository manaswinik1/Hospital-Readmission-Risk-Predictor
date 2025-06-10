

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Readmission Risk Predictor", layout="centered")

st.title("🏥 Hospital Readmission Risk Predictor")
st.write("This app uses a trained model to predict if a patient is likely to be readmitted within 30 days.")

# Load cleaned dataset structure to build form inputs
sample_data = pd.read_csv("data/processed/cleaned_data.csv")
X = sample_data.drop("readmitted", axis=1)
features = X.columns.tolist()

# Load model
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("outputs/xgb_model.json")
    return model

model = load_model()

# User inputs
st.subheader("Patient Information")
user_input = {}
for col in features:
    val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    user_input[col] = val

input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.write("🔍 **Prediction:**", "Readmitted" if prediction == 1 else "Not Readmitted")
    st.write("📊 **Probability:**", f"{probability*100:.2f}% chance of readmission")
