"""
data_cleaning.py: Cleans and preprocesses the diabetic_data.csv file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

RAW_DATA_PATH = "data/raw/diabetic_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"

def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop columns with too many missing values or IDs
    df.drop(["weight", "payer_code", "medical_specialty", "encounter_id", "patient_nbr"], axis=1, inplace=True)

    # Replace missing values marked as '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop rows with NaN
    df.dropna(inplace=True)

    # Binary target: readmitted within 30 days = 1, otherwise 0
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Save cleaned data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    clean_data()
