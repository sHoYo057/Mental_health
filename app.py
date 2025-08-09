import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 1. Load and preprocess data
# ================================
df = pd.read_csv("mental_health_clean.csv")

# Example filtering (you can adjust)
filtered_df = df.dropna(subset=['Gender', 'treatment'])

# ================================
# 2. Treatment Seeking by Gender
# ================================
treatment_by_gender = (
    filtered_df.groupby('Gender')['treatment']
    .value_counts(normalize=True)
    .mul(100)
    .unstack()
)

st.subheader("Treatment Seeking by Gender")
st.bar_chart(treatment_by_gender)

# ================================
# 3. Train Model (Logistic Regression Example)
# ================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Encode target
df['treatment'] = LabelEncoder().fit_transform(df['treatment'].astype(str))

# Handle categorical features
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop(columns=['treatment'])
y = df_encoded['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('model', LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# ================================
# 4. Feature Importance Plot
# ================================
feature_names = X.columns
coefficients = model.named_steps['model'].coef_[0]

feat_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': coefficients
}).sort_values(by='importance', key=abs, ascending=False)

st.subheader("Feature Importance")
st.dataframe(feat_importance)

# ================================
# 5. Prediction Example
# ================================
st.subheader("Make a Prediction")

# Example: Let user input values
user_input = {}
for col in X.columns:
    val = st.text_input(f"Enter value for {col}", value=str(X[col].iloc[0]))
    user_input[col] = val

# Convert to dataframe
input_df = pd.DataFrame([user_input])

# Encode input in the same way as training
for col in input_df.columns:
    input_df[col] = LabelEncoder().fit(X[col]).transform(input_df[col].astype(str))

# Make prediction
pred_prob = model.predict_proba(input_df)[0][1]
pred_class = model.predict(input_df)[0]

st.write(f"Prediction: {'Will seek treatment' if pred_class == 1 else 'Will not seek treatment'}")
st.write(f"Probability: {pred_prob:.2f}")