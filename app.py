import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
file_path = "depression_anxiety_data.csv"
data = pd.read_csv(file_path)

data.drop(columns=["id"], errors="ignore", inplace=True)

# Fill missing values
imputer = SimpleImputer(strategy="most_frequent")
data[:] = imputer.fit_transform(data)


# Identify categorical columns
categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("depression_diagnosis")

data[categorical_columns] = data[categorical_columns].astype(str)

# One-hot encoding categorical columns
one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_features = one_hot_encoder.fit_transform(data[categorical_columns])
encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Label encoding target variable
label_encoder = LabelEncoder()
data["depression_diagnosis"] = label_encoder.fit_transform(data["depression_diagnosis"])

# Combine numerical and encoded categorical features
processed_data = pd.concat([data.drop(columns=categorical_columns), encoded_df], axis=1)

# Split dataset
X = processed_data.drop(columns=["depression_diagnosis"])
y = processed_data["depression_diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Save models and encoders
pickle.dump(xgb_model, open("mental_health_model.pkl", "wb"))
pickle.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# Load models for Streamlit
model = pickle.load(open("mental_health_model.pkl", "rb"))
encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))
label_enc = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Depression Diagnosis Predictor")

# Function to capture user input
def user_input_features():
    st.sidebar.header("User Input Features")
    gender = st.sidebar.selectbox("Select Gender", ['male', 'female'])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    depression_severity = st.sidebar.selectbox("Depression Severity", ["None-minimal", "Mild", "Moderately severe", "Severe"])
    depressiveness = st.sidebar.selectbox("Depressiveness", ["True", "False"])
    suicidal = st.sidebar.selectbox("Suicidal Thoughts", ["True", "False"])
    depression_treatment = st.sidebar.selectbox("Depression Treatment", ["True", "False"])
    anxiousness = st.sidebar.selectbox("Anxiousness", ["True", "False"])
    anxiety_diagnosis = st.sidebar.selectbox("Anxiety Diagnosis", ["True", "False"])
    anxiety_treatment = st.sidebar.selectbox("Anxiety Treatment", ["True", "False"])
    sleepiness = st.sidebar.selectbox("Sleepiness", ["True", "False"])
    anxiety_severity = st.sidebar.selectbox("Anxiety Severity", ['Moderate', 'Mild', 'Severe', 'None-minimal'])
    who_bmi = st.sidebar.selectbox("Select BMI", ['Class I Obesity', 'Normal', 'Overweight', 'Not Available',
                                                  'Class III Obesity', 'Underweight', 'Class II Obesity'])
    
    return pd.DataFrame([{ "age": age, "gender": gender, "who_bmi": who_bmi, "depression_severity": depression_severity,
                          "depressiveness": depressiveness, "suicidal": suicidal, "depression_treatment": depression_treatment,
                          "anxiety_severity": anxiety_severity, "anxiousness": anxiousness, "anxiety_diagnosis": anxiety_diagnosis,
                          "anxiety_treatment": anxiety_treatment, "sleepiness": sleepiness }])

input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)

# Preprocess user input
def preprocess_input(user_df):
    categorical_columns = user_df.select_dtypes(include=["object"]).columns.tolist()
    
    for col in categorical_columns:
        user_df[col] = user_df[col].astype(str)
    
    encoded_features = encoder.transform(user_df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    
    user_df = user_df.drop(columns=categorical_columns).reset_index(drop=True)
    final_input = pd.concat([user_df, encoded_df], axis=1)
    
    missing_cols = set(model.feature_names_in_) - set(final_input.columns)
    for col in missing_cols:
        final_input[col] = 0  
    
    final_input = final_input[model.feature_names_in_]
    return final_input

processed_input = preprocess_input(input_df)

if st.sidebar.button("Predict"):
    prediction = model.predict(processed_input)
    prediction_label = label_enc.inverse_transform(prediction)[0]
    st.subheader("Prediction")
    st.write(f"The model predicts: **{prediction_label}**")