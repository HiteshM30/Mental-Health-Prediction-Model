# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import streamlit as st

file_path = "depression_anxiety_data.csv"
data = pd.read_csv(file_path)

data = data.drop(columns=["id"], errors='ignore')

imputer = SimpleImputer(strategy="most_frequent")
data["depression_severity"] = imputer.fit_transform(data[["depression_severity"]]).flatten()
data["depressiveness"] = imputer.fit_transform(data[["depressiveness"]]).flatten()
data["suicidal"] = imputer.fit_transform(data[["suicidal"]]).flatten()
data["depression_diagnosis"] = imputer.fit_transform(data[["depression_diagnosis"]]).flatten()
data["depression_treatment"] = imputer.fit_transform(data[["depression_treatment"]]).flatten()
data["anxiousness"] = imputer.fit_transform(data[["anxiousness"]]).flatten()
data["anxiety_diagnosis"] = imputer.fit_transform(data[["anxiety_diagnosis"]]).flatten()
data["anxiety_treatment"] = imputer.fit_transform(data[["anxiety_treatment"]]).flatten()
data["epworth_score"] = imputer.fit_transform(data[["epworth_score"]]).flatten()
data["sleepiness"] = imputer.fit_transform(data[["sleepiness"]]).flatten()

categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("depression_diagnosis")  # Target variable

one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_features = one_hot_encoder.fit_transform(data[categorical_columns])
encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

label_encoder = LabelEncoder()
data["depression_diagnosis"] = label_encoder.fit_transform(data["depression_diagnosis"])  # Yes = 1, No = 0

processed_data = pd.concat([data[["age", "depression_diagnosis"]], encoded_df], axis=1)

X = processed_data.drop(columns=["depression_diagnosis"])
y = processed_data["depression_diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model=XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_xgb=xgb_model.predict(X_test)

pickle.dump(xgb_model, open("mental_health_model.pkl", "wb"))
pickle.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

def predict_mental_health(symptoms_dict):
    model = pickle.load(open("mental_health_model.pkl", "rb"))
    encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))
    label_enc = pickle.load(open("label_encoder.pkl", "rb"))

    user_df = pd.DataFrame([symptoms_dict])

    missing_cols = set(encoder.get_feature_names_out()) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0  

    user_df = user_df[encoder.get_feature_names_out()]

    prediction = model.predict(user_df)
    return label_enc.inverse_transform(prediction)[0]

import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("mental_health_model.pkl", "rb"))
encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))
label_enc = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Depression Diagnosis Predictor")

st.sidebar.header("User Input Features")

def user_input_features():
    gender = st.sidebar.selectbox("Select Gender", ['Male', 'Female', 'Other'])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    depression_severity = st.sidebar.selectbox("Depression Severity", ["None", "Mild", "Moderate", "Severe"])
    depressiveness = st.sidebar.selectbox("Depressiveness", ["Not at all", "Slightly", "Moderately", "Very much"])
    suicidal = st.sidebar.selectbox("Suicidal Thoughts", ["Never", "Rarely", "Sometimes", "Often"])
    depression_treatment = st.sidebar.selectbox("Depression Treatment", ["No", "Yes"])
    anxiousness = st.sidebar.selectbox("Anxiousness", ["Not at all", "Slightly", "Moderately", "Very much"])
    anxiety_diagnosis = st.sidebar.selectbox("Anxiety Diagnosis", ["No", "Yes"])
    anxiety_treatment = st.sidebar.selectbox("Anxiety Treatment", ["No", "Yes"])
    sleepiness = st.sidebar.selectbox("Sleepiness", ["Not at all", "Slightly", "Moderately", "Very much"])
    anxiety_severity = st.sidebar.selectbox("Anxiety Severity", ['Moderate', 'Mild', 'Severe', 'None-minimal'])
    who_bmi = st.sidebar.selectbox("Select BMI", ['Class I Obesity', 'Normal', 'Overweight', 'Not Available',
                                                  'Class III Obesity', 'Underweight', 'Class II Obesity'])

    user_data = {
        "age": age,
        "gender": gender,
        "who_bmi": who_bmi,
        "depression_severity": depression_severity,
        "depressiveness": depressiveness,
        "suicidal": suicidal,
        "depression_treatment": depression_treatment,
        "anxiety_severity": anxiety_severity,
        "anxiousness": anxiousness,
        "anxiety_diagnosis": anxiety_diagnosis,
        "anxiety_treatment": anxiety_treatment,
        "sleepiness": sleepiness
    }

    return pd.DataFrame([user_data])

input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)

def preprocess_input(user_df):
    categorical_columns = user_df.select_dtypes(include=["object"]).columns.tolist()
    
    encoded_features = encoder.transform(user_df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

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


# %%
