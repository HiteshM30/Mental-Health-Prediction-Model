import pickle
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

# Load models and encoders
model = pickle.load(open("mental_health_model.pkl", "rb"))
encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))
label_enc = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Depression Diagnosis Predictor")

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
    user_df[categorical_columns] = user_df[categorical_columns].astype(str)
    
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
    
    # LIME explanation
    explainer = LimeTabularExplainer(model.feature_importances_.reshape(1, -1),
                                     feature_names=processed_input.columns.tolist(),
                                     class_names=label_enc.classes_,
                                     discretize_continuous=True)
    explanation = explainer.explain_instance(processed_input.values[0], model.predict_proba)
    
    st.subheader("LIME Explanation")
    st.write(explanation.as_list())
    st.pyplot(explanation.as_pyplot_figure())