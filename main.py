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

# %%
# Load the dataset
file_path = "depression_anxiety_data.csv"
data = pd.read_csv(file_path)
data

# %%


# %%
"""
## Preprocessing the Data
"""

# %%
data.isna().sum()

# %%
# Drop unnecessary columns
data = data.drop(columns=["id"], errors='ignore')

# Handle missing values
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

# %%
# Encode categorical variables
categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
categorical_columns.remove("depression_diagnosis")  # Target variable

one_hot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded_features = one_hot_encoder.fit_transform(data[categorical_columns])
encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Encode target variable
label_encoder = LabelEncoder()
data["depression_diagnosis"] = label_encoder.fit_transform(data["depression_diagnosis"])  # Yes = 1, No = 0

# Combine processed data
processed_data = pd.concat([data[["age", "depression_diagnosis"]], encoded_df], axis=1)

# %%
processed_data.nunique()

# %%
"""
## Splitting Data and Training all three models and comparing their scores
"""

# %%
# Split the data
X = processed_data.drop(columns=["depression_diagnosis"])
y = processed_data["depression_diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#Train an XGBoost model
xgb_model=XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb=xgb_model.predict(X_test)

# Evaluate the model
rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1-score": f1_score(y_test, y_pred_rf),
}
print('Evaluaution of Random Forest Model is as follows:\n')
print(rf_metrics,'\n\n')

xgb_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb),
    "Recall": recall_score(y_test, y_pred_xgb),
    "F1-score": f1_score(y_test, y_pred_xgb)
}
print('Evaluaution of XGBoost Model is as follows:\n')
print(xgb_metrics,'\n\n')


# %%
X

# %%
"""
### As XGBoost model has better evaluation metrics, we will use it for deployment
"""

# %%
# Save model and encoders
pickle.dump(xgb_model, open("mental_health_model.pkl", "wb"))
pickle.dump(one_hot_encoder, open("one_hot_encoder.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# Inference function
def predict_mental_health(symptoms_dict):
    model = pickle.load(open("mental_health_model.pkl", "rb"))
    encoder = pickle.load(open("one_hot_encoder.pkl", "rb"))
    label_enc = pickle.load(open("label_encoder.pkl", "rb"))

    # Convert user input into a DataFrame
    user_df = pd.DataFrame([symptoms_dict])

    # Ensure all columns match training set (handle missing columns)
    missing_cols = set(encoder.get_feature_names_out()) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0  # Fill missing columns with 0

    # Reorder columns to match the training set
    user_df = user_df[encoder.get_feature_names_out()]

    # Predict and return the result
    prediction = model.predict(user_df)
    return label_enc.inverse_transform(prediction)[0]


# %%
"""
### Building the StreamLit UI
"""

# %%
