# Depression Diagnosis Predictor

This is a Streamlit-based web application that predicts depression diagnoses based on user input features using an XGBoost model. The application also provides interpretability using LIME (Local Interpretable Model-agnostic Explanations).

## Features
- User-friendly web interface built with Streamlit.
- Inputs include various demographic and mental health-related features.
- Uses an XGBoost model for prediction.
- Implements LIME to explain individual predictions.
- Supports categorical feature encoding using OneHotEncoder.

## Dataset Preprocessing Steps

- Load dataset and drop unnecessary columns (e.g., ID).
- Handle missing values using SimpleImputer with the "most_frequent" strategy.
- Identify categorical columns and convert them to string type.
- Apply One-Hot Encoding to categorical variables using OneHotEncoder.
- Label encode the target variable (depression_diagnosis) using LabelEncoder.
- Split the dataset into training and testing sets (80-20 split).
- Model Selection Rationale
- XGBoost Classifier was chosen due to its high accuracy, ability to handle imbalanced datasets, and robustness in handling categorical and numerical features.
- The model was trained using default parameters, which can be fine-tuned for better performance.
- LIME was integrated for model interpretability to understand feature importance.

## Installation

### Prerequisites
Ensure you have Python installed. Then, install the required dependencies:
```bash
pip install pandas numpy streamlit scikit-learn xgboost lime pickle-mixin
```

## Usage

1. Clone the repository:
```bash
git clone <https://github.com/HiteshM30/Mental-Health-Prediction-Model.git>
cd <repository_folder>
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Enter the required input features in the sidebar and click on the **Predict** button.
4. The model will display the predicted depression diagnosis along with a LIME explanation.

## File Structure
- `app.py` - Main Streamlit application file.
- `depression_anxiety_data.csv` - Sample dataset used for training.
- `mental_health_model.pkl` - Trained XGBoost model.
- `one_hot_encoder.pkl` - Pre-trained OneHotEncoder for categorical variables.
- `label_encoder.pkl` - Pre-trained LabelEncoder for target variable.

## Model Training
The model is trained using the following steps:
- Preprocessing categorical and numerical features.
- Encoding categorical variables using OneHotEncoder.
- Splitting data into training and test sets.
- Training an XGBoost classifier.
- Saving the trained model and encoders using Pickle.

## Explanation with LIME
LIME is used to explain the model's predictions by showing the most influential features.
- After making a prediction, a LIME explanation is generated.
- The explanation highlights key features affecting the model's decision.
- The results are displayed as text and a visual plot.

## Author
Hitesh Matharu

## Acknowledgments
Special thanks to the open-source community for providing valuable tools and frameworks like Streamlit, XGBoost, and LIME.

