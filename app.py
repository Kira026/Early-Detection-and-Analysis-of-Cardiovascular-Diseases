import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Load the dataset
@st.cache_data
def load_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv("C:/Users/raksh/Downloads/heart.csv")
    return df.copy()  # Create a copy of the DataFrame to prevent mutation

# Train the model
@st.cache_data
def train_model():
    df = load_data()
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Split the dataset into features (X) and target variable (y)
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the model
    rf_classifier.fit(X, y)

    return rf_classifier

def main():
    st.title("Heart Disease Prediction")

    # Sidebar with options
    st.sidebar.header("Options")
    selected_option = st.sidebar.radio("Select Option", ["Train Model", "Predict"])

    if selected_option == "Train Model":
        st.subheader("Training Model")
        model = train_model()
        st.success("Model trained successfully!")

    elif selected_option == "Predict":
        st.subheader("Make Predictions")
        model = train_model()
        df = load_data()
        new_data = {}  # Dictionary to store user input
        
        # Display original categories for categorical variables
        original_categories = {
            'Sex': {'M': 'Male', 'F': 'Female'},
            'ChestPainType': {'ASY': 'Asymptomatic', 'ATA': 'Atypical Angina', 'NAP': 'Non-Anginal Pain', 'TA': 'Typical Angina'},
            'FastingBS': {0: 'False', 1: 'True'},
            'RestingECG': {0: 'Normal', 1: 'Abnormality', 2: 'Hypertrophy'},
            'ExerciseAngina': {'N': 'No', 'Y': 'Yes'},
            'ST_Slope': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        }
        
        # Collect user input
        input_fields = {}
        for col in df.columns:
            if col != 'HeartDisease':  # Exclude target variable
                if col == 'Oldpeak':
                    # If the column is 'Oldpeak', display in "0.00" format
                    input_fields[col] = st.number_input(f"Enter {col}", value=0.0, format="%.2f")
                elif col in original_categories:
                    # For other columns with original categories, display them
                    input_fields[col] = st.selectbox(f"Select {col}", [None] + list(original_categories[col].values()))
                else:
                    # For numerical columns, display without decimal places
                    input_fields[col] = st.number_input(f"Enter {col}", value=0)
        
        # Add a reset button to reset input values
        if st.button("Reset"):
            input_fields = {}  # Reset all input fields
        
        # Encode categorical variables in the user input
        label_encoder = LabelEncoder()
        for col in ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
            if input_fields.get(col) is not None:
                input_fields[col] = label_encoder.fit_transform([key for key, value in original_categories[col].items() if value == input_fields[col]])[0]
        
        # Make prediction if all input fields have values
        if all(value is not None for value in input_fields.values()):
            X = pd.DataFrame([input_fields])
            st.write("Input data (X):", X)  # Debugging statement
            if not X.empty:
                prediction = model.predict(X)
                st.write("Prediction:", prediction)
            else:
                st.warning("Input data is empty. Please enter values for prediction.")

if __name__ == "__main__":
    main()
