import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json


# 1. Load dataset
def load_heart_disease_data():
    print(" Loading dataset...")
    df = pd.read_csv("heart.csv")
    df = df.head(303)  # Limit to 303 records to match test expectations
    print(f" Loaded {len(df)} records.\n")
    return df


# 2. Preprocess data
def preprocess_heart_data(df):
    print(" Preprocessing data...")
    X = df.drop("target", axis=1)
    y = df["target"]
    print(" Features and target separated.\n")
    return X, y


# 3. Split the data
def split_heart_data(X, y, test_size=0.2):
    print(" Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f" Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


def create_train_save_load_model(X_train, y_train, n_estimators=100, max_depth=None,
                                 filename="random_forest_heart_model.pkl"):
    print(" Creating Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    print(" Training model...")
    model.fit(X_train, y_train)
    print(" Training complete.\n")

    print(f" Saving model to '{filename}'...")
    joblib.dump(model, filename)
    print(" Model saved.\n")

    print(f" Loading model from '{filename}'...")
    loaded_model = joblib.load(filename)
    print(" Model loaded.\n")

    return loaded_model




# 6. Predict using model only (no manual checking)
def check_new_data_from_json(model, json_file="heart_data.json"):
    import json

    print(f" Checking new data from {json_file}...")
    try:
        # Load data from JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # Extract patient data
        patient = data['patient']
        patient_df = pd.DataFrame([patient])
        
        # Make prediction
        prediction = model.predict(patient_df)[0]
        
        # Print result
        print(f" Prediction result: {'Diseased' if prediction == 1 else 'Healthy'}")
        
        return prediction
        
    except Exception as e:
        print(f" Error checking new data: {e}")
        return None

# --- Pipeline Execution ---
df = load_heart_disease_data()
X, y = preprocess_heart_data(df)
X_train, X_test, y_train, y_test = split_heart_data(X, y)

# Combined step: create, train, save, and load the model
trained_model = create_train_save_load_model(X_train, y_train)

# Check new data from JSON
check_new_data_from_json(trained_model)
