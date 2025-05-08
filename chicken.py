import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np


# 1. Load chicken disease dataset
def load_chicken_disease_data():
    print(" Loading dataset...")
    df = pd.read_csv("chicken_disease_data.csv")
    print(f" Loaded {len(df)} records.\n")
    return df


# 2. EDA Function to count chickens with age > 2
def perform_eda_on_age(df):
    print(" Performing EDA on Age column...")
    if 'Age' not in df.columns:
        print(" 'Age' column not found.\n")
        return
    count_over_2 = df[df['Age'] > 2].shape[0]
    print(f" Number of chickens with age > 2: {count_over_2}\n")


# 3. Preprocess data with explicit label encoding
def preprocess_chicken_data(df):
    print("️ Preprocessing data...")

    if "Disease Predicted" not in df.columns:
        raise ValueError("Target column 'Disease Predicted' not found.")

    # Clean and normalize values
    df["Disease Predicted"] = df["Disease Predicted"].astype(str).str.strip().str.title()

    print(" Unique target values after cleaning:", df["Disease Predicted"].unique())

    # Map Healthy = 0, Sick = 1
    df["target"] = df["Disease Predicted"].map({"Healthy": 0, "Sick": 1})

    # Check for unmapped labels
    if df["target"].isnull().any():
        raise ValueError("Unrecognized values in 'Disease Predicted'. Expected 'Healthy' or 'Sick'.")

    df = df.drop("Disease Predicted", axis=1)

    # One-hot encode features
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    print(" Features and target separated.\n")
    return X, y, df


# 4. Split data
def split_chicken_data(X, y, test_size=0.2):
    print(" Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f" Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# 5. Train model
def create_and_train_model(X_train, y_train):
    print(" Creating Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(" Model trained.\n")
    return model


# 6. Predict from new JSON data
def check_new_data_from_json(model, json_file="chicken_data.json"):
    import json

    print(f" Checking new data from {json_file}...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        chicken = data['chicken']

        original_df = load_chicken_disease_data()

        # Construct new row
        temp_df = pd.DataFrame([{
            'Age': chicken['Age'],
            'Breed': chicken['Breed'],
            'Temperature': chicken['Temperature'],
            'Eating Behavior': chicken['Eating_Behavior'],
            'Coughing': chicken['Coughing'],
            'Feces Appearance': chicken['Feces_Appearance'],
            'Water Consumption': chicken['Water_Consumption'],
            'Disease Predicted': 'Healthy'  # Placeholder
        }])

        combined_df = pd.concat([original_df, temp_df], ignore_index=True)
        X_combined, _, _ = preprocess_chicken_data(combined_df)

        new_chicken_features = X_combined.iloc[[-1]]

        prediction = model.predict(new_chicken_features)[0]
        final_prediction = int(prediction)

        print(f" Prediction result: {'Sick' if final_prediction == 1 else 'Healthy'}")

    except Exception as e:
        print(f" Error checking new data: {e}")
        final_prediction = None

    return final_prediction


# --- Pipeline Execution ---
df = load_chicken_disease_data()
perform_eda_on_age(df)
X, y, df_encoded = preprocess_chicken_data(df)
X_train, X_test, y_train, y_test = split_chicken_data(X, y)
model = create_and_train_model(X_train, y_train)

# Save model
joblib.dump(model, 'decision_tree_chicken_disease_model.pkl')
print(" Model saved as 'decision_tree_chicken_disease_model.pkl'")

# Predict from JSON
check_new_data_from_json(model)
