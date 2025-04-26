import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


# 1. Load real-world dataset
def load_heart_disease_data():
    print("📥 Loading dataset...")
    url = "heart.csv"  # Make sure this file is in your working directory
    df = pd.read_csv(url)
    # Don't limit the rows, we need all 303 records for the test
    print(f"✅ Loaded {len(df)} records.\n")
    return df


# 2. Preprocess data
def preprocess_heart_data(df):
    print("🛠️ Preprocessing data...")
    X = df.drop("target", axis=1)
    y = df["target"]
    print("✅ Features and target separated.\n")
    return X, y


# 3. Split the data
def split_heart_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# 4. Create model
def create_model(n_estimators=100, max_depth=None):
    print("🔧 Creating Random Forest model...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return model


# 5. Train model
def train_model(model, X_train, y_train):
    print("🏋️ Training model...")
    model.fit(X_train, y_train)
    print("✅ Training complete.\n")
    return model


# 6. Save model
def save_model(model, filename="random_forest_heart_model.pkl"):
    print(f"💾 Saving model as '{filename}'...")
    joblib.dump(model, filename)
    print("✅ Model saved.\n")


# 7. Load model
def load_model(filename="random_forest_heart_model.pkl"):
    print(f"📦 Loading model from '{filename}'...")
    model = joblib.load(filename)
    print("✅ Model loaded.\n")
    return model


# 8. Check prediction for new data from JSON
def check_new_data_from_json(model, json_file="heart_data.json"):
    import json

    print(f"📄 Checking new data from {json_file}...")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Extract patient data
        patient = data['patient']

        # Convert to DataFrame for prediction
        patient_df = pd.DataFrame([patient])

        # Make prediction directly
        print(f"🔍 Making prediction on NEW PATIENT from JSON...")
        prediction = model.predict(patient_df)[0]

        # Calculate prediction probability if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(patient_df)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            confidence_str = f" (Confidence: {confidence:.2f})"
        else:
            confidence_str = ""

        # Convert prediction to integer if it's boolean
        is_diseased = int(prediction) if isinstance(prediction, (bool, np.bool_)) else prediction

        # Manual prediction based on symptoms and risk factors
        manual_prediction = 0  # Default to no disease

        # Check for high-risk indicators
        high_risk = False
        risk_factors = []

        # Age > 60 is a risk factor
        if patient['age'] > 60:
            risk_factors.append("Age > 60")

        # Chest pain type 3 or 4 (severe) is a risk factor
        if patient['cp'] >= 2:
            risk_factors.append("Severe chest pain (type 3-4)")

        # High cholesterol is a risk factor
        if patient['chol'] > 240:
            risk_factors.append("High cholesterol (>240)")

        # ST depression > 2.0 is a significant risk factor
        if patient['oldpeak'] > 2.0:
            risk_factors.append("Significant ST depression (>2.0)")
            high_risk = True

        # Multiple blocked vessels is a significant risk factor
        if patient['ca'] >= 2:
            risk_factors.append("Multiple blocked vessels")
            high_risk = True

        # Thalassemia type 2 or 3 is a risk factor
        if patient['thal'] >= 2:
            risk_factors.append("Abnormal thalassemia")

        # If multiple risk factors or high-risk indicators are present
        if len(risk_factors) >= 3 or high_risk:
            manual_prediction = 1

        # Use model prediction, but if it contradicts obvious symptoms, use manual prediction
        final_prediction = is_diseased
        if is_diseased == 0 and manual_prediction == 1:
            print("⚠️ Warning: Model prediction contradicts risk factor assessment.")
            print("⚠️ Risk factors identified:")
            for factor in risk_factors:
                print(f"  - {factor}")
            print("⚠️ Using risk factor-based prediction instead.")
            final_prediction = manual_prediction

        print("🧠 Patient Input:")
        print(patient_df.to_string(index=False))
        print(f"\n🔮 Model Confidence Level: {confidence:.2f}")

        if manual_prediction != is_diseased:
            print(
                f"🔄 Adjusted Prediction: {final_prediction} --> {'❤️ Heart Disease' if final_prediction == 1 else '💚 No Heart Disease'}\n")
        else:
            print("")  # Extra line for formatting

        # Print final result
        print("\n📋 FINAL HEART DISEASE PREDICTION RESULT:")
        print(f"🔍 Patient has heart disease: {'YES' if final_prediction == 1 else 'NO'}")

        if risk_factors:
            print("⚠️ Risk factors identified:")
            for factor in risk_factors:
                print(f"  - {factor}")

        print(f"📝 Diagnosis: The patient is predicted to " +
              ("have heart disease" if final_prediction == 1 else "not have heart disease") +
              " based on the provided features.\n")

    except Exception as e:
        print(f"❌ Error checking new data: {e}\n")


# --- Pipeline Execution ---
df = load_heart_disease_data()
X, y = preprocess_heart_data(df)
X_train, X_test, y_train, y_test = split_heart_data(X, y)
model = create_model()
trained_model = train_model(model, X_train, y_train)
save_model(trained_model)

# Check new data from JSON
check_new_data_from_json(trained_model)
