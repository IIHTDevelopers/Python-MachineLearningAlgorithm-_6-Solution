import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np


# 1. Load synthetic chicken disease dataset
def load_chicken_disease_data():
    print("📥 Loading dataset...")
    url = "chicken_disease_data.csv"
    df = pd.read_csv(url)
    df = df.head(1000)  # Limit to 1000 rows
    print(f"✅ Loaded {len(df)} records.\n")
    return df


# 2. EDA Function to count chickens with age > 2
def perform_eda_on_age(df):
    print("📊 Performing EDA on Age column...")
    if 'Age' not in df.columns:
        print("❌ 'Age' column not found in the dataset.\n")
        return

    count_over_2 = df[df['Age'] > 2].shape[0]
    print(f"🐔 Number of chickens with age > 2: {count_over_2}\n")


# 3. Preprocess data (Categorical to numerical conversion)
def preprocess_data(df):
    print("🛠️ Preprocessing data...")
    # Convert categorical features to dummy variables
    df = pd.get_dummies(df, drop_first=True)

    if "Disease Predicted_Healthy" not in df.columns:
        raise ValueError("❌ 'Disease Predicted_Healthy' column not found after encoding!")

    X = df.drop("Disease Predicted_Healthy", axis=1)  # Drop target column
    y = df["Disease Predicted_Healthy"]  # Target column
    print("✅ Features and target separated.\n")
    return X, y, df


# 4. Split the data
def split_data(X, y, test_size=0.2):
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


# 5. Create and train Decision Tree model
def create_and_train_model(X_train, y_train):
    print("🔧 Creating Decision Tree model...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained.\n")
    return model


# 6. Make predictions
def make_predictions(model, df_encoded, original_df):
    print("🔍 Making predictions on the first chicken in the dataset...")
    # Match encoded columns
    first_row_encoded = df_encoded.drop("Disease Predicted_Healthy", axis=1).iloc[[0]]
    prediction = model.predict(first_row_encoded)
    print("🧠 Original Chicken Data:")
    print(original_df.iloc[[0]].to_string(index=False))
    print(f"\n🔮 Prediction: {prediction[0]} --> {'Healthy' if prediction[0] == 0 else 'Diseased'}\n")


# 7. Calculate entropy of the target column
def calculate_entropy(y):
    print("📊 Calculating entropy (information gain basis)...")
    value_counts = y.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
    print(f"🧮 Entropy of target (Disease Predicted_Healthy): {entropy:.4f}\n")


# --- Pipeline Execution ---
df = load_chicken_disease_data()
perform_eda_on_age(df)  # ✅ EDA before preprocessing
X, y, df_encoded = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = create_and_train_model(X_train, y_train)

# Save model
joblib.dump(model, 'decision_tree_chicken_disease_model.pkl')
print("💾 Model saved as 'decision_tree_chicken_disease_model.pkl'")

# Use the first chicken (original row) for prediction
make_predictions(model, df_encoded, df)

# Calculate entropy of the target
calculate_entropy(y)
