import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load real-world dataset
def load_heart_disease_data():
    print("📥 Loading dataset...")
    url = "heart.csv"  # Make sure this file is in your working directory
    df = pd.read_csv(url)
    df = df.head(1000)  # Limit to 1000 rows
    print(f"✅ Loaded {len(df)} records.\n")
    return df

# 2. Preprocess data
def preprocess_data(df):
    print("🛠️ Preprocessing data...")
    X = df.drop("target", axis=1)
    y = df["target"]
    print("✅ Features and target separated.\n")
    return X, y

# 3. Split the data
def split_data(X, y, test_size=0.2):
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

# 8. Make a prediction
def make_prediction(model, X_sample, source_label=""):
    print(f"🔍 Making prediction on {source_label}...")
    prediction = model.predict(X_sample)
    print("🧠 Sample Input:")
    print(X_sample.to_string(index=False))
    print(f"\n🔮 Prediction: {prediction[0]} --> {'❤️ Heart Disease' if prediction[0] == 1 else '💚 No Heart Disease'}\n")

# --- Pipeline Execution ---
df = load_heart_disease_data()
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
model = create_model()
trained_model = train_model(model, X_train, y_train)
save_model(trained_model)

# 📌 Use FIRST PERSON from original dataset
sample_first_person = df.drop("target", axis=1).iloc[[0]]
make_prediction(trained_model, sample_first_person, source_label="FIRST PATIENT in dataset")
