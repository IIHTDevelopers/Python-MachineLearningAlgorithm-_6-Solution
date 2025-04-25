import unittest
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Path setup for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test.TestUtils import TestUtils
import heart

class TestHeartDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = heart.load_heart_disease_data()

    def test_heart_csv_loaded_correctly(self):
        try:
            if not self.df.empty and len(self.df) == 303:
                self.test_obj.yakshaAssert("TestCSVLoadedCorrectlyheart", True, "functional")
                print("TestCSVLoadedCorrectlyheart = Passed")
            else:
                self.test_obj.yakshaAssert("TestCSVLoadedCorrectlyheart", False, "functional")
                print("TestCSVLoadedCorrectlyheart = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestCSVLoadedCorrectlyheart", False, "functional")
            print("TestCSVLoadedCorrectlyheart = Failed")

    def test_preprocessing_separates_target(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            print(f"X columns: {X.columns}")
            print(f"y name: {y.name}")
            if not X.empty and not y.empty and "target" not in X.columns:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTarget", True, "functional")
                print("TestPreprocessingSeparatesTarget = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTarget", False, "functional")
                print("TestPreprocessingSeparatesTarget = Failed")
        except Exception as e:
            print(f"Exception: {str(e)}")
            self.test_obj.yakshaAssert("TestPreprocessingSeparatesTarget", False, "functional")
            print("TestPreprocessingSeparatesTarget = Failed")

    def test_split_data_counts(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            X_train, X_test, y_train, y_test = heart.split_heart_data(X, y)
            print(f"X_train length: {len(X_train)}, X_test length: {len(X_test)}")
            if len(X_train) == 242 and len(X_test) == 61:
                self.test_obj.yakshaAssert("TestSplitDataCounts", True, "functional")
                print("TestSplitDataCounts = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitDataCounts", False, "functional")
                print("TestSplitDataCounts = Failed")
        except Exception as e:
            print(f"Exception: {str(e)}")
            self.test_obj.yakshaAssert("TestSplitDataCounts", False, "functional")
            print("TestSplitDataCounts = Failed")

    def test_model_trains_and_saves(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            X_train, X_test, y_train, y_test = heart.split_heart_data(X, y)
            model = heart.train_model(heart.create_model(), X_train, y_train)
            heart.save_model(model, "random_forest_heart_model.pkl")
            print(f"Model file exists: {os.path.exists('random_forest_heart_model.pkl')}")
            if os.path.exists("random_forest_heart_model.pkl"):
                self.test_obj.yakshaAssert("TestModelTrainsAndSaves", True, "functional")
                print("TestModelTrainsAndSaves = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelTrainsAndSaves", False, "functional")
                print("TestModelTrainsAndSaves = Failed")
        except Exception as e:
            print(f"Exception: {str(e)}")
            self.test_obj.yakshaAssert("TestModelTrainsAndSaves", False, "functional")
            print("TestModelTrainsAndSaves = Failed")

    def test_prediction_for_first_patient(self):
        try:
            X, y = heart.preprocess_heart_data(self.df)
            X_train, X_test, y_train, y_test = heart.split_heart_data(X, y)
            model = heart.train_model(heart.create_model(), X_train, y_train)

            first_patient = self.df.drop(columns=["target"]).iloc[[0]]
            prediction = model.predict(first_patient)

            expected_prediction = 1  # From your console output
            print(f"Prediction: {prediction[0]}, Expected: {expected_prediction}")

            if prediction[0] == expected_prediction:
                self.test_obj.yakshaAssert("TestPredictionForFirstPatient", True, "functional")
                print("TestPredictionForFirstPatient = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictionForFirstPatient", False, "functional")
                print("TestPredictionForFirstPatient = Failed")
        except Exception as e:
            print(f"Exception: {str(e)}")
            self.test_obj.yakshaAssert("TestPredictionForFirstPatient", False, "functional")
            print("TestPredictionForFirstPatient = Failed")


import unittest
import os
import sys
import pandas as pd
import numpy as np

# Adjust path to access your main code and TestUtils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test.TestUtils import TestUtils
import chicken

class TestChickenDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = chicken.load_chicken_disease_data()

    def test_chicken_csv_loaded_correctly(self):
        try:
            if not self.df.empty and len(self.df) == 1000:
                self.test_obj.yakshaAssert("TestCSVLoadedCorrectly", True, "functional")
                print("TestCSVLoadedCorrectly = Passed")
            else:
                self.test_obj.yakshaAssert("TestCSVLoadedCorrectly", False, "functional")
                print("TestCSVLoadedCorrectly = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestCSVLoadedCorrectly", False, "functional")
            print("TestCSVLoadedCorrectly = Failed")

    def test_eda_on_age_count(self):
        try:
            count = self.df[self.df["Age"] > 2].shape[0]
            if count == 493:
                self.test_obj.yakshaAssert("TestEDAOnAgeCount", True, "functional")
                print("TestEDAOnAgeCount = Passed")
            else:
                self.test_obj.yakshaAssert("TestEDAOnAgeCount", False, "functional")
                print("TestEDAOnAgeCount = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestEDAOnAgeCount", False, "functional")
            print("TestEDAOnAgeCount = Failed")

    def test_preprocessing_output(self):
        try:
            X, y, df_encoded = chicken.preprocess_chicken_data(self.df)
            if "Disease Predicted_Healthy" in y.name and not X.empty:
                self.test_obj.yakshaAssert("TestPreprocessingOutput", True, "functional")
                print("TestPreprocessingOutput = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessingOutput", False, "functional")
                print("TestPreprocessingOutput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPreprocessingOutput", False, "functional")
            print("TestPreprocessingOutput = Failed")

    def test_model_trains_successfully(self):
        try:
            X, y, df_encoded = chicken.preprocess_chicken_data(self.df)
            X_train, X_test, y_train, y_test = chicken.split_chicken_data(X, y)
            model = chicken.create_and_train_model(X_train, y_train)
            if model and hasattr(model, "predict"):
                self.test_obj.yakshaAssert("TestModelTrainsSuccessfully", True, "functional")
                print("TestModelTrainsSuccessfully = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelTrainsSuccessfully", False, "functional")
                print("TestModelTrainsSuccessfully = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestModelTrainsSuccessfully", False, "functional")
            print("TestModelTrainsSuccessfully = Failed")

    def test_entropy_calculation(self):
        try:
            _, y, _ = chicken.preprocess_chicken_data(self.df)
            value_counts = y.value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)

            if np.isclose(entropy, 0.8016, atol=0.001):
                self.test_obj.yakshaAssert("TestEntropyCalculation", True, "functional")
                print("TestEntropyCalculation = Passed")
            else:
                self.test_obj.yakshaAssert("TestEntropyCalculation", False, "functional")
                print("TestEntropyCalculation = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestEntropyCalculation", False, "functional")
            print("TestEntropyCalculation = Failed")
