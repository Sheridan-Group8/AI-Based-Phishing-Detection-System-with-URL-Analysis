# test_svm.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load test dataset
test_df = pd.read_csv("phishing_emails_test.csv")
X_test_text = test_df["email_text"]
y_test = test_df["label"]

# Load saved model and vectorizer
svm_model = joblib.load("svm_phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Transform test data using the saved vectorizer
X_test = vectorizer.transform(X_test_text)

# Predict on the test dataset
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
