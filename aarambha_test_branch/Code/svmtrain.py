import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load training dataset
train_df = pd.read_csv("phishing_emails_clean_no_punct.csv")

# Features and labels
X_train_text = train_df["email_text"]
y_train = train_df["label"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1)
X_train = vectorizer.fit_transform(X_train_text)

# Train SVM model
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(svm_model, "svm_phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Training complete!")
print("Trained model saved as 'svm_phishing_model.pkl'")
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
