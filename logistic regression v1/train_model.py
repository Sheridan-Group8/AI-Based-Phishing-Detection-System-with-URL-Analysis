import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# 1. Load dataset
df = pd.read_csv("data/emails.csv")

X = df["text"]
y = df["label"]

# 2. Initialize TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=3000
)

# 3. Convert text → TF-IDF
X_tfidf = vectorizer.fit_transform(X)

# 4. Train ML model
model = LogisticRegression()
model.fit(X_tfidf, y)

# 5. Save vectorizer + model
os.makedirs("model", exist_ok=True)  # create folder if not exists
with open("model/email_classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved successfully as model/email_classifier.pkl")
