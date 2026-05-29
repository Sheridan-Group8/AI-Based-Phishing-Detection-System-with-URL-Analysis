import re
import pickle
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class PhishingDetector:
    """
    Phishing email detector using Naive Bayes with text features only.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = MultinomialNB()
        self.is_trained = False

    def preprocess_email(self, email_text):
        """Clean and preprocess email text."""
        if email_text is None:
            return ""
        text = str(email_text).lower()
        text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_features(self, emails, fit_vectorizer=False):
        """Extract text features (TF-IDF)."""
        processed_emails = [self.preprocess_email(email) for email in emails]
        if fit_vectorizer:
            return self.vectorizer.fit_transform(processed_emails)
        return self.vectorizer.transform(processed_emails)

    def train(self, emails, labels):
        """Train the Naive Bayes classifier."""
        print(f"Extracting text features from {len(emails)} emails...")
        X = self._extract_features(emails, fit_vectorizer=True)

        print(f"Feature dimensions: {X.shape[1]} (text only)")
        self.classifier.fit(X, labels)
        self.is_trained = True
        print("Model trained successfully!")

    def predict(self, email_text):
        """
        Predict if an email is phishing.

        Returns:
            tuple: (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = self._extract_features([email_text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)

        return prediction, confidence

    def evaluate(self, emails, labels):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = self._extract_features(emails)
        predictions = self.classifier.predict(X)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions,
                                    target_names=['Legitimate', 'Phishing']))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, predictions))

        return predictions

    def save_model(self, filepath='phishing_model.pkl'):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='phishing_model.pkl'):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = True

        print(f"Model loaded from {filepath}")


def load_dataset_from_huggingface():
    """
    Load the phishing email dataset from HuggingFace.
    Returns:
        (emails, labels) where `labels` are 1 for phishing and 0 for safe.
    """
    print("Loading phishing dataset from Hugging Face...")
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")

    # Inspect expected columns
    # The dataset contains: "Email Text", "Email Type" (two class strings) :contentReference[oaicite:1]{index=1}

    emails = [str(x) for x in dataset["Email Text"]]
    labels = [
        1 if str(lbl).lower() == "phishing email" else 0
        for lbl in dataset["Email Type"]
    ]

    phishing_count = sum(labels)
    safe_count = len(labels) - phishing_count
    print(f"Loaded {len(emails)} emails: {phishing_count} phishing, {safe_count} safe")

    return emails, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phishing Email Detector using Naive Bayes (Text Only)"
    )
    parser.add_argument("--csv", type=str, help="Path to Kaggle CSV file")
    parser.add_argument("--load-model", type=str, help="Load a pre-trained model")
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("  Phishing Email Detector (Text-Only Naive Bayes)")
    print("=" * 50 + "\n")

    detector = PhishingDetector()

    if args.load_model:
        detector.load_model(args.load_model)
    else:
        #if not args.csv:
            #raise ValueError("Please provide --csv dataset path")

        emails, labels = load_dataset_from_huggingface()

        print("\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=0.2, random_state=42, stratify=labels
        )

        print("\nTraining Naive Bayes classifier...")
        detector.train(X_train, y_train)
        detector.evaluate(X_test, y_test)
        detector.save_model()

    if not args.no_interactive:
        print("\nEnter an email to check (or 'quit' to exit):\n")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            prediction, confidence = detector.predict(user_input)
            status = "PHISHING" if prediction == 1 else "SAFE"
            print(f"\nResult: {status} (confidence: {confidence:.1%})\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
