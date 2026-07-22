"""
Integrated Phishing Detection Toolkit
- URL phishing detector using Random Forest + structural/reputation features
- Email phishing detector using Naive Bayes + TF-IDF text features

This file combines the two provided scripts into one runnable module.
"""

import re
import pickle
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract
import whois
from datasets import load_dataset
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# URL PHISHING DETECTOR
# ============================================================

class URLAnalyzer:
    SUSPICIOUS_TLDS = {
        ".xyz", ".top", ".club", ".work", ".click", ".link", ".gq", ".ml",
        ".cf", ".tk", ".ga", ".pw", ".cc", ".buzz", ".info", ".biz", ".ru"
    }

    SUSPICIOUS_KEYWORDS = {
        "login", "signin", "verify", "secure", "account", "update",
        "confirm", "password", "credential", "authenticate",
        "suspend", "locked", "urgent", "expire", "billing", "payment"
    }

    def analyze(self, url: str) -> np.ndarray:
        features = {
            "url_length": len(url),
            "num_dots": url.count("."),
            "num_digits": sum(c.isdigit() for c in url),
            "num_hyphens": url.count("-"),
            "has_at_symbol": 1 if "@" in url else 0,
            "has_ip": 0,
            "suspicious_tld": 0,
            "keyword": 0,
        }

        normalized = url
        if not normalized.startswith(("http://", "https://")):
            normalized = "http://" + normalized

        parsed = urlparse(normalized)
        domain = parsed.netloc.lower()

        if re.search(r"\d+\.\d+\.\d+\.\d+", domain):
            features["has_ip"] = 1

        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                features["suspicious_tld"] = 1
                break

        lowered = normalized.lower()
        for word in self.SUSPICIOUS_KEYWORDS:
            if word in lowered:
                features["keyword"] = 1
                break

        return np.array(list(features.values()))


class DomainReputationAnalyzer:
    def __init__(self, phishtank_file: str = "phishtank.csv"):
        self.bad_urls = set()
        self.load_phishtank(phishtank_file)

    def load_phishtank(self, file: str) -> None:
        try:
            df = pd.read_csv(file)
            if "url" in df.columns:
                self.bad_urls = set(df["url"].astype(str).str.lower())
            elif "phish_url" in df.columns:
                self.bad_urls = set(df["phish_url"].astype(str).str.lower())
            print(f"Loaded {len(self.bad_urls)} PhishTank URLs")
        except Exception as e:
            print(f"PhishTank not loaded: {e}")

    def extract_domain(self, url: str):
        try:
            ext = tldextract.extract(url)
            return f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else None
        except Exception:
            return None

    def domain_age(self, domain: str):
        try:
            w = whois.whois(domain)
            creation = w.creation_date

            if isinstance(creation, list):
                creation = creation[0]

            if not creation:
                return None

            return (datetime.now() - creation).days
        except Exception:
            return None

    def score(self, url: str) -> dict:
        result = {
            "listed_phishtank": False,
            "domain_age": None,
            "score": 0
        }

        lowered = url.lower()

        if lowered in self.bad_urls:
            result["listed_phishtank"] = True
            result["score"] -= 0.9
            return result

        domain = self.extract_domain(lowered)
        age = self.domain_age(domain) if domain else None
        result["domain_age"] = age

        if age is not None:
            if age < 30:
                result["score"] -= 0.4
            elif age < 90:
                result["score"] -= 0.2
            else:
                result["score"] += 0.1

        return result


class URLPhishingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=5000
        )

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )

        self.analyzer = URLAnalyzer()
        self.reputation = DomainReputationAnalyzer()
        self.trained = False

    def preprocess(self, url: str) -> str:
        url = url.lower()
        url = re.sub(r"https?://", " ", url)
        url = re.sub(r"[^a-z0-9]", " ", url)
        return url

    def extract_features(self, urls, fit=False):
        text = [self.preprocess(u) for u in urls]

        if fit:
            text_features = self.vectorizer.fit_transform(text)
        else:
            text_features = self.vectorizer.transform(text)

        url_features = [self.analyzer.analyze(u) for u in urls]
        url_features = csr_matrix(np.array(url_features))

        return hstack([text_features, url_features])

    def train(self, urls, labels):
        X = self.extract_features(urls, fit=True)
        self.model.fit(X, labels)
        self.trained = True
        print("URL model trained successfully")
        print("Feature dimension:", X.shape)

    def predict(self, url: str):
        if not self.trained:
            raise ValueError("URL model not trained")

        X = self.extract_features([url], fit=False)
        prediction = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        confidence = max(prob)

        rep = self.reputation.score(url)
        adjusted_conf = confidence + (-rep["score"] * 0.3)
        adjusted_conf = max(0, min(1, adjusted_conf))

        return prediction, adjusted_conf, rep


# ============================================================
# EMAIL PHISHING DETECTOR
# ============================================================

class EmailPhishingDetector:
    """
    Phishing email detector using Naive Bayes with text features only.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.classifier = MultinomialNB()
        self.is_trained = False

    def preprocess_email(self, email_text: str) -> str:
        if email_text is None:
            return ""
        text = str(email_text).lower()
        text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_features(self, emails, fit_vectorizer=False):
        processed_emails = [self.preprocess_email(email) for email in emails]
        if fit_vectorizer:
            return self.vectorizer.fit_transform(processed_emails)
        return self.vectorizer.transform(processed_emails)

    def train(self, emails, labels):
        print(f"Extracting text features from {len(emails)} emails...")
        X = self._extract_features(emails, fit_vectorizer=True)

        print(f"Feature dimensions: {X.shape[1]} (text only)")
        self.classifier.fit(X, labels)
        self.is_trained = True
        print("Email model trained successfully!")

    def predict(self, email_text: str):
        if not self.is_trained:
            raise ValueError("Email model not trained. Call train() first.")

        X = self._extract_features([email_text])
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)

        return prediction, confidence

    def evaluate(self, emails, labels):
        if not self.is_trained:
            raise ValueError("Email model not trained. Call train() first.")

        X = self._extract_features(emails)
        predictions = self.classifier.predict(X)

        print("\n=== Email Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=["Legitimate", "Phishing"]))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, predictions))

        return predictions

    def save_model(self, filepath="phishing_email_model.pkl"):
        if not self.is_trained:
            raise ValueError("Email model not trained. Nothing to save.")

        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier
            }, f)

        print(f"Email model saved to {filepath}")

    def load_model(self, filepath="phishing_email_model.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_trained = True

        print(f"Email model loaded from {filepath}")


def load_email_dataset_from_huggingface():
    print("Loading phishing email dataset from Hugging Face...")
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")

    emails = [str(x) for x in dataset["Email Text"]]
    labels = [
        1 if str(lbl).lower() == "phishing email" else 0
        for lbl in dataset["Email Type"]
    ]

    phishing_count = sum(labels)
    safe_count = len(labels) - phishing_count
    print(f"Loaded {len(emails)} emails: {phishing_count} phishing, {safe_count} safe")

    return emails, labels


# ============================================================
# DEMO / CLI
# ============================================================

def train_demo_url_detector():
    detector = URLPhishingDetector()
    urls = [
        "paypal-secure-login.xyz/verify",
        "bank-account-update.top/login",
        "secure-amazon-confirm.click",
        "microsoft-support-login.ru/reset",
        "google.com",
        "github.com/login",
        "wikipedia.org/wiki/python",
        "amazon.com/product"
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    detector.train(urls, labels)
    return detector


def train_email_detector():
    detector = EmailPhishingDetector()
    emails, labels = load_email_dataset_from_huggingface()

    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("\nTraining Naive Bayes classifier...")
    detector.train(X_train, y_train)
    detector.evaluate(X_test, y_test)
    return detector


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrated phishing detector for URLs and emails"
    )
    parser.add_argument(
        "--mode",
        choices=["url", "email"],
        help="Choose which detector to run"
    )
    parser.add_argument(
        "--load-email-model",
        type=str,
        help="Path to a saved email model (.pkl)"
    )
    parser.add_argument(
        "--save-email-model",
        type=str,
        help="Path to save the trained email model (.pkl)"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Train/evaluate only without interactive input"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Integrated Phishing Detection Toolkit ")
    print("=" * 60)

    mode = args.mode
    if not mode:
        mode = input("Choose detector mode (url/email): ").strip().lower()

    if mode == "url":
        detector = train_demo_url_detector()

        if not args.no_interactive:
            while True:
                url = input("\nEnter URL to check (or quit): ").strip()
                if url.lower() in ["quit", "exit", "q"]:
                    break
                if not url:
                    continue

                pred, conf, rep = detector.predict(url)
                status = "PHISHING" if pred == 1 else "SAFE"
                print(f"\nResult: {status} ({conf:.1%} confidence)")
                print("Reputation:", rep)

    elif mode == "email":
        detector = EmailPhishingDetector()

        if args.load_email_model:
            detector.load_model(args.load_email_model)
        else:
            detector = train_email_detector()
            if args.save_email_model:
                detector.save_model(args.save_email_model)

        if not args.no_interactive:
            print("\nEnter an email to check (or 'quit' to exit):\n")
            while True:
                user_input = input("> ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if not user_input:
                    continue

                prediction, confidence = detector.predict(user_input)
                status = "PHISHING" if prediction == 1 else "SAFE"
                print(f"\nResult: {status} (confidence: {confidence:.1%})\n")
    else:
        raise ValueError("Mode must be 'url' or 'email'.")

    print("Goodbye!")


if __name__ == "__main__":
    main()
