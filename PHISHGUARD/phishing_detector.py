"""
Phishing Email Detector — Dual-Model Architecture

Models:
- Text model: MultinomialNB on TF-IDF (10k) + structural features (12)
- URL model: Logistic Regression on URL features (17), trained on 108k+ URLs

Combined at prediction time via weighted scoring.

Trained on multiple datasets (see training_datasets.pdf).
"""

import re
import pickle
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler


class URLAnalyzer:
    """Analyzes URLs for phishing indicators."""

    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = {
        '.xyz', '.top', '.club', '.work', '.click', '.link', '.gq', '.ml',
        '.cf', '.tk', '.ga', '.pw', '.cc', '.buzz', '.info', '.biz', '.ru'
    }

    # Known URL shorteners
    URL_SHORTENERS = {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'shorte.st'
    }

    # Brands commonly impersonated in phishing
    IMPERSONATED_BRANDS = {
        'paypal', 'apple', 'amazon', 'microsoft', 'google', 'netflix',
        'facebook', 'instagram', 'whatsapp', 'linkedin', 'twitter',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank',
        'dropbox', 'docusign', 'adobe', 'office365', 'outlook', 'icloud'
    }

    # Suspicious keywords in URLs
    SUSPICIOUS_KEYWORDS = {
        'login', 'signin', 'verify', 'secure', 'account', 'update',
        'confirm', 'password', 'credential', 'authenticate', 'suspend',
        'locked', 'urgent', 'expire', 'billing', 'payment', 'wallet'
    }

    # URL pattern to extract URLs from text. Bounded character class and
    # explicit length limit keep this linear-time even on adversarial bodies.
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"\'`]{1,2048}|www\.[^\s<>"\'`]{1,2048}',
        re.IGNORECASE
    )
    _MAX_TEXT_FOR_REGEX = 200_000

    def extract_urls(self, text):
        """Extract all URLs from text (ReDoS-safe, bounded work)."""
        if not text:
            return []
        urls = self.URL_PATTERN.findall(str(text)[:self._MAX_TEXT_FOR_REGEX])
        cleaned = []
        for url in urls:
            url = url.rstrip('.,;:!?)>')
            if url:
                cleaned.append(url)
        return cleaned

    def analyze_url(self, url):
        """
        Analyze a single URL for phishing indicators.

        Returns dict of features.
        """
        features = {
            'has_ip_address': 0,
            'url_length': 0,
            'num_dots': 0,
            'num_hyphens': 0,
            'num_underscores': 0,
            'num_slashes': 0,
            'num_digits': 0,
            'has_at_symbol': 0,
            'is_https': 0,
            'suspicious_tld': 0,
            'is_shortened': 0,
            'has_brand_in_subdomain': 0,
            'has_suspicious_keyword': 0,
            'subdomain_count': 0,
            'path_length': 0,
            'has_port': 0,
            'double_slash_redirect': 0,
        }

        if not url:
            return features

        try:
            url_lower = url.lower()

            # Add scheme if missing
            if not url_lower.startswith(('http://', 'https://')):
                url_lower = 'http://' + url_lower

            parsed = urlparse(url_lower)
            domain = parsed.netloc or ''
            path = parsed.path or ''
        except Exception:
            return features

        try:
            # Basic URL characteristics
            features['url_length'] = min(len(url), 200) / 200  # Normalize
            features['num_dots'] = min(url.count('.'), 10) / 10
            features['num_hyphens'] = min(url.count('-'), 10) / 10
            features['num_underscores'] = min(url.count('_'), 10) / 10
            features['num_slashes'] = min(url.count('/'), 15) / 15
            features['num_digits'] = min(sum(c.isdigit() for c in url), 20) / 20
            features['has_at_symbol'] = 1 if '@' in url else 0
            features['is_https'] = 1 if parsed.scheme == 'https' else 0
            features['path_length'] = min(len(path), 100) / 100

            # Check for IP address in domain
            ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
            features['has_ip_address'] = 1 if ip_pattern.search(domain) else 0

            # Check for suspicious TLD
            for tld in self.SUSPICIOUS_TLDS:
                if domain.endswith(tld):
                    features['suspicious_tld'] = 1
                    break

            # Check for URL shortener
            for shortener in self.URL_SHORTENERS:
                if shortener in domain:
                    features['is_shortened'] = 1
                    break

            # Check for brand impersonation in subdomain
            domain_parts = domain.split('.')
            if len(domain_parts) > 2:
                subdomain = '.'.join(domain_parts[:-2])
                for brand in self.IMPERSONATED_BRANDS:
                    if brand in subdomain and brand not in domain_parts[-2]:
                        features['has_brand_in_subdomain'] = 1
                        break

            # Count subdomains
            features['subdomain_count'] = min(len(domain_parts) - 2, 5) / 5 if len(domain_parts) > 2 else 0

            # Check for suspicious keywords in URL
            for keyword in self.SUSPICIOUS_KEYWORDS:
                if keyword in url_lower:
                    features['has_suspicious_keyword'] = 1
                    break

            # Check for port number (with error handling for malformed URLs)
            try:
                features['has_port'] = 1 if parsed.port and parsed.port not in [80, 443] else 0
            except ValueError:
                features['has_port'] = 1

            # Check for double slash redirect trick
            features['double_slash_redirect'] = 1 if '//' in path else 0

        except Exception:
            # If any parsing fails, return features with defaults
            pass

        return features

    def analyze_email_urls(self, email_text):
        """
        Analyze all URLs in an email and return aggregated features.

        Returns numpy array of features.
        """
        urls = self.extract_urls(email_text)

        # Initialize aggregated features
        agg_features = {
            'url_count': min(len(urls), 10) / 10,
            'has_ip_address': 0,
            'avg_url_length': 0,
            'max_url_length': 0,
            'total_dots': 0,
            'total_hyphens': 0,
            'has_at_symbol': 0,
            'all_https': 1 if urls else 0,
            'any_suspicious_tld': 0,
            'any_shortened': 0,
            'any_brand_impersonation': 0,
            'any_suspicious_keyword': 0,
            'max_subdomain_count': 0,
            'any_port': 0,
            'any_double_slash': 0,
        }

        if not urls:
            return np.array(list(agg_features.values()))

        url_lengths = []
        for url in urls:
            features = self.analyze_url(url)
            url_lengths.append(features['url_length'] * 200)

            # Aggregate using max/any logic for suspicious indicators
            agg_features['has_ip_address'] = max(agg_features['has_ip_address'], features['has_ip_address'])
            agg_features['total_dots'] = min(agg_features['total_dots'] + features['num_dots'], 1)
            agg_features['total_hyphens'] = min(agg_features['total_hyphens'] + features['num_hyphens'], 1)
            agg_features['has_at_symbol'] = max(agg_features['has_at_symbol'], features['has_at_symbol'])
            agg_features['all_https'] = min(agg_features['all_https'], features['is_https'])
            agg_features['any_suspicious_tld'] = max(agg_features['any_suspicious_tld'], features['suspicious_tld'])
            agg_features['any_shortened'] = max(agg_features['any_shortened'], features['is_shortened'])
            agg_features['any_brand_impersonation'] = max(agg_features['any_brand_impersonation'], features['has_brand_in_subdomain'])
            agg_features['any_suspicious_keyword'] = max(agg_features['any_suspicious_keyword'], features['has_suspicious_keyword'])
            agg_features['max_subdomain_count'] = max(agg_features['max_subdomain_count'], features['subdomain_count'])
            agg_features['any_port'] = max(agg_features['any_port'], features['has_port'])
            agg_features['any_double_slash'] = max(agg_features['any_double_slash'], features['double_slash_redirect'])

        agg_features['avg_url_length'] = (sum(url_lengths) / len(url_lengths)) / 200 if url_lengths else 0
        agg_features['max_url_length'] = max(url_lengths) / 200 if url_lengths else 0

        return np.array(list(agg_features.values()))

    def get_feature_names(self):
        """Return list of feature names."""
        return [
            'url_count', 'has_ip_address', 'avg_url_length', 'max_url_length',
            'total_dots', 'total_hyphens', 'has_at_symbol', 'all_https',
            'any_suspicious_tld', 'any_shortened', 'any_brand_impersonation',
            'any_suspicious_keyword', 'max_subdomain_count', 'any_port',
            'any_double_slash'
        ]


class URLClassifier:
    """
    Standalone URL phishing classifier using Logistic Regression.
    Trained separately on URL-specific datasets.
    """

    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.analyzer = URLAnalyzer()
        self.is_trained = False

    def _url_to_features(self, url):
        """Extract feature vector from a single URL."""
        feats = self.analyzer.analyze_url(url)
        return np.array(list(feats.values()))

    def train(self, urls, labels):
        """Train on lists of URLs and binary labels (1=phishing, 0=safe)."""
        print(f"  Extracting features from {len(urls)} URLs...")
        X = np.array([self._url_to_features(url) for url in urls])
        self.classifier.fit(X, labels)
        self.is_trained = True
        print(f"  URL model trained ({X.shape[1]} features)")

    def predict_url(self, url):
        """Returns (prediction, phishing_probability) for a single URL."""
        if not self.is_trained:
            return 0, 0.0
        X = self._url_to_features(url).reshape(1, -1)
        prob = self.classifier.predict_proba(X)[0]
        phish_prob = prob[1] if len(prob) > 1 else 0.0
        return (1 if phish_prob >= 0.5 else 0), phish_prob

    def predict_email_urls(self, email_text):
        """Analyze all URLs in an email. Returns max phishing probability across all URLs."""
        if not self.is_trained:
            return 0, 0.0
        urls = self.analyzer.extract_urls(email_text)
        if not urls:
            return 0, 0.0
        max_prob = 0.0
        for url in urls:
            _, prob = self.predict_url(url)
            max_prob = max(max_prob, prob)
        return (1 if max_prob >= 0.5 else 0), max_prob

    def evaluate(self, urls, labels):
        """Evaluate URL model performance."""
        X = np.array([self._url_to_features(url) for url in urls])
        preds = self.classifier.predict(X)
        print("\n  === URL Model Evaluation ===")
        print(f"  Accuracy: {accuracy_score(labels, preds):.4f}")
        print(classification_report(labels, preds,
                                    target_names=['Safe URL', 'Phishing URL']))

    def save(self, filepath='url_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({'classifier': self.classifier}, f)
        print(f"  URL model saved to {filepath}")

    def load(self, filepath='url_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.is_trained = True
        print(f"  URL model loaded from {filepath}")


class StructuralAnalyzer:
    """Extracts structural/stylistic features from raw email text."""

    URGENCY_WORDS = {
        'urgent', 'immediately', 'asap', 'right away', 'act now', 'hurry',
        'expire', 'expiring', 'expires', 'suspended', 'suspend', 'locked',
        'disabled', 'limited time', 'deadline', 'final notice', 'last chance',
        'warning', 'alert', 'attention', 'important', 'critical',
        'action required', 'verify now', 'confirm now', 'update now',
        'click now', 'respond immediately',
    }

    MONEY_WORDS = {
        'free', 'winner', 'won', 'prize', 'reward', 'gift', 'bonus', 'cash',
        'dollar', 'money', 'credit', 'refund', 'payment', 'invoice',
        'billing', 'transaction', 'transfer', 'bank', 'account', 'wallet',
    }

    THREAT_WORDS = {
        'unauthorized', 'suspicious', 'fraud', 'illegal', 'violation',
        'compromised', 'breach', 'hacked', 'stolen', 'blocked',
        'restricted', 'terminated', 'closed', 'deactivated', 'penalty',
    }

    def analyze(self, email_text):
        """Extract structural features from email text. Returns numpy array (12 features)."""
        if not email_text:
            return np.zeros(12)

        text = str(email_text)
        text_lower = text.lower()
        length = max(len(text), 1)

        # 1. Caps ratio
        alpha_chars = [c for c in text if c.isalpha()]
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)

        # 2. Exclamation mark density
        excl_density = min(text.count('!') / (length / 100), 1.0)

        # 3. Question mark density
        ques_density = min(text.count('?') / (length / 100), 1.0)

        # 4. Urgency word count (normalized)
        urgency_count = sum(1 for w in self.URGENCY_WORDS if w in text_lower)
        urgency_score = min(urgency_count / 5, 1.0)

        # 5. Money/reward word count (normalized)
        money_count = sum(1 for w in self.MONEY_WORDS if w in text_lower)
        money_score = min(money_count / 5, 1.0)

        # 6. Threat word count (normalized)
        threat_count = sum(1 for w in self.THREAT_WORDS if w in text_lower)
        threat_score = min(threat_count / 4, 1.0)

        # 7. Email length (normalized)
        length_score = min(length / 5000, 1.0)

        # 8. Number of lines
        num_lines = max(text.count('\n') + 1, 1)
        lines_score = min(num_lines / 50, 1.0)

        # 9. Special character density ($, %, *, etc.)
        special_chars = sum(1 for c in text if c in '$%*~^{}[]|\\')
        special_density = min(special_chars / (length / 50), 1.0)

        # 10. Digit ratio
        digit_ratio = sum(1 for c in text if c.isdigit()) / length

        # 11. Has HTML tags
        html_pattern = re.compile(r'<[a-zA-Z][^>]*>')
        has_html = 1.0 if html_pattern.search(text) else 0.0

        # 12. Link-to-text ratio
        url_count = len(re.findall(r'https?://[^\s]+', text))
        words = text_lower.split()
        link_text_ratio = min(url_count / max(len(words) / 20, 1), 1.0)

        return np.array([
            caps_ratio, excl_density, ques_density,
            urgency_score, money_score, threat_score,
            length_score, lines_score, special_density,
            digit_ratio, has_html, link_text_ratio,
        ])

    def get_feature_names(self):
        return [
            'caps_ratio', 'excl_density', 'ques_density',
            'urgency_score', 'money_score', 'threat_score',
            'length_score', 'lines_score', 'special_density',
            'digit_ratio', 'has_html', 'link_text_ratio',
        ]


class HeaderAnalyzer:
    """Analyzes email authentication headers (SPF/DKIM/DMARC)."""

    # Regex to extract SPF/DKIM/DMARC results from Authentication-Results header
    AUTH_RESULT_PATTERN = re.compile(
        r'(spf|dkim|dmarc)\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)',
        re.IGNORECASE
    )

    # Regex to extract associated domains
    DOMAIN_PATTERNS = {
        'spf_domain': re.compile(r'smtp\.mailfrom\s*=\s*([^\s;]+)', re.IGNORECASE),
        'dkim_domain': re.compile(r'header\.d\s*=\s*([^\s;]+)', re.IGNORECASE),
        'dmarc_domain': re.compile(r'header\.from\s*=\s*([^\s;]+)', re.IGNORECASE),
    }

    # Scoring weights for each mechanism
    SCORE_MAP = {
        'spf': {'pass': 0.3, 'fail': -0.4, 'softfail': -0.2, 'neutral': 0.0,
                'none': 0.0, 'temperror': -0.1, 'permerror': -0.2},
        'dkim': {'pass': 0.35, 'fail': -0.4, 'softfail': -0.2, 'neutral': 0.0,
                 'none': 0.0, 'temperror': -0.1, 'permerror': -0.2},
        'dmarc': {'pass': 0.35, 'fail': -0.5, 'softfail': -0.25, 'neutral': 0.0,
                  'none': 0.0, 'temperror': -0.1, 'permerror': -0.3},
    }

    def analyze(self, headers):
        """
        Analyze email authentication headers.

        Args:
            headers: List of header dicts from Graph API (internetMessageHeaders)
                     with 'name' and 'value' keys, or None.

        Returns:
            dict with keys: spf, dkim, dmarc, spf_domain, dkim_domain,
                  dmarc_domain, auth_score, has_auth_headers, details
        """
        result = {
            'spf': None,
            'dkim': None,
            'dmarc': None,
            'spf_domain': None,
            'dkim_domain': None,
            'dmarc_domain': None,
            'auth_score': 0.0,
            'has_auth_headers': False,
            'details': [],
        }

        if not headers:
            return result

        # Find Authentication-Results header(s)
        auth_header_value = None
        for header in headers:
            if isinstance(header, dict) and header.get('name', '').lower() == 'authentication-results':
                auth_header_value = header.get('value', '')
                break

        if not auth_header_value:
            return result

        result['has_auth_headers'] = True

        # Parse SPF/DKIM/DMARC results
        matches = self.AUTH_RESULT_PATTERN.findall(auth_header_value)
        score = 0.0

        for mechanism, status in matches:
            mechanism = mechanism.lower()
            status = status.lower()
            result[mechanism] = status

            mechanism_score = self.SCORE_MAP.get(mechanism, {}).get(status, 0.0)
            score += mechanism_score
            result['details'].append(f"{mechanism.upper()}={status} (score: {mechanism_score:+.2f})")

        # Extract associated domains
        for key, pattern in self.DOMAIN_PATTERNS.items():
            match = pattern.search(auth_header_value)
            if match:
                result[key] = match.group(1)

        # Clamp score to [-1, +1]
        result['auth_score'] = max(-1.0, min(1.0, score))

        return result


class PhishingDetector:
    """
    Dual-model phishing detector:
    - Text model: MultinomialNB on TF-IDF + structural features
    - URL model: Logistic Regression on URL features (separate classifier)
    Combined at prediction time via weighted scoring.
    """

    # Weight for combining text and URL model scores
    TEXT_WEIGHT = 0.55
    URL_WEIGHT = 0.45

    def __init__(self, use_url_features=True):
        self.use_url_features = use_url_features
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.classifier = MultinomialNB(alpha=0.1)
        self.url_classifier = URLClassifier() if use_url_features else None
        self.url_analyzer = URLAnalyzer() if use_url_features else None
        self.structural_analyzer = StructuralAnalyzer()
        self.header_analyzer = HeaderAnalyzer()
        self.is_trained = False

    def preprocess_email(self, email_text):
        """Clean and preprocess email text."""
        if email_text is None:
            return ""
        text = str(email_text).lower()
        # Keep URL tokens for context but normalize them
        text = re.sub(r'https?://[^\s]+', ' URLTOKEN ', text)
        text = re.sub(r'www\.[^\s]+', ' URLTOKEN ', text)
        text = re.sub(r'\S+@\S+', ' EMAILTOKEN ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_features(self, emails, fit_vectorizer=False):
        """Extract text + structural features (URL analysis handled by separate model)."""
        # Text features (TF-IDF)
        processed_emails = [self.preprocess_email(email) for email in emails]
        if fit_vectorizer:
            text_features = self.vectorizer.fit_transform(processed_emails)
        else:
            text_features = self.vectorizer.transform(processed_emails)

        # Structural features (12)
        structural_list = []
        for email in emails:
            struct_feats = self.structural_analyzer.analyze(email)
            structural_list.append(struct_feats)

        structural_features = csr_matrix(np.array(structural_list))
        combined = hstack([text_features, structural_features])
        return combined

    def train(self, emails, labels):
        """
        Train the text model (Naive Bayes) on email text + structural features.
        The URL model is trained separately via train_url_model().

        Args:
            emails: List of email texts
            labels: List of labels (1 for phishing, 0 for legitimate)
        """
        print(f"\n--- Text Model (Naive Bayes) ---")
        print(f"Extracting features from {len(emails)} emails...")
        X = self._extract_features(emails, fit_vectorizer=True)

        struct_count = 12
        print(f"Feature dimensions: {X.shape[1]} (text: {self.vectorizer.max_features}, structural: {struct_count})")

        print("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(self.classifier, X, labels, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.classifier.fit(X, labels)
        self.is_trained = True
        print(f"Text model trained successfully!")

    def train_url_model(self, urls, labels):
        """Train the URL model (Logistic Regression) on URL datasets."""
        if not self.url_classifier:
            print("URL features disabled, skipping URL model training.")
            return
        print(f"\n--- URL Model (Logistic Regression) ---")
        self.url_classifier.train(urls, labels)

    def predict(self, email_text, headers=None):
        """
        Predict if an email is phishing using dual-model scoring.

        Text model (NB) and URL model (LR) produce independent phishing
        probabilities, which are combined via weighted average.

        Args:
            email_text: The email text to analyze
            headers: Optional list of header dicts from Graph API

        Returns:
            tuple: (prediction, confidence, url_analysis, header_result)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # --- Text model score ---
        X = self._extract_features([email_text], fit_vectorizer=False)
        probabilities = self.classifier.predict_proba(X)[0]
        text_phish_prob = probabilities[1] if len(probabilities) > 1 else 0.0

        # --- URL model score ---
        url_phish_prob = 0.0
        has_urls = False
        if self.url_classifier and self.url_classifier.is_trained:
            urls = self.url_analyzer.extract_urls(email_text)
            if urls:
                has_urls = True
                _, url_phish_prob = self.url_classifier.predict_email_urls(email_text)

        # --- Combine scores ---
        if has_urls:
            combined_prob = (self.TEXT_WEIGHT * text_phish_prob +
                             self.URL_WEIGHT * url_phish_prob)
        else:
            combined_prob = text_phish_prob

        # --- Auth header adjustment ---
        header_result = self.header_analyzer.analyze(headers)
        if header_result['has_auth_headers']:
            combined_prob -= header_result['auth_score'] * 0.15
            combined_prob = max(0.0, min(1.0, combined_prob))

        prediction = 1 if combined_prob >= 0.5 else 0
        confidence = combined_prob if prediction == 1 else (1.0 - combined_prob)

        # --- URL analysis details for display ---
        url_analysis = None
        if self.use_url_features:
            urls = self.url_analyzer.extract_urls(email_text)
            if urls:
                url_analysis = {
                    'urls_found': urls,
                    'count': len(urls),
                    'features': {},
                    'url_model_prob': url_phish_prob,
                }
                for url in urls[:5]:
                    url_analysis['features'][url] = self.url_analyzer.analyze_url(url)

        return prediction, confidence, url_analysis, header_result

    def evaluate(self, emails, labels):
        """Evaluate model performance on test data."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = self._extract_features(emails, fit_vectorizer=False)
        predictions = self.classifier.predict(X)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, predictions,
                                    target_names=['Legitimate', 'Phishing']))
        print("Confusion Matrix:")
        print(confusion_matrix(labels, predictions))

        return predictions

    def analyze_urls_only(self, email_text):
        """
        Analyze URLs in an email without making a prediction.

        Returns detailed URL analysis.
        """
        if not self.url_analyzer:
            return None

        urls = self.url_analyzer.extract_urls(email_text)
        if not urls:
            return {'urls_found': [], 'risk_indicators': []}

        analysis = {
            'urls_found': urls,
            'url_details': [],
            'risk_indicators': []
        }

        for url in urls:
            features = self.url_analyzer.analyze_url(url)
            detail = {'url': url, 'risks': []}

            # Check for specific risks
            if features['has_ip_address']:
                detail['risks'].append('IP address instead of domain name')
            if features['suspicious_tld']:
                detail['risks'].append('Suspicious top-level domain')
            if features['is_shortened']:
                detail['risks'].append('URL shortener (hides destination)')
            if features['has_brand_in_subdomain']:
                detail['risks'].append('Brand name in subdomain (possible impersonation)')
            if features['has_suspicious_keyword']:
                detail['risks'].append('Contains suspicious keywords (login, verify, etc.)')
            if features['has_at_symbol']:
                detail['risks'].append('Contains @ symbol (can hide real destination)')
            if features['has_port']:
                detail['risks'].append('Non-standard port number')
            if features['double_slash_redirect']:
                detail['risks'].append('Double-slash redirect pattern')
            if not features['is_https']:
                detail['risks'].append('Not using HTTPS')
            if features['url_length'] > 0.5:  # > 100 chars
                detail['risks'].append('Unusually long URL')

            analysis['url_details'].append(detail)
            analysis['risk_indicators'].extend(detail['risks'])

        analysis['risk_indicators'] = list(set(analysis['risk_indicators']))
        return analysis

    def save_model(self, filepath='phishing_model.pkl'):
        """Save both text and URL models to disk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        save_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'use_url_features': self.use_url_features,
            'version': 3,
        }
        if self.url_classifier and self.url_classifier.is_trained:
            save_data['url_classifier'] = self.url_classifier.classifier

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='phishing_model.pkl'):
        """Load both text and URL models from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.use_url_features = data.get('use_url_features', True)
            if self.use_url_features:
                if not self.url_analyzer:
                    self.url_analyzer = URLAnalyzer()
                if not self.url_classifier:
                    self.url_classifier = URLClassifier()
                if 'url_classifier' in data:
                    self.url_classifier.classifier = data['url_classifier']
                    self.url_classifier.is_trained = True
            self.structural_analyzer = StructuralAnalyzer()
            self.is_trained = True
        print(f"Model loaded from {filepath} (v{data.get('version', 1)})")


def load_dataset_from_kaggle_csv(filepath):
    """
    Load dataset from Kaggle CSV file.

    Expected format (from https://www.kaggle.com/datasets/subhajournal/phishingemails):
    - Column 'Email Text': the email content
    - Column 'Email Type': 'Safe Email' or 'Phishing Email'

    Args:
        filepath: Path to the CSV file

    Returns:
        tuple: (emails, labels) where labels are 1 for phishing, 0 for safe
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # Handle different possible column names
    text_col = None
    label_col = None

    for col in df.columns:
        if 'text' in col.lower() or 'body' in col.lower() or 'content' in col.lower():
            text_col = col
        if 'type' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
            label_col = col

    if text_col is None or label_col is None:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not identify text and label columns")

    print(f"Using columns: text='{text_col}', label='{label_col}'")

    # Drop rows with missing values
    df = df.dropna(subset=[text_col, label_col])

    emails = df[text_col].astype(str).tolist()

    # Convert labels to binary (1 for phishing, 0 for safe)
    labels = df[label_col].apply(
        lambda x: 1 if 'phish' in str(x).lower() or 'spam' in str(x).lower() else 0
    ).tolist()

    phishing_count = sum(labels)
    safe_count = len(labels) - phishing_count
    print(f"Loaded {len(emails)} emails: {phishing_count} phishing, {safe_count} safe")

    return emails, labels


def load_dataset_from_huggingface():
    """
    Load dataset directly from HuggingFace (no login required).

    Uses: https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
    (Mirror of Kaggle dataset)

    Returns:
        tuple: (emails, labels) where labels are 1 for phishing, 0 for safe
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")

    # Filter out None values
    emails = []
    labels = []
    for item in dataset:
        if item["Email Text"] is not None and item["Email Type"] is not None:
            emails.append(str(item["Email Text"]))
            labels.append(1 if item["Email Type"] == "Phishing Email" else 0)

    phishing_count = sum(labels)
    safe_count = len(labels) - phishing_count
    print(f"Loaded {len(emails)} emails: {phishing_count} phishing, {safe_count} safe")

    return emails, labels


def load_dataset_from_cybersectony():
    """
    Load email-only entries from cybersectony/PhishingEmailDetectionv2.0.

    Labels: 0 = legitimate email, 1 = phishing email, 2/3 = URLs (skipped).

    Returns:
        tuple: (emails, labels) where labels are 1 for phishing, 0 for safe
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the datasets library: pip install datasets"
        )

    print("Loading dataset from cybersectony/PhishingEmailDetectionv2.0...")
    dataset = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")

    emails = []
    labels = []
    for item in dataset:
        # Only keep email entries (labels 0 and 1), skip URL entries (2 and 3)
        if item["label"] in (0, 1) and item["content"] is not None:
            text = str(item["content"]).strip()
            if len(text) > 20:
                emails.append(text)
                labels.append(item["label"])  # 0=safe, 1=phishing already

    phishing_count = sum(labels)
    safe_count = len(labels) - phishing_count
    print(f"Loaded {len(emails)} emails: {phishing_count} phishing, {safe_count} safe")

    return emails, labels


def load_url_datasets():
    """
    Load URL datasets for training the URL model.

    Sources:
    - cybersectony/PhishingEmailDetectionv2.0 (labels 2=safe URL, 3=phishing URL)
    - shawhin/phishing-site-classification (labels 0=safe, 1=phishing)

    Returns:
        tuple: (urls, labels) where labels are 1 for phishing, 0 for safe
    """
    from datasets import load_dataset

    urls = []
    labels = []

    # Source 1: cybersectony URL entries
    print("Loading URLs from cybersectony/PhishingEmailDetectionv2.0...")
    ds = load_dataset("cybersectony/PhishingEmailDetectionv2.0", split="train")
    for item in ds:
        if item["label"] in (2, 3) and item["content"]:
            url = str(item["content"]).strip()
            if len(url) > 5:
                urls.append(url)
                labels.append(1 if item["label"] == 3 else 0)

    phish_1 = sum(labels)
    safe_1 = len(labels) - phish_1
    print(f"  cybersectony: {len(labels)} URLs ({phish_1} phishing, {safe_1} safe)")

    # Source 2: shawhin/phishing-site-classification
    print("Loading URLs from shawhin/phishing-site-classification...")
    ds2 = load_dataset("shawhin/phishing-site-classification", split="train")
    count_before = len(urls)
    for item in ds2:
        if item["text"]:
            url = str(item["text"]).strip()
            if len(url) > 5:
                urls.append(url)
                labels.append(item["labels"])

    added = len(urls) - count_before
    phish_2 = sum(labels[count_before:])
    safe_2 = added - phish_2
    print(f"  shawhin: {added} URLs ({phish_2} phishing, {safe_2} safe)")

    phish_total = sum(labels)
    safe_total = len(labels) - phish_total
    print(f"  Combined: {len(urls)} URLs ({phish_total} phishing, {safe_total} safe)")

    return urls, labels


def main():
    """Train and demo the phishing detector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phishing Email Detector — Dual Model"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to Kaggle CSV file (e.g., Phishing_Email.csv)"
    )
    parser.add_argument(
        "--huggingface",
        action="store_true",
        help="Load dataset from HuggingFace (default if no --csv provided)"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        help="Load a pre-trained model instead of training"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Phishing Email Detector (Dual Model Architecture)")
    print("=" * 55 + "\n")

    detector = PhishingDetector()

    if args.load_model:
        # Load existing model
        detector.load_model(args.load_model)
    else:
        # --- Load email datasets ---
        if args.csv:
            emails, labels = load_dataset_from_kaggle_csv(args.csv)
        else:
            emails_1, labels_1 = load_dataset_from_huggingface()
            emails_2, labels_2 = load_dataset_from_cybersectony()
            emails = emails_1 + emails_2
            labels = labels_1 + labels_2
            phishing_total = sum(labels)
            safe_total = len(labels) - phishing_total
            print(f"\nCombined emails: {len(emails)} ({phishing_total} phishing, {safe_total} safe)")

        # Split email data
        print("\nSplitting email data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Train text model
        detector.train(X_train, y_train)
        detector.evaluate(X_test, y_test)

        # --- Load URL datasets and train URL model ---
        url_data, url_labels = load_url_datasets()
        url_train, url_test, uy_train, uy_test = train_test_split(
            url_data, url_labels, test_size=0.2, random_state=42, stratify=url_labels
        )
        detector.train_url_model(url_train, uy_train)
        if detector.url_classifier and detector.url_classifier.is_trained:
            detector.url_classifier.evaluate(url_test, uy_test)

        # Save combined model
        detector.save_model()

    # Test with sample emails (including ones with URLs)
    print("\n" + "=" * 50)
    print("  Testing with Sample Emails")
    print("=" * 50 + "\n")

    test_emails = [
        "URGENT! Your account will be closed. Click here to verify your identity now!",
        "Hi Mike, Can we reschedule our meeting to next Tuesday? Let me know what works.",
        "You've won a free iPhone! Claim your prize at http://192.168.1.1/claim-prize",
        "Dear customer, verify your PayPal account: https://paypal-secure-login.xyz/verify",
        "Check out our new blog post: https://www.techcompany.com/blog/2024/update",
        "Your package is waiting! Track it here: http://bit.ly/3xFake2",
        "Meeting link: https://zoom.us/j/123456789",
        "Reset your password immediately: http://microsoft-support.click/reset?user=you@email.com",
    ]

    for email in test_emails:
        prediction, confidence, url_analysis, header_result = detector.predict(email)
        status = "\033[91mPHISHING\033[0m" if prediction == 1 else "\033[92mSAFE\033[0m"
        print(f"[{status}] ({confidence:.1%}) {email[:60]}...")

        # Show URL analysis if URLs were found
        if url_analysis and url_analysis['urls_found']:
            for url in url_analysis['urls_found'][:2]:
                feats = url_analysis['features'].get(url, {})
                risks = []
                if feats.get('has_ip_address'):
                    risks.append('IP-addr')
                if feats.get('suspicious_tld'):
                    risks.append('bad-TLD')
                if feats.get('is_shortened'):
                    risks.append('shortened')
                if feats.get('has_brand_in_subdomain'):
                    risks.append('brand-spoof')
                if feats.get('has_suspicious_keyword'):
                    risks.append('sus-keyword')
                risk_str = ', '.join(risks) if risks else 'none detected'
                print(f"    URL: {url[:50]}{'...' if len(url) > 50 else ''}")
                print(f"    Risks: {risk_str}")

        # Show header analysis if available
        if header_result and header_result.get('has_auth_headers'):
            spf = header_result.get('spf', 'N/A')
            dkim = header_result.get('dkim', 'N/A')
            dmarc = header_result.get('dmarc', 'N/A')
            print(f"    Auth: SPF={spf}, DKIM={dkim}, DMARC={dmarc} (score: {header_result['auth_score']:+.2f})")
        print()

    if not args.no_interactive:
        print("=" * 50)
        print("  Interactive Mode")
        print("=" * 50)
        print("Enter an email to check (or 'quit' to exit):\n")

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            prediction, confidence, url_analysis, header_result = detector.predict(user_input)
            status = "\033[91mPHISHING\033[0m" if prediction == 1 else "\033[92mSAFE\033[0m"
            print(f"\nResult: {status} (confidence: {confidence:.1%})")

            # Show detailed URL analysis
            if url_analysis and url_analysis['urls_found']:
                print(f"\nURLs detected: {len(url_analysis['urls_found'])}")
                for url in url_analysis['urls_found']:
                    feats = url_analysis['features'].get(url, {})
                    print(f"  - {url}")
                    risks = []
                    if feats.get('has_ip_address'):
                        risks.append("Uses IP address instead of domain")
                    if feats.get('suspicious_tld'):
                        risks.append("Suspicious top-level domain")
                    if feats.get('is_shortened'):
                        risks.append("URL shortener hides destination")
                    if feats.get('has_brand_in_subdomain'):
                        risks.append("Brand name in subdomain (impersonation)")
                    if feats.get('has_suspicious_keyword'):
                        risks.append("Contains suspicious keywords")
                    if feats.get('has_at_symbol'):
                        risks.append("@ symbol can hide real destination")
                    if not feats.get('is_https'):
                        risks.append("Not using HTTPS")
                    if risks:
                        for risk in risks:
                            print(f"    \033[93m⚠ {risk}\033[0m")
            print()

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
