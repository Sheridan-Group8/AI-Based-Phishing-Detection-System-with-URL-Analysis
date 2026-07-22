"""
Phishing Email Detector â€” Dual-Model Architecture

Runtime models:
- Text model: ONNX big model.
- URL model: ONNX URL model when present, plus deterministic URL heuristics.

The legacy scikit-learn pickle model was removed from the runtime. Training
for the current models lives in bigmodel/.
"""

import hashlib
import hmac
import json
import os
import re
import sys
from urllib.parse import urlparse

import numpy as np


APP_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


def load_detection_rules():
    """Load config/detection_rules.json with built-in fallbacks elsewhere.

    Rules are configuration, not code. If the JSON is absent or malformed,
    callers continue using their local defaults so detection stays enabled.
    """
    path = os.path.join(APP_ROOT, "config", "detection_rules.json")
    try:
        with open(path, encoding="utf-8") as f:
            rules = json.load(f)
        return rules if isinstance(rules, dict) else {}
    except Exception as exc:
        print(f"detection_rules.json not loaded ({exc}); using built-in lists.")
        return {}


_DETECTION_RULES = load_detection_rules()


class URLAnalyzer:
    """Analyzes URLs for phishing indicators."""

    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = set(_DETECTION_RULES.get("url_suspicious_tlds") or {
        '.xyz', '.top', '.club', '.work', '.click', '.link', '.gq', '.ml',
        '.cf', '.tk', '.ga', '.pw', '.cc', '.buzz', '.info', '.biz', '.ru'
    })

    # Known URL shorteners
    URL_SHORTENERS = set(_DETECTION_RULES.get("url_shorteners") or {
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'shorte.st'
    })

    # Brands commonly impersonated in phishing
    IMPERSONATED_BRANDS = set(_DETECTION_RULES.get("url_impersonated_brands") or {
        'paypal', 'apple', 'amazon', 'microsoft', 'google', 'netflix',
        'facebook', 'instagram', 'whatsapp', 'linkedin', 'twitter',
        'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank',
        'dropbox', 'docusign', 'adobe', 'office365', 'outlook', 'icloud'
    })

    # Suspicious keywords in URLs
    SUSPICIOUS_KEYWORDS = set(_DETECTION_RULES.get("url_suspicious_keywords") or {
        'login', 'signin', 'verify', 'secure', 'account', 'update',
        'confirm', 'password', 'credential', 'authenticate', 'suspend',
        'locked', 'urgent', 'expire', 'billing', 'payment', 'wallet'
    })

    # URL pattern to extract URLs from text. Bounded character class and
    # explicit length limit keep this linear-time even on adversarial bodies.
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"\'`]{1,2048}|www\.[^\s<>"\'`]{1,2048}',
        re.IGNORECASE
    )

    # Cap how much text we hand to the regex per call â€” bodies above this are
    # truncated before matching. 200 KB is well above any real email.
    _MAX_TEXT_FOR_REGEX = 200_000

    def extract_urls(self, text):
        """Extract all URLs from text (ReDoS-safe, bounded work)."""
        if not text:
            return []
        s = str(text)[:self._MAX_TEXT_FOR_REGEX]
        urls = self.URL_PATTERN.findall(s)
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
            domain = parsed.hostname or parsed.netloc or ''
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


class OnnxClassifier:
    """ONNX-runtime classifier: the local-ship backend for the big models.

    Runs an INT8 ONNX model on `onnxruntime` and tokenizes with the lightweight
    `tokenizers` library. No torch or transformers runtime is required.
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.session = None
        self.tokenizer = None
        self.is_loaded = False
        self.threshold = 0.5
        self.max_length = 512
        self.phishing_index = 1
        self._np = None

    def _find_onnx(self):
        import os
        if not os.path.isdir(self.model_dir):
            return None
        # Prefer a quantized file if present.
        files = [f for f in os.listdir(self.model_dir) if f.endswith(".onnx")]
        if not files:
            return None
        files.sort(key=lambda f: (0 if "quant" in f or "int8" in f else 1, f))
        return os.path.join(self.model_dir, files[0])

    def load(self):
        try:
            import os, json
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as e:
            print(f"OnnxClassifier: runtime unavailable ({e}); falling back.")
            return False
        try:
            onnx_path = self._find_onnx()
            tok_path = os.path.join(self.model_dir, "tokenizer.json")
            if not onnx_path or not os.path.exists(tok_path):
                print(f"OnnxClassifier: model files not found in {self.model_dir}; falling back.")
                return False
            meta_path = os.path.join(self.model_dir, "phishguard_metadata.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.threshold = float(meta.get("threshold", 0.5))
                self.max_length = int(meta.get("max_length", 512))
                for k, v in (meta.get("labels") or {}).items():
                    if str(v).upper().startswith("PHISH"):
                        self.phishing_index = int(k)
            expected_hashes = meta.get("model_sha256") or {}
            if expected_hashes:
                base = os.path.basename(onnx_path)
                expected = expected_hashes.get(base)
                if not expected:
                    print(f"OnnxClassifier: {base} has no expected hash in "
                          "phishguard_metadata.json; refusing to load it.")
                    return False
                digest = hashlib.sha256()
                with open(onnx_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        digest.update(chunk)
                if not hmac.compare_digest(digest.hexdigest().lower(), str(expected).lower()):
                    print(f"OnnxClassifier: SHA-256 mismatch for {base} "
                          "(possible tamper/corruption); refusing to load.")
                    return False
            else:
                print("OnnxClassifier: no model_sha256 in metadata - "
                      "loading unverified (re-export to add hashes).")
            self._np = np
            self.tokenizer = Tokenizer.from_file(tok_path)
            self.tokenizer.enable_truncation(max_length=self.max_length)
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._input_names = {i.name for i in self.session.get_inputs()}
            self.is_loaded = True
            print(f"OnnxClassifier: loaded {os.path.basename(onnx_path)} "
                  f"(threshold={self.threshold}).")
            return True
        except Exception as e:
            print(f"OnnxClassifier: failed to load ({e}); falling back.")
            self.is_loaded = False
            return False

    def predict_proba(self, text):
        """Return raw P(phishing) in [0, 1], or None if unavailable."""
        if not self.is_loaded:
            return None
        try:
            np = self._np
            enc = self.tokenizer.encode(text or "")
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            feeds = {"input_ids": ids}
            if "attention_mask" in self._input_names:
                feeds["attention_mask"] = mask
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            logits = self.session.run(None, feeds)[0][0]
            m = float(np.max(logits))
            ex = np.exp(logits - m)
            probs = ex / ex.sum()
            return float(probs[self.phishing_index])
        except Exception as e:
            print(f"OnnxClassifier: inference error ({e}).")
            return None

    def remap_to_centered(self, prob):
        """Remap P(phishing) so the configured threshold lands at 0.5."""
        t = self.threshold
        if t <= 0.0 or t >= 1.0:
            return prob
        if prob >= t:
            return 0.5 + 0.5 * (prob - t) / (1.0 - t)
        return 0.5 * (prob / t)


class PhishingDetector:
    """
    Dual-model phishing detector:
    - Text model: ONNX classifier attached as `self.text_model`.
    - URL model: ONNX URL model attached as `self.url_model`, plus URL
      heuristics.
    Combined at prediction time via weighted scoring.
    """

    # Weight for combining text and URL model scores
    TEXT_WEIGHT = 0.55
    URL_WEIGHT = 0.45
    # Recall-first escalation. When a body URL trips an unambiguous red flag
    # (see _has_hard_url_redflag), the email's score is floored to this value
    # so a "safe" text read can't wave through a clearly-malicious link. We
    # deliberately do NOT escalate on the URL model's probability â€” it is
    # miscalibrated and flags ordinary long URLs (mail.google.com, docs.*) at
    # >0.99, which would wreck precision.
    URL_REDFLAG_FLOOR = 0.90
    # Structural URL features that are near-zero on legitimate URLs but strong
    # phishing tells: raw-IP host, user@host credential trick, and a real
    # brand name living in the subdomain (paypal.account-verify.ru).
    _HARD_URL_REDFLAGS = ('has_ip_address', 'has_at_symbol',
                          'has_brand_in_subdomain')

    def __init__(self, use_url_features=True):
        self.use_url_features = use_url_features
        self.url_analyzer = URLAnalyzer() if use_url_features else None
        self.structural_analyzer = StructuralAnalyzer()
        self.header_analyzer = HeaderAnalyzer()
        # Runtime text classifier. Attached by app.py at startup.
        self.text_model = None
        # Optional trained URL classifier (OnnxClassifier on the URL model).
        # When set + loaded, _score_link uses its per-URL probability.
        self.url_model = None
        self.is_trained = False

    def _urls_have_hard_redflag(self, urls):
        """True if any URL in the list trips a deterministic phishing tell
        (raw-IP host, user@host trick, brand-in-subdomain). These are
        essentially absent from legitimate URLs, so they're safe to escalate
        on â€” unlike the URL model's miscalibrated probability."""
        if not self.url_analyzer or not urls:
            return False
        try:
            for url in urls:
                feats = self.url_analyzer.analyze_url(url)
                if any(feats.get(f) for f in self._HARD_URL_REDFLAGS):
                    return True
        except Exception:
            pass
        return False

    def predict(self, email_text, headers=None, extra_urls=None):
        """
        Predict if an email is phishing using dual-model scoring.

        The runtime text model produces the primary phishing probability.
        URL and auth-header signals can raise the score.

        Args:
            email_text: The email text to analyze
            headers: Optional list of header dicts from Graph API
            extra_urls: Optional list of additional URLs to score that aren't
                present in the plain text â€” e.g. HTML hyperlink destinations
                (href/src) that text extraction discards. These are unioned
                with URLs found in email_text for the URL model + display.

        Returns:
            tuple: (prediction, confidence, url_analysis, header_result)
        """
        if not (self.text_model is not None and self.text_model.is_loaded):
            raise ValueError("No runtime text model loaded.")

        # --- Text model score ---
        # Use the attached ONNX text model. Its raw probability
        # is remapped onto the 0.5-centered scale the rest of this method
        # expects.
        text_phish_prob = None
        raw = self.text_model.predict_proba(email_text)
        if raw is not None:
            text_phish_prob = self.text_model.remap_to_centered(raw)
        if text_phish_prob is None:
            raise ValueError("Runtime text model could not score this message.")

        # --- Gather URLs once: plain-text URLs + any passed-in (HTML hrefs) ---
        urls = []
        if self.url_analyzer:
            urls = list(self.url_analyzer.extract_urls(email_text))
        if extra_urls:
            for u in extra_urls:
                if u and u not in urls:
                    urls.append(u)

        # --- URL display probability ---
        # The authoritative per-link verdicts come from app._score_link (the
        # trained ONNX URL model + threat-feed/structural flags), surfaced via
        # the assessment's Links dimension. Re-running URL scoring here would
        # duplicate that work, so url_phish_prob stays a stable display
        # placeholder and does not feed the verdict.
        url_phish_prob = 0.0
        has_urls = bool(urls)

        # --- Combine scores (text-authoritative, recall-first) ---
        # The text model is the primary verdict. URL signals and auth headers
        # may only RAISE the risk score, never lower it. Rationale:
        #  * The URL model is miscalibrated, so a low URL score must not be
        #    allowed to cancel a phishing text verdict.
        #  * Passing SPF/DKIM must NOT wave through phishing content â€”
        #    phishing routinely comes from auth-passing compromised or
        #    lookalike domains. So auth only adds risk on FAILURE.
        combined_prob = text_phish_prob

        header_result = self.header_analyzer.analyze(headers)
        if header_result['has_auth_headers'] and header_result['auth_score'] < 0:
            combined_prob -= header_result['auth_score'] * 0.15  # neg score -> raises

        # Deterministic hard URL red flag (IP host / user@host / brand-in-
        # subdomain) floors the score to phishing on its own.
        if has_urls and self._urls_have_hard_redflag(urls):
            combined_prob = max(combined_prob, self.URL_REDFLAG_FLOOR)

        combined_prob = max(0.0, min(1.0, combined_prob))

        prediction = 1 if combined_prob >= 0.5 else 0
        confidence = combined_prob if prediction == 1 else (1.0 - combined_prob)

        # --- URL analysis details for display ---
        url_analysis = None
        if self.use_url_features and urls:
            url_analysis = {
                'urls_found': urls,
                'count': len(urls),
                'features': {},
                'url_model_prob': url_phish_prob,
            }
            for url in urls[:5]:
                url_analysis['features'][url] = self.url_analyzer.analyze_url(url)

        # text_phish_prob is the centered (0.5-boundary) text-model score,
        # returned so the caller can build a unified, decomposable risk score
        # without running the model a second time.
        return prediction, confidence, url_analysis, header_result, text_phish_prob


