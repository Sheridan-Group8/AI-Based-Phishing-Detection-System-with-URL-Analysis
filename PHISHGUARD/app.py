"""
PhishGuard Web Dashboard — Flask Backend
Serves the SPA and provides REST API for phishing email analysis.
"""

import base64
import hashlib
import html
import json
import os
import re
import secrets
import socket
import threading
import time as _time
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urlparse

from flask import Flask, jsonify, redirect, request, send_from_directory, session

from phishing_detector import PhishingDetector

try:
    import requests as http_requests
except ImportError:
    http_requests = None


def _load_dotenv():
    """Load optional local API keys from PHISHGUARD/.env without a dependency."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and not os.environ.get(key):
                os.environ[key] = value
    except Exception as exc:
        print(f"WARNING: could not load .env: {exc}")


_load_dotenv()

HTTP_TIMEOUT = (5, 15)
_MAX_BODY_FOR_REGEX = 200_000
_MAX_URLS_PER_EMAIL = 100

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = secrets.token_hex(32)

# Supabase database integration. The browser sends a Supabase access token in
# Authorization; Flask uses it with the anon key so database RLS scopes writes.
def _load_supabase_config():
    path = Path(__file__).parent / "static" / "js" / "supabase-config.js"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", ""

    url_match = re.search(r"SUPABASE_URL:\s*['\"]([^'\"]+)['\"]", text)
    key_match = re.search(r"SUPABASE_ANON_KEY:\s*['\"]([^'\"]+)['\"]", text)
    url = url_match.group(1).strip() if url_match else ""
    key = key_match.group(1).strip() if key_match else ""
    if "YOUR_PROJECT_REF" in url or "YOUR_ANON" in key:
        return "", ""
    return url, key


SUPABASE_URL, SUPABASE_ANON_KEY = _load_supabase_config()


def _request_jwt():
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    return token or None


def _jwt_user_id(jwt):
    if not jwt or "." not in jwt:
        return None
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        return data.get("sub")
    except Exception:
        return None


def _supabase_request(method, path, jwt, json_body=None, params=None,
                      prefer="return=minimal"):
    if not (SUPABASE_URL and SUPABASE_ANON_KEY and jwt and http_requests):
        return None

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer

    try:
        return http_requests.request(
            method, url, headers=headers, json=json_body, params=params,
            timeout=(3.05, 10),
        )
    except Exception as exc:
        print(f"WARNING: Supabase {method} {path} failed: {exc}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────────────────────────────────────
detector = PhishingDetector()
MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"
try:
    detector.load_model(str(MODEL_PATH))
    print("PhishGuard model loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load model: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  Server-side scan results cache
# ─────────────────────────────────────────────────────────────────────────────
scan_results = {}

# ─────────────────────────────────────────────────────────────────────────────
#  Session Log
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "phishguard_log.txt"


def _log_session_event(event, details=None):
    """Append a formatted entry to the session log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        f"  {event}",
        f"  Time: {timestamp}",
    ]
    if details:
        for key, value in details.items():
            lines.append(f"  {key}: {value}")
    lines.append("=" * 60)
    lines.append("")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Domain Reputation System
# ─────────────────────────────────────────────────────────────────────────────
_TRUSTED_DOMAINS = frozenset({
    # Major email providers
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "msn.com", "yahoo.com", "ymail.com", "aol.com", "icloud.com", "me.com",
    "mac.com", "protonmail.com", "proton.me", "zoho.com", "mail.com",
    "gmx.com", "gmx.net", "fastmail.com", "tutanota.com", "hey.com",
    # Tech companies
    "google.com", "microsoft.com", "apple.com", "amazon.com", "meta.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "linkedin.com",
    "github.com", "gitlab.com", "atlassian.com", "slack.com", "zoom.us",
    "dropbox.com", "salesforce.com", "adobe.com", "oracle.com", "ibm.com",
    "intel.com", "cisco.com", "vmware.com", "shopify.com", "stripe.com",
    "twilio.com", "cloudflare.com", "digitalocean.com", "heroku.com",
    "notion.so", "figma.com", "canva.com", "spotify.com", "netflix.com",
    # Financial / payments
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "citibank.com", "americanexpress.com", "discover.com", "capitalone.com",
    "fidelity.com", "schwab.com", "venmo.com", "squareup.com",
    # Services / social
    "uber.com", "lyft.com", "airbnb.com", "doordash.com", "grubhub.com",
    "reddit.com", "pinterest.com", "tiktok.com", "snapchat.com",
    "whatsapp.com", "telegram.org", "signal.org", "discord.com",
    # Enterprise / productivity
    "jira.com", "trello.com", "asana.com", "monday.com",
    "hubspot.com", "zendesk.com", "intercom.com", "mailchimp.com",
    "sendgrid.net", "amazonaws.com", "azure.com",
    # Education / government
    "edu", "gov", "mil",
})

_SUSPICIOUS_TLDS = frozenset({
    ".xyz", ".top", ".buzz", ".click", ".club", ".info", ".win", ".bid",
    ".loan", ".racing", ".review", ".stream", ".gq", ".cf", ".tk", ".ml",
    ".ga", ".pw", ".work", ".party", ".date", ".trade", ".webcam",
    ".science", ".accountant", ".cricket", ".faith", ".zip", ".mov",
})

_FREEMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "ymail.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "zoho.com", "mail.com", "gmx.com",
    "gmx.net", "fastmail.com", "tutanota.com",
})

# Cache for WHOIS / DNS lookups (domain -> result dict)
_domain_rep_cache = {}
_domain_rep_cache_lock = threading.Lock()


def _get_domain_age(domain):
    """Try to get domain age in days via python-whois. Returns None if unavailable."""
    try:
        import whois
        from datetime import datetime, timezone
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - created).days
            return max(0, age)
    except Exception:
        pass
    return None


def _check_dns(domain):
    """Check if domain resolves via DNS. Returns True/False/None."""
    try:
        socket.getaddrinfo(domain, None, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False
    except Exception:
        return None


VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "").strip()
VT_API_BASE = "https://www.virustotal.com/api/v3"
VT_CACHE_TTL_SECONDS = 24 * 3600
_vt_cache_lock = threading.Lock()
_vt_cache = {}

URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "").strip()
URLSCAN_CACHE_TTL_SECONDS = 12 * 3600
_urlscan_cache_lock = threading.Lock()
_urlscan_cache = {}

PHISHTANK_API_KEY = os.environ.get("PHISHTANK_API_KEY", "").strip()
PHISHTANK_FEED_URL_AUTH = "http://data.phishtank.com/data/{key}/online-valid.json"
PHISHTANK_FEED_URL_NOAUTH = "http://data.phishtank.com/data/online-valid.json"
URLHAUS_FEED_URL = "https://urlhaus.abuse.ch/downloads/json_online/"
PHISHTANK_REFRESH_INTERVAL = 6 * 3600
PHISHTANK_DOWNLOAD_TIMEOUT = (10, 120)
_phishtank_lock = threading.Lock()
_phishtank_data = {
    "domains": set(),
    "urls": set(),
    "updated_at": None,
    "count": 0,
}

_runtime_dir = Path(os.environ.get("PHISHGUARD_USER_DATA") or Path(__file__).parent)
_COMMUNITY_DB_PATH = _runtime_dir / "local_scan_history.json"
_community_db_lock = threading.Lock()


def _vt_lookup_hash(sha256):
    """Look up a SHA-256 file hash on VirusTotal without uploading content."""
    if not VIRUSTOTAL_API_KEY or not http_requests:
        return None

    with _vt_cache_lock:
        cached = _vt_cache.get(sha256)
        if cached:
            age = (datetime.now() - cached["cached_at"]).total_seconds()
            if age < VT_CACHE_TTL_SECONDS:
                return cached["result"]

    try:
        resp = http_requests.get(
            f"{VT_API_BASE}/files/{sha256}",
            headers={
                "x-apikey": VIRUSTOTAL_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception as exc:
        return {"error": f"network: {exc}", "status": 0}

    if resp.status_code == 404:
        result = {
            "sha256": sha256,
            "found": False,
            "malicious": 0,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
            "total": 0,
        }
    elif resp.status_code == 200:
        try:
            payload = resp.json()
            attrs = payload.get("data", {}).get("attributes", {}) or {}
            stats = attrs.get("last_analysis_stats", {}) or {}
            result = {
                "sha256": sha256,
                "found": True,
                "malicious": int(stats.get("malicious", 0) or 0),
                "suspicious": int(stats.get("suspicious", 0) or 0),
                "harmless": int(stats.get("harmless", 0) or 0),
                "undetected": int(stats.get("undetected", 0) or 0),
                "total": int(sum(v or 0 for v in stats.values())),
                "meaningful_name": attrs.get("meaningful_name", "") or "",
                "type_description": attrs.get("type_description", "") or "",
                "first_submission": attrs.get("first_submission_date"),
                "permalink": f"https://www.virustotal.com/gui/file/{sha256}",
            }
        except Exception as exc:
            return {"error": f"parse: {exc}", "status": 200}
    elif resp.status_code == 401:
        return {"error": "VT api key invalid or revoked", "status": 401}
    elif resp.status_code == 429:
        return {"error": "VT rate limit hit; try again in a minute", "status": 429}
    else:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}

    with _vt_cache_lock:
        _vt_cache[sha256] = {"result": result, "cached_at": datetime.now()}
    return result


def _check_urlscan(domain):
    """Query urlscan.io public search for prior scans of a domain."""
    if not URLSCAN_API_KEY or not http_requests or not domain:
        return None
    key = domain.lower().strip()

    with _urlscan_cache_lock:
        entry = _urlscan_cache.get(key)
        if entry:
            age = (datetime.now() - entry["cached_at"]).total_seconds()
            if age < URLSCAN_CACHE_TTL_SECONDS:
                return entry["result"]

    try:
        resp = http_requests.get(
            "https://urlscan.io/api/v1/search/",
            params={"q": f"domain:{key}", "size": 5},
            headers={
                "API-Key": URLSCAN_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception as exc:
        return {"error": f"network: {exc}"}

    if resp.status_code == 401:
        return {"error": "urlscan key invalid", "status": 401}
    if resp.status_code == 429:
        return {"error": "urlscan rate limited", "status": 429}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}

    try:
        data = resp.json()
    except ValueError:
        return {"error": "not JSON"}

    items = data.get("results", []) or []
    if not items:
        result = {"found": False, "scan_count": 0, "malicious": False}
    else:
        malicious = False
        for item in items[:5]:
            verdicts = item.get("verdicts", {}) or {}
            overall = verdicts.get("overall", {}) or {}
            if overall.get("malicious"):
                malicious = True
                break
        latest = items[0]
        result = {
            "found": True,
            "scan_count": len(items),
            "malicious": malicious,
            "latest_result": latest.get("result", ""),
            "latest_screenshot": latest.get("screenshot", ""),
        }

    with _urlscan_cache_lock:
        _urlscan_cache[key] = {"result": result, "cached_at": datetime.now()}
    return result


def _phishtank_domain_key(host):
    """Normalize a host for local threat-feed comparison."""
    if not host:
        return ""
    host = host.lower().strip()
    host = host.split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _download_phishtank_entries():
    """Download PhishTank feed entries when available."""
    if not http_requests:
        return []
    url = (PHISHTANK_FEED_URL_AUTH.format(key=PHISHTANK_API_KEY)
           if PHISHTANK_API_KEY else PHISHTANK_FEED_URL_NOAUTH)
    label = "PhishTank-authed" if PHISHTANK_API_KEY else "PhishTank-anon"
    try:
        print(f"[{label}] downloading feed...")
        resp = http_requests.get(
            url,
            headers={"User-Agent": "PhishGuard/2.0 (+phishguard-capstone)"},
            timeout=PHISHTANK_DOWNLOAD_TIMEOUT,
        )
    except Exception as exc:
        print(f"[{label}] feed download failed: {exc}")
        return []

    if resp.status_code != 200:
        print(f"[{label}] feed returned HTTP {resp.status_code}")
        return []
    try:
        data = resp.json()
    except ValueError as exc:
        print(f"[{label}] feed not JSON: {exc}")
        return []
    return data if isinstance(data, list) else []


def _download_urlhaus_entries():
    """Download URLhaus online URL feed. No API key required."""
    if not http_requests:
        return []
    try:
        print("[URLhaus] downloading feed...")
        resp = http_requests.get(
            URLHAUS_FEED_URL,
            headers={"User-Agent": "PhishGuard/2.0 (+phishguard-capstone)"},
            timeout=PHISHTANK_DOWNLOAD_TIMEOUT,
        )
    except Exception as exc:
        print(f"[URLhaus] feed download failed: {exc}")
        return []

    if resp.status_code != 200:
        print(f"[URLhaus] feed returned HTTP {resp.status_code}")
        return []
    try:
        raw = resp.json()
    except ValueError as exc:
        print(f"[URLhaus] feed not JSON: {exc}")
        return []

    out = []
    if isinstance(raw, dict):
        for rec in raw.values():
            if isinstance(rec, list) and rec:
                out.append(rec[0])
            elif isinstance(rec, dict):
                out.append(rec)
    elif isinstance(raw, list):
        out = raw
    return out


def _load_phishtank_feed():
    """Load PhishTank and URLhaus feeds into an in-memory lookup table."""
    combined_entries = []
    combined_entries.extend(_download_phishtank_entries())
    combined_entries.extend(_download_urlhaus_entries())
    if not combined_entries:
        print("[PhishTank/URLhaus] no entries loaded")
        return

    domains = set()
    urls = set()
    for entry in combined_entries:
        if not isinstance(entry, dict):
            continue
        phish_url = entry.get("url", "")
        if not phish_url:
            continue
        url_l = phish_url.lower().strip()
        urls.add(url_l)
        try:
            host = _phishtank_domain_key(urlparse(url_l).netloc)
            if host:
                domains.add(host)
        except Exception:
            pass

    with _phishtank_lock:
        _phishtank_data["domains"] = domains
        _phishtank_data["urls"] = urls
        _phishtank_data["updated_at"] = datetime.now()
        _phishtank_data["count"] = len(combined_entries)
    print(f"[Threat feed] active: {len(combined_entries)} entries, "
          f"{len(domains)} domains, {len(urls)} urls")


def _start_phishtank_refresher():
    """Start a background refresh loop for local threat feeds."""
    def _loop():
        while True:
            try:
                _load_phishtank_feed()
            except Exception as exc:
                print(f"[PhishTank] refresh error: {exc}")
            _time.sleep(PHISHTANK_REFRESH_INTERVAL)

    thread = threading.Thread(target=_loop, name="phishtank-refresher", daemon=True)
    thread.start()


def _check_phishtank(domain):
    """Check a domain locally against the loaded PhishTank/URLhaus feed."""
    if not domain:
        return None
    with _phishtank_lock:
        if not _phishtank_data["updated_at"]:
            return None
        return _phishtank_domain_key(domain) in _phishtank_data["domains"]


def _check_phishtank_url(url):
    """Check a specific URL locally against the loaded threat feed."""
    if not url:
        return None
    with _phishtank_lock:
        if not _phishtank_data["updated_at"]:
            return None
        return url.lower().strip() in _phishtank_data["urls"]


def _check_urlhaus(url_or_domain):
    """URLhaus host lookup. Sends only the host/domain."""
    if not http_requests:
        return None
    try:
        resp = http_requests.post(
            "https://urlhaus-api.abuse.ch/v1/host/",
            data={"host": url_or_domain},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("query_status") == "no_results":
                return False
            urls_online = data.get("urls", [])
            return any(u.get("url_status") == "online" for u in urls_online)
    except Exception:
        pass
    return None


def _check_abuseipdb(domain):
    """AbuseIPDB IP reputation lookup; enabled only when ABUSEIPDB_KEY exists."""
    api_key = os.environ.get("ABUSEIPDB_KEY", "").strip()
    if not api_key or not http_requests:
        return None
    try:
        ips = socket.getaddrinfo(domain, None, socket.AF_INET)
        ip = ips[0][4][0] if ips else None
        if not ip:
            return None
        resp = http_requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers={"Key": api_key, "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            score = data.get("abuseConfidenceScore", 0)
            return {"score": score, "is_abusive": score > 50}
    except Exception:
        pass
    return None


def _check_domain_reputation(domain):
    """
    Compute a worldwide reputation score for a domain (0.0-1.0).
    Uses multiple heuristics without requiring API keys:
      - Trusted domain list match
      - TLD risk analysis
      - Domain age via WHOIS (if available)
      - Structural analysis (impersonation patterns)
      - DNS existence check
    Results are cached to avoid repeated lookups.
    """
    if not domain:
        return {"score": 0.5, "signals": [], "category": "unknown"}

    domain = domain.lower().strip()
    with _domain_rep_cache_lock:
        cached = _domain_rep_cache.get(domain)
    if cached is not None:
        return cached

    score = 0.5
    signals = []

    # 1. Trusted domain list
    is_trusted = (
        domain in _TRUSTED_DOMAINS
        or any(domain.endswith("." + td) for td in _TRUSTED_DOMAINS)
        or domain.split(".")[-1] in _TRUSTED_DOMAINS  # .edu, .gov, .mil
    )
    if is_trusted:
        score += 0.35
        signals.append(("trusted", "Globally recognized domain"))

    is_freemail = domain in _FREEMAIL_DOMAINS
    if is_freemail:
        signals.append(("freemail", "Free email provider — anyone can register"))
        score -= 0.05  # slight reduction since freemail can be abused

    # 2. TLD risk analysis
    tld = "." + domain.split(".")[-1] if "." in domain else ""
    if tld in _SUSPICIOUS_TLDS:
        score -= 0.25
        signals.append(("bad_tld", f"Uses high-risk TLD ({tld})"))
    elif tld in (".com", ".org", ".net", ".co"):
        score += 0.05
        signals.append(("good_tld", "Common, established TLD"))

    # 3. Structural analysis
    name_part = domain.split(".")[0] if "." in domain else domain

    # Excessive hyphens -> impersonation / DGA
    if name_part.count("-") >= 2:
        score -= 0.15
        signals.append(("hyphens", "Multiple hyphens — common in fake domains"))

    # Numbers mixed with letters in the name (e.g., g00gle, paypa1)
    has_digits = bool(re.search(r'\d', name_part))
    has_letters = bool(re.search(r'[a-z]', name_part))
    if has_digits and has_letters and len(name_part) > 6:
        score -= 0.15
        signals.append(("mixedchars", "Mix of letters and numbers — possible impersonation"))

    # Very long domain name
    if len(name_part) > 20:
        score -= 0.1
        signals.append(("long_name", "Unusually long domain name"))

    # Brand impersonation patterns
    _brands = ["paypal", "apple", "google", "microsoft", "amazon", "chase",
               "wellsfargo", "bankofamerica", "netflix", "facebook", "instagram"]
    for brand in _brands:
        if brand in name_part and domain not in _TRUSTED_DOMAINS:
            score -= 0.25
            signals.append(("impersonation", f"Contains '{brand}' but isn't the real domain"))
            break

    # 4. Domain age via WHOIS
    domain_age_days = _get_domain_age(domain)
    if domain_age_days is not None:
        if domain_age_days < 30:
            score -= 0.25
            signals.append(("new_domain", f"Domain is only {domain_age_days} days old"))
        elif domain_age_days < 180:
            score -= 0.1
            signals.append(("young_domain", f"Domain is {domain_age_days} days old"))
        elif domain_age_days > 365 * 3:
            score += 0.1
            years = domain_age_days // 365
            signals.append(("established", f"Domain has been active for {years}+ years"))

    # 5. DNS existence check
    dns_exists = _check_dns(domain)
    if dns_exists is False:
        score -= 0.3
        signals.append(("no_dns", "Domain does not resolve — may not exist"))
    elif dns_exists is True and not is_trusted:
        signals.append(("dns_ok", "Domain resolves in DNS"))

    # 6. PhishTank database check
    phishtank_hit = _check_phishtank(domain)
    if phishtank_hit is True:
        score -= 0.5
        signals.append(("phishtank", "Listed in PhishTank as a known phishing site"))
    elif phishtank_hit is False:
        signals.append(("phishtank_clean", "Not found in PhishTank database"))

    # Clamp
    score = max(0.0, min(1.0, score))

    # Categorize
    if score >= 0.7:
        category = "trusted"
    elif score >= 0.4:
        category = "neutral"
    else:
        category = "suspicious"

    result = {"score": score, "signals": signals, "category": category}
    with _domain_rep_cache_lock:
        _domain_rep_cache[domain] = result
    return result


def _load_community_db():
    """Load local scan history. Informational only, never a verdict source."""
    try:
        if _COMMUNITY_DB_PATH.exists():
            with open(_COMMUNITY_DB_PATH, encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        pass
    return {"domains": {}, "urls": {}, "updated": ""}


def _save_community_db(db):
    try:
        _COMMUNITY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db["updated"] = datetime.utcnow().isoformat()
        with open(_COMMUNITY_DB_PATH, "w", encoding="utf-8") as handle:
            json.dump(db, handle, indent=2)
    except Exception as exc:
        print(f"WARNING: could not write local scan history: {exc}")


def _community_report(domain, is_phishing, *, user_confirmed=False):
    """Record explicit user reports only, avoiding self-poisoned model feedback."""
    if not user_confirmed or not domain:
        return
    with _community_db_lock:
        db = _load_community_db()
        if domain not in db["domains"]:
            db["domains"][domain] = {
                "phishing": 0,
                "safe": 0,
                "first_seen": datetime.utcnow().isoformat(),
            }
        if is_phishing:
            db["domains"][domain]["phishing"] += 1
        else:
            db["domains"][domain]["safe"] += 1
        _save_community_db(db)


def _community_check(domain):
    """Return local report counts only; callers must not treat as a verdict."""
    db = _load_community_db()
    entry = db.get("domains", {}).get(domain)
    if not entry:
        return None
    total = entry.get("phishing", 0) + entry.get("safe", 0)
    if total < 1:
        return None
    return {
        "reports": total,
        "phishing_reports": entry.get("phishing", 0),
        "safe_reports": entry.get("safe", 0),
    }


def _sender_domain_from_email(email):
    sender = ""
    sender_from = email.get("from", {})
    if isinstance(sender_from, dict) and "emailAddress" in sender_from:
        sender = sender_from["emailAddress"].get("address", "")
    elif isinstance(sender_from, dict):
        sender = sender_from.get("address", "") or sender_from.get("email", "")
    elif isinstance(email.get("sender"), str):
        sender = email.get("sender", "")
    return sender.split("@")[-1].lower() if "@" in sender else ""


def run_threat_intel(email, urls_in_email=None):
    """Run public URL/domain checks before the AI model verdict is finalized."""
    domain = _sender_domain_from_email(email)
    results = {
        "domain": domain,
        "checks": {},
        "confirmed_phishing": False,
        "confidence_boost": 0.0,
        "signals": [],
    }
    if not domain:
        return results

    pt = _check_phishtank(domain)
    results["checks"]["phishtank"] = pt
    if pt is True:
        results["confirmed_phishing"] = True
        results["signals"].append("PhishTank: Domain is a confirmed phishing site")

    for url in (urls_in_email or [])[:20]:
        pt_url = _check_phishtank_url(url)
        if pt_url is True:
            results["confirmed_phishing"] = True
            results["signals"].append("Threat feed: URL is confirmed phishing")
            break

    uh = _check_urlhaus(domain)
    results["checks"]["urlhaus"] = uh
    if uh is True:
        results["confirmed_phishing"] = True
        results["signals"].append("URLhaus: Domain hosts active malware/phishing URLs")

    aip = _check_abuseipdb(domain)
    results["checks"]["abuseipdb"] = aip
    if aip and isinstance(aip, dict) and aip.get("is_abusive"):
        results["confidence_boost"] += 0.15
        results["signals"].append(
            f"AbuseIPDB: Sender IP has {aip['score']}% abuse confidence"
        )

    us = _check_urlscan(domain)
    results["checks"]["urlscan"] = us
    if us and isinstance(us, dict):
        if us.get("malicious"):
            results["confirmed_phishing"] = True
            results["signals"].append(
                "urlscan.io: domain flagged as malicious in prior public scans"
            )
        elif us.get("found"):
            results["confidence_boost"] += 0.05
            results["signals"].append(
                f"urlscan.io: domain has {us.get('scan_count', 0)} prior public scans on file"
            )

    local_history = _community_check(domain)
    results["checks"]["local_history"] = local_history

    rep = _check_domain_reputation(domain)
    results["checks"]["domain_reputation"] = rep
    if rep["category"] == "suspicious":
        results["confidence_boost"] += 0.10
        top_signal = rep["signals"][0][1] if rep["signals"] else "Suspicious domain"
        results["signals"].append(f"Domain analysis: {top_signal}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  HTML to Text
# ─────────────────────────────────────────────────────────────────────────────
class _HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and decode entities, skipping non-visible content."""
    _SKIP_TAGS = {"style", "script", "head", "title"}
    _BLOCK_TAGS = {
        "p", "div", "br", "hr", "li", "tr", "h1", "h2", "h3", "h4", "h5",
        "h6", "blockquote", "pre", "table", "thead", "tbody", "tfoot",
        "section", "article", "header", "footer", "nav", "ul", "ol",
    }

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag_l in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0 and tag_l in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def handle_entityref(self, name):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&#{name};"))

    def get_text(self):
        return "".join(self._parts)


def _html_to_text(raw_html):
    """Safely convert HTML to clean plain text."""
    try:
        parser = _HTMLTextExtractor()
        parser.feed(raw_html)
        text = parser.get_text()
    except Exception:
        text = re.sub(r'<[^>]+>', ' ', raw_html)
    # Collapse spaces/tabs within each line, then collapse 3+ newlines to 2
    lines = [re.sub(r'[^\S\n]+', ' ', ln).strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the SPA."""
    return send_from_directory("templates", "index.html")


@app.route("/api/csrf", methods=["GET"])
def get_csrf_token():
    """Compatibility token for the Electron renderer's secure fetch wrapper."""
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return jsonify({"csrf": token})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Return minimal app settings expected by the newer renderer."""
    return jsonify({
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "graph_connected": bool(_graph_token["access_token"] and _live_mode),
    })


@app.route("/api/log", methods=["GET"])
def get_log():
    """Return the session log file contents."""
    if not LOG_FILE.exists():
        return jsonify({"log": "No session activity recorded yet."})
    text = LOG_FILE.read_text(encoding="utf-8")
    return jsonify({"log": text})


@app.route("/api/log/download", methods=["GET"])
def download_log():
    """Download the session log as a text file."""
    if not LOG_FILE.exists():
        return "No log file yet.", 404
    return send_from_directory(
        str(LOG_FILE.parent), LOG_FILE.name,
        as_attachment=True, mimetype="text/plain"
    )


@app.route("/api/reputation/<domain>", methods=["GET"])
def get_reputation(domain):
    """Return worldwide domain reputation."""
    rep = _check_domain_reputation(domain)
    # Convert signal tuples to serializable dicts
    serializable_signals = [
        {"type": sig[0], "message": sig[1]} for sig in rep.get("signals", [])
    ]
    return jsonify({
        "domain": domain,
        "score": rep["score"],
        "signals": serializable_signals,
        "category": rep["category"],
    })


@app.route("/api/phishtank/status", methods=["GET"])
def phishtank_status():
    """Return local threat-feed loading status for diagnostics."""
    with _phishtank_lock:
        updated = _phishtank_data["updated_at"]
        return jsonify({
            "loaded": bool(updated),
            "updated_at": updated.isoformat() if updated else None,
            "entries": _phishtank_data["count"],
            "unique_domains": len(_phishtank_data["domains"]),
            "unique_urls": len(_phishtank_data["urls"]),
        })


@app.route("/api/db/status", methods=["GET"])
def db_status():
    """Show whether Supabase config and a browser JWT are available."""
    jwt = _request_jwt()
    return jsonify({
        "configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "authenticated": bool(_jwt_user_id(jwt)),
    })


@app.route("/api/report-sender", methods=["POST"])
def report_sender():
    """Persist an explicit sender-domain report to Supabase threat_reports."""
    data = request.get_json(silent=True) or {}
    domain = (data.get("domain") or "").strip().lower()
    is_phishing = bool(data.get("is_phishing", True))
    if not domain:
        return jsonify({"error": "domain required"}), 400

    jwt = _request_jwt()
    user_id = _jwt_user_id(jwt)
    if not (jwt and user_id):
        return jsonify({"error": "Supabase sign-in required"}), 401

    _community_report(domain, is_phishing, user_confirmed=True)

    resp = _supabase_request(
        "POST", "threat_reports", jwt,
        json_body={
            "reporter_id": user_id,
            "domain": domain,
            "category": "phishing" if is_phishing else "safe",
        },
        prefer="resolution=ignore-duplicates,return=minimal",
    )
    if resp is not None and resp.status_code >= 400 and resp.status_code != 409:
        return jsonify({"error": "Supabase insert failed"}), 502
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
#  Microsoft Graph API Client (OAuth2 Auth Code + PKCE)
# ─────────────────────────────────────────────────────────────────────────────
AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_URL = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Read", "User.Read"]
REDIRECT_URI = "http://localhost:5050/auth/callback"

# Server-side state for the authenticated session
_graph_token = {"access_token": None, "expiry": None}
_graph_user = {"name": "", "email": ""}  # user info — stored server-side, not in session
_live_emails = []  # emails fetched from real Outlook
_pending_oauth = {}  # state -> {client_id, verifier, expires} — avoids session cookie issues in popup
_OAUTH_TIMEOUT = timedelta(minutes=10)  # pending auth requests expire after 10 min
_live_mode = False  # True when connected to real Outlook

_external_oauth_lock = threading.Lock()
_external_oauth_pending = {}
_EXTERNAL_OAUTH_TTL_SECONDS = 600


def _generate_pkce():
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _graph_request(endpoint, params=None):
    if not _graph_token["access_token"]:
        raise ValueError("Not authenticated")
    if _graph_token["expiry"] and datetime.now() >= _graph_token["expiry"]:
        _graph_token["access_token"] = None
        _graph_token["expiry"] = None
        raise ValueError("Token expired")
    headers = {
        "Authorization": f"Bearer {_graph_token['access_token']}",
        "Content-Type": "application/json",
    }
    r = http_requests.get(f"{GRAPH_URL}/{endpoint}", headers=headers, params=params)
    if r.status_code == 401:
        _graph_token["access_token"] = None
        _graph_token["expiry"] = None
        raise ValueError("Token expired")
    r.raise_for_status()
    return r.json()


def _fetch_all_emails():
    """Fetch all inbox emails via pagination."""
    all_emails = []
    params = {
        "$top": 250,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,body,isRead,internetMessageHeaders,hasAttachments",
        "$orderby": "receivedDateTime desc",
    }
    result = _graph_request("me/mailFolders/inbox/messages", params)
    all_emails.extend(result.get("value", []))

    # Follow @odata.nextLink until no more pages
    next_link = result.get("@odata.nextLink")
    while next_link:
        headers = {
            "Authorization": f"Bearer {_graph_token['access_token']}",
            "Content-Type": "application/json",
        }
        r = http_requests.get(next_link, headers=headers)
        if r.status_code == 401:
            _graph_token["access_token"] = None
            _graph_token["expiry"] = None
            raise ValueError("Token expired")
        r.raise_for_status()
        data = r.json()
        all_emails.extend(data.get("value", []))
        next_link = data.get("@odata.nextLink")

    return all_emails


# ── OAuth Routes ─────────────────────────────────────────────────────────────

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Return current connection status."""
    connected = bool(_graph_token["access_token"] and _live_mode)
    return jsonify({
        "connected": connected,
        "live_mode": _live_mode,
        "user_name": _graph_user["name"],
        "user_email": _graph_user["email"],
    })


@app.route("/api/auth/connect", methods=["POST"])
def auth_connect():
    """Start OAuth flow — returns the Microsoft login URL."""
    global _live_mode
    if http_requests is None:
        return jsonify({"error": "requests library not installed"}), 500

    data = request.get_json(silent=True) or {}
    client_id = data.get("client_id", "").strip()
    if not client_id:
        return jsonify({"error": "Client ID required"}), 400

    # Store client_id and PKCE server-side (not in session — popup may lose cookie)
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    # Clean up expired pending requests
    now = datetime.now()
    expired = [k for k, v in _pending_oauth.items() if now >= v["expires"]]
    for k in expired:
        del _pending_oauth[k]

    _pending_oauth[state] = {
        "client_id": client_id,
        "verifier": verifier,
        "expires": now + _OAUTH_TIMEOUT,
    }

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "response_mode": "query",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{AUTHORITY}/oauth2/v2.0/authorize?{urlencode(params)}"
    return jsonify({"auth_url": auth_url})


@app.route("/api/auth/external-start", methods=["POST"])
def auth_external_start():
    """Register the renderer nonce before browser-based Supabase OAuth."""
    data = request.get_json(silent=True) or {}
    nonce = str(data.get("nonce", "")).strip()
    if not nonce or len(nonce) < 16 or len(nonce) > 128:
        return jsonify({"error": "invalid nonce"}), 400

    now = datetime.now()
    with _external_oauth_lock:
        stale = [
            k for k, v in _external_oauth_pending.items()
            if (now - v["created"]).total_seconds() > _EXTERNAL_OAUTH_TTL_SECONDS
        ]
        for k in stale:
            _external_oauth_pending.pop(k, None)
        if len(_external_oauth_pending) > 200:
            return jsonify({"error": "too many pending"}), 429
        _external_oauth_pending[nonce] = {"created": now, "tokens": None}
    return jsonify({"ok": True})


_EXTERNAL_CALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>PhishGuard - Sign-in</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html,body { height: 100%; margin: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #06060C; color: #F0F0F8; display: flex; align-items: center;
         justify-content: center; padding: 24px; }
  .box { max-width: 440px; text-align: center; }
  h1 { font-size: 22px; margin: 0 0 14px; font-weight: 700; }
  p { color: #B0B2CC; font-size: 14px; line-height: 1.55; margin: 8px 0; }
  .spinner { margin: 32px auto 24px; width: 36px; height: 36px;
             border: 3px solid rgba(129,140,248,0.18);
             border-top-color: #818CF8; border-radius: 50%;
             animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .ok { color: #34D399; font-weight: 600; }
  .err { color: #FB7185; font-weight: 600; }
  .small { color: #7B7D98; font-size: 12px; margin-top: 24px; }
</style></head>
<body>
<div class="box">
  <h1 id="title">Completing sign-in...</h1>
  <div class="spinner" id="spinner"></div>
  <p id="status">Securely transferring your session to PhishGuard.</p>
  <p class="small">This window will close automatically.</p>
</div>
<script>
(function () {
  function escapeHtml(s) { return String(s).replace(/[<>"&]/g, ''); }
  var hash = new URLSearchParams((window.location.hash || '').replace(/^#/, ''));
  var query = new URLSearchParams(window.location.search || '');
  var nonce = query.get('nonce');
  var accessToken = hash.get('access_token');
  var refreshToken = hash.get('refresh_token') || '';
  var providerToken = hash.get('provider_token') || '';
  var providerRefreshToken = hash.get('provider_refresh_token') || '';
  var expiresIn = parseInt(hash.get('expires_in') || '3600', 10);
  var err = hash.get('error') || query.get('error');
  var errDesc = hash.get('error_description') || query.get('error_description');

  var spinnerEl = document.getElementById('spinner');
  var statusEl = document.getElementById('status');
  var titleEl = document.getElementById('title');

  function fail(msg) {
    spinnerEl.style.display = 'none';
    titleEl.textContent = 'Sign-in failed';
    statusEl.innerHTML = '<span class="err">' + escapeHtml(msg) + '</span>';
  }

  if (err) { fail((errDesc || err) + ' - you can close this tab.'); return; }
  if (!nonce || !accessToken) {
    fail('Missing session data. You can close this tab and try again.');
    return;
  }

  fetch('/api/auth/external-deliver', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nonce: nonce,
      access_token: accessToken,
      refresh_token: refreshToken,
      provider_token: providerToken,
      provider_refresh_token: providerRefreshToken,
      expires_in: expiresIn,
    }),
  }).then(function (r) { return r.json(); })
    .then(function (data) {
      spinnerEl.style.display = 'none';
      if (data && data.ok) {
        titleEl.textContent = 'Signed in';
        statusEl.innerHTML = '<span class="ok">You can close this tab and return to PhishGuard.</span>';
        setTimeout(function () { try { window.close(); } catch (e) {} }, 1500);
      } else {
        fail((data && data.error) || 'Delivery failed.');
      }
    }).catch(function (e) { fail('Delivery failed: ' + e.message); });
})();
</script>
</body></html>
"""


@app.route("/auth/external-callback")
def auth_external_callback():
    """Receive Supabase OAuth redirect and deliver URL hash tokens to Flask."""
    return _EXTERNAL_CALLBACK_HTML


@app.route("/api/auth/external-deliver", methods=["POST"])
def auth_external_deliver():
    """Browser callback page posts Supabase session tokens for Electron to poll."""
    data = request.get_json(silent=True) or {}
    nonce = str(data.get("nonce", "")).strip()
    if not nonce:
        return jsonify({"error": "nonce required"}), 400

    access_token = str(data.get("access_token", "")).strip()
    if not access_token:
        return jsonify({"error": "missing access_token"}), 400

    try:
        expires_in = int(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600

    with _external_oauth_lock:
        entry = _external_oauth_pending.get(nonce)
        if not entry:
            return jsonify({"error": "unknown or expired nonce"}), 404
        if entry.get("tokens"):
            return jsonify({"error": "already delivered"}), 409
        entry["tokens"] = {
            "access_token": access_token,
            "refresh_token": str(data.get("refresh_token", "")).strip(),
            "provider_token": str(data.get("provider_token", "")).strip(),
            "provider_refresh_token": str(data.get("provider_refresh_token", "")).strip(),
            "expires_in": expires_in,
        }
        entry["delivered_at"] = datetime.now()
    return jsonify({"ok": True})


@app.route("/api/auth/external-poll", methods=["GET"])
def auth_external_poll():
    """Return pending until browser OAuth delivers tokens, then return once."""
    nonce = (request.args.get("nonce") or "").strip()
    if not nonce:
        return jsonify({"status": "invalid"}), 400

    now = datetime.now()
    with _external_oauth_lock:
        entry = _external_oauth_pending.get(nonce)
        if not entry:
            return jsonify({"status": "expired"}), 404
        if (now - entry["created"]).total_seconds() > _EXTERNAL_OAUTH_TTL_SECONDS:
            _external_oauth_pending.pop(nonce, None)
            return jsonify({"status": "expired"}), 404
        if not entry.get("tokens"):
            return jsonify({"status": "pending"})
        tokens = entry["tokens"]
        _external_oauth_pending.pop(nonce, None)
    return jsonify({"status": "ready", **tokens})


@app.route("/api/auth/supabase-provider", methods=["POST"])
def auth_supabase_provider():
    """Use Supabase's Microsoft provider token as the Graph token."""
    global _graph_token, _live_emails, _live_mode

    if http_requests is None:
        return jsonify({"error": "requests library not installed"}), 500

    data = request.get_json(silent=True) or {}
    provider_token = (data.get("provider_token") or "").strip()
    if not provider_token:
        return jsonify({"error": "provider_token required"}), 400

    try:
        expires_in = int(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600

    _graph_token = {
        "access_token": provider_token,
        "expiry": datetime.now() + timedelta(seconds=max(60, expires_in)),
    }

    user_hint = data.get("user") or {}
    try:
        user_info = _graph_request("me")
        _graph_user["name"] = user_info.get("displayName") or user_hint.get("name") or "User"
        _graph_user["email"] = (
            user_info.get("mail")
            or user_info.get("userPrincipalName")
            or user_hint.get("email")
            or ""
        )
    except Exception:
        _graph_user["name"] = user_hint.get("name") or "Connected"
        _graph_user["email"] = user_hint.get("email") or ""

    try:
        _live_emails = _fetch_all_emails()
    except Exception as exc:
        _live_emails = []
        _live_mode = False
        return jsonify({"error": f"Could not fetch Outlook emails: {exc}"}), 502

    _live_mode = True
    scan_results.clear()
    _log_session_event("LOGIN", {
        "User": _graph_user["name"],
        "Email": _graph_user["email"],
        "Emails Loaded": len(_live_emails),
        "Source": "Supabase Microsoft provider",
    })
    return jsonify({
        "ok": True,
        "connected": True,
        "count": len(_live_emails),
        "user_name": _graph_user["name"],
        "user_email": _graph_user["email"],
    })


@app.route("/auth/callback")
def auth_callback():
    """Handle Microsoft OAuth redirect callback."""
    global _graph_token, _live_emails, _live_mode

    error = request.args.get("error")
    if error:
        desc = request.args.get("error_description", error)
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Sign-in Failed</h2><p>{html.escape(str(desc))}</p>
        <p>You can close this tab.</p></body></html>"""

    received_state = request.args.get("state")
    pending = _pending_oauth.pop(received_state, None)
    if not pending or datetime.now() >= pending["expires"]:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>Invalid or expired state parameter.</p>
        <p>You can close this tab.</p></body></html>"""

    code = request.args.get("code")
    if not code:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>No authorization code received.</p>
        <p>You can close this tab.</p></body></html>"""

    # Exchange code for token
    client_id = pending["client_id"]
    verifier = pending["verifier"]
    token_url = f"{AUTHORITY}/oauth2/v2.0/token"
    token_data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
        "scope": " ".join(SCOPES),
    }

    try:
        r = http_requests.post(token_url, data=token_data)
        if r.status_code != 200:
            err = r.json()
            desc = err.get("error_description", err.get("error", "Token exchange failed"))
            short = desc.split("\r\n")[0] if "\r\n" in desc else desc
            return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
            <h2>Sign-in Failed</h2><p>{html.escape(str(short))}</p>
            <p>You can close this tab.</p></body></html>"""

        tokens = r.json()
        access_token = tokens.get("access_token")
        expires_in = int(tokens.get("expires_in", 3600))

        # Temporarily set token so _graph_request works, but don't expose
        # to the poll yet (poll checks _graph_token to decide "connected")
        _graph_token["access_token"] = access_token
        _graph_token["expiry"] = datetime.now() + timedelta(seconds=expires_in)

        # Get user info
        try:
            user_info = _graph_request("me")
            _graph_user["name"] = user_info.get("displayName", "User")
            _graph_user["email"] = user_info.get("mail") or user_info.get("userPrincipalName", "")
        except Exception:
            _graph_user["name"] = "Connected"
            _graph_user["email"] = ""

        # Fetch emails before marking live mode — prevents race where
        # the main window poll sees "connected" but emails aren't loaded yet
        try:
            _live_emails = _fetch_all_emails()
        except Exception as e:
            _live_emails = []

        # Now mark live — the poll will see "connected" and loadEmails()
        # will return the real emails
        _live_mode = True

        # Clear any previous scan results
        scan_results.clear()

        # Log the login
        _log_session_event("LOGIN", {
            "User": _graph_user["name"],
            "Email": _graph_user["email"],
            "Emails Loaded": len(_live_emails),
        })

    except Exception as e:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>{html.escape(str(e))}</p>
        <p>You can close this tab.</p></body></html>"""

    return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
    <h2>Connected to Outlook</h2>
    <p>Signed in as {_graph_user['name'] or 'User'}</p>
    <p>You can close this tab and return to PhishGuard.</p>
    <script>window.close();</script></body></html>"""


@app.route("/api/auth/disconnect", methods=["POST"])
def auth_disconnect():
    """Disconnect from Outlook and log session summary."""
    global _graph_token, _live_emails, _live_mode

    # Compute scan stats before clearing
    total_emails = len(_live_emails)
    total_scanned = len(scan_results)
    threats = sum(1 for r in scan_results.values() if r.get("prediction") == 1)
    safe = total_scanned - threats

    _log_session_event("LOGOUT", {
        "User": _graph_user["name"],
        "Email": _graph_user["email"],
        "Total Emails": total_emails,
        "Emails Scanned": total_scanned,
        "Phishing Detected": threats,
        "Safe Emails": safe,
    })

    _graph_token = {"access_token": None, "expiry": None}
    _graph_user["name"] = ""
    _graph_user["email"] = ""
    _live_emails = []
    _live_mode = False
    scan_results.clear()
    _pending_oauth.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/refresh", methods=["POST"])
def auth_refresh_emails():
    """Re-fetch emails from Outlook."""
    global _live_emails
    if not _live_mode or not _graph_token["access_token"]:
        return jsonify({"error": "Not connected"}), 401
    try:
        _live_emails = _fetch_all_emails()
        scan_results.clear()
        return jsonify({"count": len(_live_emails)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Patch email endpoints to support live mode ───────────────────────────────

def _get_email_list():
    """Return the active email list (live emails when connected, empty otherwise)."""
    return _live_emails if _live_mode else []


def _message_key(email, idx):
    """Return the stable key used by the current renderer."""
    return str(email.get("id") or email.get("messageId") or idx)


def _find_email_by_message_id(message_id):
    """Find an email by Graph message id, falling back to numeric index strings."""
    emails = _get_email_list()
    target = str(message_id)
    for idx, email in enumerate(emails):
        if _message_key(email, idx) == target:
            return idx, email
    if target.isdigit():
        idx = int(target)
        if 0 <= idx < len(emails):
            return idx, emails[idx]
    return None, None


def _parse_auth_headers(internet_headers):
    """Parse internetMessageHeaders list into {spf, dkim, dmarc}."""
    result = {"spf": "none", "dkim": "none", "dmarc": "none"}
    if not internet_headers:
        return result
    for h in internet_headers:
        if h.get("name", "").lower() == "authentication-results":
            val = h.get("value", "").lower()
            for key in ("spf", "dkim", "dmarc"):
                if f"{key}=pass" in val:
                    result[key] = "pass"
                elif f"{key}=fail" in val or f"{key}=softfail" in val:
                    result[key] = "fail"
            break
    return result


def _normalize_email(email, idx):
    """Convert Graph API / mock email to flat format the frontend expects."""
    message_id = _message_key(email, idx)

    # Extract sender info
    sender = email.get("from", {})
    if "emailAddress" in sender:
        sender_name = sender["emailAddress"].get("name", "")
        sender_addr = sender["emailAddress"].get("address", "")
    else:
        sender_name = sender.get("name", "")
        sender_addr = sender.get("address", "")

    # Extract body — same approach as the desktop GUI
    body_raw = email.get("body", {})
    if isinstance(body_raw, dict):
        body_content = body_raw.get("content", "") or ""
    elif isinstance(body_raw, str):
        body_content = body_raw
    else:
        body_content = email.get("bodyPreview", "") or ""
    body_text = _html_to_text(body_content) if body_content else ""

    # Parse auth headers
    headers = _parse_auth_headers(email.get("internetMessageHeaders"))

    return {
        "id": message_id,
        "messageId": message_id,
        "idx": idx,
        "folder": email.get("folder", "inbox"),
        "subject": email.get("subject", "(No Subject)"),
        "sender_name": sender_name,
        "sender": sender_addr,
        "date": email.get("receivedDateTime", ""),
        "isRead": bool(email.get("isRead", True)),
        "bodyPreview": email.get("bodyPreview", ""),
        "body": body_text,
        "hasAttachments": email.get("hasAttachments", False),
        "attachments": email.get("attachments", []),
        "headers": headers,
        "scanned": message_id in scan_results or idx in scan_results,
        "scanResult": scan_results.get(message_id) or scan_results.get(idx),
    }


@app.route("/api/emails", methods=["GET"])
def get_emails_v2():
    emails = _get_email_list()
    emails_summary = [_normalize_email(email, i) for i, email in enumerate(emails)]
    return jsonify({"emails": emails_summary})


@app.route("/api/emails/<int:idx>", methods=["GET"])
def get_email_v2(idx):
    emails = _get_email_list()
    if idx < 0 or idx >= len(emails):
        return jsonify({"error": "Email not found"}), 404
    return jsonify(_normalize_email(emails[idx], idx))


@app.route("/api/emails/junk", methods=["GET"])
def get_junk_emails():
    """Junk folder support is not part of the restored backend yet."""
    return jsonify({"emails": []})


@app.route("/api/messages/<path:message_id>", methods=["GET"])
def get_message(message_id):
    idx, email = _find_email_by_message_id(message_id)
    if email is None:
        return jsonify({"error": "Email not found"}), 404
    return jsonify(_normalize_email(email, idx))


@app.route("/api/messages/<path:message_id>/headers", methods=["GET"])
def get_message_headers(message_id):
    idx, email = _find_email_by_message_id(message_id)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    headers = email.get("internetMessageHeaders") or []
    if not isinstance(headers, list):
        headers = []
    return jsonify({
        "headers": headers[:100],
        "count": len(headers),
        "truncated": len(headers) > 100,
    })


def _fetch_attachment_bytes(message_id, attachment_id):
    """Fetch Outlook attachment bytes for local hash-based analysis."""
    if not _live_mode or not _graph_token["access_token"] or not http_requests:
        return None
    url = f"{GRAPH_URL}/me/messages/{message_id}/attachments/{attachment_id}/$value"
    try:
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {_graph_token['access_token']}"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 401:
            _graph_token["access_token"] = None
            _graph_token["expiry"] = None
    except Exception as exc:
        print(f"WARNING: attachment fetch failed for {message_id}: {exc}")
    return None


@app.route("/api/messages/<path:message_id>/attachments/<path:attachment_id>/analyze", methods=["POST"])
def analyze_message_attachment(message_id, attachment_id):
    idx, email = _find_email_by_message_id(message_id)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    attachments = email.get("attachments", []) or []
    att = None
    for item in attachments:
        if not isinstance(item, dict):
            continue
        if item.get("id") == attachment_id or item.get("name") == attachment_id:
            att = item
            break

    if att is None:
        att = {"id": attachment_id, "name": attachment_id}

    name = att.get("name", "") or att.get("filename", "") or "file"
    if not VIRUSTOTAL_API_KEY:
        return jsonify({
            "configured": False,
            "name": name,
            "message": "VirusTotal API key not configured. Set VIRUSTOTAL_API_KEY to enable.",
        })

    content_bytes = _fetch_attachment_bytes(message_id, att.get("id") or attachment_id)
    if not content_bytes:
        return jsonify({
            "configured": True,
            "analyzed": False,
            "name": name,
            "message": "Could not fetch attachment bytes; analysis requires real Outlook mail.",
        })

    sha = hashlib.sha256(content_bytes).hexdigest()
    result = _vt_lookup_hash(sha)
    if result is None:
        return jsonify({
            "configured": False,
            "analyzed": False,
            "name": name,
            "sha256": sha,
            "message": "VT lookup returned no configuration.",
        })

    if "error" in result:
        return jsonify({
            "configured": True,
            "analyzed": False,
            "name": name,
            "sha256": sha,
            "error": result.get("error", "unknown error"),
            "status": result.get("status", 0),
        })

    return jsonify({
        "configured": True,
        "analyzed": True,
        "name": name,
        "size": len(content_bytes),
        **result,
    })


@app.route("/api/messages/<path:message_id>/move-to-junk", methods=["POST"])
def move_message_to_junk(message_id):
    idx, email = _find_email_by_message_id(message_id)
    if email is None:
        return jsonify({"error": "Email not found"}), 404
    return jsonify({
        "ok": False,
        "old_id": _message_key(email, idx),
        "new_id": _message_key(email, idx),
        "message": "Move to junk is not enabled in the lightweight backend.",
    }), 501


@app.route("/api/auth/photo", methods=["GET"])
def auth_photo():
    """The original backend does not fetch profile photos."""
    return "", 404


@app.route("/api/sender-dna/<path:sender_addr>", methods=["GET"])
def sender_dna(sender_addr):
    """Minimal sender DNA response; full profiling can be restored later."""
    return jsonify({
        "sender": sender_addr,
        "status": "unknown",
        "profile": None,
        "comparison": None,
    })


def _to_native(obj):
    """Recursively convert numpy/non-JSON types to native Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _make_serializable(url_analysis, header_result):
    """Convert all scan result data to JSON-serializable Python types."""
    return _to_native(url_analysis), _to_native(header_result)


def _sender_address(email):
    sender = email.get("from", {})
    if isinstance(sender, dict) and "emailAddress" in sender:
        return sender["emailAddress"].get("address", "") or ""
    if isinstance(sender, dict):
        return sender.get("email", "") or sender.get("address", "") or ""
    return ""


def _scan_history_row(email, result):
    """Build DB-safe scan metadata. Never stores message body or attachments."""
    sender_addr = _sender_address(email)
    domain = sender_addr.split("@")[-1].lower() if "@" in sender_addr else None
    return {
        "message_id": str(email.get("id") or result.get("idx") or ""),
        "sender_domain": domain,
        "prediction": int(result.get("prediction", 0)),
        "confidence": float(result.get("confidence", 0) or 0),
        "signals": {
            "intel_signals": list(
                (result.get("threat_intel") or {}).get("signals", []) or []
            )[:20],
            "checks_run": list(
                (result.get("threat_intel") or {}).get("checks", {}).keys()
            ),
            "confirmed": bool(
                (result.get("threat_intel") or {}).get("confirmed", False)
            ),
        },
    }


def _persist_scan_history(email, result):
    jwt = _request_jwt()
    user_id = _jwt_user_id(jwt)
    if not (jwt and user_id):
        return

    row = _scan_history_row(email, result)
    row["user_id"] = user_id
    resp = _supabase_request("POST", "scan_history", jwt, json_body=row)
    if resp is not None and resp.status_code >= 400:
        print(f"WARNING: scan_history insert failed "
              f"[{resp.status_code}]: {resp.text[:200]}")


def _scan_email_common(email, idx):
    if not detector.is_trained:
        return None, (jsonify({"error": "Model not loaded"}), 503)

    subject = email.get("subject", "") or ""
    body_raw = email.get("body")
    if isinstance(body_raw, dict):
        body = body_raw.get("content", "") or ""
    elif isinstance(body_raw, str):
        body = body_raw
    else:
        body = email.get("bodyPreview", "") or ""
    body_t = _html_to_text(body) if body else ""
    full = f"Subject: {subject}\n\n{body_t}"
    scan_body = body_t[:_MAX_BODY_FOR_REGEX]
    url_pattern = re.compile(r'https?://[^\s<>"\'`]{1,2048}')
    urls_found = url_pattern.findall(scan_body)[:_MAX_URLS_PER_EMAIL]
    threat_intel = run_threat_intel(email, urls_found)

    try:
        prediction, confidence, url_analysis, header_result = detector.predict(
            full, headers=email.get("internetMessageHeaders"))
        url_analysis, header_result = _make_serializable(url_analysis, header_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, (jsonify({"error": str(e)}), 500)

    prediction = _to_native(prediction)
    confidence = _to_native(confidence)
    if threat_intel["confirmed_phishing"]:
        prediction = 1
        confidence = max(confidence, 0.99)
    if prediction == 1 and threat_intel["confidence_boost"] > 0:
        confidence = min(1.0, confidence + threat_intel["confidence_boost"])

    message_id = _message_key(email, idx)
    result = {
        "id": message_id,
        "messageId": message_id,
        "idx": idx,
        "prediction": prediction,
        "confidence": confidence,
        "url_analysis": url_analysis,
        "header_result": header_result,
        "threat_intel": {
            "confirmed": threat_intel["confirmed_phishing"],
            "signals": threat_intel["signals"],
            "checks": _to_native(threat_intel["checks"]),
            "checks_run": list(threat_intel["checks"].keys()),
        },
    }
    scan_results[message_id] = result
    _persist_scan_history(email, result)
    return result, None


@app.route("/api/scan/<int:idx>", methods=["POST"])
def scan_email_v2(idx):
    emails = _get_email_list()
    if idx < 0 or idx >= len(emails):
        return jsonify({"error": "Email not found"}), 404

    result, error = _scan_email_common(emails[idx], idx)
    if error:
        return error
    return jsonify(result)


@app.route("/api/messages/<path:message_id>/scan", methods=["POST"])
def scan_message(message_id):
    idx, email = _find_email_by_message_id(message_id)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    result, error = _scan_email_common(email, idx)
    if error:
        return error
    return jsonify(result)


@app.route("/api/scan-all", methods=["POST"])
def scan_all_v2():
    emails = _get_email_list()
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503
    results = []
    for i, email in enumerate(emails):
        try:
            result, error = _scan_email_common(email, i)
            if error:
                continue
            results.append(result)
        except Exception:
            pass
    results_dict = {str(r["idx"]): r for r in results}
    return jsonify({"results": results_dict})


@app.route("/api/stats", methods=["GET"])
def get_stats_v2():
    scanned = len(scan_results)
    threats = sum(1 for r in scan_results.values() if r.get("prediction") == 1)
    safe = scanned - threats
    return jsonify({
        "scanned": scanned,
        "threats": threats,
        "safe": safe,
        "total": len(_get_email_list()),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5050"))
    debug = os.environ.get("PHISHGUARD_DEBUG") == "1"
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_phishtank_refresher()
    app.run(host="127.0.0.1", debug=debug, port=port, use_reloader=debug)
