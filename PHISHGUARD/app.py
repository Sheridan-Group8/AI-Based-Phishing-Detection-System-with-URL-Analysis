"""
PhishGuard Web Dashboard — Flask Backend
Serves the SPA and provides REST API for phishing email analysis.
"""

import base64
import hashlib
import hmac
import html
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
import threading
import time as _time
from datetime import datetime, timedelta
from functools import wraps
from html.parser import HTMLParser
from pathlib import Path
import tempfile as _tempfile
from urllib.parse import urlparse, urlsplit, unquote

from flask import Flask, Response, jsonify, request, send_from_directory

from phishing_detector import OnnxClassifier, PhishingDetector, load_detection_rules

try:
    import requests as http_requests
except ImportError:
    http_requests = None


APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _load_dotenv():
    """Load optional local API keys from PHISHGUARD/.env without a dependency."""
    env_path = APP_ROOT / ".env"
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
_DETECTION_RULES = load_detection_rules()

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=str(APP_ROOT / "static"),
    template_folder=str(APP_ROOT / "templates"),
)
app.secret_key = secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

_FLASK_PORT = int(os.environ.get("FLASK_PORT", "5050"))
ALLOWED_ORIGINS = frozenset({
    f"http://127.0.0.1:{_FLASK_PORT}",
    f"http://localhost:{_FLASK_PORT}",
})
CSRF_HEADER = "X-CSRF-Token"
LAUNCH_HEADER = "X-Launch-Secret"
LAUNCH_SECRET = os.environ.get("PHISHGUARD_LAUNCH_SECRET", "")
_DOMAIN_RE = re.compile(r"^(?=.{1,253}\.?$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\.?$")


def _resolve_user_data_dir():
    safe_roots = [Path.home().resolve(), Path(_tempfile.gettempdir()).resolve()]
    candidates = []
    env_val = os.environ.get("PHISHGUARD_USER_DATA")
    if env_val:
        try:
            candidates.append(Path(env_val).expanduser().resolve())
        except (OSError, RuntimeError) as exc:
            print(f"WARNING: PHISHGUARD_USER_DATA invalid ({exc}); ignoring.")
    candidates.append((Path.home() / ".phishguard").resolve())
    candidates.append((Path(_tempfile.gettempdir()) / "phishguard").resolve())

    for candidate in candidates:
        try:
            under_safe_root = any(
                candidate == root or root in candidate.parents
                for root in safe_roots
            )
        except Exception:
            under_safe_root = False
        if not under_safe_root:
            print(f"WARNING: Refusing unsafe data dir {candidate}")
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:
            print(f"WARNING: Could not create {candidate}: {exc}")

    emergency = Path(_tempfile.mkdtemp(prefix="phishguard-"))
    print(f"WARNING: Falling back to emergency data dir {emergency}")
    return emergency


def _is_public_domain(value):
    domain = (value or "").strip().lower().rstrip(".")
    if not domain or len(domain) > 253 or domain in {"localhost", "local"}:
        return False
    try:
        ip = ipaddress.ip_address(domain)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        pass
    return bool(_DOMAIN_RE.fullmatch(domain))


USER_DATA_DIR = _resolve_user_data_dir()

# Supabase database integration. The browser sends a Supabase access token in
# Authorization; Flask uses it with the anon key so database RLS scopes writes.
def _load_supabase_config():
    path = APP_ROOT / "static" / "js" / "supabase-config.js"
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
ONNX_DIR = APP_ROOT / "bigmodel" / "onnx-model"
URL_ONNX_DIR = APP_ROOT / "bigmodel" / "onnx-url-model"
_onnx_enabled = os.environ.get("PHISHGUARD_ONNX") != "0"

if _onnx_enabled:
    try:
        text_model = OnnxClassifier(str(ONNX_DIR))
        if text_model.load():
            detector.text_model = text_model
            detector.is_trained = True
            print("PhishGuard text AI active (local ONNX model).")
    except Exception as e:
        print(f"WARNING: ONNX text model init failed: {e}")

    try:
        url_model = OnnxClassifier(str(URL_ONNX_DIR))
        if url_model.load():
            detector.url_model = url_model
            print("PhishGuard URL AI active (local ONNX model).")
    except Exception as e:
        print(f"WARNING: ONNX URL model init failed: {e}")

if not detector.is_trained:
    print("WARNING: No runtime text model loaded. Scanning will be unavailable.")

# ─────────────────────────────────────────────────────────────────────────────
#  Per-session server-side state
#
#  Mailbox/OAuth state is keyed by a server-generated session ID carried in an
#  HttpOnly, SameSite=Strict cookie. This prevents two renderer/browser clients
#  from sharing the same Outlook token, email list, and scan results.
# ─────────────────────────────────────────────────────────────────────────────
SESSION_COOKIE = "pg_sid"
SESSION_IDLE_TTL = timedelta(hours=8)
_SCAN_ALL_COOLDOWN = 30


class SessionState:
    """Per-browser/Electron-window mailbox and auth state."""

    __slots__ = (
        "sid", "created", "last_seen", "csrf_token",
        "graph_token", "graph_user", "live_emails", "junk_emails",
        "live_mode", "scan_results", "_last_scan_all",
        "_scan_all_in_flight",
    )

    def __init__(self, sid):
        self.sid = sid
        self.created = datetime.now()
        self.last_seen = self.created
        self.csrf_token = secrets.token_urlsafe(32)
        self.graph_token = {"access_token": None, "expiry": None}
        self.graph_user = {"name": "", "email": ""}
        self.live_emails = []
        self.junk_emails = []
        self.live_mode = False
        self.scan_results = {}
        self._last_scan_all = None
        self._scan_all_in_flight = False


_sessions = {}
_sessions_lock = threading.Lock()


def _get_or_create_session():
    sid = request.cookies.get(SESSION_COOKIE)
    needs_new_cookie = False
    now = datetime.now()
    with _sessions_lock:
        stale = [
            key for key, value in _sessions.items()
            if now - value.last_seen > SESSION_IDLE_TTL
        ]
        for key in stale:
            _sessions.pop(key, None)

        sess = _sessions.get(sid) if sid else None
        if sess is None:
            sid = secrets.token_urlsafe(32)
            sess = SessionState(sid)
            _sessions[sid] = sess
            needs_new_cookie = True
        sess.last_seen = now

    request.environ["phishguard.session"] = sess
    request.environ["phishguard.sid"] = sid
    request.environ["phishguard.set_cookie"] = needs_new_cookie
    return sess


def _current_session():
    return request.environ.get("phishguard.session") or _get_or_create_session()


@app.after_request
def _emit_session_cookie(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=()",
    )
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store, max-age=0")
        resp.headers.setdefault("Pragma", "no-cache")

    sid = request.environ.get("phishguard.sid")
    if sid and request.environ.get("phishguard.set_cookie"):
        resp.set_cookie(
            SESSION_COOKIE,
            sid,
            httponly=True,
            samesite="Strict",
            secure=False,
            max_age=int(SESSION_IDLE_TTL.total_seconds()),
        )
    return resp

# ─────────────────────────────────────────────────────────────────────────────
#  Session Log
# ─────────────────────────────────────────────────────────────────────────────
def _origin_ok():
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")
    if origin:
        return origin.rstrip("/") in ALLOWED_ORIGINS
    if referer:
        parsed = urlparse(referer)
        ref_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return ref_origin in ALLOWED_ORIGINS
    if LAUNCH_SECRET:
        provided = request.headers.get(LAUNCH_HEADER, "")
        if provided and hmac.compare_digest(provided, LAUNCH_SECRET):
            return True
    return False


def _launch_secret_ok():
    if not LAUNCH_SECRET:
        return True
    provided = request.headers.get(LAUNCH_HEADER, "")
    return bool(provided) and hmac.compare_digest(provided, LAUNCH_SECRET)


def require_csrf(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _origin_ok():
            return jsonify({"error": "Forbidden (bad origin)"}), 403
        sess = _current_session()
        token = request.headers.get(CSRF_HEADER, "")
        if not token or not hmac.compare_digest(token, sess.csrf_token):
            return jsonify({"error": "Forbidden (bad CSRF token)"}), 403
        return view(*args, **kwargs)
    return wrapped


_sender_profiles = {}
_sender_profiles_lock = threading.Lock()


def _persist_sender_profile_to_supabase(sender_email, profile, jwt):
    """Upsert one sender DNA profile into Supabase for the signed-in user."""
    if not jwt or not profile:
        return
    user_id = _jwt_user_id(jwt)
    if not user_id:
        return
    body = {
        "user_id": user_id,
        "sender_email": sender_email,
        "profile": profile,
        "email_count": int(profile.get("email_count", 0) or 0),
    }
    resp = _supabase_request(
        "POST", "sender_profiles", jwt,
        json_body=body,
        prefer="resolution=merge-duplicates,return=minimal",
    )
    if resp is not None and resp.status_code >= 400:
        print(f"WARNING: sender_profiles upsert failed [{resp.status_code}]: {resp.text[:200]}")


def _analyze_writing_style(text):
    """Extract writing-style features from email text."""
    if not text or not text.strip():
        return None

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return None

    avg_sentence_len = len(words) / max(len(sentences), 1)
    avg_word_len = sum(len(word) for word in words) / word_count
    caps_ratio = sum(1 for ch in text if ch.isupper()) / max(len(text), 1)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    comma_density = text.count(",") / max(word_count, 1)

    formal_words = {
        "regarding", "furthermore", "consequently", "therefore", "hereby",
        "pursuant", "accordingly", "sincerely", "respectfully", "cordially",
        "attached", "enclosed",
    }
    informal_words = {
        "hey", "hi", "lol", "btw", "fyi", "gonna", "wanna", "yeah",
        "yep", "nope", "cool", "awesome", "thanks",
    }
    lower_words = {word.lower() for word in words}
    formality = (len(lower_words & formal_words) - len(lower_words & informal_words)) / max(word_count, 1)

    first_line = text.strip().split("\n")[0].lower().strip()
    if any(first_line.startswith(g) for g in ("dear", "good morning", "good afternoon", "good evening")):
        greeting = "formal"
    elif any(first_line.startswith(g) for g in ("hi", "hey", "hello")):
        greeting = "casual"
    else:
        greeting = "none"

    has_signature = (
        bool(re.search(r"(?:regards|best|sincerely|thanks|cheers|sent from)", text[-200:].lower()))
        if len(text) > 50 else False
    )

    return {
        "word_count": word_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "avg_word_len": round(avg_word_len, 2),
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "comma_density": round(comma_density, 4),
        "formality": round(formality, 4),
        "greeting": greeting,
        "has_signature": has_signature,
    }


def _extract_send_hour(datetime_str):
    try:
        dt = datetime.fromisoformat(str(datetime_str).replace("Z", "+00:00"))
        return dt.hour
    except Exception:
        return None


def _sender_addr_from_email(email):
    sender = email.get("from", {})
    if isinstance(sender, dict) and "emailAddress" in sender:
        return (sender.get("emailAddress") or {}).get("address", "") or ""
    if isinstance(sender, dict):
        return sender.get("address", "") or sender.get("email", "") or ""
    if isinstance(email.get("sender"), str):
        return email.get("sender", "")
    return ""


def _build_sender_profile(sender_addr, emails):
    """Build a behavioral profile from all messages by this sender."""
    sender_addr = (sender_addr or "").lower().strip()
    sender_emails = [
        email for email in emails
        if _sender_addr_from_email(email).lower() == sender_addr
    ]
    if not sender_emails:
        return None

    styles, send_hours = [], []
    for email in sender_emails:
        body = email.get("body", {})
        content = body.get("content", "") if isinstance(body, dict) else str(body or "")
        clean = _html_to_text(content) if "<" in content else re.sub(r"<[^>]+>", " ", content)
        style = _analyze_writing_style(clean)
        if style:
            styles.append(style)
        hour = _extract_send_hour(email.get("receivedDateTime", ""))
        if hour is not None:
            send_hours.append(hour)

    if not styles:
        return None

    return {
        "email": sender_addr,
        "email_count": len(sender_emails),
        "last_updated": datetime.now().isoformat(),
        "avg_word_count": round(sum(s["word_count"] for s in styles) / len(styles), 1),
        "avg_sentence_len": round(sum(s["avg_sentence_len"] for s in styles) / len(styles), 2),
        "avg_word_len": round(sum(s["avg_word_len"] for s in styles) / len(styles), 2),
        "avg_caps_ratio": round(sum(s["caps_ratio"] for s in styles) / len(styles), 4),
        "avg_formality": round(sum(s["formality"] for s in styles) / len(styles), 4),
        "typical_greeting": max(
            {s["greeting"] for s in styles},
            key=lambda g: sum(1 for s in styles if s["greeting"] == g),
        ),
        "usually_has_signature": sum(1 for s in styles if s["has_signature"]) > len(styles) / 2,
        "typical_send_hours": sorted(set(send_hours)) if send_hours else [],
        "avg_exclamations": round(sum(s["exclamation_count"] for s in styles) / len(styles), 1),
    }


def _compare_to_profile(profile, email):
    """Compare a message against an established sender DNA profile."""
    body = email.get("body", {})
    content = body.get("content", "") if isinstance(body, dict) else str(body or "")
    clean = _html_to_text(content) if "<" in content else re.sub(r"<[^>]+>", " ", content)
    style = _analyze_writing_style(clean)
    if not style or not profile:
        return {"score": 0, "flags": [], "status": "insufficient_data"}

    flags, deviation = [], 0
    avg_word_count = profile["avg_word_count"]
    if avg_word_count > 0:
        ratio = style["word_count"] / avg_word_count
        if ratio < 0.3 or ratio > 3.0:
            flags.append({
                "type": "length",
                "message": f"Email length is unusual ({style['word_count']} words vs typical {int(avg_word_count)})",
                "severity": "medium",
            })
            deviation += 20

    if abs(style["avg_sentence_len"] - profile["avg_sentence_len"]) > 8:
        flags.append({
            "type": "sentence_style",
            "message": "Writing style differs from this sender's usual pattern",
            "severity": "medium",
        })
        deviation += 15

    if style["caps_ratio"] - profile["avg_caps_ratio"] > 0.05:
        flags.append({
            "type": "caps",
            "message": "Unusually high use of capital letters",
            "severity": "low",
        })
        deviation += 10

    formality_diff = abs(style["formality"] - profile["avg_formality"])
    if formality_diff > 0.01:
        flags.append({
            "type": "formality",
            "message": "Tone is more "
            + ("formal" if style["formality"] > profile["avg_formality"] else "casual")
            + " than usual",
            "severity": "low",
        })
        deviation += 10

    if style["greeting"] != profile["typical_greeting"] and profile["typical_greeting"] != "none":
        flags.append({
            "type": "greeting",
            "message": f"Greeting style changed (usually '{profile['typical_greeting']}', now '{style['greeting']}')",
            "severity": "low",
        })
        deviation += 10

    if profile["usually_has_signature"] and not style["has_signature"]:
        flags.append({
            "type": "signature",
            "message": "Email signature is missing (this sender usually includes one)",
            "severity": "medium",
        })
        deviation += 15

    hour = _extract_send_hour(email.get("receivedDateTime", ""))
    if hour is not None and profile["typical_send_hours"]:
        min_dist = min(abs(hour - h) for h in profile["typical_send_hours"])
        min_dist = min(min_dist, 24 - min_dist)
        if min_dist > 6:
            flags.append({
                "type": "send_time",
                "message": f"Sent at an unusual time ({hour}:00 - typically active around {profile['typical_send_hours'][0]}:00)",
                "severity": "low",
            })
            deviation += 10

    if style["exclamation_count"] > profile["avg_exclamations"] + 3:
        flags.append({
            "type": "urgency",
            "message": "More exclamation marks than usual (possible urgency manipulation)",
            "severity": "medium",
        })
        deviation += 15

    deviation = min(100, deviation)
    status = "suspicious" if deviation >= 50 else "minor_deviation" if deviation >= 25 else "matches"
    return {
        "score": deviation,
        "flags": flags,
        "status": status,
        "profile_emails": profile["email_count"],
    }


LOG_FILE = USER_DATA_DIR / "phishguard_log.txt"


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
_SUSPICIOUS_TLDS = frozenset(
    t if str(t).startswith(".") else "." + str(t)
    for t in (_DETECTION_RULES.get("risky_tlds") or _SUSPICIOUS_TLDS)
)

_FREEMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "ymail.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "zoho.com", "mail.com", "gmx.com",
    "gmx.net", "fastmail.com", "tutanota.com",
})

# Cache for RDAP / DNS lookups.
_domain_rep_cache = {}
_domain_rep_cache_lock = threading.Lock()
_domain_age_cache = {}  # registrable domain -> {"days": int|None, "at": datetime}
_domain_age_lock = threading.Lock()
_DOMAIN_AGE_OK_TTL = 7 * 24 * 3600
_DOMAIN_AGE_FAIL_TTL = 3600


def _get_domain_age(domain):
    """Return domain age in days via RDAP, or None if unavailable.

    RDAP is HTTPS JSON and avoids the optional WHOIS dependency. The
    lookup is best-effort; failures briefly cache as None so scans keep moving.
    """
    reg = _reg_domain(domain)
    if not reg or "." not in reg or not http_requests:
        return None

    now = datetime.now()
    with _domain_age_lock:
        cached = _domain_age_cache.get(reg)
        if cached:
            ttl = _DOMAIN_AGE_OK_TTL if cached["days"] is not None else _DOMAIN_AGE_FAIL_TTL
            if (now - cached["at"]).total_seconds() < ttl:
                return cached["days"]

    days = None
    try:
        from datetime import timezone
        resp = http_requests.get(
            "https://rdap.org/domain/" + reg,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "PhishGuard/2.0"},
        )
        if resp.status_code == 200:
            for event in (resp.json().get("events") or []):
                action = str(event.get("eventAction", "")).lower()
                if action not in ("registration", "registered"):
                    continue
                raw = str(event.get("eventDate", "")).strip()
                if not raw:
                    break
                timestamp = raw.replace("Z", "+00:00")
                try:
                    created = datetime.fromisoformat(timestamp)
                except Exception:
                    created = datetime.fromisoformat(timestamp[:19] + "+00:00")
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days = max(0, (datetime.now(timezone.utc) - created).days)
                break
    except Exception:
        days = None

    with _domain_age_lock:
        _domain_age_cache[reg] = {"days": days, "at": now}
    return days


def _check_dns(domain):
    """Check if domain resolves via DNS. Returns True/False/None."""
    try:
        socket.getaddrinfo(domain, None, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except socket.gaierror:
        return False
    except Exception:
        return None


_PROVIDER_DEFINITIONS = {
    "virustotal": {
        "name": "VirusTotal",
        "privacy": (
            "Receives extracted URLs and SHA-256 attachment hashes. Attachment "
            "contents are uploaded only after a separate confirmation."
        ),
    },
    "google_safe_browsing": {
        "name": "Google Safe Browsing",
        "privacy": (
            "Receives 4-byte SHA-256 prefixes derived from URLs in the message; "
            "PhishGuard does not send the complete URL in this lookup."
        ),
    },
    "urlscan": {
        "name": "urlscan.io",
        "privacy": (
            "Receives sender/link domains for search. Unfamiliar link hosts may "
            "be submitted as an unlisted URL for a live scan."
        ),
    },
    "abuseipdb": {
        "name": "AbuseIPDB",
        "privacy": "Receives the public originating IP address from message headers.",
    },
}

VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "").strip()
GSB_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_KEY", "").strip()
URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "").strip()
ABUSEIPDB_API_KEY = os.environ.get("ABUSEIPDB_KEY", "").strip()
_DEPLOYER_PROVIDER_KEYS = {
    "virustotal": VIRUSTOTAL_API_KEY,
    "google_safe_browsing": GSB_API_KEY,
    "urlscan": URLSCAN_API_KEY,
    "abuseipdb": ABUSEIPDB_API_KEY,
}
_PROVIDER_KEY_SOURCES = {
    provider_id: ("deployer" if key else "none")
    for provider_id, key in _DEPLOYER_PROVIDER_KEYS.items()
}
_provider_config_lock = threading.RLock()
_provider_enabled = {provider_id: True for provider_id in _PROVIDER_DEFINITIONS}
_provider_runtime = {
    provider_id: {
        "status": "enabled" if _DEPLOYER_PROVIDER_KEYS[provider_id] else "not_configured",
        "reason": "",
        "updated_at": None,
    }
    for provider_id in _PROVIDER_DEFINITIONS
}


def _provider_key(provider_id):
    return {
        "virustotal": VIRUSTOTAL_API_KEY,
        "google_safe_browsing": GSB_API_KEY,
        "urlscan": URLSCAN_API_KEY,
        "abuseipdb": ABUSEIPDB_API_KEY,
    }.get(provider_id, "")


def _set_provider_key(provider_id, value):
    global VIRUSTOTAL_API_KEY, GSB_API_KEY, URLSCAN_API_KEY, ABUSEIPDB_API_KEY
    if provider_id == "virustotal":
        VIRUSTOTAL_API_KEY = value
    elif provider_id == "google_safe_browsing":
        GSB_API_KEY = value
    elif provider_id == "urlscan":
        URLSCAN_API_KEY = value
    elif provider_id == "abuseipdb":
        ABUSEIPDB_API_KEY = value


def _provider_can_run(provider_id):
    with _provider_config_lock:
        return bool(
            _provider_enabled.get(provider_id)
            and _provider_key(provider_id)
            and http_requests
        )


def _record_provider_status(provider_id, status, reason=""):
    if provider_id not in _PROVIDER_DEFINITIONS:
        return
    if status not in {"enabled", "unavailable", "rate_limited"}:
        status = "unavailable"
    with _provider_config_lock:
        _provider_runtime[provider_id] = {
            "status": status,
            "reason": str(reason or "")[:80],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }


def _record_provider_http_status(provider_id, status_code):
    if status_code in (200, 201, 202, 204):
        _record_provider_status(provider_id, "enabled")
    elif status_code == 429:
        _record_provider_status(provider_id, "rate_limited", "Request quota reached")
    elif status_code in (401, 403):
        _record_provider_status(provider_id, "unavailable", "API key rejected or revoked")
    else:
        _record_provider_status(provider_id, "unavailable", "Provider request failed")


def _provider_public_statuses():
    statuses = []
    with _provider_config_lock:
        for provider_id, definition in _PROVIDER_DEFINITIONS.items():
            configured = bool(_provider_key(provider_id))
            enabled = bool(configured and _provider_enabled[provider_id])
            runtime = dict(_provider_runtime[provider_id])
            if not configured:
                status, reason = "not_configured", "No API key is configured"
            elif not enabled:
                status, reason = "disabled", "Disabled by the user"
            elif not http_requests:
                status, reason = "unavailable", "HTTP client is unavailable"
            else:
                status = runtime.get("status") or "enabled"
                reason = runtime.get("reason") or ""
                if status == "not_configured":
                    status = "enabled"
            statuses.append({
                "id": provider_id,
                "name": definition["name"],
                "configured": configured,
                "enabled": enabled,
                "status": status,
                "reason": reason,
                "key_source": _PROVIDER_KEY_SOURCES[provider_id],
                "privacy": definition["privacy"],
                "updated_at": runtime.get("updated_at"),
            })
    return statuses


def _clear_provider_cache(provider_id):
    if provider_id == "virustotal":
        with _vt_cache_lock:
            _vt_cache.clear()
        with _vt_url_cache_lock:
            _vt_url_cache.clear()
    elif provider_id == "google_safe_browsing":
        with _gsb_cache_lock:
            _gsb_cache.clear()
    elif provider_id == "urlscan":
        with _urlscan_cache_lock:
            _urlscan_cache.clear()


def _apply_provider_configuration(message):
    """Apply a validated command received from Electron's private stdin pipe."""
    providers = message.get("providers") if isinstance(message, dict) else None
    if not isinstance(providers, dict):
        return False
    changed = []
    with _provider_config_lock:
        for provider_id in _PROVIDER_DEFINITIONS:
            update = providers.get(provider_id)
            if not isinstance(update, dict):
                continue
            old_key = _provider_key(provider_id)
            old_enabled = _provider_enabled[provider_id]
            if isinstance(update.get("enabled"), bool):
                _provider_enabled[provider_id] = update["enabled"]
            if update.get("use_deployer_key") is True:
                _set_provider_key(provider_id, _DEPLOYER_PROVIDER_KEYS[provider_id])
                _PROVIDER_KEY_SOURCES[provider_id] = (
                    "deployer" if _DEPLOYER_PROVIDER_KEYS[provider_id] else "none"
                )
            elif isinstance(update.get("key"), str):
                key = update["key"].strip()
                if (
                    key
                    and len(key) <= 4096
                    and all(0x21 <= ord(ch) <= 0x7E for ch in key)
                ):
                    _set_provider_key(provider_id, key)
                    _PROVIDER_KEY_SOURCES[provider_id] = "user"
            if old_key != _provider_key(provider_id) or old_enabled != _provider_enabled[provider_id]:
                changed.append(provider_id)
            configured = bool(_provider_key(provider_id))
            _provider_runtime[provider_id] = {
                "status": (
                    "enabled" if configured and _provider_enabled[provider_id]
                    else "disabled" if configured
                    else "not_configured"
                ),
                "reason": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
    for provider_id in changed:
        _clear_provider_cache(provider_id)
    return True


def _provider_control_loop(stream):
    """Consume secret configuration without echoing commands or parse errors."""
    for raw_line in stream:
        if len(raw_line) > 32_768:
            continue
        try:
            message = json.loads(raw_line)
        except (TypeError, ValueError):
            continue
        if isinstance(message, dict) and message.get("type") == "provider_config":
            _apply_provider_configuration(message)


def _start_provider_control_channel():
    if os.environ.get("ELECTRON_MODE") != "1" or not sys.stdin:
        return
    thread = threading.Thread(
        target=_provider_control_loop,
        args=(sys.stdin,),
        name="provider-control",
        daemon=True,
    )
    thread.start()


VT_API_BASE = "https://www.virustotal.com/api/v3"
VT_CACHE_TTL_SECONDS = 24 * 3600
_vt_cache_lock = threading.Lock()
_vt_cache = {}
VT_URL_MAX_PER_EMAIL = 10
VT_URL_SUBMIT_MAX_PER_EMAIL = 3
VT_URL_SUBMIT_WAIT = 45
VT_URL_SUBMIT_POLL = 6
_vt_url_cache_lock = threading.Lock()
_vt_url_cache = {}

URLSCAN_CACHE_TTL_SECONDS = 12 * 3600
_urlscan_cache_lock = threading.Lock()
_urlscan_cache = {}
URLSCAN_VISIBILITY = "unlisted"
URLSCAN_LIVE_MAX_PER_EMAIL = 3
URLSCAN_LIVE_WAIT_SECONDS = 45
URLSCAN_LIVE_POLL_INTERVAL = 6

_runtime_dir = USER_DATA_DIR
_COMMUNITY_DB_PATH = _runtime_dir / "local_scan_history.json"
_community_db_lock = threading.Lock()


def _vt_lookup_hash(sha256):
    """Look up a SHA-256 file hash on VirusTotal without uploading content."""
    if not _provider_can_run("virustotal"):
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
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Network request failed")
        return {"error": "VirusTotal unavailable", "status": 0}

    if resp.status_code == 404:
        _record_provider_status("virustotal", "enabled")
    else:
        _record_provider_http_status("virustotal", resp.status_code)
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
        except Exception:
            _record_provider_status("virustotal", "unavailable", "Invalid provider response")
            return {"error": "VirusTotal returned an invalid response", "status": 200}
    elif resp.status_code == 401:
        return {"error": "VT api key invalid or revoked", "status": 401}
    elif resp.status_code == 429:
        return {"error": "VT rate limit hit; try again in a minute", "status": 429}
    else:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}

    with _vt_cache_lock:
        _vt_cache[sha256] = {"result": result, "cached_at": datetime.now()}
    return result


VT_MAX_FILES_PER_SCAN = 5
VT_MAX_FILE_BYTES = 32 * 1024 * 1024


def _vt_upload_file(content_bytes, filename="file"):
    """Upload file bytes to VirusTotal for an explicit user-requested scan."""
    if not _provider_can_run("virustotal"):
        return None
    if not content_bytes:
        return {"error": "no file bytes", "status": 0}
    if len(content_bytes) > VT_MAX_FILE_BYTES:
        return {
            "error": f"file too large to upload (> {VT_MAX_FILE_BYTES // (1024 * 1024)} MB)",
            "status": 413,
        }
    try:
        resp = http_requests.post(
            f"{VT_API_BASE}/files",
            headers={
                "x-apikey": VIRUSTOTAL_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            files={"file": (filename or "file", content_bytes)},
            timeout=(5, 60),
        )
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Network request failed")
        return {"error": "VirusTotal unavailable", "status": 0}
    _record_provider_http_status("virustotal", resp.status_code)
    if resp.status_code == 401:
        return {"error": "VT api key invalid or revoked", "status": 401}
    if resp.status_code == 429:
        return {"error": "VT rate limit hit; try again in a minute", "status": 429}
    if resp.status_code not in (200, 201):
        return {"error": f"http {resp.status_code}", "status": resp.status_code}
    try:
        analysis_id = resp.json().get("data", {}).get("id", "")
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Invalid provider response")
        return {"error": "VirusTotal returned an invalid response", "status": 200}
    if not analysis_id:
        return {"error": "no analysis id returned", "status": 200}
    return {"analysis_id": analysis_id}


def _vt_poll_analysis(analysis_id):
    """Poll a VirusTotal file analysis and return a normalized verdict."""
    if not _provider_can_run("virustotal") or not analysis_id:
        return None
    try:
        resp = http_requests.get(
            f"{VT_API_BASE}/analyses/{analysis_id}",
            headers={
                "x-apikey": VIRUSTOTAL_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Network request failed")
        return {"error": "VirusTotal unavailable", "status": 0}
    _record_provider_http_status("virustotal", resp.status_code)
    if resp.status_code == 429:
        return {"error": "VT rate limit hit; try again in a minute", "status": 429}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}
    try:
        payload = resp.json()
        attrs = payload.get("data", {}).get("attributes", {}) or {}
        status = attrs.get("status", "queued")
        sha = ((payload.get("meta", {}) or {}).get("file_info", {}) or {}).get("sha256", "")
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Invalid provider response")
        return {"error": "VirusTotal returned an invalid response", "status": 200}
    if status != "completed":
        return {"status": status}
    stats = attrs.get("stats", {}) or {}
    return {
        "status": "completed",
        "found": True,
        "sha256": sha,
        "malicious": int(stats.get("malicious", 0) or 0),
        "suspicious": int(stats.get("suspicious", 0) or 0),
        "harmless": int(stats.get("harmless", 0) or 0),
        "undetected": int(stats.get("undetected", 0) or 0),
        "total": int(sum(v or 0 for v in stats.values())),
        "permalink": f"https://www.virustotal.com/gui/file/{sha}" if sha else "",
    }


def _check_virustotal_url(url):
    """Look up exact URL reputation on VirusTotal v3."""
    if not _provider_can_run("virustotal") or not url:
        return None
    with _vt_url_cache_lock:
        cached = _vt_url_cache.get(url)
        if cached and (datetime.now() - cached["cached_at"]).total_seconds() < VT_CACHE_TTL_SECONDS:
            return cached["result"]

    url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").strip("=")
    try:
        resp = http_requests.get(
            f"{VT_API_BASE}/urls/{url_id}",
            headers={
                "x-apikey": VIRUSTOTAL_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Network request failed")
        return {"error": "VirusTotal unavailable", "status": 0}

    if resp.status_code == 404:
        _record_provider_status("virustotal", "enabled")
    else:
        _record_provider_http_status("virustotal", resp.status_code)
    if resp.status_code == 404:
        result = {"found": False, "malicious": 0, "suspicious": 0, "total": 0}
    elif resp.status_code == 200:
        try:
            attrs = resp.json().get("data", {}).get("attributes", {}) or {}
            stats = attrs.get("last_analysis_stats", {}) or {}
            result = {
                "found": True,
                "malicious": int(stats.get("malicious", 0) or 0),
                "suspicious": int(stats.get("suspicious", 0) or 0),
                "harmless": int(stats.get("harmless", 0) or 0),
                "undetected": int(stats.get("undetected", 0) or 0),
                "total": int(sum(v or 0 for v in stats.values())),
            }
        except Exception:
            _record_provider_status("virustotal", "unavailable", "Invalid provider response")
            return {"error": "VirusTotal returned an invalid response", "status": 200}
    elif resp.status_code in (401, 429):
        return {
            "error": "VT key invalid" if resp.status_code == 401 else "VT rate limit",
            "status": resp.status_code,
        }
    else:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}

    with _vt_url_cache_lock:
        _vt_url_cache[url] = {"result": result, "cached_at": datetime.now()}
    return result


def _vt_submit_url(url):
    """Submit a URL to VirusTotal for a fresh scan."""
    if not _provider_can_run("virustotal") or not url:
        return None
    try:
        resp = http_requests.post(
            f"{VT_API_BASE}/urls",
            headers={
                "x-apikey": VIRUSTOTAL_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            data={"url": url},
            timeout=(5, 30),
        )
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Network request failed")
        return {"error": "VirusTotal unavailable", "status": 0}
    _record_provider_http_status("virustotal", resp.status_code)
    if resp.status_code in (401, 429):
        return {
            "error": "VT key invalid" if resp.status_code == 401 else "VT rate limit",
            "status": resp.status_code,
        }
    if resp.status_code not in (200, 201):
        return {"error": f"http {resp.status_code}", "status": resp.status_code}
    try:
        analysis_id = resp.json().get("data", {}).get("id", "")
    except Exception:
        _record_provider_status("virustotal", "unavailable", "Invalid provider response")
        return {"error": "VirusTotal returned an invalid response", "status": 200}
    return {"analysis_id": analysis_id} if analysis_id else {"error": "no analysis id", "status": 200}


def _vt_scan_url(url):
    """Submit a URL to VirusTotal and wait briefly for a completed verdict."""
    submitted = _vt_submit_url(url)
    if not isinstance(submitted, dict) or "analysis_id" not in submitted:
        return submitted if isinstance(submitted, dict) else None

    analysis_id = submitted["analysis_id"]
    waited = 0
    while waited < VT_URL_SUBMIT_WAIT:
        _time.sleep(VT_URL_SUBMIT_POLL)
        waited += VT_URL_SUBMIT_POLL
        result = _vt_poll_analysis(analysis_id)
        if result is None:
            return None
        if "error" in result:
            return result
        if result.get("status") != "completed":
            continue
        normalized = {
            "found": True,
            "malicious": int(result.get("malicious", 0) or 0),
            "suspicious": int(result.get("suspicious", 0) or 0),
            "harmless": int(result.get("harmless", 0) or 0),
            "undetected": int(result.get("undetected", 0) or 0),
            "total": int(result.get("total", 0) or 0),
        }
        with _vt_url_cache_lock:
            _vt_url_cache[url] = {"result": normalized, "cached_at": datetime.now()}
        return normalized
    return {"pending": True, "timeout": True}


def _check_urlscan(domain):
    """Query urlscan.io public search for prior scans of a domain."""
    if not _provider_can_run("urlscan") or not domain:
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
            params={"q": f"domain:{key}", "size": 20},
            headers={
                "API-Key": URLSCAN_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        _record_provider_status("urlscan", "unavailable", "Network request failed")
        return {"error": "urlscan.io unavailable", "status": 0}

    _record_provider_http_status("urlscan", resp.status_code)
    if resp.status_code == 401:
        return {"error": "urlscan key invalid", "status": 401}
    if resp.status_code == 429:
        return {"error": "urlscan rate limited", "status": 429}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}

    try:
        data = resp.json()
    except ValueError:
        _record_provider_status("urlscan", "unavailable", "Invalid provider response")
        return {"error": "not JSON"}

    items = data.get("results", []) or []
    if not items:
        result = {"found": False, "scan_count": 0, "malicious": False,
                  "verdict": "unknown"}
    else:
        malicious_count = 0
        max_score = 0
        brands, categories = set(), set()
        for item in items:
            overall = ((item.get("verdicts") or {}).get("overall") or {})
            if overall.get("malicious"):
                malicious_count += 1
            try:
                max_score = max(max_score, int(overall.get("score") or 0))
            except (TypeError, ValueError):
                pass
            for brand in (overall.get("brands") or []):
                brands.add(brand.get("name") if isinstance(brand, dict) else str(brand))
            for category in (overall.get("categories") or []):
                categories.add(str(category))
        if malicious_count > 0 or max_score >= 60:
            verdict = "malicious"
        elif max_score >= 30 or brands:
            verdict = "suspicious"
        else:
            verdict = "clean"
        latest = items[0]
        result = {
            "found": True,
            "scan_count": len(items),
            "malicious": verdict == "malicious",
            "verdict": verdict,
            "malicious_scans": malicious_count,
            "max_score": max_score,
            "brands": sorted(brands)[:5],
            "categories": sorted(categories)[:5],
            "latest_result": latest.get("result", ""),
            "latest_screenshot": latest.get("screenshot", ""),
        }

    with _urlscan_cache_lock:
        _urlscan_cache[key] = {"result": result, "cached_at": datetime.now()}
    return result


def _urlscan_submit(url):
    """Submit a URL to urlscan.io for a bounded live scan."""
    if not _provider_can_run("urlscan") or not url:
        return None
    try:
        resp = http_requests.post(
            "https://urlscan.io/api/v1/scan/",
            headers={
                "API-Key": URLSCAN_API_KEY,
                "Content-Type": "application/json",
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            json={"url": url, "visibility": URLSCAN_VISIBILITY},
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        _record_provider_status("urlscan", "unavailable", "Network request failed")
        return {"error": "urlscan.io unavailable", "status": 0}
    if resp.status_code == 400:
        _record_provider_status("urlscan", "enabled")
    else:
        _record_provider_http_status("urlscan", resp.status_code)
    if resp.status_code in (400, 401, 429):
        return {
            "error": {
                400: "urlscan rejected the URL",
                401: "urlscan key invalid",
                429: "urlscan submission rate limited",
            }[resp.status_code],
            "status": resp.status_code,
        }
    if resp.status_code not in (200, 201):
        return {"error": f"http {resp.status_code}", "status": resp.status_code}
    try:
        uuid = resp.json().get("uuid", "")
    except Exception:
        _record_provider_status("urlscan", "unavailable", "Invalid provider response")
        return {"error": "urlscan.io returned an invalid response", "status": 200}
    return {"uuid": uuid} if uuid else {"error": "no uuid returned", "status": 200}


def _urlscan_result(uuid):
    """Fetch a urlscan result by uuid and normalize it to the search verdict shape."""
    if not _provider_can_run("urlscan") or not uuid:
        return None
    try:
        resp = http_requests.get(
            f"https://urlscan.io/api/v1/result/{uuid}/",
            headers={
                "API-Key": URLSCAN_API_KEY,
                "User-Agent": "PhishGuard/2.0 (+phishguard-capstone)",
            },
            timeout=HTTP_TIMEOUT,
        )
    except Exception:
        _record_provider_status("urlscan", "unavailable", "Network request failed")
        return {"error": "urlscan.io unavailable", "status": 0}
    if resp.status_code == 404:
        _record_provider_status("urlscan", "enabled")
    else:
        _record_provider_http_status("urlscan", resp.status_code)
    if resp.status_code == 404:
        return {"pending": True}
    if resp.status_code != 200:
        return {"error": f"http {resp.status_code}", "status": resp.status_code}
    try:
        overall = ((resp.json().get("verdicts") or {}).get("overall") or {})
        score = int(overall.get("score") or 0)
        malicious = bool(overall.get("malicious"))
        brands = sorted({
            brand.get("name") if isinstance(brand, dict) else str(brand)
            for brand in (overall.get("brands") or [])
        })
        categories = sorted({str(cat) for cat in (overall.get("categories") or [])})
    except Exception:
        _record_provider_status("urlscan", "unavailable", "Invalid provider response")
        return {"error": "urlscan.io returned an invalid response", "status": 200}
    verdict = ("malicious" if (malicious or score >= 60)
               else "suspicious" if (score >= 30 or brands) else "clean")
    return {
        "found": True,
        "live": True,
        "verdict": verdict,
        "malicious": verdict == "malicious",
        "scan_count": 1,
        "max_score": score,
        "brands": brands[:5],
        "categories": categories[:5],
    }


def _urlscan_live_scan(url, host=""):
    """Submit a host URL to urlscan and wait briefly for a domain-level verdict."""
    submitted = _urlscan_submit(url)
    if not isinstance(submitted, dict) or "uuid" not in submitted:
        return submitted if isinstance(submitted, dict) else None

    uuid = submitted["uuid"]
    waited = 0
    while waited < URLSCAN_LIVE_WAIT_SECONDS:
        _time.sleep(URLSCAN_LIVE_POLL_INTERVAL)
        waited += URLSCAN_LIVE_POLL_INTERVAL
        result = _urlscan_result(uuid)
        if result is None:
            return None
        if "error" in result:
            return result
        if result.get("pending"):
            continue
        if host:
            with _urlscan_cache_lock:
                _urlscan_cache[host] = {"result": result, "cached_at": datetime.now()}
        return result
    return {"pending": True, "timeout": True}


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


GSB_CACHE_TTL_SECONDS = 6 * 3600
GSB_MAX_URLS = 500
GSB_MAX_PREFIXES = 1000
_gsb_cache_lock = threading.Lock()
_gsb_cache = {}
_GSB_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_GSB_THREAT_TYPES = {
    1: "MALWARE",
    2: "SOCIAL_ENGINEERING",
    3: "UNWANTED_SOFTWARE",
    4: "POTENTIALLY_HARMFUL_APPLICATION",
}


def _gsb_canonicalize(url):
    """Canonicalize a URL for Google Safe Browsing hash expressions."""
    if not url:
        return ""
    text = url.strip().replace("\t", "").replace("\r", "").replace("\n", "")
    text = text.split("#", 1)[0]
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    host, path, query = (parts.hostname or ""), (parts.path or "/"), parts.query

    def unquote_all(value):
        previous = None
        while previous != value:
            previous, value = value, unquote(value)
        return value

    host, path = unquote_all(host), unquote_all(path)
    host = re.sub(r"\.{2,}", ".", host.strip(".")).lower()
    if not path.startswith("/"):
        path = "/" + path
    trailing = path.endswith("/")
    segments = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
        else:
            segments.append(segment)
    path = "/" + "/".join(segments) + ("/" if trailing and segments else "")
    if not path:
        path = "/"

    def escape(value):
        out = []
        for ch in value:
            code = ord(ch)
            if code <= 0x20 or code >= 0x7F or ch in ("#", "%"):
                out.append("".join(f"%{b:02X}" for b in ch.encode("utf-8")))
            else:
                out.append(ch)
        return "".join(out)

    canonical = escape(host) + escape(path)
    if query:
        canonical += "?" + escape(query)
    return canonical


def _gsb_expressions(url):
    """Return Safe Browsing host-suffix/path-prefix expressions for a URL."""
    canonical = _gsb_canonicalize(url)
    if not canonical:
        return []
    slash = canonical.find("/")
    host, pathq = (canonical, "/") if slash == -1 else (canonical[:slash], canonical[slash:])
    qpos = pathq.find("?")
    path, query = (pathq, "") if qpos == -1 else (pathq[:qpos], pathq[qpos + 1:])
    path = path or "/"

    if _GSB_IP_RE.match(host):
        hosts = [host]
    else:
        labels = host.split(".")
        hosts = [host]
        for i in range(max(0, len(labels) - 5), len(labels) - 1):
            candidate = ".".join(labels[i:])
            if candidate not in hosts and len(candidate.split(".")) >= 2:
                hosts.append(candidate)
        hosts = hosts[:5]

    paths = []
    if query:
        paths.append(path + "?" + query)
    paths.append(path)
    paths.append("/")
    acc, dirs = "", 0
    segments = [seg for seg in path.split("/") if seg]
    walk = segments if path.endswith("/") else segments[:-1]
    for segment in walk:
        acc += "/" + segment
        if acc + "/" not in paths:
            paths.append(acc + "/")
            dirs += 1
        if dirs >= 4:
            break

    seen_paths, deduped_paths = set(), []
    for path_item in paths:
        if path_item not in seen_paths:
            seen_paths.add(path_item)
            deduped_paths.append(path_item)
    paths = deduped_paths[:6]

    expressions, seen = [], set()
    for host_item in hosts:
        for path_item in paths:
            expr = host_item + path_item
            if expr not in seen:
                seen.add(expr)
                expressions.append(expr)
    return expressions[:30]


def _gsb_varint(data, pos):
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _gsb_parse_response(body):
    """Parse the small protobuf response shape returned by hashes:search."""
    results, pos, end = [], 0, len(body)
    try:
        while pos < end:
            tag, pos = _gsb_varint(body, pos)
            field, wire = tag >> 3, tag & 7
            if field == 1 and wire == 2:
                length, pos = _gsb_varint(body, pos)
                sub, pos = body[pos:pos + length], pos + length
                full_hash, threat, q = None, 0, 0
                while q < len(sub):
                    tag2, q = _gsb_varint(sub, q)
                    field2, wire2 = tag2 >> 3, tag2 & 7
                    if field2 == 1 and wire2 == 2:
                        length2, q = _gsb_varint(sub, q)
                        full_hash, q = sub[q:q + length2], q + length2
                    elif field2 == 2 and wire2 == 2:
                        length2, q = _gsb_varint(sub, q)
                        detail, q = sub[q:q + length2], q + length2
                        dpos = 0
                        while dpos < len(detail):
                            tag3, dpos = _gsb_varint(detail, dpos)
                            field3, wire3 = tag3 >> 3, tag3 & 7
                            if field3 == 1 and wire3 == 0:
                                threat, dpos = _gsb_varint(detail, dpos)
                            elif wire3 == 2:
                                length3, dpos = _gsb_varint(detail, dpos)
                                dpos += length3
                            elif wire3 == 0:
                                _, dpos = _gsb_varint(detail, dpos)
                            else:
                                break
                    elif wire2 == 2:
                        length2, q = _gsb_varint(sub, q)
                        q += length2
                    elif wire2 == 0:
                        _, q = _gsb_varint(sub, q)
                    else:
                        break
                if full_hash is not None:
                    results.append((full_hash, threat))
            elif wire == 2:
                length, pos = _gsb_varint(body, pos)
                pos += length
            elif wire == 0:
                _, pos = _gsb_varint(body, pos)
            elif wire == 5:
                pos += 4
            elif wire == 1:
                pos += 8
            else:
                break
    except Exception:
        pass
    return results


def _check_safe_browsing(urls):
    """Google Safe Browsing v5 hash-prefix lookup for a batch of URLs."""
    if not _provider_can_run("google_safe_browsing") or not urls:
        return {}
    now = datetime.now()
    to_query, cached_hits, seen = [], {}, set()
    with _gsb_cache_lock:
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            cached = _gsb_cache.get(url)
            if cached and (now - cached["cached_at"]).total_seconds() < GSB_CACHE_TTL_SECONDS:
                if cached["threat"]:
                    cached_hits[url] = cached["threat"]
            else:
                to_query.append(url)
    to_query = to_query[:GSB_MAX_URLS]
    if not to_query:
        return cached_hits

    full_to_urls, prefixes = {}, set()
    for url in to_query:
        for expr in _gsb_expressions(url):
            digest = hashlib.sha256(expr.encode("utf-8")).digest()
            full_to_urls.setdefault(digest, set()).add(url)
            prefixes.add(digest[:4])
            if len(prefixes) >= GSB_MAX_PREFIXES:
                break
    if not prefixes:
        return cached_hits

    prefixes_b64 = [
        base64.b64encode(prefix).decode("ascii")
        for prefix in list(prefixes)[:GSB_MAX_PREFIXES]
    ]
    try:
        resp = http_requests.get(
            "https://safebrowsing.googleapis.com/v5/hashes:search",
            params={"key": GSB_API_KEY, "hashPrefixes": prefixes_b64},
            timeout=HTTP_TIMEOUT,
        )
        body = resp.content if resp.status_code == 200 else b""
    except Exception:
        _record_provider_status(
            "google_safe_browsing", "unavailable", "Network request failed"
        )
        return cached_hits
    _record_provider_http_status("google_safe_browsing", resp.status_code)
    if resp.status_code != 200:
        return cached_hits

    matched = {}
    for full_hash, threat_int in _gsb_parse_response(body):
        owners = full_to_urls.get(full_hash)
        if not owners:
            continue
        threat = _GSB_THREAT_TYPES.get(threat_int, "THREAT")
        for owner in owners:
            matched[owner] = threat

    with _gsb_cache_lock:
        for url in to_query:
            _gsb_cache[url] = {"threat": matched.get(url), "cached_at": now}
    hits = dict(cached_hits)
    hits.update(matched)
    return hits


_IPV4_RE = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")


def _extract_sender_ip(headers):
    """Return a public originating sender IP from message headers."""
    import ipaddress

    header_map = _hdr_dict(headers)
    if not header_map:
        return None

    def is_public(ip):
        try:
            addr = ipaddress.ip_address(ip)
            return not (
                addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast
            )
        except ValueError:
            return False

    for source in (
        header_map.get("x-sender-ip"),
        header_map.get("x-ms-exchange-crosstenant-originalattributedtenantconnectingip"),
        header_map.get("x-originating-ip"),
        header_map.get("received"),
    ):
        if not source:
            continue
        for candidate in _IPV4_RE.findall(source):
            if is_public(candidate):
                return candidate
    return None


def _check_abuseipdb_ip(ip):
    """AbuseIPDB reputation for the originating mail-sender IP."""
    if not _provider_can_run("abuseipdb") or not ip:
        return None
    try:
        resp = http_requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        _record_provider_http_status("abuseipdb", resp.status_code)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            score = data.get("abuseConfidenceScore", 0)
            return {"ip": ip, "score": score, "is_abusive": score > 50}
        return {
            "error": "AbuseIPDB unavailable",
            "status": resp.status_code,
        }
    except Exception:
        _record_provider_status("abuseipdb", "unavailable", "Network request failed")
        return {"error": "AbuseIPDB unavailable", "status": 0}


def _check_abuseipdb(domain):
    """AbuseIPDB IP reputation lookup; enabled only when ABUSEIPDB_KEY exists."""
    if not _provider_can_run("abuseipdb"):
        return None
    try:
        ips = socket.getaddrinfo(domain, None, socket.AF_INET)
        ip = ips[0][4][0] if ips else None
        if not ip:
            return None
        resp = http_requests.get(
            "https://api.abuseipdb.com/api/v2/check",
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
            timeout=HTTP_TIMEOUT,
        )
        _record_provider_http_status("abuseipdb", resp.status_code)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            score = data.get("abuseConfidenceScore", 0)
            return {"score": score, "is_abusive": score > 50}
        return {"error": "AbuseIPDB unavailable", "status": resp.status_code}
    except Exception:
        _record_provider_status("abuseipdb", "unavailable", "Network request failed")
        return {"error": "AbuseIPDB unavailable", "status": 0}


def _check_domain_reputation(domain):
    """
    Compute a worldwide reputation score for a domain (0.0-1.0).
    Uses multiple heuristics without requiring API keys:
      - Trusted domain list match
      - TLD risk analysis
    - Domain age via RDAP (if available)
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
    _brands = list(_BRAND_NAME_TOKENS.keys()) or [
        "paypal", "apple", "google", "microsoft", "amazon", "chase",
        "wellsfargo", "bankofamerica", "netflix", "facebook", "instagram",
    ]
    for brand in _brands:
        if brand in name_part and domain not in _TRUSTED_DOMAINS:
            score -= 0.25
            signals.append(("impersonation", f"Contains '{brand}' but isn't the real domain"))
            break

    # 4. Domain age via RDAP
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


# OpenPhish community feed: free active-phishing URL list, fetched in bulk.
# Matching is exact-URL only and happens locally, so individual email links are
# never submitted to OpenPhish. Host-level matching would be too broad because
# phishing pages are often hosted on shared legitimate infrastructure.
OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
OPENPHISH_CACHE_TTL_SECONDS = 6 * 3600
OPENPHISH_RETRY_SECONDS = 600
_openphish_lock = threading.Lock()
_openphish = {"urls": frozenset(), "fetched_at": None, "ok": False}


def _openphish_urls():
    """Return cached OpenPhish URLs, refreshing when stale. Best-effort."""
    if not http_requests:
        return _openphish["urls"]

    with _openphish_lock:
        now = datetime.now()
        last = _openphish["fetched_at"]
        ttl = OPENPHISH_CACHE_TTL_SECONDS if _openphish["ok"] else OPENPHISH_RETRY_SECONDS
        if last and (now - last).total_seconds() < ttl:
            return _openphish["urls"]

        _openphish["fetched_at"] = now
        try:
            resp = http_requests.get(OPENPHISH_FEED_URL, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200 and resp.text:
                urls = frozenset(
                    line.strip()
                    for line in resp.text.splitlines()
                    if line.strip().startswith("http")
                )
                if urls:
                    _openphish["urls"] = urls
                    _openphish["ok"] = True
                    print(f"OpenPhish feed refreshed: {len(urls)} active URLs")
                else:
                    _openphish["ok"] = False
            else:
                _openphish["ok"] = False
        except Exception as exc:
            _openphish["ok"] = False
            print(
                f"OpenPhish feed refresh failed ({exc}); "
                f"using {len(_openphish['urls'])} cached URLs"
            )
        return _openphish["urls"]


def _check_openphish(url):
    """Return True if the exact URL appears in the active OpenPhish feed."""
    feed = _openphish_urls()
    if not feed:
        return None
    candidate = (url or "").strip()
    if not candidate:
        return None
    return (
        candidate in feed
        or candidate.rstrip("/") in feed
        or (candidate + "/") in feed
    )


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


_PROVIDER_OUTCOME_PRIORITY = {
    "not_checked": 0,
    "not_found": 1,
    "not_flagged": 2,
    "provider_unavailable": 3,
    # Preserve a positive threat match even if a later indicator hits a
    # provider outage or rate limit.
    "flagged": 4,
}


def _record_scan_provider_outcome(results, provider_id, outcome, provider_result=None):
    """Aggregate a provider's per-indicator results into one explicit outcome."""
    if provider_id not in _PROVIDER_DEFINITIONS:
        return
    if isinstance(provider_result, dict) and provider_result.get("error"):
        outcome = "provider_unavailable"
    runtime = next(
        item for item in _provider_public_statuses() if item["id"] == provider_id
    )
    if runtime["status"] in {"unavailable", "rate_limited"}:
        outcome = "provider_unavailable"
    if outcome not in _PROVIDER_OUTCOME_PRIORITY:
        outcome = "not_checked"
    current = results["provider_outcomes"][provider_id]["outcome"]
    if _PROVIDER_OUTCOME_PRIORITY[outcome] >= _PROVIDER_OUTCOME_PRIORITY[current]:
        results["provider_outcomes"][provider_id] = {
            "outcome": outcome,
            "reason": runtime["reason"] if outcome == "provider_unavailable" else "",
        }


def run_threat_intel(email, urls_in_email=None):
    """Run public URL/domain checks before the AI model verdict is finalized."""
    domain = _sender_domain_from_email(email)
    results = {
        "domain": domain,
        "checks": {},
        "confirmed_phishing": False,
        "signals": [],
        "url_threats": {},
        "provider_outcomes": {
            provider_id: {"outcome": "not_checked", "reason": ""}
            for provider_id in _PROVIDER_DEFINITIONS
        },
    }

    if urls_in_email:
        gsb_hits = _check_safe_browsing(urls_in_email)
        if _provider_can_run("google_safe_browsing"):
            _record_scan_provider_outcome(
                results,
                "google_safe_browsing",
                "flagged" if gsb_hits else "not_flagged",
            )
        seen_hosts = set()
        urlhaus_calls = 0
        urlscan_live_submits = 0
        urlscan_rate_limited = False
        vt_url_calls = 0
        vt_submits = 0
        vt_rate_limited = False

        for raw_url in urls_in_email[:_MAX_URLS_PER_EMAIL]:
            url = (raw_url or "").strip()
            if not url:
                continue
            try:
                host = (urlparse(url).hostname or "").lower()
            except Exception:
                host = ""

            if url in gsb_hits and url not in results["url_threats"]:
                threat_type = str(gsb_hits[url]).replace("_", " ").title()
                results["url_threats"][url] = {
                    "malicious": True,
                    "sources": [f"Google Safe Browsing ({threat_type})"],
                }
                results["confirmed_phishing"] = True
                results["signals"].append(
                    f"Google Safe Browsing: a link is flagged as {threat_type} ({url})"
                )

            # OpenPhish exact-URL hit. The feed is downloaded in bulk and
            # matched locally, so this check does not submit user links.
            if url not in results["url_threats"] and _check_openphish(url) is True:
                results["url_threats"][url] = {
                    "malicious": True,
                    "sources": ["OpenPhish"],
                }
                results["confirmed_phishing"] = True
                results["signals"].append(
                    f"OpenPhish: a link is on the active phishing feed ({url})"
                )

            existing = results["url_threats"].get(url)
            if (
                vt_url_calls < VT_URL_MAX_PER_EMAIL
                and not vt_rate_limited
                and not (isinstance(existing, dict) and existing.get("malicious"))
                and host
                and _link_domain_untrusted(host)
            ):
                vt_url_calls += 1
                vt_result = _check_virustotal_url(url)
                if isinstance(vt_result, dict):
                    _record_scan_provider_outcome(
                        results,
                        "virustotal",
                        (
                            "not_found" if vt_result.get("found") is False
                            else "flagged" if (
                                vt_result.get("malicious", 0)
                                or vt_result.get("suspicious", 0)
                            )
                            else "not_flagged"
                        ),
                        vt_result,
                    )
                if isinstance(vt_result, dict) and vt_result.get("status") == 429:
                    vt_rate_limited = True
                if (
                    isinstance(vt_result, dict)
                    and vt_result.get("found") is False
                    and vt_submits < VT_URL_SUBMIT_MAX_PER_EMAIL
                    and not vt_rate_limited
                ):
                    vt_submits += 1
                    scanned = _vt_scan_url(url)
                    if (
                        isinstance(scanned, dict)
                        and ("error" in scanned or scanned.get("found") is not None)
                    ):
                        _record_scan_provider_outcome(
                            results,
                            "virustotal",
                            (
                                "not_found" if scanned.get("found") is False
                                else "flagged" if (
                                    scanned.get("malicious", 0)
                                    or scanned.get("suspicious", 0)
                                )
                                else "not_flagged"
                            ),
                            scanned,
                        )
                    if isinstance(scanned, dict) and scanned.get("found"):
                        vt_result = scanned
                    elif isinstance(scanned, dict) and scanned.get("status") == 429:
                        vt_rate_limited = True
                if isinstance(vt_result, dict) and vt_result.get("found") is not None:
                    entry = results["url_threats"].get(url)
                    entry = entry if isinstance(entry, dict) else {"malicious": False}
                    if vt_result.get("found") and vt_result.get("malicious", 0) >= 1:
                        entry["malicious"] = True
                        entry.setdefault("sources", []).append("VirusTotal")
                        entry["vt"] = "malicious"
                        results["confirmed_phishing"] = True
                        results["signals"].append(
                            f"VirusTotal: {vt_result['malicious']} engines flag this link as malicious ({url})"
                        )
                    elif vt_result.get("found") and vt_result.get("suspicious", 0) >= 2:
                        entry["vt"] = "suspicious"
                    elif vt_result.get("found"):
                        entry["vt"] = "clean"
                    else:
                        entry["vt"] = "unknown"
                    results["url_threats"][url] = entry

            if (
                host
                and host not in seen_hosts
                and host not in results["url_threats"]
                and urlhaus_calls < 10
            ):
                seen_hosts.add(host)
                urlhaus_calls += 1
                sources = []
                suspicious_note = None
                urlscan_clean = False
                if _check_urlhaus(host) is True:
                    sources.append("URLhaus")
                us_host = _check_urlscan(host)
                if isinstance(us_host, dict):
                    _record_scan_provider_outcome(
                        results,
                        "urlscan",
                        (
                            "not_found" if us_host.get("found") is False
                            else "flagged" if (
                                us_host.get("malicious")
                                or us_host.get("verdict") == "suspicious"
                            )
                            else "not_flagged"
                        ),
                        us_host,
                    )
                    if us_host.get("malicious"):
                        sources.append("urlscan.io")
                    elif us_host.get("verdict") == "suspicious":
                        brand_note = (
                            " impersonating " + ", ".join(us_host["brands"])
                            if us_host.get("brands") else ""
                        )
                        suspicious_note = (
                            "urlscan.io: prior scans of this link's host "
                            f"scored it risky (score {us_host.get('max_score', 0)}{brand_note})"
                        )
                    elif us_host.get("verdict") == "clean":
                        urlscan_clean = True
                    elif us_host.get("found") is False:
                        if (
                            _link_domain_untrusted(host)
                            and not sources
                            and not urlscan_rate_limited
                            and urlscan_live_submits < URLSCAN_LIVE_MAX_PER_EMAIL
                        ):
                            urlscan_live_submits += 1
                            try:
                                scheme = urlparse(url).scheme or "https"
                            except Exception:
                                scheme = "https"
                            live = _urlscan_live_scan(f"{scheme}://{host}", host)
                            if isinstance(live, dict):
                                if "error" in live or live.get("verdict"):
                                    _record_scan_provider_outcome(
                                        results,
                                        "urlscan",
                                        (
                                            "flagged" if live.get("verdict") in {
                                                "malicious", "suspicious"
                                            }
                                            else "not_flagged"
                                        ),
                                        live,
                                    )
                                if live.get("status") == 429:
                                    urlscan_rate_limited = True
                                elif live.get("verdict") == "malicious":
                                    sources.append("urlscan.io")
                                elif live.get("verdict") == "suspicious":
                                    brand_note = (
                                        " impersonating " + ", ".join(live["brands"])
                                        if live.get("brands") else ""
                                    )
                                    suspicious_note = (
                                        "urlscan.io live scan scored this link risky "
                                        f"(score {live.get('max_score', 0)}{brand_note})"
                                    )
                                elif live.get("verdict") == "clean":
                                    urlscan_clean = True
                                    results["signals"].append(
                                        f"urlscan.io: live scan of {host} found no threats"
                                    )

                if sources:
                    results["url_threats"][host] = {
                        "malicious": True,
                        "sources": sources,
                    }
                    results["confirmed_phishing"] = True
                    results["signals"].append(
                        f"{'/'.join(sources)}: a link host in this email is confirmed malicious ({host})"
                    )
                elif suspicious_note:
                    results["url_threats"][host] = {
                        "malicious": False,
                        "suspicious": True,
                        "sources": ["urlscan.io"],
                    }
                    results["signals"].append(suspicious_note)
                elif urlscan_clean:
                    results["url_threats"][host] = {
                        "malicious": False,
                        "urlscan_clean": True,
                    }

    if not domain:
        return results

    uh = _check_urlhaus(domain)
    results["checks"]["urlhaus"] = uh
    if uh is True:
        results["confirmed_phishing"] = True
        results["signals"].append("URLhaus: Domain hosts active malware/phishing URLs")

    aip = _check_abuseipdb_ip(_extract_sender_ip(email.get("internetMessageHeaders")))
    results["checks"]["abuseipdb"] = aip
    if isinstance(aip, dict):
        _record_scan_provider_outcome(
            results,
            "abuseipdb",
            "flagged" if aip.get("is_abusive") else "not_flagged",
            aip,
        )
    if aip and isinstance(aip, dict) and aip.get("is_abusive"):
        results["signals"].append(
            f"AbuseIPDB: sending IP {aip.get('ip')} has {aip['score']}% abuse confidence"
        )

    us = _check_urlscan(domain)
    results["checks"]["urlscan"] = us
    if isinstance(us, dict):
        _record_scan_provider_outcome(
            results,
            "urlscan",
            (
                "not_found" if us.get("found") is False
                else "flagged" if (
                    us.get("malicious") or us.get("verdict") == "suspicious"
                )
                else "not_flagged"
            ),
            us,
        )
    if us and isinstance(us, dict) and us.get("found"):
        brands = us.get("brands") or []
        brand_note = (" impersonating " + ", ".join(brands)) if brands else ""
        if us.get("verdict") == "malicious":
            results["confirmed_phishing"] = True
            results["signals"].append(
                f"urlscan.io: {us.get('malicious_scans', 0)} prior scan(s) flagged this domain as malicious{brand_note}"
            )
        elif us.get("verdict") == "suspicious":
            results["signals"].append(
                f"urlscan.io: prior scans scored this domain risky (score {us.get('max_score', 0)}{brand_note})"
            )
        else:
            results["signals"].append(
                f"urlscan.io: {us.get('scan_count', 0)} prior scans, none flagged malicious"
            )

    local_history = _community_check(domain)
    results["checks"]["local_history"] = local_history

    rep = _check_domain_reputation(domain)
    results["checks"]["domain_reputation"] = rep
    if rep["category"] == "suspicious":
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


class _HTMLLinkExtractor(HTMLParser):
    """Collect real destinations hidden in HTML attributes."""
    _URL_ATTRS = {
        "a": ("href",),
        "area": ("href",),
        "img": ("src",),
        "iframe": ("src",),
        "form": ("action",),
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = []

    def handle_starttag(self, tag, attrs):
        wanted = self._URL_ATTRS.get(tag.lower())
        if not wanted:
            return
        for name, value in attrs:
            if name.lower() in wanted and value:
                v = value.strip()
                if v.lower().startswith(("http://", "https://")):
                    self.urls.append(v)


def _extract_html_link_urls(raw_html, limit=_MAX_URLS_PER_EMAIL):
    """Return deduplicated http(s) URLs from href/src/action HTML attrs."""
    if not raw_html or "<" not in raw_html:
        return []
    try:
        parser = _HTMLLinkExtractor()
        parser.feed(raw_html)
    except Exception:
        return []
    seen, out = set(), []
    for url in parser.urls:
        url = url[:2048]
        if url not in seen:
            seen.add(url)
            out.append(url)
        if len(out) >= limit:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the SPA."""
    return send_from_directory(str(APP_ROOT / "templates"), "index.html")


@app.route("/api/csrf", methods=["GET"])
def get_csrf_token():
    """Compatibility token for the Electron renderer's secure fetch wrapper."""
    sess = _current_session()
    return jsonify({"csrf": sess.csrf_token, "launchSecret": bool(LAUNCH_SECRET)})


@app.route("/api/detection-rules", methods=["GET"])
def get_detection_rules():
    """Expose non-secret detection config needed by the renderer."""
    return jsonify({
        "trusted_domains": sorted(
            _DETECTION_RULES.get("trusted_domains")
            or ["company.com", "amazon.com", "chase.com", "atlassian.net"]),
        "brand_name_tokens": _DETECTION_RULES.get("brand_name_tokens") or _BRAND_NAME_TOKENS,
        "brand_domains": sorted(_DETECTION_RULES.get("brand_domains") or _BRAND_DOMAINS),
        "risky_tlds": sorted(_DETECTION_RULES.get("risky_tlds") or _RISKY_TLDS),
    })


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Return non-secret app and provider status for the renderer."""
    sess = _current_session()
    return jsonify({
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "graph_connected": bool(sess.graph_token["access_token"] and sess.live_mode),
        "logging_enabled": False,
        "user_data_dir": str(USER_DATA_DIR),
        "providers": _provider_public_statuses(),
    })


@app.route("/api/log", methods=["GET"])
def get_log():
    """Return the session log file contents."""
    if not _origin_ok() or not _launch_secret_ok():
        return jsonify({"error": "Forbidden"}), 403
    if not LOG_FILE.exists():
        return jsonify({"log": "No session activity recorded yet."})
    text = LOG_FILE.read_text(encoding="utf-8")
    return jsonify({"log": text})


@app.route("/api/log/download", methods=["GET"])
def download_log():
    """Download the session log as a text file."""
    if not _origin_ok() or not _launch_secret_ok():
        return jsonify({"error": "Forbidden"}), 403
    if not LOG_FILE.exists():
        return "No log file yet.", 404
    return send_from_directory(
        str(LOG_FILE.parent), LOG_FILE.name,
        as_attachment=True, mimetype="text/plain"
    )


@app.route("/api/reputation/<domain>", methods=["GET"])
def get_reputation(domain):
    """Return worldwide domain reputation."""
    if not _is_public_domain(domain):
        return jsonify({"error": "invalid domain"}), 400
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


@app.route("/api/community-stats", methods=["GET"])
def get_community_stats():
    """Return local scan history stats (per-device, informational only)."""
    db = _load_community_db()
    domains = db.get("domains", {})
    total_reports = sum(
        item.get("phishing", 0) + item.get("safe", 0)
        for item in domains.values()
    )
    phishing_domains = sum(
        1 for item in domains.values()
        if item.get("phishing", 0) > item.get("safe", 0)
    )
    return jsonify({
        "total_domains_tracked": len(domains),
        "total_scan_reports": total_reports,
        "known_phishing_domains": phishing_domains,
        "updated": db.get("updated", ""),
        "note": "Local scan history - shown for info only, does not affect verdicts.",
    })


@app.route("/api/report-sender", methods=["POST"])
@require_csrf
def report_sender():
    """Persist an explicit sender-domain report locally and, when signed in, to Supabase."""
    data = request.get_json(silent=True) or {}
    domain = (data.get("domain") or "").strip().lower()
    is_phishing = bool(data.get("is_phishing"))
    if not domain:
        return jsonify({"error": "domain required"}), 400
    if not _is_public_domain(domain):
        return jsonify({"error": "invalid domain"}), 400

    _community_report(domain, is_phishing, user_confirmed=True)

    jwt = _request_jwt()
    user_id = _jwt_user_id(jwt)
    if jwt and user_id:
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
            print(f"WARNING: threat_reports insert failed [{resp.status_code}]: {resp.text[:200]}")
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────────────────────
#  Microsoft Graph API Client
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_URL = "https://graph.microsoft.com/v1.0"

_external_oauth_lock = threading.Lock()
_external_oauth_pending = {}
_EXTERNAL_OAUTH_TTL_SECONDS = 600


def _graph_request(sess, endpoint, params=None):
    token = sess.graph_token
    if not token["access_token"]:
        raise ValueError("Not authenticated")
    if token["expiry"] and datetime.now() >= token["expiry"]:
        token["access_token"] = None
        token["expiry"] = None
        raise ValueError("Token expired")
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Content-Type": "application/json",
    }
    r = http_requests.get(
        f"{GRAPH_URL}/{endpoint}", headers=headers, params=params,
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code == 401:
        token["access_token"] = None
        token["expiry"] = None
        raise ValueError("Token expired")
    r.raise_for_status()
    return r.json()


def _fetch_folder_emails(sess, folder):
    """Fetch all emails from a mail folder via Graph pagination."""
    all_emails = []
    params = {
        "$top": 250,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,body,isRead,internetMessageHeaders,hasAttachments",
        "$orderby": "receivedDateTime desc",
    }
    result = _graph_request(sess, f"me/mailFolders/{folder}/messages", params)
    all_emails.extend(result.get("value", []))

    # Follow @odata.nextLink until no more pages
    next_link = result.get("@odata.nextLink")
    while next_link:
        token = sess.graph_token
        headers = {
            "Authorization": f"Bearer {token['access_token']}",
            "Content-Type": "application/json",
        }
        r = http_requests.get(next_link, headers=headers, timeout=HTTP_TIMEOUT)
        if r.status_code == 401:
            token["access_token"] = None
            token["expiry"] = None
            raise ValueError("Token expired")
        r.raise_for_status()
        data = r.json()
        all_emails.extend(data.get("value", []))
        next_link = data.get("@odata.nextLink")

    return all_emails


def _fetch_all_emails(sess):
    return _fetch_folder_emails(sess, "inbox")


def _fetch_junk_emails(sess):
    return _fetch_folder_emails(sess, "junkemail")


def _graph_post(sess, endpoint, json_data=None):
    token = sess.graph_token
    if not token["access_token"]:
        raise ValueError("Not authenticated")
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Content-Type": "application/json",
    }
    r = http_requests.post(
        f"{GRAPH_URL}/{endpoint}", headers=headers, json=json_data,
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code == 401:
        token["access_token"] = None
        token["expiry"] = None
        raise ValueError("Token expired")
    r.raise_for_status()
    return r


# ── OAuth Routes ─────────────────────────────────────────────────────────────

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Return current connection status."""
    sess = _current_session()
    connected = bool(sess.graph_token["access_token"] and sess.live_mode)
    return jsonify({
        "connected": connected,
        "live_mode": sess.live_mode,
        "user_name": sess.graph_user["name"],
        "user_email": sess.graph_user["email"],
    })


# Legacy direct-Microsoft OAuth routes (/api/auth/connect + /auth/callback)
# are intentionally removed. Microsoft sign-in is Supabase-brokered via the
# external-browser flow below, so the backend no longer depends on a fixed
# http://localhost:5050/auth/callback redirect URI.


@app.route("/api/auth/external-start", methods=["POST"])
@require_csrf
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
  var code = query.get('code') || hash.get('code') || '';
  var accessToken = hash.get('access_token') || '';
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
  if (!nonce || (!accessToken && !code)) {
    fail('Missing session data. You can close this tab and try again.');
    return;
  }

  fetch('/api/auth/external-deliver', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      nonce: nonce,
      code: code,
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
    """Receive Supabase OAuth redirect and deliver PKCE code/tokens to Flask."""
    return _EXTERNAL_CALLBACK_HTML


@app.route("/api/auth/external-deliver", methods=["POST"])
def auth_external_deliver():
    """Browser callback page posts Supabase PKCE code or tokens for Electron to poll."""
    data = request.get_json(silent=True) or {}
    def clean(value):
        return "" if value is None else str(value).strip()

    nonce = clean(data.get("nonce"))
    if not nonce:
        return jsonify({"error": "nonce required"}), 400

    access_token = clean(data.get("access_token"))
    code = clean(data.get("code"))
    if not access_token and not code:
        return jsonify({"error": "missing access_token or code"}), 400

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
            "code": code,
            "refresh_token": clean(data.get("refresh_token")),
            "provider_token": clean(data.get("provider_token")),
            "provider_refresh_token": clean(data.get("provider_refresh_token")),
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
@require_csrf
def auth_supabase_provider():
    """Use Supabase's Microsoft provider token as the Graph token."""
    sess = _current_session()

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

    sess.graph_token = {
        "access_token": provider_token,
        "expiry": datetime.now() + timedelta(seconds=max(60, expires_in)),
    }

    user_hint = data.get("user") or {}
    try:
        user_info = _graph_request(sess, "me")
        sess.graph_user["name"] = user_info.get("displayName") or user_hint.get("name") or "User"
        sess.graph_user["email"] = (
            user_info.get("mail")
            or user_info.get("userPrincipalName")
            or user_hint.get("email")
            or ""
        )
    except Exception:
        sess.graph_user["name"] = user_hint.get("name") or "Connected"
        sess.graph_user["email"] = user_hint.get("email") or ""

    try:
        sess.live_emails = _fetch_all_emails(sess)
        sess.junk_emails = _fetch_junk_emails(sess)
    except Exception as exc:
        sess.live_emails = []
        sess.junk_emails = []
        sess.live_mode = False
        return jsonify({"error": f"Could not fetch Outlook emails: {exc}"}), 502

    sess.live_mode = True
    sess.scan_results.clear()
    _log_session_event("LOGIN", {
        "User": sess.graph_user["name"],
        "Email": sess.graph_user["email"],
        "Emails Loaded": len(sess.live_emails),
        "Source": "Supabase Microsoft provider",
    })
    return jsonify({
        "ok": True,
        "connected": True,
        "count": len(sess.live_emails),
        "user_name": sess.graph_user["name"],
        "user_email": sess.graph_user["email"],
    })


@app.route("/api/auth/disconnect", methods=["POST"])
@require_csrf
def auth_disconnect():
    """Disconnect from Outlook and log session summary."""
    sess = _current_session()

    # Compute scan stats before clearing
    total_emails = len(sess.live_emails)
    total_scanned = len(sess.scan_results)
    threats = sum(1 for r in sess.scan_results.values() if r.get("prediction") == 1)
    safe = total_scanned - threats

    _log_session_event("LOGOUT", {
        "User": sess.graph_user["name"],
        "Email": sess.graph_user["email"],
        "Total Emails": total_emails,
        "Emails Scanned": total_scanned,
        "Phishing Detected": threats,
        "Safe Emails": safe,
    })

    sess.graph_token = {"access_token": None, "expiry": None}
    sess.graph_user["name"] = ""
    sess.graph_user["email"] = ""
    sess.live_emails = []
    sess.junk_emails = []
    sess.live_mode = False
    sess.scan_results.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/refresh", methods=["POST"])
@require_csrf
def auth_refresh_emails():
    """Re-fetch emails from Outlook."""
    sess = _current_session()
    if not sess.live_mode or not sess.graph_token["access_token"]:
        return jsonify({"error": "Not connected"}), 401
    try:
        sess.live_emails = _fetch_all_emails(sess)
        sess.junk_emails = _fetch_junk_emails(sess)
        sess.scan_results.clear()
        return jsonify({"count": len(sess.live_emails)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Patch email endpoints to support live mode ───────────────────────────────

def _get_email_list(sess=None, folder="inbox"):
    """Return the active email list (live emails when connected, empty otherwise)."""
    sess = sess or _current_session()
    if folder == "junk":
        return sess.junk_emails
    return sess.live_emails if sess.live_mode else []


def _message_key(email, idx):
    """Return the stable key used by the current renderer."""
    return str(email.get("id") or email.get("messageId") or idx)


def _find_email_by_message_id(message_id, sess=None, folder=None):
    """Find an email by stable message id in this session."""
    sess = sess or _current_session()
    target = str(message_id)
    folders = ["inbox", "junk"] if folder is None else [folder]
    for folder_key in folders:
        emails = _get_email_list(sess, folder_key)
        for idx, email in enumerate(emails):
            if _message_key(email, idx) == target:
                return idx, email
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


def _normalize_email(sess, email, idx):
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
        "scanned": message_id in sess.scan_results,
        "scanResult": sess.scan_results.get(message_id),
    }


@app.route("/api/emails", methods=["GET"])
def get_emails_v2():
    sess = _current_session()
    folder = (request.args.get("folder") or "inbox").lower()
    if folder not in ("inbox", "junk"):
        return jsonify({"error": "Unknown folder"}), 400
    emails = _get_email_list(sess, folder)
    emails_summary = [
        _normalize_email(sess, email, i) for i, email in enumerate(emails)
    ]
    return jsonify({"emails": emails_summary, "folder": folder})


@app.route("/api/emails/junk", methods=["GET"])
def get_junk_emails():
    sess = _current_session()
    emails = _get_email_list(sess, "junk")
    return jsonify({
        "emails": [_normalize_email(sess, email, i) for i, email in enumerate(emails)],
        "folder": "junk",
    })


@app.route("/api/messages/<path:message_id>", methods=["GET"])
def get_message(message_id):
    sess = _current_session()
    folder = request.args.get("folder")
    idx, email = _find_email_by_message_id(message_id, sess, folder)
    if email is None:
        return jsonify({"error": "Email not found"}), 404
    return jsonify(_normalize_email(sess, email, idx))


def _fetch_attachment_bytes(sess, message_id, attachment_id):
    """Fetch Outlook attachment bytes for local hash-based analysis."""
    token = sess.graph_token
    if not sess.live_mode or not token["access_token"] or not http_requests:
        return None
    url = f"{GRAPH_URL}/me/messages/{message_id}/attachments/{attachment_id}/$value"
    try:
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {token['access_token']}"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 401:
            token["access_token"] = None
            token["expiry"] = None
    except Exception as exc:
        print(f"WARNING: attachment fetch failed for {message_id}: {exc}")
    return None


def _vt_scan_attachments(sess, message_id, attachments):
    """Hash attachments locally and look up the hash on VirusTotal."""
    if not _provider_can_run("virustotal") or not attachments:
        return attachments
    scanned = 0
    for attachment in attachments:
        if not isinstance(attachment, dict) or "vt" in attachment:
            continue
        if scanned >= VT_MAX_FILES_PER_SCAN:
            break
        size = attachment.get("size") or 0
        if size and size > VT_MAX_FILE_BYTES:
            continue
        attachment_id = attachment.get("id", "")
        if not attachment_id:
            continue
        content_bytes = _fetch_attachment_bytes(sess, message_id, attachment_id)
        if not content_bytes:
            continue
        scanned += 1
        sha = hashlib.sha256(content_bytes).hexdigest()
        result = _vt_lookup_hash(sha)
        if isinstance(result, dict) and "error" not in result:
            attachment["vt"] = result
    return attachments


def _attachment_name(attachment):
    """Return a display name for either Graph attachment dicts or mock strings."""
    if isinstance(attachment, dict):
        return attachment.get("name") or attachment.get("filename") or "file"
    return str(attachment or "file")


def _classify_attachment(name):
    """Return (risk, reason): dangerous | caution | safe."""
    lower = (name or "").lower()
    parts = [p for p in lower.split(".") if p]
    ext = parts[-1] if parts else ""
    dangerous = {
        "exe", "scr", "bat", "cmd", "com", "pif", "js", "jse", "vbs",
        "vbe", "wsf", "wsh", "hta", "jar", "msi", "msix", "ps1",
        "lnk", "iso", "img", "reg", "cpl", "dll", "vbscript", "scf",
    }
    macro = {"docm", "xlsm", "pptm", "dotm", "xltm", "potm", "xlam", "ppam"}
    archive = {"zip", "rar", "7z", "gz", "tar", "cab", "ace", "bz2", "tgz"}
    doc = {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv",
           "jpg", "jpeg", "png", "gif", "zip", "rtf"}

    if len(parts) >= 3 and parts[-2] in doc and ext in dangerous:
        return "dangerous", f"Double extension - disguised executable (.{parts[-2]}.{ext})."
    if ext in dangerous:
        return "dangerous", f"Executable/script file (.{ext}) - can run code if opened."
    if ext in macro:
        return "caution", f"Macro-enabled Office document (.{ext}) - can run malicious macros."
    if ext in archive:
        return "caution", f"Archive (.{ext}) - may hide a malicious file inside."
    return "safe", ""


def _find_attachment(email, attachment_id):
    for attachment in (email.get("attachments", []) or []):
        if not isinstance(attachment, dict):
            continue
        if attachment.get("id") == attachment_id or attachment.get("name") == attachment_id:
            return attachment
    return None


def _load_attachment_metadata(sess, message_id, email):
    """Fetch and cache Graph attachment metadata for the newer frontend rail."""
    if email.get("attachments"):
        return email["attachments"]
    if not email.get("hasAttachments"):
        return []
    if not sess.live_mode or not sess.graph_token["access_token"] or not http_requests:
        return email.get("attachments", [])

    url = f"{GRAPH_URL}/me/messages/{message_id}/attachments"
    try:
        resp = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {sess.graph_token['access_token']}"},
            params={"$select": "id,name,contentType,size"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 401:
            sess.graph_token["access_token"] = None
            sess.graph_token["expiry"] = None
            return email.get("attachments", [])
        resp.raise_for_status()
        attachments = [
            {
                "id": item.get("id", ""),
                "name": item.get("name", "file"),
                "contentType": item.get("contentType", ""),
                "size": item.get("size", 0),
            }
            for item in (resp.json().get("value") or [])[:25]
        ]
        email["attachments"] = attachments
        return attachments
    except Exception as exc:
        print(f"WARNING: attachment metadata fetch failed for {message_id}: {exc}")
        return email.get("attachments", [])


@app.route("/api/messages/<path:message_id>/attachments", methods=["GET"])
def get_message_attachments(message_id):
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    attachments = _load_attachment_metadata(sess, message_id, email)
    out = []
    for attachment in attachments:
        name = _attachment_name(attachment)
        risk, reason = _classify_attachment(name)
        item = {"name": name, "risk": risk, "reason": reason}
        if isinstance(attachment, dict):
            item["id"] = attachment.get("id", "")
            item["size"] = attachment.get("size", 0)
            item["contentType"] = attachment.get("contentType", "")
        out.append(item)
    return jsonify({"attachments": out})


@app.route("/api/messages/<path:message_id>/attachments/<path:attachment_id>/analyze", methods=["POST"])
@require_csrf
def analyze_message_attachment(message_id, attachment_id):
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    _load_attachment_metadata(sess, message_id, email)
    att = _find_attachment(email, attachment_id)
    if att is None:
        return jsonify({"error": "Attachment not found"}), 404

    name = att.get("name", "") or att.get("filename", "") or "file"
    vt_status = next(
        item for item in _provider_public_statuses() if item["id"] == "virustotal"
    )
    if not vt_status["configured"] or not vt_status["enabled"]:
        return jsonify({
            "configured": vt_status["configured"],
            "enabled": vt_status["enabled"],
            "outcome": "not_checked",
            "name": name,
            "message": (
                "VirusTotal is disabled."
                if vt_status["configured"]
                else "VirusTotal API key is not configured."
            ),
        })

    content_bytes = _fetch_attachment_bytes(sess, message_id, att.get("id") or attachment_id)
    if not content_bytes:
        return jsonify({
            "configured": True,
            "enabled": True,
            "analyzed": False,
            "outcome": "not_checked",
            "name": name,
            "message": "Could not fetch attachment bytes; analysis requires real Outlook mail.",
        })

    sha = hashlib.sha256(content_bytes).hexdigest()
    result = _vt_lookup_hash(sha)
    if result is None:
        return jsonify({
            "configured": False,
            "enabled": False,
            "analyzed": False,
            "outcome": "not_checked",
            "name": name,
            "sha256": sha,
            "message": "VT lookup returned no configuration.",
        })

    if "error" in result:
        return jsonify({
            "configured": True,
            "enabled": True,
            "analyzed": False,
            "outcome": "provider_unavailable",
            "name": name,
            "sha256": sha,
            "error": result.get("error", "unknown error"),
            "status": result.get("status", 0),
        })

    return jsonify({
        "configured": True,
        "enabled": True,
        "analyzed": True,
        "outcome": (
            "not_found" if result.get("found") is False
            else "flagged" if result.get("malicious", 0) or result.get("suspicious", 0)
            else "not_flagged"
        ),
        "name": name,
        "size": len(content_bytes),
        **result,
    })


@app.route("/api/messages/<path:message_id>/attachments/<path:attachment_id>/deepscan", methods=["POST"])
@require_csrf
def deepscan_message_attachment(message_id, attachment_id):
    """Explicit VirusTotal upload. Sends file contents only on user action."""
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    _load_attachment_metadata(sess, message_id, email)
    attachment = _find_attachment(email, attachment_id)
    if attachment is None:
        return jsonify({"error": "Attachment not found"}), 404

    name = _attachment_name(attachment)
    vt_status = next(
        item for item in _provider_public_statuses() if item["id"] == "virustotal"
    )
    if not vt_status["configured"] or not vt_status["enabled"]:
        return jsonify({
            "configured": vt_status["configured"],
            "enabled": vt_status["enabled"],
            "outcome": "not_checked",
            "name": name,
            "message": (
                "VirusTotal is disabled."
                if vt_status["configured"]
                else "VirusTotal API key is not configured."
            ),
        })

    confirmation = request.get_json(silent=True) or {}
    if confirmation.get("confirm_upload") is not True:
        return jsonify({
            "error": "Explicit upload confirmation required",
            "confirmation_required": True,
        }), 400

    content_bytes = _fetch_attachment_bytes(sess, message_id, attachment.get("id", ""))
    if not content_bytes:
        return jsonify({"configured": True, "analyzed": False, "name": name,
                        "message": "Could not fetch attachment bytes - deep scan is only available for real Outlook mail."})

    sha = hashlib.sha256(content_bytes).hexdigest()
    known = _vt_lookup_hash(sha)
    if isinstance(known, dict) and known.get("found"):
        attachment["vt"] = known
        return jsonify({"configured": True, "analyzed": True, "uploaded": False,
                        "name": name, "size": len(content_bytes), **known})

    upload = _vt_upload_file(content_bytes, name)
    if upload is None:
        return jsonify({"configured": False, "name": name})
    if "error" in upload:
        return jsonify({"configured": True, "analyzed": False, "name": name,
                        "error": upload["error"], "status": upload.get("status", 0)})
    return jsonify({"configured": True, "uploaded": True, "name": name,
                    "sha256": sha, "status": "pending",
                    "analysis_id": upload["analysis_id"]})


@app.route("/api/messages/<path:message_id>/attachments/<path:attachment_id>/deepscan/<path:analysis_id>", methods=["GET"])
def deepscan_message_attachment_status(message_id, attachment_id, analysis_id):
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess)
    if email is None:
        return jsonify({"error": "Email not found"}), 404
    result = _vt_poll_analysis(analysis_id)
    if result is None:
        return jsonify({"configured": False})
    if "error" in result:
        return jsonify({"analyzed": False, "error": result["error"],
                        "status": result.get("status", 0)})
    if result.get("status") != "completed":
        return jsonify({"analyzed": False, "status": result.get("status", "queued")})
    _load_attachment_metadata(sess, message_id, email)
    attachment = _find_attachment(email, attachment_id)
    if attachment is not None:
        attachment["vt"] = {
            key: result.get(key)
            for key in ("found", "malicious", "suspicious", "harmless",
                        "undetected", "total", "permalink", "sha256")
        }
    return jsonify({"analyzed": True, **result})


@app.route("/api/messages/<path:message_id>/move-to-junk", methods=["POST"])
@require_csrf
def move_message_to_junk(message_id):
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess, "inbox")
    if email is None:
        return jsonify({"error": "Message not found in inbox"}), 404
    try:
        resp = _graph_post(
            sess,
            f"me/messages/{message_id}/move",
            {"destinationId": "junkemail"},
        )
        new_id = message_id
        try:
            moved = resp.json() if resp is not None else None
            if isinstance(moved, dict) and moved.get("id"):
                new_id = moved["id"]
        except Exception:
            pass

        sess.live_emails = [
            e for e in sess.live_emails
            if _message_key(e, -1) != message_id
        ]
        email["id"] = new_id
        email["folder"] = "junk"
        sess.junk_emails.insert(0, email)

        if message_id in sess.scan_results and new_id != message_id:
            sr = sess.scan_results.pop(message_id)
            sr["id"] = new_id
            sr["messageId"] = new_id
            sess.scan_results[new_id] = sr

        return jsonify({
            "ok": True,
            "message": "Moved to junk",
            "old_id": message_id,
            "new_id": new_id,
        })
    except Exception as exc:
        print(f"WARNING: move-to-junk failed for {message_id}: {exc}")
        return jsonify({"error": "Move failed"}), 500


@app.route("/api/auth/photo", methods=["GET"])
def auth_photo():
    sess = _current_session()
    if not sess.graph_token["access_token"] or not http_requests:
        return "", 404
    try:
        resp = http_requests.get(
            f"{GRAPH_URL}/me/photo/$value",
            headers={"Authorization": f"Bearer {sess.graph_token['access_token']}"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code == 200:
            return Response(
                resp.content,
                mimetype=resp.headers.get("Content-Type", "image/jpeg"),
            )
        if resp.status_code == 401:
            sess.graph_token["access_token"] = None
            sess.graph_token["expiry"] = None
    except Exception:
        pass
    return "", 404


@app.route("/api/sender-dna/<path:sender_addr>", methods=["GET"])
def sender_dna(sender_addr):
    """Build/retrieve sender DNA profile and compare against a referenced email."""
    sess = _current_session()
    addr = sender_addr.lower().strip()
    emails = _get_email_list(sess, "inbox") + _get_email_list(sess, "junk")

    profile = _build_sender_profile(addr, emails)
    if profile:
        with _sender_profiles_lock:
            _sender_profiles[addr] = profile
        _persist_sender_profile_to_supabase(addr, profile, _request_jwt())

    message_id = request.args.get("message_id") or request.args.get("id")
    comparison = None
    if message_id and profile:
        target = next((email for email in emails if _message_key(email, -1) == message_id), None)
        if target is not None:
            comparison = _compare_to_profile(profile, target)

    if not profile:
        with _sender_profiles_lock:
            profile = _sender_profiles.get(addr)

    if not profile:
        return jsonify({
            "status": "unknown",
            "message": "Not enough emails from this sender to build a profile",
            "profile": None,
            "comparison": None,
        })

    return jsonify({
        "status": "ok",
        "profile": profile,
        "comparison": comparison,
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


_CRED_KEYWORDS = (
    "login", "signin", "sign-in", "verify", "confirm", "password",
    "credential", "authenticate", "reactivate", "revalidate", "validate",
)
_LURE_PHRASES = (
    "inactive", "deactivat", "suspend", "reactivat", "verify your",
    "confirm your", "confirm that you", "unusual activity", "unusual sign",
    "validate your", "re-validate", "revalidate", "password expires",
    "password will expire", "storage is full", "mailbox is full",
    "update your billing", "update your payment", "avoid suspension",
    "account has been", "still use this account", "flagged as",
)
_RISKY_TLDS = {
    "test", "example", "invalid", "localhost", "zip", "mov", "xyz", "top",
    "click", "link", "buzz", "tk", "ml", "ga", "cf", "gq", "work", "fit",
    "rest", "country", "kim", "loan", "men", "review", "date", "racing",
    "stream", "download", "win", "bid", "trade",
}
_RISKY_TLDS = set(_DETECTION_RULES.get("risky_tlds") or _RISKY_TLDS)
_CRED_LURE = (
    "verify your", "confirm your", "confirm that you", "log in", "login",
    "sign in", "signin", "update your password", "reset your password",
    "validate your", "re-validate", "revalidate", "verify your identity",
    "confirm your identity", "unusual activity", "unusual sign-in",
    "account has been", "account is", "suspend", "deactivat", "reactivat",
    "still use this account", "secure your account", "billing portal",
)
_CRED_LURE = tuple(_DETECTION_RULES.get("cred_lure_phrases") or _CRED_LURE)
_BRAND_DOMAINS = {
    "paypal.com", "microsoft.com", "apple.com", "amazon.com", "google.com",
    "netflix.com", "facebook.com", "instagram.com", "linkedin.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com",
    "americanexpress.com", "dhl.com", "fedex.com", "ups.com",
    "docusign.com", "dropbox.com", "adobe.com", "coinbase.com",
    "outlook.com", "office.com", "icloud.com",
}
_BRAND_DOMAINS = set(_DETECTION_RULES.get("brand_domains") or _BRAND_DOMAINS)
_BRAND_NAME_TOKENS = {
    "paypal": "paypal.com", "microsoft": "microsoft.com",
    "office 365": "office.com", "office365": "office.com",
    "microsoft 365": "microsoft.com", "apple": "apple.com",
    "amazon": "amazon.com", "google": "google.com",
    "netflix": "netflix.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "linkedin": "linkedin.com",
    "chase": "chase.com", "wells fargo": "wellsfargo.com",
    "bank of america": "bankofamerica.com",
    "american express": "americanexpress.com", "amex": "americanexpress.com",
    "dhl": "dhl.com", "fedex": "fedex.com",
    "docusign": "docusign.com", "dropbox": "dropbox.com",
    "adobe": "adobe.com", "coinbase": "coinbase.com",
}
_BRAND_NAME_TOKENS = dict(_DETECTION_RULES.get("brand_name_tokens") or _BRAND_NAME_TOKENS)
URL_MODEL_SOLO_CAP = 45


class _AnchorPairExtractor(HTMLParser):
    """Collect visible link text and href pairs from HTML email bodies."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pairs = []
        self._href = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        if self._href is not None:
            self._flush()
        href = ""
        for name, value in attrs:
            if name.lower() == "href" and value:
                href = value.strip()
        self._href = href
        self._buf = []

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            self._flush()

    def _flush(self):
        text = "".join(self._buf).strip()
        if self._href:
            self.pairs.append((text, self._href))
        self._href, self._buf = None, []


_DISPLAYED_DOMAIN_RE = re.compile(
    r'^\s*(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)', re.I)
_HOMOGLYPH_MAP = str.maketrans({
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "b",
    "7": "t", "8": "b", "9": "g", "$": "s", "|": "l",
})


def _risk_level(score):
    return "high" if score >= 65 else ("elevated" if score >= 40 else "low")


def _join_and(items):
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _link_domain_untrusted(host):
    host = (host or "").lower().strip(".")
    if not host:
        return False
    if host in _TRUSTED_DOMAINS:
        return False
    if any(host.endswith("." + td) for td in _TRUSTED_DOMAINS):
        return False
    if host.split(".")[-1] in _TRUSTED_DOMAINS:
        return False
    return True


def _reg_domain(value):
    value = (value or "").strip().lower().strip("<>").strip()
    if "@" in value:
        value = value.rsplit("@", 1)[-1]
    if "://" in value:
        try:
            value = urlparse(value).hostname or value
        except Exception:
            pass
    value = value.split("/")[0].split(":")[0].strip(".")
    parts = [p for p in value.split(".") if p]
    return ".".join(parts[-2:]) if len(parts) >= 2 else value


def _anchor_text_mismatch_urls(raw_html):
    """Return href URLs whose visible text displays a different untrusted site."""
    out = set()
    if not raw_html or "<a" not in raw_html.lower():
        return out
    try:
        parser = _AnchorPairExtractor()
        parser.feed(raw_html)
    except Exception:
        return out
    for text, href in parser.pairs:
        if not href.lower().startswith(("http://", "https://")):
            continue
        match = _DISPLAYED_DOMAIN_RE.match(text or "")
        if not match:
            continue
        shown_reg = _reg_domain(match.group(1))
        if not shown_reg or "." not in shown_reg:
            continue
        try:
            href_host = (urlparse(href).hostname or "").lower()
        except Exception:
            continue
        if href_host and shown_reg != _reg_domain(href_host) and _link_domain_untrusted(href_host):
            out.add(href)
    return out


def _deglyph(value):
    return (value or "").lower().translate(_HOMOGLYPH_MAP).replace("rn", "m").replace("vv", "w")


def _edit_distance(a, b, cap=2):
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                val = min(val, prev2[j - 2] + 1)
            cur[j] = val
            best = min(best, val)
        if best > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[lb]


def _lookalike_brand(domain):
    """Return the known brand domain mimicked by an untrusted look-alike."""
    domain = _reg_domain(domain)
    if not domain or "." not in domain:
        return None
    if not _link_domain_untrusted(domain) or domain in _BRAND_DOMAINS:
        return None
    core = domain.split(".", 1)[0]
    deglyph_core = _deglyph(core)
    for brand_domain in _BRAND_DOMAINS:
        brand_core = brand_domain.split(".", 1)[0]
        if deglyph_core == brand_core and core != brand_core:
            return brand_domain
        if brand_core in deglyph_core and deglyph_core != brand_core:
            parts = [p for p in re.split(r"[^a-z0-9]+", deglyph_core) if p]
            if brand_core in parts or deglyph_core.startswith(brand_core) or deglyph_core.endswith(brand_core):
                return brand_domain
        if len(brand_core) >= 6 and _edit_distance(deglyph_core, brand_core, cap=1) <= 1:
            return brand_domain
    return None


def _hdr_dict(headers):
    out = {}
    if isinstance(headers, list):
        for header in headers:
            if isinstance(header, dict):
                name = (header.get("name") or "").strip().lower()
                value = (header.get("value") or "").strip()
                if name:
                    out[name] = (out[name] + " " + value) if name in out else value
    return out


def _score_link(url, url_threats, analyzer, is_deceptive=False):
    try:
        host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    except Exception:
        host = ""
    feats = analyzer.analyze_url(url) if analyzer else {}
    url_feed = (url_threats or {}).get(url) or {}
    host_feed = ((url_threats.get(host) if (host and url_threats) else None)) or {}
    tld = host.rsplit(".", 1)[-1] if "." in host else ""

    model_score = None
    url_model = getattr(detector, "url_model", None)
    if url_model is not None and getattr(url_model, "is_loaded", False):
        try:
            prob = url_model.predict_proba(url)
            if prob is not None:
                model_score = int(round(prob * 100))
        except Exception:
            model_score = None

    feed_malicious = bool(url_feed.get("malicious") or host_feed.get("malicious"))
    feed_suspicious = bool(url_feed.get("suspicious") or host_feed.get("suspicious"))
    urlscan_clean = bool(host_feed.get("urlscan_clean"))
    trusted = bool(host and not _link_domain_untrusted(host))

    if trusted:
        structural, reason = 5, "Goes to a recognized, trusted domain."
    elif feats.get("has_ip_address"):
        structural, reason = 90, "Uses a raw IP address instead of a domain name."
    elif feats.get("has_at_symbol"):
        structural, reason = 90, "Contains an '@' that hides the link's true destination."
    elif feats.get("has_brand_in_subdomain"):
        structural, reason = 90, "Puts a well-known brand name in the subdomain to look legitimate."
    elif host.startswith("xn--") or ".xn--" in host:
        structural, reason = 85, "Uses punycode - a possible look-alike of a real domain."
    elif feats.get("is_shortened"):
        structural, reason = 65, "Shortened link that hides where it actually goes."
    elif feats.get("suspicious_tld") or tld in _RISKY_TLDS:
        structural, reason = 65, "Unfamiliar domain on a high-risk top-level domain (." + tld + ")."
    elif feats.get("has_suspicious_keyword"):
        structural, reason = 40, "Unfamiliar domain using security/account wording in the link."
    else:
        structural, reason = 22, "Goes to an unfamiliar domain."

    reputation = ("malicious" if feed_malicious else "suspicious" if feed_suspicious
                  else "trusted" if trusted else "clean" if urlscan_clean else "unknown")
    sources = (url_feed.get("sources") or []) + (host_feed.get("sources") or [])
    checks = []
    if _provider_can_run("google_safe_browsing"):
        checks.append({
            "src": "Google Safe Browsing",
            "result": "flagged" if any("Google Safe Browsing" in s for s in sources) else "clean",
        })
    if any("urlscan" in s.lower() for s in sources):
        checks.append({"src": "urlscan.io", "result": "malicious" if feed_malicious else "suspicious"})
    elif host_feed.get("suspicious"):
        checks.append({"src": "urlscan.io", "result": "suspicious"})
    elif host_feed.get("urlscan_clean"):
        checks.append({"src": "urlscan.io", "result": "clean"})
    if url_feed.get("vt"):
        checks.append({"src": "VirusTotal", "result": url_feed["vt"]})
    if any("URLhaus" in s for s in sources):
        checks.append({"src": "URLhaus", "result": "flagged"})
    if any("OpenPhish" in s for s in sources):
        checks.append({"src": "OpenPhish", "result": "flagged"})

    def result(score, msg, decided_by):
        score = int(max(0, min(100, score)))
        return score, msg, {
            "score": score,
            "model": model_score,
            "structural": structural,
            "reputation": reputation,
            "in_malicious_db": feed_malicious,
            "checks": checks,
            "decided_by": decided_by,
        }

    if feed_malicious:
        srcs = "/".join(dict.fromkeys(sources)) or "threat feed"
        return result(100, "Confirmed malicious - listed on " + srcs + ".", "threat_db")
    if is_deceptive:
        checks.append({"src": "Deceptive link", "result": "malicious"})
        return result(
            88,
            "The link's visible text shows a different website than where it actually leads - a hallmark of phishing.",
            "link_text_mismatch",
        )
    lookalike = _lookalike_brand(host)
    if lookalike:
        checks.append({"src": "Look-alike of " + lookalike, "result": "malicious"})
        return result(
            88,
            "Link domain '" + host + "' is a look-alike of " + lookalike + " - likely brand impersonation.",
            "lookalike_domain",
        )
    if trusted:
        return result(5, reason, "trusted_domain")

    score, decided = structural, "structural"
    corroborated = structural >= 65 or feed_suspicious
    if model_score is not None and model_score > score:
        if urlscan_clean:
            pass
        elif corroborated:
            score, reason, decided = model_score, "The URL model flags this link as likely malicious.", "model+corroboration"
        else:
            capped = min(model_score, URL_MODEL_SOLO_CAP)
            if capped > score:
                score, reason, decided = capped, (
                    "The URL model finds this link's pattern unusual, but no reputation source corroborates it - treat with caution."
                ), "model_uncorroborated"
    if feed_suspicious and score < 70:
        score, reason, decided = 70, "Prior urlscan analyses scored this link's host risky.", "reputation_suspicious"
    gsb_active = _provider_can_run("google_safe_browsing")
    if urlscan_clean and decided == "structural" and score <= 25:
        vetted = (["Google Safe Browsing"] if gsb_active else []) + ["urlscan"]
        score, reason, decided = (
            10,
            f"Checked against {_join_and(vetted)} - no threats found.",
            "reputation_clean",
        )
    elif gsb_active and decided == "structural" and score <= 22:
        reason = "Checked against Google Safe Browsing - not flagged; domain is unfamiliar."
        decided = "reputation_gsb_clean"
    return result(score, reason, decided)


def _assess_links(urls, url_threats, analyzer, deceptive_urls=None):
    if not urls:
        return {"score": 0, "level": "low", "summary": "No links in this message.", "links": []}
    deceptive_urls = set(deceptive_urls or [])
    scored = []
    for url in urls[:_MAX_URLS_PER_EMAIL]:
        score, reason, factors = _score_link(
            url, url_threats, analyzer, is_deceptive=url in deceptive_urls)
        scored.append({"url": url, "score": score, "level": _risk_level(score),
                       "reason": reason, "factors": factors})
    top = max(scored, key=lambda item: item["score"])
    high = sum(1 for item in scored if item["score"] >= 65)
    if top["score"] >= 65:
        summary = (f"{high} dangerous link(s). " if high > 1 else "") + top["reason"]
    elif top["score"] >= 40:
        summary = top["reason"]
    else:
        summary = f"{len(scored)} link(s), none clearly dangerous."
    return {"score": top["score"], "level": _risk_level(top["score"]),
            "summary": summary, "links": scored}


def _assess_content(model_score, full_text, struct):
    text = (full_text or "").lower()
    model_pct = int(round(max(0.0, min(1.0, model_score)) * 100))
    heuristic = 0
    findings, tags = [], []
    urgency = [word for word in struct.URGENCY_WORDS if word in text]
    threat = [word for word in struct.THREAT_WORDS if word in text]
    money = [word for word in struct.MONEY_WORDS if word in text]
    credential = any(phrase in text for phrase in _CRED_LURE)
    if credential:
        heuristic += 45
    if urgency:
        heuristic += 25
    if threat:
        heuristic += 30
    if money:
        heuristic += 20
    score = max(model_pct, min(100, heuristic))
    if model_pct >= 60:
        findings.append("Wording matches patterns the AI model links to phishing.")
        tags.append("ai_content")
    if credential:
        findings.append("Asks you to confirm, verify, or sign in to an account.")
        tags.append("credential_harvesting")
    if urgency or threat:
        findings.append("Uses urgency or pressure language.")
        tags.append("urgency")
    if money:
        findings.append("References payments, invoices, or money.")
        tags.append("financial")
    if not findings:
        findings.append("No strong phishing language detected.")
    return {"score": score, "level": _risk_level(score),
            "summary": findings[0], "findings": findings, "tags": tags}


def analyze_email_auth(headers, from_addr, header_result, from_name=""):
    out = {"risk": 0, "signals": [], "positives": [], "tags": [],
           "sender_ip": None, "aligned": None, "reply_to_mismatch": False,
           "provider_signal": None, "evidence": {}, "impersonated_brand": None}
    header_map = _hdr_dict(headers)
    if not header_map and not header_result:
        return out

    auth_results = " ".join(
        value for key, value in header_map.items()
        if key in ("authentication-results", "arc-authentication-results")
        or key.startswith("x-ms-exchange-authentication-results")
    )

    def grab(pattern, source=auth_results):
        match = re.search(pattern, source, re.I)
        return match.group(1).strip().lower() if match else ""

    from_dom = _reg_domain(from_addr) or _reg_domain(header_map.get("from", ""))
    mailfrom = _reg_domain(grab(r"smtp\.mailfrom=([^\s;]+)"))
    header_d = _reg_domain(grab(r"header\.d=([^\s;]+)"))
    return_path = _reg_domain(header_map.get("return-path", ""))
    authed = {domain for domain in (mailfrom, header_d, return_path) if domain}
    dmarc = str((header_result or {}).get("dmarc") or "").lower()

    if dmarc == "pass":
        out["aligned"] = True
    elif from_dom and authed:
        out["aligned"] = from_dom in authed
    if out["aligned"] is True:
        out["positives"].append("From aligns with the authenticated sender domain.")
    elif out["aligned"] is False:
        is_brand = from_dom in _BRAND_DOMAINS or any(from_dom.endswith("." + brand) for brand in _BRAND_DOMAINS)
        if is_brand:
            out["risk"] = max(out["risk"], 85)
            out["tags"].append("domain_spoof")
            out["signals"].append(
                f"Brand spoof: visible sender '{from_dom}' does not align with the authenticated domain ({', '.join(sorted(authed)) or 'none'})."
            )
        else:
            out["risk"] = max(out["risk"], 45)
            out["signals"].append(
                f"Sender domain '{from_dom}' does not align with the authenticated domain ({', '.join(sorted(authed))})."
            )

    reply_to = _reg_domain(header_map.get("reply-to", ""))
    if reply_to and from_dom and reply_to != from_dom and reply_to not in authed:
        out["reply_to_mismatch"] = True
        out["risk"] = max(out["risk"], 45)
        out["tags"].append("reply_to_mismatch")
        out["signals"].append(f"Replies go to a different domain ('{reply_to}') than the sender.")
        out["evidence"]["reply_to_mismatch"] = (
            f"A reply would go to '{reply_to}', not the sender's domain '{from_dom}'.")

    name_l = (from_name or "").lower()
    for token, brand_domain in _BRAND_NAME_TOKENS.items():
        if token in name_l:
            if (from_dom != brand_domain and not from_dom.endswith("." + brand_domain)
                    and _link_domain_untrusted(from_dom or "")):
                strong = dmarc != "pass"
                out["risk"] = max(out["risk"], 80 if strong else 65)
                if "display_name_spoof" not in out["tags"]:
                    out["tags"].append("display_name_spoof")
                out["signals"].append(
                    "The display name claims to be " + token.title()
                    + ", but the sender domain is '" + (from_dom or "unknown")
                    + "', not " + brand_domain + ".")
                out["evidence"]["display_name_spoof"] = (
                    "The display name says '" + token.title() + "', but the address is '"
                    + (from_dom or "unknown") + "' - not " + brand_domain + ".")
                out["impersonated_brand"] = token.title()
            break

    lookalike = _lookalike_brand(from_dom)
    if lookalike:
        out["risk"] = max(out["risk"], 85)
        if "lookalike_domain" not in out["tags"]:
            out["tags"].append("lookalike_domain")
        out["signals"].append(
            "Sender domain '" + from_dom + "' is a look-alike of "
            + lookalike + " - likely impersonation.")
        out["evidence"]["lookalike_domain"] = (
            "The sender domain '" + from_dom + "' is a near-copy of "
            + lookalike + ".")
        out["impersonated_brand"] = lookalike.split(".")[0].title()

    forefront = header_map.get("x-forefront-antispam-report", "")
    compauth = grab(r"compauth=(\w+)")
    category = grab(r"CAT:(\w+)", forefront)
    sfv = grab(r"SFV:(\w+)", forefront)
    try:
        scl = int(header_map.get("x-ms-exchange-organization-scl", grab(r"SCL:(-?\d+)", forefront) or "999"))
    except Exception:
        scl = None

    if category in ("phsh", "hphsh", "spoof") or sfv == "phsh":
        out["risk"] = max(out["risk"], 60)
        out["tags"].append("provider_flagged")
        msg = f"Microsoft classified this message as {category or sfv}."
        out["signals"].append(msg)
        out["provider_signal"] = {"status": "fail", "label": "Flagged", "message": msg}
    elif compauth == "fail":
        out["risk"] = max(out["risk"], 55)
        out["tags"].append("provider_flagged")
        msg = "Microsoft composite authentication (compauth) failed."
        out["signals"].append(msg)
        out["provider_signal"] = {"status": "fail", "label": "CompAuth fail", "message": msg}
    elif scl is not None and 0 <= scl <= 99 and scl >= 5:
        out["risk"] = max(out["risk"], 40)
        msg = f"Elevated Microsoft spam confidence (SCL {scl})."
        out["signals"].append(msg)
        out["provider_signal"] = {"status": "warn", "label": f"SCL {scl}", "message": msg}
    elif sfv == "spm" or category == "spm":
        out["risk"] = max(out["risk"], 35)
        msg = "Microsoft marked this message as spam."
        out["signals"].append(msg)
        out["provider_signal"] = {"status": "warn", "label": "Spam", "message": msg}

    if compauth == "pass":
        out["positives"].append("Microsoft composite authentication: pass.")
    if scl is not None and -1 <= scl <= 0:
        out["positives"].append(f"Low Microsoft spam confidence (SCL {scl}).")
    out["sender_ip"] = (
        header_map.get("x-sender-ip")
        or header_map.get("x-ms-exchange-crosstenant-originalattributedtenantconnectingip")
        or None
    )
    return out


def _assess_auth(header_result, headers=None, from_addr="", from_name=""):
    score, findings, tags = 0, [], []
    header_result = header_result or {}

    def value(key):
        return str(header_result.get(key) or "").lower()

    failures = [key.upper() for key in ("spf", "dkim", "dmarc") if value(key) == "fail"]
    missing = [key.upper() for key in ("spf", "dkim", "dmarc") if value(key) in ("none", "")]
    if failures:
        score = max(score, 70)
        findings.append("Failed authentication: " + ", ".join(failures) + ".")
        tags.append("auth_fail")
    elif len(missing) == 3:
        score = max(score, 15)
        findings.append("No SPF/DKIM/DMARC records present.")
    else:
        findings.append("SPF/DKIM/DMARC passed.")

    auth = analyze_email_auth(headers, from_addr, header_result, from_name)
    score = max(score, auth["risk"])
    findings.extend(auth["signals"])
    for tag in auth["tags"]:
        if tag not in tags:
            tags.append(tag)
    if auth["aligned"] is True and not auth["signals"] and score < 30:
        findings.insert(0, "Fully verified - " + " ".join(auth["positives"][:2]))
    return {"score": score, "level": _risk_level(score),
            "summary": findings[0] if findings else "No authentication issues.",
            "findings": findings, "tags": tags,
            "detail": {"aligned": auth["aligned"],
                       "reply_to_mismatch": auth["reply_to_mismatch"],
                       "provider_signal": auth["provider_signal"],
                       "sender_ip": auth["sender_ip"],
                       "evidence": auth.get("evidence") or {},
                       "impersonated_brand": auth.get("impersonated_brand")}}


def _assess_sender(threat_intel):
    score, findings, tags = 0, [], []
    checks = (threat_intel or {}).get("checks", {}) or {}
    rep = checks.get("domain_reputation")
    if isinstance(rep, dict) and rep.get("category") == "suspicious":
        score = max(score, 35)
        signals = rep.get("signals") or []
        findings.append("Sender domain looks unfamiliar" + (": " + signals[0][1] if signals and len(signals[0]) > 1 else "."))
        tags.append("suspicious_domain")
    abuse = checks.get("abuseipdb")
    if isinstance(abuse, dict) and abuse.get("is_abusive"):
        score = max(score, 75)
        findings.append(f"Sender IP has a high abuse score ({abuse.get('score')}%).")
        tags.append("abusive_ip")
    if not findings:
        findings.append("No sender-reputation concerns.")
    return {"score": score, "level": _risk_level(score),
            "summary": findings[0], "findings": findings, "tags": tags}


def _assess_attachments(attachments):
    items, score, findings, tags = [], 0, [], []
    vt_checked = False
    for attachment in (attachments or []):
        name = _attachment_name(attachment)
        risk, reason = _classify_attachment(name)
        vt = attachment.get("vt") if isinstance(attachment, dict) else None
        vt_out = None
        if isinstance(vt, dict) and vt.get("found") is not None:
            vt_checked = True
            malicious = int(vt.get("malicious", 0) or 0)
            suspicious = int(vt.get("suspicious", 0) or 0)
            total = int(vt.get("total", 0) or 0)
            vt_out = {
                "malicious": malicious,
                "suspicious": suspicious,
                "total": total,
                "found": bool(vt.get("found")),
                "permalink": vt.get("permalink", ""),
            }
            if malicious >= 1:
                risk = "malicious"
                reason = f"VirusTotal: {malicious} of {total} security engines flagged this file as malicious."
            elif suspicious >= 2:
                if risk == "safe":
                    risk = "caution"
                reason = reason or f"VirusTotal: {suspicious} engines flagged this file as suspicious."
        items.append({"name": name, "risk": risk, "reason": reason, "vt": vt_out})
        if risk == "malicious":
            score = max(score, 96)
        elif risk == "dangerous":
            score = max(score, 90)
        elif risk == "caution":
            score = max(score, 55)
    malicious = [item["name"] for item in items if item["risk"] == "malicious"]
    danger = [item["name"] for item in items if item["risk"] == "dangerous"]
    caution = [item["name"] for item in items if item["risk"] == "caution"]
    if malicious:
        findings.append("VirusTotal flagged a malicious attachment: " + malicious[0])
        tags.append("malicious_attachment")
    if danger:
        findings.append("Dangerous attachment: " + danger[0])
        tags.append("dangerous_attachment")
    if caution:
        findings.append("Risky attachment: " + caution[0])
        tags.append("risky_attachment")
    if not items:
        summary = "No attachments."
    elif not findings:
        summary = (f"{len(items)} attachment(s) - checked on VirusTotal, none flagged."
                   if vt_checked else f"{len(items)} attachment(s), none risky.")
    else:
        summary = findings[0]
    return {"score": score, "level": _risk_level(score), "summary": summary,
            "findings": findings, "tags": tags, "items": items}


def _compute_assessment(model_score, full_text, urls, url_threats,
                        header_result, threat_intel, struct,
                        headers=None, from_addr="", attachments=None,
                        from_name="", deceptive_links=None):
    content = _assess_content(model_score, full_text, struct)
    links = _assess_links(urls, url_threats, detector.url_analyzer, deceptive_links)
    sender = _assess_sender(threat_intel)
    auth = _assess_auth(header_result, headers, from_addr, from_name)
    files = _assess_attachments(attachments)
    dimensions = [content, links, sender, auth, files]
    scores = [dim["score"] for dim in dimensions]
    base_max = max(scores)
    if base_max >= 50:
        prod = 1.0
        for score in scores:
            prod *= (1.0 - score / 100.0)
        overall = round((1.0 - prod) * 100)
    else:
        overall = base_max
    if threat_intel.get("confirmed_phishing"):
        overall = max(overall, 97)
    overall = int(max(0, min(100, overall)))
    verdict = 1 if overall >= 50 else 0
    driver = max(dimensions, key=lambda dim: dim["score"])
    if verdict:
        summary = "Likely phishing - " + driver["summary"]
    elif overall >= 30:
        summary = "Probably safe, but worth a look - " + driver["summary"]
    else:
        summary = "No phishing indicators found."

    actions = []
    if verdict:
        if links["score"] >= 50:
            actions.append("Do not click any links in this message.")
        actions.append("Do not reply or share passwords, codes, or payment details.")
        if sender["score"] >= 50:
            actions.append("Verify the sender through a channel you already trust.")
        actions.append("Report the message and delete it.")

    threats = []
    concern = 50
    if content["score"] >= concern:
        tags = content.get("tags", [])
        if "credential_harvesting" in tags:
            threats.append({"title": "Credential Harvesting", "severity": "high",
                            "desc": "Requests personal identity or account verification through the message or a link."})
        if "urgency" in tags:
            threats.append({"title": "Urgency Manipulation", "severity": "medium",
                            "desc": "Creates artificial urgency or pressure to force a hasty action."})
        if "financial" in tags:
            threats.append({"title": "Financial Lure", "severity": "medium",
                            "desc": "References payments, invoices, refunds, or money to bait a response."})
        if "ai_content" in tags and "credential_harvesting" not in tags and "urgency" not in tags:
            threats.append({"title": "Suspicious Wording", "severity": "medium",
                            "desc": "The message text matches patterns the AI model associates with phishing."})
    for link in links.get("links", [])[:3]:
        if link.get("score", 0) < concern:
            continue
        reason = link.get("reason", "")
        title = "Known Malicious Link" if "Confirmed malicious" in reason else (
            "Domain Spoofing" if "brand" in reason.lower() else "Suspicious Link")
        threats.append({"title": title,
                        "severity": "high" if link["score"] >= 65 else "medium",
                        "desc": reason})
    if sender["score"] >= concern:
        tags = sender.get("tags", [])
        if "abusive_ip" in tags:
            threats.append({"title": "Abusive Sender IP", "severity": "high",
                            "desc": "The sending IP address has a high abuse score."})
        if "suspicious_domain" in tags:
            threats.append({"title": "Suspicious Sender Domain", "severity": "medium",
                            "desc": "The sender domain is unfamiliar or has a low reputation."})
    if auth["score"] >= concern:
        tags = auth.get("tags", [])
        evidence = ((auth.get("detail") or {}).get("evidence") or {})
        if "lookalike_domain" in tags:
            threats.append({"title": "Look-Alike Sender Domain", "severity": "high",
                            "desc": evidence.get("lookalike_domain") or "The sender domain is a near-identical look-alike of a trusted brand."})
        if "display_name_spoof" in tags:
            threats.append({"title": "Display Name Spoofing", "severity": "high",
                            "desc": evidence.get("display_name_spoof") or "The display name impersonates a brand the sender domain does not belong to."})
        if "domain_spoof" in tags:
            threats.append({"title": "Domain Spoofing", "severity": "high",
                            "desc": "The visible sender does not align with the authenticated sending domain."})
        if "auth_fail" in tags:
            threats.append({"title": "Failed Authentication", "severity": "high",
                            "desc": "The sender failed SPF/DKIM/DMARC checks."})
        if "reply_to_mismatch" in tags:
            threats.append({"title": "Reply Address Mismatch", "severity": "medium",
                            "desc": "Replies would go to a different domain than the sender."})
        if "provider_flagged" in tags:
            threats.append({"title": "Provider-Flagged Sender", "severity": "medium",
                            "desc": "Microsoft's mail protection flagged this message."})
    file_tags = files.get("tags", [])
    if "malicious_attachment" in file_tags:
        threats.append({"title": "Malicious Attachment", "severity": "high",
                        "desc": files["summary"] + " - do not open it."})
    elif "dangerous_attachment" in file_tags:
        threats.append({"title": "Dangerous Attachment", "severity": "high",
                        "desc": files["summary"] + " - do not open it."})
    elif "risky_attachment" in file_tags:
        threats.append({"title": "Risky Attachment", "severity": "medium",
                        "desc": files["summary"]})

    seen, deduped = set(), []
    for threat in threats:
        if threat["title"] in seen:
            continue
        seen.add(threat["title"])
        deduped.append(threat)
    threats = deduped[:6]
    if verdict and not threats:
        threats.append({"title": "Suspicious Message", "severity": "high",
                        "desc": driver.get("summary") or "This message matches patterns associated with phishing."})
    return {
        "overall": overall,
        "verdict": verdict,
        "summary": summary,
        "dimensions": {"content": content, "links": links, "sender": sender,
                       "auth": auth, "files": files},
        "actions": actions,
        "threats": threats,
    }


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
    signals = {
        "intel_signals": list(
            (result.get("threat_intel") or {}).get("signals", []) or []
        )[:20],
        "checks_run": list(
            (result.get("threat_intel") or {}).get("checks", {}).keys()
        ),
        "confirmed": bool(
            (result.get("threat_intel") or {}).get("confirmed", False)
        ),
    }
    if result.get("assessment") is not None:
        signals["assessment"] = _to_native(result["assessment"])
    if result.get("risk_score") is not None:
        signals["risk_score"] = int(result["risk_score"])
    if result.get("header_result") is not None:
        header_result = result["header_result"] if isinstance(result["header_result"], dict) else {}
        signals["header_result"] = {
            key: header_result.get(key)
            for key in ("spf", "dkim", "dmarc")
            if header_result.get(key) is not None
        }
    return {
        "message_id": str(email.get("id") or result.get("idx") or ""),
        "sender_domain": domain,
        "prediction": int(result.get("prediction", 0)),
        "confidence": float(result.get("confidence", 0) or 0),
        "signals": signals,
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


def _scan_email_common(sess, email, idx):
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
    text_urls = url_pattern.findall(scan_body)
    href_urls = _extract_html_link_urls(body)
    deceptive_links = _anchor_text_mismatch_urls(body)
    seen_urls, urls_found = set(), []
    for url in text_urls + href_urls:
        if url not in seen_urls:
            seen_urls.add(url)
            urls_found.append(url)
        if len(urls_found) >= _MAX_URLS_PER_EMAIL:
            break
    threat_intel = run_threat_intel(email, urls_found)

    try:
        prediction, confidence, url_analysis, header_result, model_score = detector.predict(
            full, headers=email.get("internetMessageHeaders"), extra_urls=href_urls)
        url_analysis, header_result = _make_serializable(url_analysis, header_result)
        model_score = float(_to_native(model_score))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, (jsonify({"error": str(e)}), 500)

    sender = email.get("from", {})
    from_name = ""
    if isinstance(sender, dict):
        email_address = sender.get("emailAddress", {}) or {}
        from_addr = email_address.get("address", "") or sender.get("address", "")
        from_name = email_address.get("name", "") or sender.get("name", "")
    else:
        from_addr = email.get("sender", "") if isinstance(email.get("sender"), str) else ""
        from_name = email.get("sender_name", "") if isinstance(email.get("sender_name"), str) else ""

    message_id = _message_key(email, idx)
    attachments = _load_attachment_metadata(sess, message_id, email)
    attachments = _vt_scan_attachments(sess, message_id, attachments)
    assessment = _compute_assessment(
        model_score,
        full,
        urls_found,
        threat_intel.get("url_threats") or {},
        header_result,
        threat_intel,
        detector.structural_analyzer,
        headers=email.get("internetMessageHeaders"),
        from_addr=from_addr,
        attachments=attachments,
        from_name=from_name,
        deceptive_links=deceptive_links,
    )

    prediction = int(assessment["verdict"])
    risk_score = int(assessment["overall"])
    confidence = risk_score / 100.0 if prediction == 1 else (1.0 - risk_score / 100.0)

    result = {
        "id": message_id,
        "messageId": message_id,
        "idx": idx,
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": risk_score,
        "assessment": _to_native(assessment),
        "model_score": model_score,
        "url_analysis": url_analysis,
        "header_result": header_result,
        "threat_intel": {
            "confirmed": threat_intel["confirmed_phishing"],
            "signals": threat_intel["signals"],
            "checks": _to_native(threat_intel["checks"]),
            "checks_run": list(threat_intel["checks"].keys()),
            "provider_outcomes": _to_native(
                threat_intel.get("provider_outcomes") or {}
            ),
        },
    }
    sess.scan_results[message_id] = result
    _persist_scan_history(email, result)
    return result, None


@app.route("/api/messages/<path:message_id>/scan", methods=["POST"])
@require_csrf
def scan_message(message_id):
    sess = _current_session()
    idx, email = _find_email_by_message_id(message_id, sess)
    if email is None:
        return jsonify({"error": "Email not found"}), 404

    result, error = _scan_email_common(sess, email, idx)
    if error:
        return error
    return jsonify(result)


@app.route("/api/scan-all", methods=["POST"])
@require_csrf
def scan_all_v2():
    sess = _current_session()
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503
    folder = (request.args.get("folder") or "inbox").lower()
    if folder not in ("inbox", "junk"):
        return jsonify({"error": "Unknown folder"}), 400

    now = datetime.now()
    if sess._scan_all_in_flight:
        return jsonify({"error": "A scan-all is already running for this session"}), 429
    if sess._last_scan_all and (now - sess._last_scan_all).total_seconds() < _SCAN_ALL_COOLDOWN:
        wait = int(_SCAN_ALL_COOLDOWN - (now - sess._last_scan_all).total_seconds())
        return jsonify({"error": f"Please wait {wait}s before scanning again"}), 429

    emails = _get_email_list(sess, folder)
    results = []
    sess._scan_all_in_flight = True
    try:
        for i, email in enumerate(emails):
            try:
                result, error = _scan_email_common(sess, email, i)
                if error:
                    continue
                results.append(result)
            except Exception:
                pass
        results_dict = {r["id"]: r for r in results if r.get("id")}
        return jsonify({"results": results_dict})
    finally:
        sess._scan_all_in_flight = False
        sess._last_scan_all = datetime.now()


@app.route("/api/stats", methods=["GET"])
def get_stats_v2():
    sess = _current_session()
    scanned = len(sess.scan_results)
    threats = sum(1 for r in sess.scan_results.values() if r.get("prediction") == 1)
    safe = scanned - threats
    return jsonify({
        "scanned": scanned,
        "threats": threats,
        "safe": safe,
        "total": len(_get_email_list(sess, "inbox")),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5050"))
    debug = os.environ.get("PHISHGUARD_DEBUG") == "1"
    _start_provider_control_channel()
    app.run(host="127.0.0.1", debug=debug, port=port, use_reloader=debug)
