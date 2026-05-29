"""
PhishGuard Web Dashboard — Flask Backend
Serves the SPA and provides REST API for phishing email analysis.
"""

import base64
import json as _json_decode
import hashlib


# ─────────────────────────────────────────────────────────────────────────────
#  .env loader (no python-dotenv dependency)
# ─────────────────────────────────────────────────────────────────────────────
# Read KEY=value pairs from .env in the app directory and inject them
# into os.environ BEFORE any code below reads from there. Existing env
# vars are not overwritten — so passing VIRUSTOTAL_API_KEY=… on the
# command line still wins over the file.
def _load_dotenv():
    import os as _os
    from pathlib import Path as _Path
    env_path = _Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip optional surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not key or _os.environ.get(key):
                continue
            _os.environ[key] = value
    except Exception as e:
        print(f"WARNING: could not load .env: {e}")


_load_dotenv()
import hmac
import html
import os
import re
import secrets
import socket
import sys
from datetime import datetime, timedelta
from functools import wraps
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urlparse

from flask import Flask, jsonify, make_response, redirect, request, send_from_directory

from phishing_detector import PhishingDetector

try:
    import requests as http_requests
except ImportError:
    http_requests = None

if sys.version_info < (3, 10):
    raise RuntimeError(
        "PhishGuard final/ requires Python 3.10+ so security-supported "
        "dependency versions can be installed."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Security configuration
# ─────────────────────────────────────────────────────────────────────────────
# All external HTTP calls use this (connect, read) timeout pair.
HTTP_TIMEOUT = (5, 15)

# Known-good SHA-256 of phishing_model.pkl. Pickle load is blocked if the
# artifact on disk does not match. Update this when intentionally retraining.
MODEL_SHA256 = "cde9377a507d3a301b6c46675bc9b00ad1e91adc5875d12feb33403bf893618f"

# Writable data directory — passed from the Electron main process via
# app.getPath('userData'). Falls back to ~/.phishguard when launched from
# the CLI.
#
# The env var is treated as untrusted: a hostile parent process could set
# PHISHGUARD_USER_DATA=/etc or /var/www to make us write our JSON state
# files into a sensitive location. Resolve the path and refuse anything
# that isn't rooted in the user's home directory or the system temp dir.
import tempfile as _tempfile


def _resolve_user_data_dir():
    safe_roots = [Path.home().resolve(), Path(_tempfile.gettempdir()).resolve()]
    env_val = os.environ.get("PHISHGUARD_USER_DATA")

    candidates = []
    if env_val:
        try:
            candidates.append(Path(env_val).expanduser().resolve())
        except (OSError, RuntimeError) as e:
            print(f"WARNING: PHISHGUARD_USER_DATA invalid ({e}); ignoring.")
    candidates.append((Path.home() / ".phishguard").resolve())
    candidates.append((Path(_tempfile.gettempdir()) / "phishguard").resolve())

    for candidate in candidates:
        # Must be anchored under home or temp — symlinks are resolved first
        # so a link from an allowed root to /etc is caught.
        try:
            under_safe_root = any(
                candidate == root or root in candidate.parents
                for root in safe_roots
            )
        except Exception:
            under_safe_root = False
        if not under_safe_root:
            print(f"WARNING: Refusing unsafe data dir {candidate} "
                  f"(not under {safe_roots[0]} or {safe_roots[1]})")
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as e:
            print(f"WARNING: Could not create {candidate}: {e}")
    # If every candidate failed, fall back to an unsafe emergency location
    # so the backend still starts — better to run with logging disabled than
    # to refuse service entirely.
    emergency = Path(_tempfile.mkdtemp(prefix="phishguard-"))
    print(f"WARNING: Falling back to emergency data dir {emergency}")
    return emergency


USER_DATA_DIR = _resolve_user_data_dir()

# Origin allowlist for mutating (POST) routes. Electron loads the SPA from
# 127.0.0.1 on FLASK_PORT, so build the allowlist from that. An additional
# per-launch token (PHISHGUARD_LAUNCH_SECRET) is honoured for Electron which
# sends it via the X-Launch-Secret header, proving the request originated in
# the process the Electron main spawned.
_FLASK_PORT = int(os.environ.get("FLASK_PORT", 5050))
ALLOWED_ORIGINS = frozenset({
    f"http://127.0.0.1:{_FLASK_PORT}",
    f"http://localhost:{_FLASK_PORT}",
})
LAUNCH_SECRET = os.environ.get("PHISHGUARD_LAUNCH_SECRET", "")

# ─────────────────────────────────────────────────────────────────────────────
#  Supabase project configuration
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for the Supabase project is the renderer-side
# `static/js/supabase-config.js`. Flask parses that file at startup so we
# never drift out of sync between renderer and backend. Both values are
# PUBLIC by design (anon key); committing them is fine.
#
# Flask uses these to make PostgREST API calls on the user's behalf for
# the Flask-side migrations (scan_history, sender_profiles, threat_reports).
# The user's JWT (forwarded in the Authorization header) is what RLS uses
# to scope rows; the anon key just identifies the project.

def _load_supabase_config():
    path = Path(__file__).parent / "static" / "js" / "supabase-config.js"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not read supabase-config.js: {e}")
        return "", ""
    url_match = re.search(r"SUPABASE_URL:\s*['\"]([^'\"]+)['\"]", text)
    key_match = re.search(r"SUPABASE_ANON_KEY:\s*['\"]([^'\"]+)['\"]", text)
    url = url_match.group(1).rstrip("/") if url_match else ""
    key = key_match.group(1) if key_match else ""
    if not url or not key or "YOUR_PROJECT_REF" in url or "YOUR_ANON_KEY" in key:
        print("WARNING: supabase-config.js has placeholder or missing values; "
              "Flask Supabase calls will be disabled until the file is filled in.")
        return "", ""
    return url, key


SUPABASE_URL, SUPABASE_ANON_KEY = _load_supabase_config()


def _request_jwt():
    """Pull the Supabase access token out of the Authorization header.

    Returns the JWT string if the renderer attached one, or None. Flask
    handlers that write to Supabase should require a non-None JWT — RLS
    needs the user identity to scope the write to the right rows.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


def _jwt_user_id(jwt):
    """Decode the user_id (sub claim) from a Supabase JWT.

    We do NOT verify the signature here — Supabase itself validates the
    JWT on every REST API call, so a forged token would be rejected by
    Postgres anyway. This decode is only used to populate body fields
    (reporter_id) when the schema doesn't default to auth.uid().
    """
    if not jwt or "." not in jwt:
        return None
    try:
        parts = jwt.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = _json_decode.loads(
            base64.urlsafe_b64decode(payload_b64).decode("utf-8")
        )
        sub = payload.get("sub")
        return sub if isinstance(sub, str) and sub else None
    except Exception:
        return None


def _supabase_request(method, path, jwt, json_body=None, params=None,
                      prefer="return=minimal"):
    """Make an authenticated PostgREST request against the user's project.

    `path` is relative to /rest/v1, e.g. "scan_history" or
    "threat_reports?domain=eq.evil.com". RLS rejects writes that don't
    match the JWT's auth.uid(), so callers don't have to do their own
    authorization checks — the database does it for them.

    Returns the requests.Response, or None on configuration / transport
    error. Caller checks status_code.
    """
    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        return None
    if not jwt:
        return None
    if not http_requests:
        return None
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }
    try:
        return http_requests.request(
            method, url, headers=headers, json=json_body,
            params=params, timeout=HTTP_TIMEOUT,
        )
    except Exception as e:
        print(f"WARNING: Supabase {method} {path} failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  Defensive helpers
# ─────────────────────────────────────────────────────────────────────────────

# Azure client IDs are UUIDs; anything else is almost certainly an attempt to
# bloat _global_pending_oauth.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_MAX_CLIENT_ID_LEN = 64

# Graph API base — any follow-up URL (next_link) must live under this origin.
_GRAPH_HOST = "graph.microsoft.com"

# Cap sizes before we hand strings to regex or tokenizers. Prevents ReDoS and
# runaway memory on hostile bodies.
_MAX_BODY_FOR_REGEX = 200_000   # ~200 KB
_MAX_URLS_PER_EMAIL = 100

# Pagination safety — a compromised tenant could return an infinite nextLink
# chain. Cap the loop regardless.
_MAX_GRAPH_PAGES = 500

# Per-session scan-all cooldown (seconds). Prevents one renderer from hogging
# the backend / model / Graph quota.
_SCAN_ALL_COOLDOWN = 30

# Render-time sanitiser — strips categories that enable bidi spoofing, hidden
# overrides, and ANSI escapes inside Graph displayName / email fields.
_UNSAFE_UNICODE_RE = re.compile(
    r"[\u0000-\u001F"      # C0 controls (incl. \x1B = ESC)
    r"\u007F"              # DEL
    r"\u2028\u2029"        # line/paragraph separators
    r"\u200E\u200F"        # LRM / RLM
    r"\u202A-\u202E"       # LRE/RLE/PDF/LRO/RLO
    r"\u2066-\u2069"       # isolate controls
    r"]"
)


def _sanitise_display(text):
    """Return `text` with bidi overrides / control / ANSI chars stripped."""
    if not text:
        return ""
    return _UNSAFE_UNICODE_RE.sub("", str(text))[:256]


def _valid_client_id(s):
    return bool(s) and len(s) <= _MAX_CLIENT_ID_LEN and bool(_UUID_RE.match(s))


def _is_graph_url(url):
    """True iff the URL is scheme=https and host is exactly graph.microsoft.com."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.scheme == "https" and p.netloc.lower() == _GRAPH_HOST


def _safe_error(exc, user_msg="Request failed"):
    """Log the real exception and return a jsonifiable generic error.

    Avoids leaking file paths, library internals, or stack fragments in
    `str(e)` to the renderer. The full traceback goes to stderr where the
    user running the app (or the Electron log capture) can still see it.
    """
    import traceback
    traceback.print_exc()
    return jsonify({"error": user_msg}), 500

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = secrets.token_hex(32)
# Cap request size — refuse giant bodies that could wedge a scan.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# ─────────────────────────────────────────────────────────────────────────────
#  Load Model — verify SHA-256 first, fail closed on mismatch
# ─────────────────────────────────────────────────────────────────────────────
detector = PhishingDetector()
MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"


def _verify_model_hash(path: Path, expected_hex: str) -> bool:
    """Return True only if SHA-256 of the pickle file matches expected_hex."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if not hmac.compare_digest(actual, expected_hex):
            print(f"FATAL: Model hash mismatch. Expected {expected_hex}, got {actual}.")
            return False
        return True
    except Exception as e:
        print(f"FATAL: Could not hash model file: {e}")
        return False


if _verify_model_hash(MODEL_PATH, MODEL_SHA256):
    try:
        detector.load_model(str(MODEL_PATH))
        print("PhishGuard model loaded (hash verified).")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
else:
    print("WARNING: Refusing to load unverified model pickle. "
          "Scanning will be unavailable until the hash matches.")

# ─────────────────────────────────────────────────────────────────────────────
#  Per-session server-side state
#
#  All mailbox/OAuth state is keyed by a server-generated session ID carried
#  in an httponly, SameSite=Strict cookie. Two browsers talking to the same
#  backend see independent state.
# ─────────────────────────────────────────────────────────────────────────────
_sessions = {}  # sid -> SessionState
_sessions_lock_import = True
import threading as _threading_early
_sessions_lock = _threading_early.Lock()

# Locks for the module-level caches touched by scan handlers. Python's GIL
# makes single dict ops atomic, but _save_sender_profiles() serialises the
# whole dict while other threads may be mutating it — that can raise
# "dictionary changed size during iteration" under concurrent scans. A
# single lock per cache keeps things simple.
_sender_profiles_lock = _threading_early.Lock()
_domain_rep_cache_lock = _threading_early.Lock()

SESSION_COOKIE = "pg_sid"
SESSION_IDLE_TTL = timedelta(hours=8)


class SessionState:
    """Per-browser/Electron-window mailbox and auth state."""

    __slots__ = (
        "sid", "created", "last_seen", "csrf_token",
        "graph_token", "graph_user",
        "live_emails", "junk_emails", "live_mode",
        "pending_oauth",
        "scan_results",          # keyed by message id
        "_last_scan_all",        # datetime of last /api/scan-all
        "_scan_all_in_flight",   # bool — serialises concurrent scan-alls
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
        self.pending_oauth = {}  # state -> {client_id, verifier, expires}
        self.scan_results = {}   # message_id -> result dict
        self._last_scan_all = None
        self._scan_all_in_flight = False


def _get_or_create_session():
    """Look up the SessionState for this request, creating it if needed.

    Also flags whether a fresh sid cookie should be written on the response.
    """
    sid = request.cookies.get(SESSION_COOKIE)
    needs_new_cookie = False
    with _sessions_lock:
        # GC stale sessions occasionally
        now = datetime.now()
        stale = [k for k, v in _sessions.items()
                 if now - v.last_seen > SESSION_IDLE_TTL]
        for k in stale:
            _sessions.pop(k, None)

        sess = _sessions.get(sid) if sid else None
        if sess is None:
            sid = secrets.token_urlsafe(32)
            sess = SessionState(sid)
            _sessions[sid] = sess
            needs_new_cookie = True
        else:
            sess.last_seen = now
    return sess, needs_new_cookie


def _attach_session_cookie(resp, sid):
    """Set the sid cookie with safe defaults."""
    resp.set_cookie(
        SESSION_COOKIE, sid,
        max_age=int(SESSION_IDLE_TTL.total_seconds()),
        httponly=True,
        samesite="Strict",
        secure=False,  # Electron uses http://127.0.0.1; TLS is not applicable
        path="/",
    )
    return resp


# Paths that don't need a per-browser session. The OAuth callback arrives in
# a popup whose cookie has been stripped by Microsoft's cross-site redirect,
# so auto-creating a SessionState for it just produces an 8-hour orphan in
# _sessions. Static files don't need sessions either.
_SESSIONLESS_PATHS = ("/auth/callback", "/static/")


@app.before_request
def _ensure_session():
    """Attach the session object to `request.pg_session` for every request,
    except for sessionless endpoints that would only accumulate orphans."""
    path = request.path or ""
    if path.startswith(_SESSIONLESS_PATHS):
        request.pg_session = None
        request.pg_needs_cookie = False
        return
    sess, needs_cookie = _get_or_create_session()
    # Stash on the request so handlers can pick it up from a single place
    request.pg_session = sess
    request.pg_needs_cookie = needs_cookie


@app.after_request
def _emit_session_cookie(resp):
    """Write the sid cookie on the first response of each new session."""
    if getattr(request, "pg_needs_cookie", False):
        sess = getattr(request, "pg_session", None)
        if sess is not None:
            _attach_session_cookie(resp, sess.sid)
    return resp


@app.after_request
def _security_headers(resp):
    """Apply defence-in-depth HTTP response headers.

    These complement the CSP meta tag in index.html. Any new endpoint added
    later inherits them automatically, which matters more than the CSP in the
    HTML: JSON responses (/api/*), redirects, and static binary payloads all
    pass through here.
    """
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Permissions-Policy",
        "geolocation=(), microphone=(), camera=(), clipboard-read=(), "
        "clipboard-write=(), usb=(), serial=(), bluetooth=(), payment=(), "
        "interest-cohort=()",
    )
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    # Don't cache API responses — they carry session-scoped mailbox data.
    if request.path.startswith("/api/") or request.path.startswith("/auth/"):
        resp.headers.setdefault("Cache-Control", "no-store, max-age=0")
        resp.headers.setdefault("Pragma", "no-cache")
    return resp


def _current_session():
    """Convenience accessor for handlers."""
    return request.pg_session


# ─────────────────────────────────────────────────────────────────────────────
#  CSRF + Origin enforcement for mutating routes
# ─────────────────────────────────────────────────────────────────────────────
CSRF_HEADER = "X-CSRF-Token"
LAUNCH_HEADER = "X-Launch-Secret"


def _origin_ok():
    """Return True if the request's Origin (or Referer) is allowlisted.

    Electron renderer running inside our BrowserWindow sends an Origin of
    http://127.0.0.1:<port>. Requests from arbitrary web pages attempting a
    cross-origin POST to localhost will have a different Origin and are
    rejected. A launch-secret header is also accepted (Electron passes it on
    every fetch) so we can still distinguish our own renderer from a stray
    browser tab even when Origin headers are stripped.
    """
    origin = request.headers.get("Origin") or ""
    referer = request.headers.get("Referer") or ""

    if origin in ALLOWED_ORIGINS:
        return True
    if referer:
        try:
            p = urlparse(referer)
            ref_origin = f"{p.scheme}://{p.netloc}"
            if ref_origin in ALLOWED_ORIGINS:
                return True
        except Exception:
            pass

    # Launch-secret proves the request came from the Electron-spawned UI.
    if LAUNCH_SECRET:
        provided = request.headers.get(LAUNCH_HEADER, "")
        if provided and hmac.compare_digest(provided, LAUNCH_SECRET):
            return True

    return False


def _launch_secret_ok():
    """Return True if no launch secret is configured or the caller supplied it.

    GET endpoints cannot rely on Origin/Referer alone because non-browser local
    processes can spoof those headers. For sensitive read paths, require the
    per-launch secret when Electron configured one.
    """
    if not LAUNCH_SECRET:
        return True
    provided = request.headers.get(LAUNCH_HEADER, "")
    return bool(provided) and hmac.compare_digest(provided, LAUNCH_SECRET)


def require_csrf(view):
    """Decorator for POST/DELETE handlers — enforces Origin + CSRF token."""

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


def require_connected(view):
    """Decorator for routes that require an active Outlook connection."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        sess = _current_session()
        if not sess.live_mode:
            return jsonify({"error": "Not connected"}), 401
        return view(*args, **kwargs)

    return wrapped


# Scan results kept here for legacy helpers. Prefer per-session storage.
scan_results = {}

# ─────────────────────────────────────────────────────────────────────────────
#  Sender DNA Fingerprinting
# ─────────────────────────────────────────────────────────────────────────────
# In-memory cache of profiles built during this Flask process. Authoritative
# storage moved to Supabase (sender_profiles table) and is keyed per-user via
# RLS. The cache is just a same-process speedup.
_sender_profiles = {}  # sender_email -> profile dict

import json as _json


def _persist_sender_profile_to_supabase(sender_email, profile, jwt):
    """Upsert one sender's profile into Supabase as the signed-in user.

    Silently skipped when there's no JWT (unsigned-in / dev mode without
    sign-in) or when Supabase config isn't loaded. RLS requires user_id
    to match auth.uid() — we include it in the body since the column has
    no default.
    """
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
        print(f"WARNING: sender_profiles upsert failed "
              f"[{resp.status_code}]: {resp.text[:200]}")


def _analyze_writing_style(text):
    """Extract writing style features from email text."""
    if not text or not text.strip():
        return None

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return None

    # Features
    avg_sentence_len = len(words) / max(len(sentences), 1)
    avg_word_len = sum(len(w) for w in words) / word_count
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count('!')
    question_count = text.count('?')
    comma_density = text.count(',') / max(word_count, 1)

    # Formality markers
    formal_words = {'regarding', 'furthermore', 'consequently', 'therefore',
                    'hereby', 'pursuant', 'accordingly', 'sincerely',
                    'respectfully', 'cordially', 'attached', 'enclosed'}
    informal_words = {'hey', 'hi', 'lol', 'btw', 'fyi', 'gonna', 'wanna',
                      'yeah', 'yep', 'nope', 'cool', 'awesome', 'thanks'}
    lower_words = set(w.lower() for w in words)
    formality = (len(lower_words & formal_words) - len(lower_words & informal_words)) / max(word_count, 1)

    # Greeting style
    first_line = text.strip().split('\n')[0].lower().strip()
    greeting = 'formal' if any(first_line.startswith(g) for g in ['dear', 'good morning', 'good afternoon', 'good evening']) \
        else 'casual' if any(first_line.startswith(g) for g in ['hi', 'hey', 'hello']) \
        else 'none'

    # Signature detection
    has_signature = bool(re.search(r'(?:regards|best|sincerely|thanks|cheers|sent from)',
                                    text[-200:].lower())) if len(text) > 50 else False

    return {
        'word_count': word_count,
        'avg_sentence_len': round(avg_sentence_len, 2),
        'avg_word_len': round(avg_word_len, 2),
        'caps_ratio': round(caps_ratio, 4),
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'comma_density': round(comma_density, 4),
        'formality': round(formality, 4),
        'greeting': greeting,
        'has_signature': has_signature,
    }


def _extract_send_hour(datetime_str):
    """Extract hour of day from ISO datetime string."""
    try:
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.hour
    except Exception:
        return None


def _build_sender_profile(sender_addr, emails):
    """Build a behavioral profile from all emails by this sender."""
    sender_emails = []
    for email in emails:
        addr = email.get('from', {}).get('emailAddress', {}).get('address', '')
        if addr.lower() == sender_addr.lower():
            sender_emails.append(email)

    if not sender_emails:
        return None

    styles = []
    send_hours = []

    for email in sender_emails:
        body = email.get('body', {})
        if isinstance(body, dict):
            content = body.get('content', '')
        else:
            content = str(body) if body else ''

        # Strip HTML for analysis
        clean = re.sub(r'<[^>]+>', ' ', content)
        style = _analyze_writing_style(clean)
        if style:
            styles.append(style)

        hour = _extract_send_hour(email.get('receivedDateTime', ''))
        if hour is not None:
            send_hours.append(hour)

    if not styles:
        return None

    # Average the style features across all emails
    profile = {
        'email': sender_addr.lower(),
        'email_count': len(sender_emails),
        'last_updated': datetime.now().isoformat(),
        'avg_word_count': round(sum(s['word_count'] for s in styles) / len(styles), 1),
        'avg_sentence_len': round(sum(s['avg_sentence_len'] for s in styles) / len(styles), 2),
        'avg_word_len': round(sum(s['avg_word_len'] for s in styles) / len(styles), 2),
        'avg_caps_ratio': round(sum(s['caps_ratio'] for s in styles) / len(styles), 4),
        'avg_formality': round(sum(s['formality'] for s in styles) / len(styles), 4),
        'typical_greeting': max(set(s['greeting'] for s in styles), key=lambda g: sum(1 for s in styles if s['greeting'] == g)),
        'usually_has_signature': sum(1 for s in styles if s['has_signature']) > len(styles) / 2,
        'typical_send_hours': sorted(set(send_hours)) if send_hours else [],
        'avg_exclamations': round(sum(s['exclamation_count'] for s in styles) / len(styles), 1),
    }

    return profile


def _compare_to_profile(profile, email):
    """Compare a single email against the sender's established profile.
    Returns a dict with deviation score (0-100) and specific flags."""
    body = email.get('body', {})
    if isinstance(body, dict):
        content = body.get('content', '')
    else:
        content = str(body) if body else ''

    clean = re.sub(r'<[^>]+>', ' ', content)
    style = _analyze_writing_style(clean)

    if not style or not profile:
        return {'score': 0, 'flags': [], 'status': 'insufficient_data'}

    flags = []
    deviation = 0

    # Word count deviation (is email much shorter or longer than usual?)
    avg_wc = profile['avg_word_count']
    if avg_wc > 0:
        wc_ratio = style['word_count'] / avg_wc
        if wc_ratio < 0.3 or wc_ratio > 3.0:
            flags.append({
                'type': 'length',
                'message': f"Email length is unusual ({style['word_count']} words vs typical {int(avg_wc)})",
                'severity': 'medium'
            })
            deviation += 20

    # Sentence length deviation
    sl_diff = abs(style['avg_sentence_len'] - profile['avg_sentence_len'])
    if sl_diff > 8:
        flags.append({
            'type': 'sentence_style',
            'message': 'Writing style differs from this sender\'s usual pattern',
            'severity': 'medium'
        })
        deviation += 15

    # Caps ratio deviation (much more CAPS than usual?)
    caps_diff = style['caps_ratio'] - profile['avg_caps_ratio']
    if caps_diff > 0.05:
        flags.append({
            'type': 'caps',
            'message': 'Unusually high use of capital letters',
            'severity': 'low'
        })
        deviation += 10

    # Formality shift
    formality_diff = abs(style['formality'] - profile['avg_formality'])
    if formality_diff > 0.01:
        flags.append({
            'type': 'formality',
            'message': 'Tone is more ' + ('formal' if style['formality'] > profile['avg_formality'] else 'casual') + ' than usual',
            'severity': 'low'
        })
        deviation += 10

    # Greeting change
    if style['greeting'] != profile['typical_greeting'] and profile['typical_greeting'] != 'none':
        flags.append({
            'type': 'greeting',
            'message': f"Greeting style changed (usually '{profile['typical_greeting']}', now '{style['greeting']}')",
            'severity': 'low'
        })
        deviation += 10

    # Signature missing when usually present
    if profile['usually_has_signature'] and not style['has_signature']:
        flags.append({
            'type': 'signature',
            'message': 'Email signature is missing (this sender usually includes one)',
            'severity': 'medium'
        })
        deviation += 15

    # Unusual send time
    hour = _extract_send_hour(email.get('receivedDateTime', ''))
    if hour is not None and profile['typical_send_hours']:
        min_dist = min(abs(hour - h) for h in profile['typical_send_hours'])
        min_dist = min(min_dist, 24 - min_dist)  # wrap around midnight
        if min_dist > 6:
            flags.append({
                'type': 'send_time',
                'message': f"Sent at an unusual time ({hour}:00 - typically active around {profile['typical_send_hours'][0]}:00)",
                'severity': 'low'
            })
            deviation += 10

    # Exclamation spike
    if style['exclamation_count'] > profile['avg_exclamations'] + 3:
        flags.append({
            'type': 'urgency',
            'message': 'More exclamation marks than usual (possible urgency manipulation)',
            'severity': 'medium'
        })
        deviation += 15

    deviation = min(100, deviation)

    if deviation >= 50:
        status = 'suspicious'
    elif deviation >= 25:
        status = 'minor_deviation'
    else:
        status = 'matches'

    return {
        'score': deviation,
        'flags': flags,
        'status': status,
        'profile_emails': profile['email_count'],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Session Log
# ─────────────────────────────────────────────────────────────────────────────
LOG_FILE = USER_DATA_DIR / "phishguard_log.txt"

# Logging is disabled by default (privacy: contains user name, email, counts).
# Users opt in via the Settings panel → toggles PHISHGUARD_LOG=1 env var at
# launch, or flips the in-memory flag via /api/settings/logging.
_LOGGING_ENABLED = os.environ.get("PHISHGUARD_LOG") == "1"


def _log_session_event(event, details=None):
    """Append a formatted entry to the session log file (opt-in only)."""
    if not _LOGGING_ENABLED:
        return
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

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        # Packaged/read-only path, disk full, etc. — degrade silently.
        print(f"WARNING: session log write failed: {e}")

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


# ─────────────────────────────────────────────────────────────────────────────
#  PhishTank — online-valid feed (bulk download, in-memory lookup)
# ─────────────────────────────────────────────────────────────────────────────
# Why this approach instead of the legacy /checkurl endpoint:
#   1. /checkurl was deprecated and now requires an API key + per-request
#      rate limits; the unauthenticated path returns sparse / unreliable
#      results.
#   2. /checkurl indexes specific phishing URLs (with paths), not bare
#      domains. Checking `http://<domain>/` produces lots of false
#      negatives even for known phishing hosts.
#   3. Sending user email domains to PhishTank in real time is a privacy
#      leak. The feed download sends ZERO user data — it just pulls a
#      public list of currently-active phishing URLs.
#
# The feed is ~30 MB JSON, refreshed every ~hour upstream. We re-download
# every 6h. In-memory lookups are O(1). First ~30s of Flask startup the
# feed is empty (download running in background), so _check_phishtank
# returns None until then; callers treat that as "unknown".
# ─────────────────────────────────────────────────────────────────────────────
#  VirusTotal — file hash lookup for attachment analysis
# ─────────────────────────────────────────────────────────────────────────────
# We do NOT upload files. We hash them locally (SHA-256) and look up the
# hash against VirusTotal's existing reports. This is the lowest-privacy
# option: only the hash leaves the device, never the file contents. If
# VT has no report for the hash, we return "unknown" rather than
# uploading.
#
# Free public API key: register at virustotal.com (4 req/min, 500/day).
# Set VIRUSTOTAL_API_KEY env var to enable. Without a key, the analyze
# endpoint returns a configured=False stub.
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "").strip()
VT_API_BASE = "https://www.virustotal.com/api/v3"
VT_CACHE_TTL_SECONDS = 24 * 3600  # results don't change minute-to-minute

_vt_cache_lock = _threading_early.Lock()
_vt_cache = {}  # sha256 -> {"result": dict, "cached_at": datetime}


def _vt_lookup_hash(sha256):
    """Look up a file hash on VirusTotal. Returns one of:
      - dict with stats fields when VT has (or doesn't have) a report
      - dict with {"error": ..., "status": ...} on transport/parse failure
      - None when not configured (no API key)
    Cached in-process for VT_CACHE_TTL_SECONDS to avoid burning quota."""
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
    except Exception as e:
        return {"error": f"network: {e}", "status": 0}

    if resp.status_code == 404:
        # Hash not in VT's database — not malicious, just unknown.
        result = {
            "sha256": sha256, "found": False,
            "malicious": 0, "suspicious": 0, "harmless": 0, "undetected": 0,
            "total": 0,
        }
    elif resp.status_code == 200:
        try:
            payload = resp.json()
            attrs = payload.get("data", {}).get("attributes", {}) or {}
            stats = attrs.get("last_analysis_stats", {}) or {}
            result = {
                "sha256": sha256, "found": True,
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
        except Exception as e:
            return {"error": f"parse: {e}", "status": 200}
    elif resp.status_code == 401:
        return {"error": "VT api key invalid or revoked", "status": 401}
    elif resp.status_code == 429:
        return {"error": "VT rate limit hit; try again in a minute",
                "status": 429}
    else:
        return {"error": f"http {resp.status_code}",
                "status": resp.status_code}

    with _vt_cache_lock:
        _vt_cache[sha256] = {"result": result, "cached_at": datetime.now()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  urlscan.io — historical scans for a domain
# ─────────────────────────────────────────────────────────────────────────────
# We use the SEARCH endpoint (not /scan), so no scans are SUBMITTED on
# the user's behalf — only public, already-scanned domains are queried.
# That keeps the privacy profile the same as PhishTank/URLhaus: data
# leaving the device is just the domain, and only when a key is set.
URLSCAN_API_KEY = os.environ.get("URLSCAN_API_KEY", "").strip()
URLSCAN_CACHE_TTL_SECONDS = 12 * 3600
_urlscan_cache_lock = _threading_early.Lock()
_urlscan_cache = {}  # domain -> {"result": ..., "cached_at": datetime}


def _check_urlscan(domain):
    """Query urlscan.io for prior public scans of `domain`. Returns:
      - dict with the latest scan's verdict
      - dict with an 'error' key on transport/parse failure
      - None when not configured / no domain / no http_requests
    Results cached in-memory for 12 hours."""
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
    except Exception as e:
        return {"error": f"network: {e}"}
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
        for r in items[:5]:
            v = r.get("verdicts", {}) or {}
            ov = v.get("overall", {}) or {}
            if ov.get("malicious"):
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


def _fetch_attachment_bytes(sess, message_id, attachment_id):
    """Pull the raw bytes of one Outlook attachment via Graph.
    Returns bytes on success or None when no token / not in live mode /
    transport error / non-2xx response."""
    if not sess.live_mode or not sess.graph_token["access_token"]:
        return None
    if not http_requests:
        return None
    token = sess.graph_token["access_token"]
    url = (f"{GRAPH_URL}/me/messages/{message_id}/attachments/"
           f"{attachment_id}/$value")
    try:
        r = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 200:
            return r.content
        if r.status_code == 401:
            sess.graph_token["access_token"] = None
            sess.graph_token["expiry"] = None
    except Exception as e:
        print(f"WARNING: attachment fetch failed for {message_id}: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Original PhishTank section follows
# ─────────────────────────────────────────────────────────────────────────────
# PhishTank now requires a (free) API key for bulk downloads. Set
# PHISHTANK_API_KEY in the environment to enable it. Without a key we
# fall back to URLhaus's free, no-auth feed.
PHISHTANK_API_KEY = os.environ.get("PHISHTANK_API_KEY", "").strip()
PHISHTANK_FEED_URL_AUTH = "http://data.phishtank.com/data/{key}/online-valid.json"
PHISHTANK_FEED_URL_NOAUTH = "http://data.phishtank.com/data/online-valid.json"
URLHAUS_FEED_URL = "https://urlhaus.abuse.ch/downloads/json_online/"
PHISHTANK_REFRESH_INTERVAL = 6 * 3600   # 6 hours
PHISHTANK_DOWNLOAD_TIMEOUT = (10, 120)  # connect + read

_phishtank_lock = _threading_early.Lock()
_phishtank_data = {
    "domains": set(),     # set of host names known to host phishing
    "urls": set(),        # set of full phishing URLs (lower-cased)
    "updated_at": None,   # datetime of last successful load, or None
    "count": 0,           # total entries in the feed
}


def _phishtank_domain_key(host):
    """Normalize a host for comparison against the PhishTank domain set."""
    if not host:
        return ""
    h = host.lower().strip()
    # Strip credentials and port
    h = h.split("@")[-1].split(":")[0]
    if h.startswith("www."):
        h = h[4:]
    return h


def _download_phishtank_entries():
    """Return PhishTank entries as a list of dicts with a 'url' key, or []
    if download failed. Uses the authed URL if PHISHTANK_API_KEY is set,
    otherwise tries the anonymous URL (which PhishTank has been
    progressively restricting; expect 404 without a key)."""
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
    except Exception as e:
        print(f"[{label}] feed download failed (network): {e}")
        return []

    if resp.status_code != 200:
        print(f"[{label}] feed returned HTTP {resp.status_code} "
              f"({len(resp.content)} bytes). "
              + ("" if PHISHTANK_API_KEY else
                 "Anon access has been retired; set PHISHTANK_API_KEY "
                 "env var with a free key from phishtank.org."))
        return []

    try:
        data = resp.json()
    except ValueError as e:
        print(f"[{label}] feed not JSON: {e}")
        return []

    if not isinstance(data, list):
        print(f"[{label}] unexpected feed shape: {type(data).__name__}")
        return []

    print(f"[{label}] downloaded {len(data)} entries.")
    return data


def _download_urlhaus_entries():
    """Return URLhaus entries as a list of dicts with a 'url' key, or []
    if download failed. No API key needed."""
    if not http_requests:
        return []
    try:
        print("[URLhaus] downloading feed...")
        resp = http_requests.get(
            URLHAUS_FEED_URL,
            headers={"User-Agent": "PhishGuard/2.0 (+phishguard-capstone)"},
            timeout=PHISHTANK_DOWNLOAD_TIMEOUT,
        )
    except Exception as e:
        print(f"[URLhaus] feed download failed (network): {e}")
        return []

    if resp.status_code != 200:
        print(f"[URLhaus] feed returned HTTP {resp.status_code}.")
        return []

    try:
        raw = resp.json()
    except ValueError as e:
        print(f"[URLhaus] feed not JSON: {e}")
        return []

    # URLhaus shapes the file as a dict keyed by record id, each value a
    # single-element list. Flatten to a list of dicts.
    out = []
    if isinstance(raw, dict):
        for rec in raw.values():
            if isinstance(rec, list) and rec:
                out.append(rec[0])
            elif isinstance(rec, dict):
                out.append(rec)
    elif isinstance(raw, list):
        out = raw

    print(f"[URLhaus] downloaded {len(out)} entries.")
    return out


def _load_phishtank_feed():
    """Combine PhishTank (if API key configured) + URLhaus into one
    in-memory phishing/malware URL set. Either source contributing is
    enough to produce a useful signal; both is best."""
    combined_entries = []
    combined_entries.extend(_download_phishtank_entries())
    combined_entries.extend(_download_urlhaus_entries())

    if not combined_entries:
        print("[PhishTank/URLhaus] no entries from either source — "
              "_check_phishtank will return None.")
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
            netloc = urlparse(url_l).netloc
            host = _phishtank_domain_key(netloc)
            if host:
                domains.add(host)
        except Exception:
            pass

    with _phishtank_lock:
        _phishtank_data["domains"] = domains
        _phishtank_data["urls"] = urls
        _phishtank_data["updated_at"] = datetime.now()
        _phishtank_data["count"] = len(combined_entries)

    print(f"[Threat feed] active — {len(combined_entries)} entries combined, "
          f"{len(domains)} unique domains, {len(urls)} unique URLs.")


def _start_phishtank_refresher():
    """Background thread that keeps the PhishTank feed fresh."""
    def _loop():
        while True:
            try:
                _load_phishtank_feed()
            except Exception as e:
                print(f"[PhishTank] refresh loop error: {e}")
            _time.sleep(PHISHTANK_REFRESH_INTERVAL)
    t = _threading_early.Thread(
        target=_loop, name="phishtank-refresher", daemon=True,
    )
    t.start()


def _check_phishtank(domain):
    """Check `domain` against the in-memory PhishTank feed.

    Returns:
      True  — host is on the PhishTank phishing list
      False — feed is loaded and the host is NOT on the list
      None  — feed not loaded yet (typically only during the first
              ~30 s of Flask startup, before the background download
              finishes)

    Always enabled — the privacy concern that gated the legacy
    implementation (sending user domains to PhishTank) does not apply
    here, because we only download the public list and check locally.
    """
    if not domain:
        return None
    with _phishtank_lock:
        if not _phishtank_data["updated_at"]:
            return None
        return _phishtank_domain_key(domain) in _phishtank_data["domains"]


def _check_phishtank_url(url):
    """Check a specific URL against the PhishTank list. Returns True if
    that exact URL is listed, False if the feed is loaded but the URL
    isn't, None if the feed hasn't loaded yet. Used by the URL-analysis
    path to flag known phishing URLs even when the host itself is not
    yet listed."""
    if not url:
        return None
    with _phishtank_lock:
        if not _phishtank_data["updated_at"]:
            return None
        return url.lower().strip() in _phishtank_data["urls"]


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


# ─────────────────────────────────────────────────────────────────────────────
#  Threat Intelligence Aggregator
#  Checks multiple public databases BEFORE the AI model runs.
#  If any database confirms phishing, we can be 100% confident.
# ─────────────────────────────────────────────────────────────────────────────
import json as _json
import threading
import time as _time

# Local scan history — previously labelled "community threat database". It is
# NOT a real crowd-sourced feed: it is a per-device record of the user's own
# scans. To prevent self-report poisoning, it is never used to boost or
# override security verdicts. It is shown in the UI as "Local Scan History".
_COMMUNITY_DB_PATH = USER_DATA_DIR / "local_scan_history.json"
_community_db_lock = threading.Lock()


def _load_community_db():
    """Load the local scan history from disk."""
    try:
        if _COMMUNITY_DB_PATH.exists():
            with open(_COMMUNITY_DB_PATH) as f:
                return _json.load(f)
    except Exception:
        pass
    return {"domains": {}, "urls": {}, "updated": ""}


def _save_community_db(db):
    try:
        db["updated"] = datetime.utcnow().isoformat()
        with open(_COMMUNITY_DB_PATH, "w") as f:
            _json.dump(db, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write local scan history: {e}")


def _community_report(domain, is_phishing, *, user_confirmed=False):
    """Record a scan into local history. Only writes when the user explicitly
    confirms (e.g. clicks "Report this sender"). Automatic writes are ignored
    so model predictions can never poison the database.
    """
    if not user_confirmed:
        return
    with _community_db_lock:
        db = _load_community_db()
        if domain not in db["domains"]:
            db["domains"][domain] = {
                "phishing": 0, "safe": 0,
                "first_seen": datetime.utcnow().isoformat(),
            }
        if is_phishing:
            db["domains"][domain]["phishing"] += 1
        else:
            db["domains"][domain]["safe"] += 1
        _save_community_db(db)


def _community_check(domain):
    """Return raw local-history counts for a domain. Does NOT produce a
    verdict — callers must treat this as informational only.
    """
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


def _check_urlhaus(url_or_domain):
    """URLhaus host lookup — free abuse.ch service, no key required.
    Sends only the domain. Returns True if the host has online malicious
    URLs on file, False if known-clean, None if the lookup failed."""
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
            active = [u for u in urls_online if u.get("url_status") == "online"]
            return len(active) > 0
    except Exception:
        pass
    return None


def _check_abuseipdb(domain):
    """AbuseIPDB IP reputation lookup. Auto-activates when ABUSEIPDB_KEY
    is set — key presence is the consent signal."""
    api_key = os.environ.get("ABUSEIPDB_KEY")
    if not api_key or not http_requests:
        return None
    try:
        import socket as _socket
        ips = _socket.getaddrinfo(domain, None, _socket.AF_INET)
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
            abuse_score = data.get("abuseConfidenceScore", 0)
            return {"score": abuse_score, "is_abusive": abuse_score > 50}
    except Exception:
        pass
    return None


def run_threat_intel(email, urls_in_email=None):
    """Run all threat intelligence checks on an email.
    Returns a dict with findings from each source.
    This runs BEFORE the AI model — confirmed hits override the model."""

    sender = ""
    sender_from = email.get("from", {})
    if isinstance(sender_from, dict) and "emailAddress" in sender_from:
        sender = sender_from["emailAddress"].get("address", "")
    elif isinstance(sender_from, dict):
        sender = sender_from.get("address", "")
    elif isinstance(email.get("sender"), str):
        sender = email.get("sender", "")

    domain = sender.split("@")[-1].lower() if "@" in sender else ""

    results = {
        "domain": domain,
        "checks": {},
        "confirmed_phishing": False,
        "confidence_boost": 0.0,
        "signals": [],
    }

    if not domain:
        return results

    # 1. PhishTank
    pt = _check_phishtank(domain)
    results["checks"]["phishtank"] = pt
    if pt is True:
        results["confirmed_phishing"] = True
        results["signals"].append("PhishTank: Domain is a confirmed phishing site")

    # 2. URLhaus — check domain and any URLs in the email
    uh = _check_urlhaus(domain)
    results["checks"]["urlhaus"] = uh
    if uh is True:
        results["confirmed_phishing"] = True
        results["signals"].append("URLhaus: Domain hosts active malware/phishing URLs")

    # 3. AbuseIPDB
    aip = _check_abuseipdb(domain)
    results["checks"]["abuseipdb"] = aip
    if aip and isinstance(aip, dict) and aip.get("is_abusive"):
        results["confidence_boost"] += 0.15
        results["signals"].append(f"AbuseIPDB: Sender IP has {aip['score']}% abuse confidence")

    # 4. urlscan.io — historical scans of the sender's domain
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

    # 5. Local scan history (informational only — no verdict boost; this is
    # per-device data that could be self-poisoned and must not feed back into
    # security decisions).
    cr = _community_check(domain)
    results["checks"]["local_history"] = cr

    # 6. Domain reputation (our existing heuristic system)
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
def get_csrf():
    """Return the CSRF token bound to this session. Must be called before
    any mutating request."""
    sess = _current_session()
    return jsonify({
        "csrf": sess.csrf_token,
        "launchSecret": bool(LAUNCH_SECRET),
    })


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Expose user-facing security/privacy toggles so the UI can reflect
    what is on and off."""
    return jsonify({
        "logging_enabled": _LOGGING_ENABLED,
        "user_data_dir": str(USER_DATA_DIR),
    })


@app.route("/api/log", methods=["GET"])
def get_log():
    """Return the session log file contents. Requires a matching session
    cookie, an allowlisted Origin, and (if configured) the launch secret."""
    if not _origin_ok() or not _launch_secret_ok():
        return jsonify({"error": "Forbidden"}), 403
    if not _LOGGING_ENABLED:
        return jsonify({"log": "Logging is disabled."})
    if not LOG_FILE.exists():
        return jsonify({"log": "No session activity recorded yet."})
    try:
        text = LOG_FILE.read_text(encoding="utf-8")
    except OSError as e:
        return jsonify({"error": f"Could not read log: {e}"}), 500
    return jsonify({"log": text})


@app.route("/api/log/download", methods=["GET"])
def download_log():
    """Download the session log. Same origin + launch-secret gates as /api/log."""
    if not _origin_ok() or not _launch_secret_ok():
        return "Forbidden", 403
    if not _LOGGING_ENABLED or not LOG_FILE.exists():
        return "No log file available.", 404
    return send_from_directory(
        str(LOG_FILE.parent), LOG_FILE.name,
        as_attachment=True, mimetype="text/plain"
    )


@app.route("/api/log/clear", methods=["POST"])
@require_csrf
def clear_log():
    """User-initiated log reset."""
    try:
        if LOG_FILE.exists():
            LOG_FILE.unlink()
    except OSError as e:
        return _safe_error(e, "Could not clear log")
    return jsonify({"ok": True})


@app.route("/api/phishtank/status", methods=["GET"])
def phishtank_status():
    """Diagnostic — shows whether the PhishTank feed is loaded, how many
    entries it has, and when it was last refreshed. No auth needed; this
    is project-public info."""
    with _phishtank_lock:
        updated = _phishtank_data["updated_at"]
        return jsonify({
            "loaded": bool(updated),
            "updated_at": updated.isoformat() if updated else None,
            "entries": _phishtank_data["count"],
            "unique_domains": len(_phishtank_data["domains"]),
            "unique_urls": len(_phishtank_data["urls"]),
        })


@app.route("/api/reputation/<domain>", methods=["GET"])
def get_reputation(domain):
    """Return worldwide domain reputation."""
    rep = _check_domain_reputation(domain)
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
    """Return local scan history stats (per-device, not a real community)."""
    db = _load_community_db()
    domains = db.get("domains", {})
    total_reports = sum(d.get("phishing", 0) + d.get("safe", 0) for d in domains.values())
    phishing_domains = sum(1 for d in domains.values()
                          if d.get("phishing", 0) > d.get("safe", 0))
    return jsonify({
        "total_domains_tracked": len(domains),
        "total_scan_reports": total_reports,
        "known_phishing_domains": phishing_domains,
        "updated": db.get("updated", ""),
        "note": "Local scan history — shown for info only, does not affect verdicts.",
    })


@app.route("/api/report-sender", methods=["POST"])
@require_csrf
def report_sender():
    """User explicitly reports a sender domain. Writes to both the local
    per-device history (legacy fallback for offline / unsigned-in use)
    and the Supabase threat_reports table (cross-user community DB)."""
    data = request.get_json(silent=True) or {}
    domain = (data.get("domain") or "").strip().lower()
    is_phishing = bool(data.get("is_phishing"))
    if not domain:
        return jsonify({"error": "domain required"}), 400

    # Local per-device history (legacy). Kept so unsigned-in / offline use
    # still has a record.
    _community_report(domain, is_phishing, user_confirmed=True)

    # Supabase cross-user community DB. Append-only via RLS; duplicate
    # (reporter_id, domain, category) tuples are ignored silently.
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
            print(f"WARNING: threat_reports insert failed "
                  f"[{resp.status_code}]: {resp.text[:200]}")

    return jsonify({"ok": True})


@app.route("/api/sender-dna/<path:sender_addr>", methods=["GET"])
def get_sender_dna(sender_addr):
    """Build/retrieve sender DNA profile and compare against the referenced
    email (by message id)."""
    sess = _current_session()
    emails = _session_email_list(sess)
    addr = sender_addr.lower().strip()

    profile = _build_sender_profile(addr, emails)
    if profile:
        with _sender_profiles_lock:
            _sender_profiles[addr] = profile
        _persist_sender_profile_to_supabase(addr, profile, _request_jwt())

    message_id = request.args.get('message_id') or request.args.get('id')
    comparison = None
    if message_id and profile:
        target = next((e for e in emails if e.get("id") == message_id), None)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Microsoft Graph API Client (OAuth2 Auth Code + PKCE)
# ─────────────────────────────────────────────────────────────────────────────
AUTHORITY = "https://login.microsoftonline.com/common"
GRAPH_URL = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.ReadWrite", "User.Read"]
REDIRECT_URI = f"http://localhost:{_FLASK_PORT}/auth/callback"

_OAUTH_TIMEOUT = timedelta(minutes=10)
_MAX_PENDING_OAUTH = 1000  # hard cap against spam

# OAuth state is keyed by the `state` parameter and carries the sid of the
# session that started the flow, so the popup (which may lose its session
# cookie on the round-trip through Microsoft) can still be routed back to the
# correct SessionState.
_global_pending_oauth = {}  # state -> {client_id, verifier, expires, sid}
_global_pending_lock = _threading_early.Lock()


def _sweep_pending_oauth():
    """Drop expired pending OAuth entries and evict oldest over the cap.

    Runs on a daemon timer so the dictionary cannot grow without bound even
    when nobody starts a new login."""
    now = datetime.now()
    with _global_pending_lock:
        expired = [k for k, v in _global_pending_oauth.items() if now >= v["expires"]]
        for k in expired:
            del _global_pending_oauth[k]
        overflow = len(_global_pending_oauth) - _MAX_PENDING_OAUTH
        if overflow > 0:
            # Evict the oldest entries first
            oldest = sorted(_global_pending_oauth.items(), key=lambda kv: kv[1]["expires"])
            for k, _ in oldest[:overflow]:
                del _global_pending_oauth[k]


def _start_background_sweeper():
    """Daemon thread that sweeps _global_pending_oauth every 60s."""
    def _loop():
        while True:
            try:
                _sweep_pending_oauth()
            except Exception as e:
                print(f"WARNING: pending OAuth sweep failed: {e}")
            _time.sleep(60)
    t = _threading_early.Thread(target=_loop, name="pg-oauth-sweeper", daemon=True)
    t.start()


def _generate_pkce():
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


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
    """Fetch all emails from a mail folder via pagination.

    Guards:
      * `next_link` must resolve to https://graph.microsoft.com/* — a hostile
        tenant cannot redirect us to 127.0.0.1 or the metadata service.
      * Pagination is capped at _MAX_GRAPH_PAGES so an infinite-chain response
        cannot wedge the backend.
    """
    all_emails = []
    params = {
        "$top": 250,
        "$select": "id,subject,from,receivedDateTime,bodyPreview,body,isRead,internetMessageHeaders,hasAttachments",
        "$orderby": "receivedDateTime desc",
    }
    result = _graph_request(sess, f"me/mailFolders/{folder}/messages", params)
    all_emails.extend(result.get("value", []))

    next_link = result.get("@odata.nextLink")
    pages = 1
    while next_link and pages < _MAX_GRAPH_PAGES:
        if not _is_graph_url(next_link):
            print(f"WARNING: rejected non-Graph nextLink: {next_link[:80]}")
            break
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
        pages += 1
    if pages >= _MAX_GRAPH_PAGES:
        print(f"WARNING: pagination cap ({_MAX_GRAPH_PAGES}) hit for {folder}")

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
        raise ValueError("Token expired")
    r.raise_for_status()
    return r


def _graph_delete(sess, endpoint):
    token = sess.graph_token
    if not token["access_token"]:
        raise ValueError("Not authenticated")
    headers = {"Authorization": f"Bearer {token['access_token']}"}
    r = http_requests.delete(
        f"{GRAPH_URL}/{endpoint}", headers=headers, timeout=HTTP_TIMEOUT,
    )
    if r.status_code == 401:
        token["access_token"] = None
        raise ValueError("Token expired")
    r.raise_for_status()
    return r


# ── OAuth Routes ─────────────────────────────────────────────────────────────

@app.route("/api/auth/status", methods=["GET"])
def auth_status():
    """Return current connection status for this session. Empty when the
    user hasn't signed in to Microsoft via Supabase yet."""
    sess = _current_session()
    connected = bool(sess.graph_token["access_token"] and sess.live_mode)
    return jsonify({
        "connected": connected,
        "live_mode": sess.live_mode,
        "user_name": sess.graph_user["name"],
        "user_email": sess.graph_user["email"],
        "user_photo": "",
    })


@app.route("/api/auth/photo", methods=["GET"])
def get_user_photo():
    sess = _current_session()
    if not sess.graph_token["access_token"]:
        return "", 404
    try:
        headers = {"Authorization": f"Bearer {sess.graph_token['access_token']}"}
        r = http_requests.get(
            f"{GRAPH_URL}/me/photo/$value", headers=headers, timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 200:
            from flask import Response
            return Response(r.content, mimetype=r.headers.get("Content-Type", "image/jpeg"))
    except Exception:
        pass
    return "", 404


@app.route("/api/auth/connect", methods=["POST"])
@require_csrf
def auth_connect():
    if http_requests is None:
        return jsonify({"error": "requests library not installed"}), 500

    data = request.get_json(silent=True) or {}
    client_id = (data.get("client_id") or "").strip()
    if not _valid_client_id(client_id):
        return jsonify({"error": "Client ID must be a valid UUID"}), 400

    sess = _current_session()
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)
    now = datetime.now()

    with _global_pending_lock:
        # Expire stale pending entries across all sessions
        stale = [k for k, v in _global_pending_oauth.items() if now >= v["expires"]]
        for k in stale:
            del _global_pending_oauth[k]

        _global_pending_oauth[state] = {
            "client_id": client_id,
            "verifier": verifier,
            "expires": now + _OAUTH_TIMEOUT,
            "sid": sess.sid,
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


@app.route("/auth/callback")
def auth_callback():
    """Handle Microsoft OAuth redirect. The callback arrives in a popup that
    may have lost its session cookie on the round-trip through Microsoft, so
    we resolve the target session via the sid stashed alongside `state` when
    /api/auth/connect was called."""
    error = request.args.get("error")
    if error:
        desc = request.args.get("error_description", error)
        # Cap the length so a pathological Microsoft error page cannot render
        # a multi-MB blob into the popup, and drop CR/LF that could confuse
        # a browser into treating the response as split.
        desc = str(desc).replace("\r", " ").replace("\n", " ")[:500]
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Sign-in Failed</h2><p>{html.escape(desc)}</p>
        <p>You can close this tab.</p></body></html>"""

    received_state = request.args.get("state")
    with _global_pending_lock:
        pending = _global_pending_oauth.pop(received_state, None)
    if not pending or datetime.now() >= pending["expires"]:
        return """<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>Invalid or expired state parameter.</p>
        <p>You can close this tab.</p></body></html>"""

    code = request.args.get("code")
    if not code:
        return """<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>No authorization code received.</p>
        <p>You can close this tab.</p></body></html>"""

    # Resolve the SessionState that started this flow
    with _sessions_lock:
        sess = _sessions.get(pending["sid"])
    if sess is None:
        return """<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Session Expired</h2><p>Please open PhishGuard and sign in again.</p>
        <p>You can close this tab.</p></body></html>"""

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
        r = http_requests.post(token_url, data=token_data, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            # Microsoft sometimes returns HTML on gateway errors (502/503).
            # Parse as JSON defensively so we don't crash on a non-JSON body.
            try:
                err = r.json()
                desc = err.get("error_description", err.get("error", "Token exchange failed"))
            except (ValueError, Exception):
                desc = f"Token exchange failed (HTTP {r.status_code})"
            short = str(desc).split("\r\n")[0] if "\r\n" in str(desc) else str(desc)
            return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
            <h2>Sign-in Failed</h2><p>{html.escape(short[:500])}</p>
            <p>You can close this tab.</p></body></html>"""

        try:
            tokens = r.json()
        except ValueError:
            return """<html><body style="font-family:system-ui;text-align:center;padding:60px">
            <h2>Sign-in Failed</h2><p>Could not parse token response.</p>
            <p>You can close this tab.</p></body></html>"""
        access_token = tokens.get("access_token")
        expires_in = int(tokens.get("expires_in", 3600))

        sess.graph_token["access_token"] = access_token
        sess.graph_token["expiry"] = datetime.now() + timedelta(seconds=expires_in)

        try:
            user_info = _graph_request(sess, "me")
            # Sanitise before storage — displayName/email land in the sidebar,
            # sender cards, and (escaped) the callback HTML. Strip bidi
            # overrides and control chars so a malicious Graph response can't
            # spoof UI text or inject terminal escapes.
            sess.graph_user["name"] = _sanitise_display(user_info.get("displayName", "User"))
            sess.graph_user["email"] = _sanitise_display(
                user_info.get("mail") or user_info.get("userPrincipalName", "")
            )
        except Exception:
            sess.graph_user["name"] = "Connected"
            sess.graph_user["email"] = ""

        try:
            sess.live_emails = _fetch_all_emails(sess)
        except Exception:
            sess.live_emails = []
        try:
            sess.junk_emails = _fetch_junk_emails(sess)
        except Exception:
            sess.junk_emails = []

        sess.live_mode = True
        sess.scan_results.clear()

        _log_session_event("LOGIN", {
            "User": sess.graph_user["name"],
            "Email": sess.graph_user["email"],
            "Emails Loaded": len(sess.live_emails),
        })

    except Exception as e:
        return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
        <h2>Error</h2><p>{html.escape(str(e))}</p>
        <p>You can close this tab.</p></body></html>"""

    return f"""<html><body style="font-family:system-ui;text-align:center;padding:60px">
    <h2>Connected to Outlook</h2>
    <p>Signed in as {html.escape(sess.graph_user['name'] or 'User')}</p>
    <p>You can close this tab and return to PhishGuard.</p>
    <script>window.close();</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  External-browser OAuth handoff
# ─────────────────────────────────────────────────────────────────────────────
# When the renderer wants the user to sign in with an existing Microsoft
# session on their system, it opens Supabase's OAuth URL in the user's
# DEFAULT browser (via shell.openExternal). The browser has the user's
# real Microsoft cookies, so Microsoft shows the account picker with
# "Signed in" labels.
#
# After auth the browser lands at /auth/external-callback?nonce=<x>#tokens.
# The hash isn't sent to Flask, so we serve a small HTML page that grabs
# the hash via JS and POSTs the tokens to /api/auth/external-deliver.
# Electron's renderer polls /api/auth/external-poll until tokens land.
#
# Security: the nonce is the only auth. It's a random UUID generated by
# the renderer, passed to Flask via /api/auth/external-start before the
# OAuth flow begins. The browser sees the nonce in the redirect URL.
# Both Flask sides validate the nonce. Tokens are one-shot — consumed
# on first successful poll so a stale URL can't deliver twice.

_external_oauth_lock = _threading_early.Lock()
_external_oauth_pending = {}   # nonce -> {"created", "tokens", ...}
_EXTERNAL_OAUTH_TTL_SECONDS = 600  # 10 minutes from start to poll


@app.route("/api/auth/external-start", methods=["POST"])
@require_csrf
def auth_external_start():
    """Renderer registers a fresh nonce before opening the browser."""
    data = request.get_json(silent=True) or {}
    nonce = str(data.get("nonce", "")).strip()
    if not nonce or len(nonce) < 16 or len(nonce) > 128:
        return jsonify({"error": "invalid nonce"}), 400
    now = datetime.now()
    with _external_oauth_lock:
        # GC expired entries.
        stale = [k for k, v in _external_oauth_pending.items()
                 if (now - v["created"]).total_seconds() > _EXTERNAL_OAUTH_TTL_SECONDS]
        for k in stale:
            _external_oauth_pending.pop(k, None)
        # Cap total entries to defang a renderer that spams nonces.
        if len(_external_oauth_pending) > 200:
            return jsonify({"error": "too many pending"}), 429
        _external_oauth_pending[nonce] = {"created": now, "tokens": None}
    return jsonify({"ok": True})


_EXTERNAL_CALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>PhishGuard — Sign-in</title>
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
  <h1 id="title">Completing sign-in…</h1>
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

  if (err) { fail((errDesc || err) + ' — you can close this tab.'); return; }
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
    """Page the system browser lands on after Microsoft → Supabase OAuth.
    Strips the tokens out of the URL hash via inline JS and POSTs them
    to /api/auth/external-deliver."""
    return _EXTERNAL_CALLBACK_HTML


@app.route("/api/auth/external-deliver", methods=["POST"])
def auth_external_deliver():
    """Browser callback page POSTs the tokens here.

    Intentionally NOT CSRF-protected: the request comes from the user's
    system browser, which has a different cookie jar than Electron's
    renderer. The nonce — unguessable, registered before the OAuth
    flow began — is what authenticates the delivery.
    """
    data = request.get_json(silent=True) or {}
    nonce = str(data.get("nonce", "")).strip()
    if not nonce:
        return jsonify({"error": "nonce required"}), 400
    access_token = str(data.get("access_token", "")).strip()
    if not access_token:
        return jsonify({"error": "missing access_token"}), 400

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
            "expires_in": int(data.get("expires_in") or 3600),
        }
        entry["delivered_at"] = datetime.now()
    return jsonify({"ok": True})


@app.route("/api/auth/external-poll", methods=["GET"])
def auth_external_poll():
    """Electron polls here. Returns 'pending' until the browser has
    delivered tokens. On 'ready' the tokens are returned ONCE and the
    entry is removed."""
    nonce = (request.args.get("nonce") or "").strip()
    if not nonce:
        return jsonify({"status": "invalid"}), 400
    with _external_oauth_lock:
        entry = _external_oauth_pending.get(nonce)
        if not entry:
            return jsonify({"status": "expired"}), 404
        if not entry.get("tokens"):
            return jsonify({"status": "pending"})
        tokens = entry["tokens"]
        # One-shot delivery.
        _external_oauth_pending.pop(nonce, None)
    return jsonify({"status": "ready", **tokens})


@app.route("/api/auth/supabase-provider", methods=["POST"])
@require_csrf
def auth_supabase_provider():
    """Renderer forwards the Supabase session's `provider_token` (the
    Microsoft Graph access token) so Flask can call Outlook on the
    user's behalf — same SessionState slots the legacy PKCE flow used
    to populate, just plumbed through Supabase instead.

    Body shape:
      {
        "provider_token": "<MS access token>",
        "provider_refresh_token": "<optional refresh>",
        "expires_in": 3600,
        "user": { "name": "...", "email": "..." }
      }
    """
    if http_requests is None:
        return jsonify({"error": "requests library not installed"}), 500

    data = request.get_json(silent=True) or {}
    token = data.get("provider_token", "")
    if not isinstance(token, str) or not token:
        return jsonify({"error": "provider_token required"}), 400
    try:
        expires_in = int(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600

    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    name = _sanitise_display(user.get("name", "") or "")
    email = _sanitise_display(user.get("email", "") or "")

    sess = _current_session()
    sess.graph_token["access_token"] = token
    sess.graph_token["expiry"] = datetime.now() + timedelta(seconds=expires_in)
    sess.graph_user["name"] = name or "Connected"
    sess.graph_user["email"] = email

    try:
        sess.live_emails = _fetch_all_emails(sess)
    except Exception:
        sess.live_emails = []
    try:
        sess.junk_emails = _fetch_junk_emails(sess)
    except Exception:
        sess.junk_emails = []

    sess.live_mode = True
    sess.scan_results.clear()

    _log_session_event("LOGIN", {
        "User": sess.graph_user["name"],
        "Email": sess.graph_user["email"],
        "Source": "Supabase Microsoft OAuth",
        "Emails Loaded": len(sess.live_emails),
    })

    return jsonify({
        "ok": True,
        "emails_loaded": len(sess.live_emails),
    })


@app.route("/api/auth/disconnect", methods=["POST"])
@require_csrf
def auth_disconnect():
    sess = _current_session()
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
    sess.graph_user = {"name": "", "email": ""}
    sess.live_emails = []
    sess.junk_emails = []
    sess.live_mode = False
    sess.scan_results.clear()
    sess.pending_oauth.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/refresh", methods=["POST"])
@require_csrf
def auth_refresh_emails():
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


# ── Email endpoints (message-id based, folder-aware) ────────────────────────

def _session_email_list(sess, folder="inbox"):
    """Return the active email list for this session. Empty unless the
    user has signed in to Microsoft via Supabase (live_mode True)."""
    if folder == "junk":
        return sess.junk_emails
    if sess.live_mode:
        return sess.live_emails
    return []


def _find_message(sess, message_id, folder=None):
    """Locate a message in this session's inbox or junk folder. If `folder`
    is given, only that folder is searched. Returns (email, folder_key)."""
    if folder == "inbox" or folder is None:
        for e in _session_email_list(sess, "inbox"):
            if e.get("id") == message_id:
                return e, "inbox"
    if folder == "junk" or folder is None:
        for e in _session_email_list(sess, "junk"):
            if e.get("id") == message_id:
                return e, "junk"
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


def _normalize_email(sess, email, idx, folder):
    """Convert Graph API / mock email to flat format the frontend expects.

    `idx` is included only as a display aid; all mutating API calls are
    keyed by the stable `id` field (Graph message id / mock id).
    """
    sender = email.get("from", {})
    if "emailAddress" in sender:
        sender_name = sender["emailAddress"].get("name", "")
        sender_addr = sender["emailAddress"].get("address", "")
    else:
        sender_name = sender.get("name", "")
        sender_addr = sender.get("address", "")

    body_raw = email.get("body", {})
    if isinstance(body_raw, dict):
        body_content = body_raw.get("content", "") or ""
    elif isinstance(body_raw, str):
        body_content = body_raw
    else:
        body_content = email.get("bodyPreview", "") or ""
    body_text = _html_to_text(body_content) if body_content else ""

    headers = _parse_auth_headers(email.get("internetMessageHeaders"))

    body_html = ""
    if isinstance(body_raw, dict) and body_raw.get("contentType", "").lower() == "html":
        body_html = body_raw.get("content", "")
    elif isinstance(body_raw, str) and "<" in body_raw and ">" in body_raw:
        body_html = body_raw

    message_id = email.get("id", "")
    scanned_result = sess.scan_results.get(message_id)

    return {
        "id": message_id,
        "messageId": message_id,
        "folder": folder,
        "idx": idx,  # kept so UI can preserve order; never used for mutations
        "subject": email.get("subject", "(No Subject)"),
        "sender_name": sender_name,
        "sender": sender_addr,
        "date": email.get("receivedDateTime", ""),
        "isRead": bool(email.get("isRead", True)),
        "bodyPreview": email.get("bodyPreview", ""),
        "body": body_text,
        "bodyHtml": body_html,
        "hasAttachments": email.get("hasAttachments", False),
        "attachments": email.get("attachments", []),
        "headers": headers,
        "scanned": scanned_result is not None,
        "scanResult": scanned_result,
    }


@app.route("/api/emails", methods=["GET"])
def get_emails_v2():
    sess = _current_session()
    folder = (request.args.get("folder") or "inbox").lower()
    if folder not in ("inbox", "junk"):
        return jsonify({"error": "Unknown folder"}), 400
    emails = _session_email_list(sess, folder)
    return jsonify({
        "emails": [_normalize_email(sess, e, i, folder) for i, e in enumerate(emails)],
        "folder": folder,
    })


@app.route("/api/emails/junk", methods=["GET"])
def get_junk_emails():
    sess = _current_session()
    junk = _session_email_list(sess, "junk")
    return jsonify({
        "emails": [_normalize_email(sess, e, i, "junk") for i, e in enumerate(junk)],
        "folder": "junk",
    })


@app.route("/api/messages/<path:message_id>", methods=["GET"])
def get_message(message_id):
    sess = _current_session()
    folder = request.args.get("folder")
    email, found_folder = _find_message(sess, message_id, folder)
    if not email:
        return jsonify({"error": "Message not found"}), 404
    return jsonify(_normalize_email(sess, email, -1, found_folder))


@app.route("/api/messages/<path:message_id>/attachments/<path:attachment_id>/analyze",
           methods=["POST"])
@require_csrf
def analyze_attachment(message_id, attachment_id):
    """Pull an attachment from Outlook (real mode only), SHA-256 hash
    it locally, and look the hash up on VirusTotal. No file content
    leaves the device — only the hash.

    Response shapes:
      • {"configured": False, ...}     — VT API key not set
      • {"analyzed": False, ...}       — couldn't fetch bytes (dev mode,
                                          Graph 401, etc.)
      • {"analyzed": True, "found": false, ...} — hashed but VT has no
                                          report; we DON'T upload.
      • {"analyzed": True, "found": true, "malicious": N, ...} — VT
                                          returned a report.
    """
    sess = _current_session()
    email, _folder = _find_message(sess, message_id)
    if not email:
        return jsonify({"error": "Message not found"}), 404

    attachments = email.get("attachments", []) or []
    att = None
    for a in attachments:
        if not isinstance(a, dict):
            continue
        if a.get("id") == attachment_id or a.get("name") == attachment_id:
            att = a
            break
    if att is None:
        return jsonify({"error": "Attachment not found"}), 404

    name = att.get("name", "") or att.get("filename", "") or "file"

    if not VIRUSTOTAL_API_KEY:
        return jsonify({
            "configured": False,
            "name": name,
            "message": ("VirusTotal API key not configured. Set "
                        "VIRUSTOTAL_API_KEY env var to enable."),
        })

    # Bytes come from Graph in live mode; dev-mode mock emails have no
    # real bytes so we can't compute a real hash.
    real_aid = att.get("id", "")
    content_bytes = _fetch_attachment_bytes(sess, message_id, real_aid)
    if not content_bytes:
        return jsonify({
            "configured": True,
            "analyzed": False,
            "name": name,
            "message": ("Could not fetch attachment bytes — analysis "
                        "is only available for real Outlook mail."),
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
@require_csrf
def move_to_junk(message_id):
    sess = _current_session()
    # Only inbox messages can be moved to junk — enforce the folder context.
    # _find_message uses _session_email_list which lazily seeds the dev-mode
    email, folder = _find_message(sess, message_id, "inbox")
    if not email:
        return jsonify({"error": "Message not found in inbox"}), 404
    try:
        resp = _graph_post(sess, f"me/messages/{message_id}/move",
                            {"destinationId": "junkemail"})
        # Graph's /move is a copy-then-delete, so the moved message gets
        # a NEW id in the destination folder. The response body is the
        # new message; pull its id out so the scan result we kept and
        # any future operations target the right row.
        new_id = message_id
        try:
            moved = resp.json() if resp is not None else None
            if isinstance(moved, dict) and moved.get("id"):
                new_id = moved["id"]
        except Exception:
            pass

        # Order matters: filter inbox FIRST (using the original id),
        # THEN mutate email["id"]. If we mutate first, the in-place
        # change makes the filter miss the email and the inbox keeps a
        # duplicate of the now-moved message.
        sess.live_emails = [e for e in sess.live_emails if e.get("id") != message_id]
        email["id"] = new_id
        sess.junk_emails.insert(0, email)

        # Remap the scan result so the score ring follows the email
        # into the junk folder instead of getting orphaned.
        if message_id in sess.scan_results and new_id != message_id:
            sr = sess.scan_results.pop(message_id)
            sr["id"] = new_id
            sr["messageId"] = new_id
            sess.scan_results[new_id] = sr

        return jsonify({"ok": True, "message": "Moved to junk",
                        "old_id": message_id, "new_id": new_id})
    except Exception as e:
        return _safe_error(e, "Move failed")


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


def _scan_history_row(email, prediction, confidence, threat_intel):
    """Shape a scan result into the row format the Supabase scan_history
    table expects. Sender body / content NEVER goes here — only metadata."""
    sender = email.get("from", {})
    sender_addr = ""
    if isinstance(sender, dict) and "emailAddress" in sender:
        sender_addr = sender["emailAddress"].get("address", "") or ""
    domain = sender_addr.split("@")[-1].lower() if "@" in sender_addr else None
    return {
        "message_id": email.get("id", "") or "",
        "sender_domain": domain or None,
        "prediction": int(prediction),
        "confidence": float(min(1.0, max(0.0, confidence))),
        "signals": {
            "intel_signals": list(threat_intel.get("signals", []) or [])[:20],
            "checks_run": list((threat_intel.get("checks", {}) or {}).keys()),
            "confirmed": bool(threat_intel.get("confirmed_phishing", False)),
        },
    }


def _scan_one(sess, email, jwt=None):
    """Run the dual-model + threat-intel pipeline on a single email. Stores
    the result in this session's scan_results keyed by message id."""
    if not detector.is_trained:
        raise RuntimeError("Model not loaded")

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

    # Cap body length before running the URL regex — the bracket-class pattern
    # below is ReDoS-safe on normal input but we still bound work so a
    # pathological 10 MB body cannot pin a scan thread.
    scan_body = body_t[:_MAX_BODY_FOR_REGEX]
    # Character class excludes common delimiters; no alternation, no nested
    # quantifiers — linear-time match.
    url_pattern = re.compile(r'https?://[^\s<>"\'`]{1,2048}')
    urls_found = url_pattern.findall(scan_body)[:_MAX_URLS_PER_EMAIL]

    threat_intel = run_threat_intel(email, urls_found)

    prediction, confidence, url_analysis, header_result = detector.predict(
        full, headers=email.get("internetMessageHeaders"))
    url_analysis, header_result = _make_serializable(url_analysis, header_result)
    prediction = _to_native(prediction)
    confidence = _to_native(confidence)

    if threat_intel["confirmed_phishing"]:
        prediction = 1
        confidence = max(confidence, 0.99)
    if prediction == 1 and threat_intel["confidence_boost"] > 0:
        confidence = min(1.0, confidence + threat_intel["confidence_boost"])

    message_id = email.get("id", "")
    result = {
        "id": message_id,
        "messageId": message_id,
        "prediction": prediction,
        "confidence": confidence,
        "url_analysis": url_analysis,
        "header_result": header_result,
        "threat_intel": {
            "confirmed": threat_intel["confirmed_phishing"],
            "signals": threat_intel["signals"],
            "checks_run": list(threat_intel["checks"].keys()),
        },
    }
    sess.scan_results[message_id] = result

    # Persist to Supabase scan_history when we have a signed-in user.
    # Metadata only — email content stays in SessionState memory.
    # RLS requires user_id = auth.uid(); column has no default, so we
    # decode the JWT to populate it.
    if jwt:
        user_id = _jwt_user_id(jwt)
        if user_id:
            row = _scan_history_row(email, prediction, confidence, threat_intel)
            row["user_id"] = user_id
            resp = _supabase_request("POST", "scan_history", jwt, json_body=row)
            if resp is not None and resp.status_code >= 400:
                print(f"WARNING: scan_history insert failed "
                      f"[{resp.status_code}]: {resp.text[:200]}")

    return result


@app.route("/api/messages/<path:message_id>/scan", methods=["POST"])
@require_csrf
def scan_message(message_id):
    sess = _current_session()
    email, folder = _find_message(sess, message_id)
    if not email:
        return jsonify({"error": "Message not found"}), 404
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503
    try:
        return jsonify(_scan_one(sess, email, jwt=_request_jwt()))
    except Exception as e:
        return _safe_error(e, "Scan failed")


@app.route("/api/scan-all", methods=["POST"])
@require_csrf
def scan_all_v2():
    sess = _current_session()
    if not detector.is_trained:
        return jsonify({"error": "Model not loaded"}), 503
    folder = (request.args.get("folder") or "inbox").lower()
    if folder not in ("inbox", "junk"):
        return jsonify({"error": "Unknown folder"}), 400

    # Per-session cooldown and concurrency guard — one scan-all at a time per
    # session, and no faster than once every _SCAN_ALL_COOLDOWN seconds.
    now = datetime.now()
    last = getattr(sess, "_last_scan_all", None)
    in_flight = getattr(sess, "_scan_all_in_flight", False)
    if in_flight:
        return jsonify({"error": "A scan-all is already running for this session"}), 429
    if last and (now - last).total_seconds() < _SCAN_ALL_COOLDOWN:
        wait = int(_SCAN_ALL_COOLDOWN - (now - last).total_seconds())
        return jsonify({"error": f"Please wait {wait}s before scanning again"}), 429

    sess._scan_all_in_flight = True
    try:
        emails = _session_email_list(sess, folder)
        jwt = _request_jwt()
        results = []
        for email in emails:
            try:
                results.append(_scan_one(sess, email, jwt=jwt))
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
        "total": len(_session_email_list(sess, "inbox")),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    is_electron = os.environ.get("ELECTRON_MODE") == "1"
    # Debug mode is opt-in via PHISHGUARD_DEBUG=1. Never on in Electron.
    debug = os.environ.get("PHISHGUARD_DEBUG") == "1" and not is_electron
    _start_background_sweeper()
    _start_phishtank_refresher()  # Downloads feed in background, refreshes every 6h.
    # Bind to 127.0.0.1 only — do not expose on all interfaces.
    app.run(host="127.0.0.1", debug=debug, port=_FLASK_PORT, use_reloader=debug)
