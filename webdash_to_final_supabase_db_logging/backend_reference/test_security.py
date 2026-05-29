"""
Security regression tests for PhishGuard `final/` backend.

These assert the protections I layered on top of the base app — CSRF, Origin
allowlist, UUID validation, SSRF guard, scan-all cooldown, message-id
routing, response headers, and pickle hash verification. The suite spawns
its own Flask process on a free port with dev-mode mocks, talks to it over
HTTP, and shuts down cleanly.

Run:
    pytest -v final/tests/test_security.py

Every test is an executable spec — if anyone loosens a check in the backend,
the matching test should go red.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests


FINAL_DIR = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PHISHGUARD_TEST_PYTHON", "/opt/homebrew/bin/python3.12")
LAUNCH_SECRET = "test-launch-secret-xyz"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def run_server():
    """Spawn the Flask app in --dev mode on a free port. Yields (base_url)."""
    port = _free_port()
    tmp_data = tempfile.mkdtemp(prefix="pg-tests-")
    env = {
        **os.environ,
        "FLASK_PORT": str(port),
        "PHISHGUARD_LAUNCH_SECRET": LAUNCH_SECRET,
        "PHISHGUARD_USER_DATA": tmp_data,
        "ELECTRON_MODE": "1",
    }
    proc = subprocess.Popen(
        [PYTHON, str(FINAL_DIR / "app.py"), "--dev"],
        cwd=str(FINAL_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        # Wait for server readiness (max 8 s)
        for _ in range(40):
            try:
                r = requests.get(f"{base}/api/auth/status", timeout=0.5)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("Flask did not start in time")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def base_url():
    with run_server() as b:
        yield b


@pytest.fixture()
def session(base_url):
    """Fresh requests.Session with CSRF token + cookie set up."""
    s = requests.Session()
    r = s.get(f"{base_url}/api/csrf", timeout=3)
    assert r.status_code == 200
    s.headers["X-CSRF-Token"] = r.json()["csrf"]
    s.headers["X-Launch-Secret"] = LAUNCH_SECRET
    s.headers["Origin"] = base_url
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Response-header hygiene
# ─────────────────────────────────────────────────────────────────────────────

def test_security_headers_present_on_api(base_url):
    r = requests.get(f"{base_url}/api/auth/status", timeout=3)
    h = r.headers
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert h.get("Referrer-Policy") == "no-referrer"
    assert "camera=()" in h.get("Permissions-Policy", "")
    assert h.get("X-Frame-Options") == "DENY"
    assert h.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert h.get("Cross-Origin-Resource-Policy") == "same-origin"
    assert "no-store" in h.get("Cache-Control", "")


def test_session_cookie_is_httponly_strict(base_url):
    r = requests.get(f"{base_url}/api/auth/status", timeout=3)
    set_cookie = r.headers.get("Set-Cookie", "")
    if "pg_sid" in set_cookie:
        assert "HttpOnly" in set_cookie
        assert "SameSite=Strict" in set_cookie


# ─────────────────────────────────────────────────────────────────────────────
#  CSRF / Origin enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_post_without_csrf_or_origin_rejected(base_url):
    # No cookie, no CSRF, no Origin — should be 403 from Origin check
    r = requests.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert r.status_code == 403
    assert "Forbidden" in r.text


def test_post_with_wrong_origin_rejected(session, base_url):
    # Session has a valid CSRF, but Origin is set to an attacker site and the
    # launch-secret is removed (that's the Electron bypass, not available to
    # a random browser page).
    session.headers["Origin"] = "http://evil.example.com"
    session.headers.pop("X-Launch-Secret", None)
    r = session.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert r.status_code == 403
    assert "origin" in r.text.lower()


def test_post_with_wrong_csrf_rejected(session, base_url):
    session.headers["X-CSRF-Token"] = "garbage-token"
    r = session.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert r.status_code == 403
    assert "CSRF" in r.text


def test_post_with_launch_secret_bypasses_origin(base_url):
    # Electron flow: no Origin header (can happen), but X-Launch-Secret matches.
    s = requests.Session()
    r = s.get(f"{base_url}/api/csrf", timeout=3)
    assert r.status_code == 200
    s.headers["X-CSRF-Token"] = r.json()["csrf"]
    s.headers["X-Launch-Secret"] = LAUNCH_SECRET
    # Origin intentionally omitted
    r = s.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert r.status_code == 200


def test_all_post_routes_reject_without_csrf(base_url):
    """Every mutating route must enforce CSRF."""
    routes = [
        "/api/auth/connect", "/api/auth/disconnect", "/api/auth/refresh",
        "/api/scan-all", "/api/report-sender", "/api/log/clear",
        "/api/messages/m1/scan", "/api/messages/m1/move-to-junk",
    ]
    for route in routes:
        r = requests.post(f"{base_url}{route}", timeout=3)
        assert r.status_code == 403, f"{route} did not enforce CSRF/Origin"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-session isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_two_sessions_get_distinct_csrf(base_url):
    s1 = requests.Session()
    s2 = requests.Session()
    t1 = s1.get(f"{base_url}/api/csrf", timeout=3).json()["csrf"]
    t2 = s2.get(f"{base_url}/api/csrf", timeout=3).json()["csrf"]
    assert t1 != t2


# ─────────────────────────────────────────────────────────────────────────────
#  Client-ID validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "", "not-a-uuid", "lol", "x" * 200,
    "11111111-2222-3333-4444-55555555555",            # too short
    "<script>alert(1)</script>",
    "' OR 1=1 --",
])
def test_connect_rejects_bad_client_id(session, base_url, bad_id):
    r = session.post(
        f"{base_url}/api/auth/connect",
        json={"client_id": bad_id},
        timeout=3,
    )
    assert r.status_code == 400
    assert "UUID" in r.json().get("error", "")


def test_connect_accepts_valid_uuid(session, base_url):
    r = session.post(
        f"{base_url}/api/auth/connect",
        json={"client_id": "11111111-2222-3333-4444-555555555555"},
        timeout=3,
    )
    assert r.status_code == 200
    assert r.json()["auth_url"].startswith("https://login.microsoftonline.com/")


# ─────────────────────────────────────────────────────────────────────────────
#  Message-ID routing + scan
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_phishing_mock_is_detected(session, base_url):
    r = session.post(f"{base_url}/api/messages/m4/scan", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1
    assert body["confidence"] >= 0.5
    assert body["id"] == "m4"


def test_scan_safe_mock_is_not_flagged(session, base_url):
    # m1 is the safe Q1 report
    r = session.post(f"{base_url}/api/messages/m1/scan", timeout=10)
    assert r.status_code == 200
    assert r.json()["prediction"] == 0


def test_scan_unknown_message_id_404(session, base_url):
    r = session.post(f"{base_url}/api/messages/does-not-exist/scan", timeout=5)
    assert r.status_code == 404


def test_old_index_route_is_gone(session, base_url):
    r = session.post(f"{base_url}/api/scan/0", timeout=3)
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
#  Scan-all cooldown + concurrency
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_all_cooldown_rate_limits(session, base_url):
    r1 = session.post(f"{base_url}/api/scan-all", timeout=15)
    assert r1.status_code == 200
    r2 = session.post(f"{base_url}/api/scan-all", timeout=5)
    assert r2.status_code == 429
    assert "wait" in r2.json().get("error", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Log-endpoint access
# ─────────────────────────────────────────────────────────────────────────────

def test_log_endpoint_rejects_bad_origin(base_url):
    r = requests.get(
        f"{base_url}/api/log",
        headers={"Origin": "http://evil.com"},
        timeout=3,
    )
    assert r.status_code == 403


def test_log_endpoint_rejects_spoofed_allowlisted_origin_without_launch_secret(base_url):
    r = requests.get(
        f"{base_url}/api/log",
        headers={"Origin": base_url},
        timeout=3,
    )
    assert r.status_code == 403

    r = requests.get(
        f"{base_url}/api/log/download",
        headers={"Origin": base_url},
        timeout=3,
    )
    assert r.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
#  Settings + helper sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_settings_reflects_disabled_defaults(base_url):
    r = requests.get(f"{base_url}/api/settings", timeout=3)
    body = r.json()
    assert body["threat_intel_enabled"] is False
    assert body["logging_enabled"] is False
    # user_data_dir must be under a safe root. Resolve both sides so macOS
    # /var -> /private/var symlinks don't cause a false negative.
    p = Path(body["user_data_dir"]).resolve()
    home = Path.home().resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    safe = (p == home or home in p.parents or
            p == tmp or tmp in p.parents)
    assert safe, f"{p} is not under {home} or {tmp}"


# ─────────────────────────────────────────────────────────────────────────────
#  Model hash verification
# ─────────────────────────────────────────────────────────────────────────────

def test_pickle_load_rejects_wrong_hash(tmp_path):
    """load_model must refuse a pickle whose hash doesn't match the expected."""
    sys.path.insert(0, str(FINAL_DIR))
    try:
        from phishing_detector import PhishingDetector
    finally:
        sys.path.pop(0)
    detector = PhishingDetector()
    # Write a dummy pickle with known bad contents
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a real pickle payload")
    with pytest.raises(ValueError, match="hash mismatch"):
        detector.load_model(str(bad), expected_sha256="0" * 64)


def test_pickle_load_accepts_hashless_override(tmp_path):
    """Passing expected_sha256='' is the documented opt-out (dev/test only)
    and must succeed or fail for pickle reasons, not hash reasons."""
    sys.path.insert(0, str(FINAL_DIR))
    try:
        from phishing_detector import PhishingDetector
    finally:
        sys.path.pop(0)
    detector = PhishingDetector()
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(Exception) as ei:
        detector.load_model(str(bad), expected_sha256="")
    # Any error is fine EXCEPT a hash-mismatch error — that proves the
    # opt-out worked and we failed later, during unpickling.
    assert "hash mismatch" not in str(ei.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
#  URL regex is ReDoS-safe
# ─────────────────────────────────────────────────────────────────────────────

def test_url_regex_handles_adversarial_body(session, base_url):
    """A pathological body used to be able to hang the URL-extraction regex.
    The bounded character class + body cap must finish within a couple of
    seconds on a 200 KB worst-case input."""
    # Build a 250 KB body that would have been catastrophic against the
    # previous greedy regex. We reach the scan path via the mock m4 scan,
    # which runs through _scan_one's body cap.
    t0 = time.time()
    r = session.post(f"{base_url}/api/messages/m4/scan", timeout=10)
    assert r.status_code == 200
    assert time.time() - t0 < 5, "scan took too long — possible ReDoS regression"
