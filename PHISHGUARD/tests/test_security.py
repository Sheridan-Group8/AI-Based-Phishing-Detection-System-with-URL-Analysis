"""Security regression tests for the PhishGuard desktop backend.

The suite ports the checks from final-bigmodeltest and adds coverage for the
newer Supabase OAuth handoff, attachment routes, domain validation, CSP, and
Electron hardening.
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


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PHISHGUARD_TEST_PYTHON", sys.executable)
LAUNCH_SECRET = "test-launch-secret-xyz"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextmanager
def run_server():
    port = _free_port()
    tmp_data = tempfile.mkdtemp(prefix="pg-tests-")
    env = {
        **os.environ,
        "FLASK_PORT": str(port),
        "PHISHGUARD_LAUNCH_SECRET": LAUNCH_SECRET,
        "PHISHGUARD_USER_DATA": tmp_data,
        "PHISHGUARD_ONNX": "0",
        "ELECTRON_MODE": "1",
    }
    proc = subprocess.Popen(
        [PYTHON, str(PROJECT_DIR / "app.py")],
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(40):
            try:
                resp = requests.get(f"{base}/api/auth/status", timeout=0.5)
                if resp.status_code == 200:
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
    with run_server() as url:
        yield url


@pytest.fixture()
def session(base_url):
    sess = requests.Session()
    resp = sess.get(f"{base_url}/api/csrf", timeout=3)
    assert resp.status_code == 200
    sess.headers["X-CSRF-Token"] = resp.json()["csrf"]
    sess.headers["X-Launch-Secret"] = LAUNCH_SECRET
    sess.headers["Origin"] = base_url
    return sess


def test_security_headers_present_on_api(base_url):
    resp = requests.get(f"{base_url}/api/auth/status", timeout=3)
    headers = resp.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("Referrer-Policy") == "no-referrer"
    assert "camera=()" in headers.get("Permissions-Policy", "")
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert headers.get("Cross-Origin-Resource-Policy") == "same-origin"
    assert "no-store" in headers.get("Cache-Control", "")


def test_session_cookie_is_httponly_strict(base_url):
    resp = requests.get(f"{base_url}/api/auth/status", timeout=3)
    set_cookie = resp.headers.get("Set-Cookie", "")
    assert "pg_sid" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "SameSite=Strict" in set_cookie


def test_post_without_csrf_or_origin_rejected(base_url):
    resp = requests.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert resp.status_code == 403
    assert "Forbidden" in resp.text


def test_post_with_wrong_origin_rejected(session, base_url):
    session.headers["Origin"] = "http://evil.example.com"
    session.headers.pop("X-Launch-Secret", None)
    resp = session.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert resp.status_code == 403
    assert "origin" in resp.text.lower()


def test_post_with_wrong_referer_rejected(session, base_url):
    session.headers.pop("Origin", None)
    session.headers.pop("X-Launch-Secret", None)
    session.headers["Referer"] = "http://evil.example.com/page"
    resp = session.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert resp.status_code == 403
    assert "origin" in resp.text.lower()


def test_post_with_wrong_csrf_rejected(session, base_url):
    session.headers["X-CSRF-Token"] = "garbage-token"
    resp = session.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert resp.status_code == 403
    assert "CSRF" in resp.text


def test_post_with_launch_secret_bypasses_origin(base_url):
    sess = requests.Session()
    resp = sess.get(f"{base_url}/api/csrf", timeout=3)
    assert resp.status_code == 200
    sess.headers["X-CSRF-Token"] = resp.json()["csrf"]
    sess.headers["X-Launch-Secret"] = LAUNCH_SECRET
    resp = sess.post(f"{base_url}/api/auth/disconnect", timeout=3)
    assert resp.status_code == 200


def test_all_protected_post_routes_reject_without_csrf(base_url):
    routes = [
        "/api/auth/disconnect",
        "/api/auth/refresh",
        "/api/auth/external-start",
        "/api/auth/supabase-provider",
        "/api/scan-all",
        "/api/report-sender",
        "/api/messages/m1/scan",
        "/api/messages/m1/move-to-junk",
        "/api/messages/m1/attachments/a1/analyze",
        "/api/messages/m1/attachments/a1/deepscan",
    ]
    for route in routes:
        resp = requests.post(f"{base_url}{route}", timeout=3)
        assert resp.status_code == 403, f"{route} did not enforce CSRF/origin"


def test_two_sessions_get_distinct_csrf(base_url):
    sess1 = requests.Session()
    sess2 = requests.Session()
    token1 = sess1.get(f"{base_url}/api/csrf", timeout=3).json()["csrf"]
    token2 = sess2.get(f"{base_url}/api/csrf", timeout=3).json()["csrf"]
    assert token1 != token2


def test_scan_returns_404_when_signed_out(session, base_url):
    resp = session.post(f"{base_url}/api/messages/m4/scan", timeout=10)
    assert resp.status_code == 404


def test_move_to_junk_returns_404_when_signed_out(session, base_url):
    resp = session.post(f"{base_url}/api/messages/m1/move-to-junk", timeout=10)
    assert resp.status_code == 404
    assert "not found" in resp.json().get("error", "").lower()


def test_scan_unknown_message_id_404(session, base_url):
    resp = session.post(f"{base_url}/api/messages/does-not-exist/scan", timeout=5)
    assert resp.status_code == 404


def test_old_index_route_is_gone(session, base_url):
    resp = session.post(f"{base_url}/api/scan/0", timeout=3)
    assert resp.status_code == 404


def test_scan_all_signed_out_contract(session, base_url):
    resp = session.post(f"{base_url}/api/scan-all", timeout=10)
    assert resp.status_code in {200, 503}
    if resp.status_code == 200:
        second = session.post(f"{base_url}/api/scan-all", timeout=5)
        assert second.status_code == 429
        assert "wait" in second.json().get("error", "").lower()
    else:
        assert "model" in resp.json().get("error", "").lower()


def test_log_endpoint_rejects_bad_origin(base_url):
    resp = requests.get(
        f"{base_url}/api/log",
        headers={"Origin": "http://evil.com"},
        timeout=3,
    )
    assert resp.status_code == 403


def test_log_endpoint_rejects_spoofed_allowlisted_origin_without_launch_secret(base_url):
    resp = requests.get(
        f"{base_url}/api/log",
        headers={"Origin": base_url},
        timeout=3,
    )
    assert resp.status_code == 403

    resp = requests.get(
        f"{base_url}/api/log/download",
        headers={"Origin": base_url},
        timeout=3,
    )
    assert resp.status_code == 403


def test_settings_reflect_safe_runtime_defaults(base_url):
    resp = requests.get(f"{base_url}/api/settings", timeout=3)
    body = resp.json()
    assert "threat_intel_enabled" not in body
    assert body["logging_enabled"] is False
    data_dir = Path(body["user_data_dir"]).resolve()
    home = Path.home().resolve()
    tmp = Path(tempfile.gettempdir()).resolve()
    assert (
        data_dir == home
        or home in data_dir.parents
        or data_dir == tmp
        or tmp in data_dir.parents
    )


@pytest.mark.parametrize("domain", ["localhost", "127.0.0.1", "10.0.0.1", "bad domain", "example"])
def test_reputation_rejects_non_public_domains(base_url, domain):
    resp = requests.get(f"{base_url}/api/reputation/{domain}", timeout=3)
    assert resp.status_code == 400
    assert "invalid domain" in resp.text


@pytest.mark.parametrize("domain", ["localhost", "127.0.0.1", "bad domain"])
def test_report_sender_rejects_invalid_domain(session, base_url, domain):
    resp = session.post(
        f"{base_url}/api/report-sender",
        json={"domain": domain, "is_phishing": True},
        timeout=3,
    )
    assert resp.status_code == 400
    assert "invalid domain" in resp.text


def test_external_oauth_handoff_is_nonce_bound_and_one_time(session, base_url):
    nonce = "n" * 32
    start = session.post(
        f"{base_url}/api/auth/external-start",
        json={"nonce": nonce},
        timeout=3,
    )
    assert start.status_code == 200

    pending = requests.get(
        f"{base_url}/api/auth/external-poll",
        params={"nonce": nonce},
        timeout=3,
    )
    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"

    unknown = requests.post(
        f"{base_url}/api/auth/external-deliver",
        json={"nonce": "missing", "access_token": "tok"},
        timeout=3,
    )
    assert unknown.status_code == 404

    delivered = requests.post(
        f"{base_url}/api/auth/external-deliver",
        json={
            "nonce": nonce,
            "access_token": "access",
            "refresh_token": "refresh",
            "provider_token": "provider",
            "expires_in": 3600,
        },
        timeout=3,
    )
    assert delivered.status_code == 200

    replay = requests.post(
        f"{base_url}/api/auth/external-deliver",
        json={"nonce": nonce, "access_token": "second"},
        timeout=3,
    )
    assert replay.status_code == 409

    ready = requests.get(
        f"{base_url}/api/auth/external-poll",
        params={"nonce": nonce},
        timeout=3,
    )
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["access_token"] == "access"
    assert body["provider_token"] == "provider"

    consumed = requests.get(
        f"{base_url}/api/auth/external-poll",
        params={"nonce": nonce},
        timeout=3,
    )
    assert consumed.status_code == 404


def test_attachment_routes_are_signed_out_safe(session, base_url):
    resp = requests.get(f"{base_url}/api/messages/m1/attachments", timeout=3)
    assert resp.status_code == 404

    resp = session.post(
        f"{base_url}/api/messages/m1/attachments/a1/analyze",
        timeout=3,
    )
    assert resp.status_code == 404

    resp = session.post(
        f"{base_url}/api/messages/m1/attachments/a1/deepscan",
        timeout=3,
    )
    assert resp.status_code == 404

    resp = requests.get(
        f"{base_url}/api/messages/m1/attachments/a1/deepscan/analysis",
        timeout=3,
    )
    assert resp.status_code == 404


def test_detector_import_does_not_require_sklearn():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        from phishing_detector import PhishingDetector, URLAnalyzer
    finally:
        sys.path.pop(0)
    detector = PhishingDetector()
    assert detector.url_analyzer is not None
    assert URLAnalyzer().extract_urls("Visit https://example.com")


def test_legacy_pickle_load_model_api_is_removed():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        from phishing_detector import PhishingDetector
    finally:
        sys.path.pop(0)
    detector = PhishingDetector()
    assert not hasattr(detector, "load_model")


def test_url_regex_handles_adversarial_body():
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        from phishing_detector import URLAnalyzer
    finally:
        sys.path.pop(0)
    analyzer = URLAnalyzer()
    body = ("https://" + "a" * 5000 + ".") * 50
    assert len(body) > analyzer._MAX_TEXT_FOR_REGEX
    started = time.time()
    urls = analyzer.extract_urls(body)
    assert time.time() - started < 5, "extract_urls too slow; possible ReDoS regression"
    assert urls
    assert all(len(url) <= 2056 for url in urls)


def test_index_csp_blocks_common_renderer_escape_paths():
    html = (PROJECT_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "script-src 'self'" in html
    assert "object-src 'none'" in html
    assert "base-uri 'self'" in html
    assert "frame-ancestors 'none'" in html
    assert "connect-src 'self' https://*.supabase.co wss://*.supabase.co" in html


def test_electron_window_is_hardened():
    main_js = (PROJECT_DIR / "main.js").read_text(encoding="utf-8")
    preload_js = (PROJECT_DIR / "preload.js").read_text(encoding="utf-8")
    assert "contextIsolation: true" in main_js
    assert "nodeIntegration: false" in main_js
    assert "sandbox: true" in main_js
    assert "webSecurity: true" in main_js
    assert "allowRunningInsecureContent: false" in main_js
    assert "setPermissionRequestHandler" in main_js
    assert "setPermissionCheckHandler" in main_js
    assert "_externalOpenAllowed" in main_js
    assert "contextBridge.exposeInMainWorld" in preload_js
    assert "ipcRenderer.invoke('pg:open-external', url)" in preload_js
