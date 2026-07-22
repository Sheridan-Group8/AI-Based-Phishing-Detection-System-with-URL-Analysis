from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta

import pytest

import app as pg_app


ORIGIN = "http://127.0.0.1:5050"


def _fake_jwt(sub="user-123"):
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": sub}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"header.{payload}.sig"


@pytest.fixture()
def client():
    pg_app.app.config["TESTING"] = True
    with pg_app._sessions_lock:
        pg_app._sessions.clear()
    with pg_app._external_oauth_lock:
        pg_app._external_oauth_pending.clear()
    with pg_app.app.test_client() as test_client:
        yield test_client


def _csrf_headers(client):
    resp = client.get("/api/csrf")
    assert resp.status_code == 200
    return {
        "Origin": ORIGIN,
        "X-CSRF-Token": resp.get_json()["csrf"],
        "X-Launch-Secret": "test-launch-secret-xyz",
    }


def test_mocked_graph_provider_connects_and_exposes_mailbox(monkeypatch, client):
    graph_email = {
        "id": "graph-1",
        "subject": "Quarterly report",
        "from": {"emailAddress": {"name": "Alice", "address": "alice@example.com"}},
        "receivedDateTime": "2026-07-09T12:00:00Z",
        "bodyPreview": "Please review.",
        "body": {"content": "<p>Please review https://example.com/report</p>"},
        "internetMessageHeaders": [
            {"name": "Authentication-Results", "value": "spf=pass dkim=pass dmarc=pass"}
        ],
        "hasAttachments": True,
        "attachments": [{"id": "att-1", "name": "report.pdf", "size": 7}],
    }
    monkeypatch.setattr(
        pg_app,
        "_graph_request",
        lambda sess, endpoint, params=None: {
            "displayName": "Test User",
            "mail": "test@example.com",
        },
    )
    monkeypatch.setattr(pg_app, "_fetch_all_emails", lambda sess: [graph_email])
    monkeypatch.setattr(pg_app, "_fetch_junk_emails", lambda sess: [])

    resp = client.post(
        "/api/auth/supabase-provider",
        headers=_csrf_headers(client),
        json={"provider_token": "graph-token", "expires_in": 3600},
    )
    assert resp.status_code == 200
    assert resp.get_json()["connected"] is True

    emails = client.get("/api/emails").get_json()["emails"]
    assert len(emails) == 1
    assert emails[0]["id"] == "graph-1"
    assert emails[0]["sender"] == "alice@example.com"
    assert emails[0]["headers"] == {"spf": "pass", "dkim": "pass", "dmarc": "pass"}


def test_mocked_virustotal_attachment_analysis(monkeypatch, client):
    test_mocked_graph_provider_connects_and_exposes_mailbox(monkeypatch, client)
    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "vt-test-key")
    monkeypatch.setattr(pg_app, "_fetch_attachment_bytes", lambda sess, mid, aid: b"payload")
    monkeypatch.setattr(
        pg_app,
        "_vt_lookup_hash",
        lambda sha: {
            "found": True,
            "malicious": 1,
            "suspicious": 0,
            "harmless": 5,
            "total": 6,
            "sha256": sha,
        },
    )

    resp = client.post(
        "/api/messages/graph-1/attachments/att-1/analyze",
        headers=_csrf_headers(client),
    )
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["configured"] is True
    assert body["analyzed"] is True
    assert body["malicious"] == 1
    assert body["size"] == len(b"payload")


def test_scan_route_uses_mocked_model_intel_and_supabase_persistence(monkeypatch, client):
    test_mocked_graph_provider_connects_and_exposes_mailbox(monkeypatch, client)

    class FakeDetector:
        is_trained = True
        structural_analyzer = object()

        def predict(self, full, headers=None, extra_urls=None):
            assert "Quarterly report" in full
            assert extra_urls == []
            return 0, 0.91, {"urls": []}, {"spf": "pass"}, 0.12

    captured = {}

    monkeypatch.setattr(pg_app, "detector", FakeDetector())
    monkeypatch.setattr(
        pg_app,
        "run_threat_intel",
        lambda email, urls: {
            "confirmed_phishing": False,
            "signals": [],
            "checks": {"mock": {"ok": True}},
            "url_threats": {},
        },
    )
    monkeypatch.setattr(pg_app, "_load_attachment_metadata", lambda sess, mid, email: [])
    monkeypatch.setattr(pg_app, "_vt_scan_attachments", lambda sess, mid, attachments: attachments)
    monkeypatch.setattr(
        pg_app,
        "_compute_assessment",
        lambda *args, **kwargs: {
            "verdict": 0,
            "overall": 12,
            "summary": "Mocked clean message",
            "factors": [],
        },
    )

    def fake_supabase_request(method, path, jwt, json_body=None, params=None, prefer="return=minimal"):
        captured.update({"method": method, "path": path, "jwt": jwt, "json_body": json_body})

        class Resp:
            status_code = 201
            text = ""

        return Resp()

    monkeypatch.setattr(pg_app, "_supabase_request", fake_supabase_request)

    headers = _csrf_headers(client)
    headers["Authorization"] = "Bearer " + _fake_jwt()
    resp = client.post("/api/messages/graph-1/scan", headers=headers)
    body = resp.get_json()
    assert resp.status_code == 200
    assert body["prediction"] == 0
    assert body["risk_score"] == 12
    assert captured["method"] == "POST"
    assert captured["path"] == "scan_history"
    assert captured["json_body"]["user_id"] == "user-123"
    assert captured["json_body"]["message_id"] == "graph-1"


def test_scan_all_uses_mocked_scanner_and_rate_limits_second_call(monkeypatch, client):
    test_mocked_graph_provider_connects_and_exposes_mailbox(monkeypatch, client)
    monkeypatch.setattr(pg_app.detector, "is_trained", True)

    def fake_scan(sess, email, idx):
        message_id = pg_app._message_key(email, idx)
        return {"id": message_id, "messageId": message_id, "prediction": 0}, None

    monkeypatch.setattr(pg_app, "_scan_email_common", fake_scan)

    first = client.post("/api/scan-all", headers=_csrf_headers(client))
    assert first.status_code == 200
    assert "graph-1" in first.get_json()["results"]

    second = client.post("/api/scan-all", headers=_csrf_headers(client))
    assert second.status_code == 429
    assert "wait" in second.get_json()["error"].lower()


def test_external_oauth_expired_nonce_is_removed(client):
    headers = _csrf_headers(client)
    nonce = "x" * 32
    resp = client.post("/api/auth/external-start", headers=headers, json={"nonce": nonce})
    assert resp.status_code == 200
    with pg_app._external_oauth_lock:
        pg_app._external_oauth_pending[nonce]["created"] = (
            datetime.now() - timedelta(seconds=pg_app._EXTERNAL_OAUTH_TTL_SECONDS + 1)
        )
    expired = client.get(f"/api/auth/external-poll?nonce={nonce}")
    assert expired.status_code == 404
    assert expired.get_json()["status"] == "expired"
