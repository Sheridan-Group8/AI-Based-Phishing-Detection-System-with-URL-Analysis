from __future__ import annotations

import app as pg_app
from phishing_detector import URLAnalyzer


def test_url_extraction_is_bounded_and_ignores_script_breakout_chars():
    analyzer = URLAnalyzer()
    urls = analyzer.extract_urls(
        'Visit https://safe.example/path, then http://evil.test/<script>'
    )
    assert urls == ["https://safe.example/path", "http://evil.test/"]


def test_url_analyzer_flags_common_phishing_indicators():
    features = URLAnalyzer().analyze_url(
        "http://paypal.login-example.xyz:8080/verify/account"
    )
    assert features["suspicious_tld"] == 1
    assert features["has_suspicious_keyword"] == 1
    assert features["has_port"] == 1
    assert features["is_https"] == 0


def test_domain_validator_rejects_unsafe_inputs():
    assert pg_app._is_public_domain("example.com")
    assert not pg_app._is_public_domain("localhost")
    assert not pg_app._is_public_domain("127.0.0.1")
    assert not pg_app._is_public_domain("bad domain")


def test_domain_age_uses_rdap_json(monkeypatch):
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "events": [
                    {
                        "eventAction": "registration",
                        "eventDate": "2020-01-01T00:00:00Z",
                    }
                ]
            }

    class FakeRequests:
        calls = []

        @classmethod
        def get(cls, url, **kwargs):
            cls.calls.append((url, kwargs))
            return FakeResponse()

    pg_app._domain_age_cache.clear()
    monkeypatch.setattr(pg_app, "http_requests", FakeRequests)

    days = pg_app._get_domain_age("login.example.com")

    assert days is not None
    assert days > 0
    assert FakeRequests.calls[0][0] == "https://rdap.org/domain/example.com"


def test_openphish_feed_matches_exact_urls_locally(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "https://evil.example/login\nhttps://evil.example/trailing/\n"

    class FakeRequests:
        calls = []

        @classmethod
        def get(cls, url, **kwargs):
            cls.calls.append((url, kwargs))
            return FakeResponse()

    pg_app._openphish.update({"urls": frozenset(), "fetched_at": None, "ok": False})
    monkeypatch.setattr(pg_app, "http_requests", FakeRequests)

    assert pg_app._check_openphish("https://evil.example/login") is True
    assert pg_app._check_openphish("https://evil.example/trailing") is True
    assert pg_app._check_openphish("https://safe.example/") is False
    assert FakeRequests.calls[0][0] == pg_app.OPENPHISH_FEED_URL


def test_threat_intel_marks_openphish_hit_as_confirmed(monkeypatch):
    pg_app._openphish.update({
        "urls": frozenset({"https://evil.example/login"}),
        "fetched_at": pg_app.datetime.now(),
        "ok": True,
    })
    monkeypatch.setattr(pg_app, "GSB_API_KEY", "")
    monkeypatch.setattr(pg_app, "_check_virustotal_url", lambda url: None)
    monkeypatch.setattr(pg_app, "_vt_scan_url", lambda url: None)
    monkeypatch.setattr(pg_app, "_check_urlhaus", lambda host: None)
    monkeypatch.setattr(pg_app, "_check_urlscan", lambda host: None)
    monkeypatch.setattr(pg_app, "_check_abuseipdb_ip", lambda ip: None)

    result = pg_app.run_threat_intel({}, ["https://evil.example/login"])

    assert result["confirmed_phishing"] is True
    assert result["url_threats"]["https://evil.example/login"]["malicious"] is True
    assert result["url_threats"]["https://evil.example/login"]["sources"] == ["OpenPhish"]
    assert any("OpenPhish" in signal for signal in result["signals"])


def test_html_to_text_strips_non_visible_content_and_decodes_entities():
    text = pg_app._html_to_text(
        "<html><head><title>secret</title></head>"
        "<body><style>.x{}</style><script>alert(1)</script>"
        "<h1>Hello&nbsp;Team</h1><p>Click &amp; review</p></body></html>"
    )
    assert "secret" not in text
    assert "alert" not in text
    assert "Hello Team" in text
    assert "Click & review" in text


def test_html_link_extractor_deduplicates_http_destinations_only():
    urls = pg_app._extract_html_link_urls(
        '<a href="https://example.com/a">A</a>'
        '<img src="https://example.com/a">'
        '<form action="http://forms.example/submit"></form>'
        '<a href="javascript:alert(1)">bad</a>'
    )
    assert urls == ["https://example.com/a", "http://forms.example/submit"]


def test_attachment_classifier_detects_dangerous_and_caution_files():
    assert pg_app._classify_attachment("invoice.pdf.exe")[0] == "dangerous"
    assert pg_app._classify_attachment("macro.xlsm")[0] == "caution"
    assert pg_app._classify_attachment("archive.zip")[0] == "caution"
    assert pg_app._classify_attachment("notes.txt")[0] == "safe"


def test_sender_profile_build_and_compare_uses_graph_sender_shape():
    emails = [
        {
            "id": "m1",
            "from": {"emailAddress": {"address": "alice@example.com"}},
            "receivedDateTime": "2026-07-09T09:15:00Z",
            "body": {"content": "Hello team. Please review the weekly notes. Thanks, Alice"},
        },
        {
            "id": "m2",
            "from": {"emailAddress": {"address": "alice@example.com"}},
            "receivedDateTime": "2026-07-09T09:45:00Z",
            "body": {"content": "Hello team. The report is attached for review. Thanks, Alice"},
        },
    ]
    profile = pg_app._build_sender_profile("alice@example.com", emails)
    assert profile["email"] == "alice@example.com"
    assert profile["email_count"] == 2
    assert profile["typical_greeting"] == "casual"
    assert 9 in profile["typical_send_hours"]

    comparison = pg_app._compare_to_profile(
        profile,
        {"body": {"content": "URGENT!!! SEND PAYMENT NOW!!!"}},
    )
    assert comparison["status"] in {"normal", "suspicious"}
    assert comparison["score"] >= 0


def test_registered_domain_extraction_handles_urls_and_addresses():
    assert pg_app._reg_domain("Security <alerts@mail.example.com>") == "example.com"
    assert pg_app._reg_domain("https://login.microsoft.com/path") == "microsoft.com"
    assert pg_app._reg_domain("localhost") == "localhost"
