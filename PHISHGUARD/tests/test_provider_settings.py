from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

import app as pg_app


PROVIDER_IDS = {
    "virustotal",
    "google_safe_browsing",
    "urlscan",
    "abuseipdb",
}


@pytest.fixture
def client():
    pg_app.app.config["TESTING"] = True
    with pg_app.app.test_client() as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def restore_provider_runtime_state():
    runtime = deepcopy(pg_app._provider_runtime)
    yield
    with pg_app._provider_config_lock:
        pg_app._provider_runtime.clear()
        pg_app._provider_runtime.update(runtime)


def _status(provider_id):
    return next(
        item for item in pg_app._provider_public_statuses()
        if item["id"] == provider_id
    )


def test_settings_lists_provider_state_without_returning_credentials(monkeypatch, client):
    secret = "user-key-that-must-not-leak"
    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", secret)
    monkeypatch.setitem(pg_app._provider_enabled, "virustotal", True)
    monkeypatch.setitem(pg_app._PROVIDER_KEY_SOURCES, "virustotal", "user")

    response = client.get("/api/settings")
    payload = response.get_json()
    serialized = json.dumps(payload)

    assert response.status_code == 200
    assert {item["id"] for item in payload["providers"]} == PROVIDER_IDS
    assert secret not in serialized
    vt = next(item for item in payload["providers"] if item["id"] == "virustotal")
    assert vt["configured"] is True
    assert vt["enabled"] is True
    assert vt["key_source"] == "user"


def test_private_control_message_updates_key_and_enabled_state(monkeypatch):
    secret = "private-control-key"
    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "")
    monkeypatch.setitem(pg_app._provider_enabled, "virustotal", True)
    monkeypatch.setitem(pg_app._PROVIDER_KEY_SOURCES, "virustotal", "none")

    assert pg_app._apply_provider_configuration({
        "providers": {
            "virustotal": {
                "key": secret,
                "use_deployer_key": False,
                "enabled": False,
            }
        }
    })

    status = _status("virustotal")
    assert pg_app.VIRUSTOTAL_API_KEY == secret
    assert status["configured"] is True
    assert status["enabled"] is False
    assert secret not in json.dumps(status)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, "unavailable"), (403, "unavailable"), (429, "rate_limited")],
)
def test_virustotal_invalid_revoked_and_rate_limited_statuses(
    monkeypatch, status_code, expected
):
    class Response:
        def __init__(self):
            self.status_code = status_code

    class Requests:
        @staticmethod
        def get(*_args, **_kwargs):
            return Response()

    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "test-key")
    monkeypatch.setitem(pg_app._provider_enabled, "virustotal", True)
    monkeypatch.setattr(pg_app, "http_requests", Requests)
    pg_app._vt_url_cache.clear()

    result = pg_app._check_virustotal_url("https://untrusted.example/login")

    assert result["status"] == status_code
    assert _status("virustotal")["status"] == expected
    if status_code in (401, 403):
        assert "rejected or revoked" in _status("virustotal")["reason"]


def test_provider_outage_is_normalized_without_exception_details(monkeypatch):
    class Requests:
        @staticmethod
        def get(*_args, **_kwargs):
            raise RuntimeError("request failed with secret-key-in-exception")

    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "test-key")
    monkeypatch.setitem(pg_app._provider_enabled, "virustotal", True)
    monkeypatch.setattr(pg_app, "http_requests", Requests)
    pg_app._vt_url_cache.clear()

    result = pg_app._check_virustotal_url("https://untrusted.example/login")

    assert result == {"error": "VirusTotal unavailable", "status": 0}
    assert "secret-key-in-exception" not in json.dumps(result)
    assert _status("virustotal")["status"] == "unavailable"


def test_all_optional_providers_disabled_make_no_provider_requests(monkeypatch):
    class Requests:
        calls = 0

        @classmethod
        def get(cls, *_args, **_kwargs):
            cls.calls += 1
            raise AssertionError("disabled provider made a request")

        @classmethod
        def post(cls, *_args, **_kwargs):
            cls.calls += 1
            raise AssertionError("disabled provider made a request")

    for provider_id in PROVIDER_IDS:
        monkeypatch.setitem(pg_app._provider_enabled, provider_id, False)
    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "vt")
    monkeypatch.setattr(pg_app, "GSB_API_KEY", "gsb")
    monkeypatch.setattr(pg_app, "URLSCAN_API_KEY", "urlscan")
    monkeypatch.setattr(pg_app, "ABUSEIPDB_API_KEY", "abuse")
    monkeypatch.setattr(pg_app, "http_requests", Requests)
    monkeypatch.setattr(pg_app, "_check_urlhaus", lambda _host: None)
    monkeypatch.setattr(pg_app, "_check_openphish", lambda _url: False)

    result = pg_app.run_threat_intel(
        {},
        ["https://untrusted.example/login"],
    )

    assert Requests.calls == 0
    assert result["confirmed_phishing"] is False
    assert {
        item["outcome"] for item in result["provider_outcomes"].values()
    } == {"not_checked"}


def test_attachment_upload_requires_explicit_confirmation(monkeypatch, client):
    monkeypatch.setattr(pg_app, "VIRUSTOTAL_API_KEY", "vt-key")
    monkeypatch.setitem(pg_app._provider_enabled, "virustotal", True)
    csrf = client.get("/api/csrf").get_json()["csrf"]
    session_id = client.get_cookie(pg_app.SESSION_COOKIE).value
    session = pg_app._sessions[session_id]
    session.live_mode = True
    session.live_emails = [{
        "id": "m1",
        "attachments": [{"id": "a1", "name": "document.bin", "size": 4}],
    }]

    response = client.post(
        "/api/messages/m1/attachments/a1/deepscan",
        headers={
            "Origin": "http://127.0.0.1:5050",
            "X-CSRF-Token": csrf,
            "X-Launch-Secret": "test-launch-secret-xyz",
        },
    )

    assert response.status_code == 400
    assert response.get_json()["confirmation_required"] is True


def test_electron_store_encrypts_keys_and_survives_restart(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    module_path = Path(__file__).resolve().parents[1] / "provider-settings.js"
    script = r"""
const fs = require('fs');
const { ProviderSettingsStore } = require(process.argv[1]);
const userDataDir = process.argv[2];
const safeStorage = {
  isEncryptionAvailable: () => true,
  encryptString: value => Buffer.from('encrypted:' + value, 'utf8'),
  decryptString: value => value.toString('utf8').slice('encrypted:'.length),
};
let store = new ProviderSettingsStore({ userDataDir, safeStorage, platform: 'win32' });
store.update('virustotal', { key: 'restart-secret', enabled: false });
const raw = fs.readFileSync(store.filePath, 'utf8');
const publicData = store.publicSettings();
store = new ProviderSettingsStore({ userDataDir, safeStorage, platform: 'win32' });
const backend = store.backendConfiguration();
process.stdout.write(JSON.stringify({
  plaintextStored: raw.includes('restart-secret'),
  publicData,
  backendKey: backend.providers.virustotal.key,
  enabled: backend.providers.virustotal.enabled,
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(module_path), str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["plaintextStored"] is False
    assert payload["publicData"]["providers"]["virustotal"]["userKeyConfigured"] is True
    assert "restart-secret" not in json.dumps(payload["publicData"])
    assert payload["backendKey"] == "restart-secret"
    assert payload["enabled"] is False


def test_electron_store_rejects_linux_basic_text_backend(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    module_path = Path(__file__).resolve().parents[1] / "provider-settings.js"
    script = r"""
const { ProviderSettingsStore } = require(process.argv[1]);
const store = new ProviderSettingsStore({
  userDataDir: process.argv[2],
  platform: 'linux',
  safeStorage: {
    isEncryptionAvailable: () => true,
    getSelectedStorageBackend: () => 'basic_text',
    encryptString: value => Buffer.from(value),
  },
});
try {
  store.update('urlscan', { key: 'must-not-be-saved' });
  process.stdout.write('accepted');
} catch (error) {
  process.stdout.write(error.code || 'error');
}
"""
    result = subprocess.run(
        [node, "-e", script, str(module_path), str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )

    assert result.stdout == "SECURE_STORAGE_UNAVAILABLE"
    assert not (tmp_path / "threat-intelligence-settings.json").exists()


def test_renderer_masks_keys_and_confirms_virustotal_upload():
    project = Path(__file__).resolve().parents[1]
    html = (project / "templates" / "index.html").read_text(encoding="utf-8")
    js = (project / "static" / "js" / "app.js").read_text(encoding="utf-8")
    preload = (project / "preload.js").read_text(encoding="utf-8")

    assert html.count('class="settings-input provider-key-input" type="password"') == 4
    assert 'autocomplete="off"' in html
    assert "window.confirm(" in js
    assert "confirm_upload: true" in js
    for label in (
        "Not flagged",
        "Not found",
        "Not checked",
        "Provider unavailable",
    ):
        assert label in js
    assert "providerSettings: Object.freeze" in preload
    assert "pg:provider-settings:update" in preload
