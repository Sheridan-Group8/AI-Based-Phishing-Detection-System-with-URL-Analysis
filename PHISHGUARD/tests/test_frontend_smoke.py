from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser

import pytest

import app as pg_app


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.scripts = []
        self.stylesheets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet" and attrs.get("href"):
            self.stylesheets.append(attrs["href"])


def _index_html():
    return (pg_app.Path(__file__).resolve().parents[1] / "templates" / "index.html").read_text(
        encoding="utf-8"
    )


def test_frontend_shell_contains_core_workflow_controls():
    parser = _IdCollector()
    parser.feed(_index_html())
    required_ids = {
        "emailList",
        "emailPreview",
        "scanBtn",
        "refreshBtn",
        "searchInput",
        "previewBody",
        "previewBodyHtml",
        "attachmentsList",
        "pgSignInMicrosoftBtn",
    }
    assert required_ids <= parser.ids
    assert "/static/css/style.css?v=123" in parser.stylesheets


def test_flask_serves_frontend_shell():
    pg_app.app.config["TESTING"] = True
    with pg_app.app.test_client() as client:
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "PhishGuard" in body
    assert "Content-Security-Policy" in body
    assert "emailList" in body


def test_renderer_fetch_wrapper_keeps_security_headers():
    js = (pg_app.Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "async function apiFetch" in js
    assert "X-CSRF-Token" in js
    assert "X-Launch-Secret" in js
    assert "Authorization" in js
    assert "credentials = opts.credentials || 'same-origin'" in js
    assert "/api/auth/external-poll?nonce=" in js


@pytest.mark.parametrize("relative_path", ["main.js", "preload.js", "static/js/app.js"])
def test_javascript_entrypoints_parse(relative_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    result = subprocess.run(
        [node, "--check", relative_path],
        cwd=str(pg_app.Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
