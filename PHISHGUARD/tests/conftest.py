from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

os.environ.setdefault("PHISHGUARD_ONNX", "0")
os.environ.setdefault("PHISHGUARD_LAUNCH_SECRET", "test-launch-secret-xyz")
os.environ.setdefault("PHISHGUARD_USER_DATA", tempfile.mkdtemp(prefix="pg-pytest-"))
os.environ.setdefault("FLASK_PORT", "5050")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
