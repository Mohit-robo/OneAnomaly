"""
Tests for the local Edge Gateway (gateway/main.py).

The gateway depends on:
- Triton inference server (mocked)
- Memory bank filesystem (tmp_path fixture)
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gateway_client():
    """Create a FastAPI TestClient for the gateway."""
    # Mock tritonclient before any import of gateway code
    mock_triton_mod = MagicMock()
    mock_triton_client = MagicMock()
    mock_triton_mod.InferenceServerClient.return_value = mock_triton_client
    mock_triton_client.is_server_ready.return_value = True

    # Build a fake inference response
    mock_response = MagicMock()
    mock_response.as_numpy.return_value = np.zeros((1, 196, 768), dtype=np.float32)
    mock_triton_client.infer.return_value = mock_response

    with patch.dict("sys.modules", {"tritonclient": MagicMock(),
                                    "tritonclient.grpc": mock_triton_mod}):
        import importlib
        import sys

        # Ensure gateway modules load cleanly
        for mod in list(sys.modules.keys()):
            if "gateway" in mod:
                del sys.modules[mod]

        from fastapi.testclient import TestClient
        from gateway.main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_gateway_health(gateway_client):
    resp = gateway_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "triton_connected" in data


# ---------------------------------------------------------------------------
# Sync Session
# ---------------------------------------------------------------------------

def test_sync_session(gateway_client):
    payload = {
        "session_name": "test_session",
        "config": {
            "phase1": {"mode": "thresholding"},
            "phase2": {"spatial_mode": "whole", "regions": []}
        }
    }
    resp = gateway_client.post("/sync_session", json=payload)
    assert resp.status_code == 200
    assert resp.json()["saved"] is True


# ---------------------------------------------------------------------------
# List Memory Banks
# ---------------------------------------------------------------------------

def test_list_memory_banks_empty(gateway_client, tmp_path, monkeypatch):
    """Should return empty list when no banks directory exists."""
    import gateway.main as gm
    monkeypatch.setattr(gm, "BASE_DIR", tmp_path)
    resp = gateway_client.get("/list_memory_banks")
    assert resp.status_code == 200
    assert resp.json()["banks"] == []


# ---------------------------------------------------------------------------
# Infer — no session
# ---------------------------------------------------------------------------

def test_infer_no_session(gateway_client):
    """Should 400 if session not synced first."""
    import base64
    dummy_b64 = "data:image/jpeg;base64," + base64.b64encode(b"x" * 100).decode()
    resp = gateway_client.post("/infer", json={
        "session_name": "nonexistent_session",
        "image_b64": dummy_b64,
        "phase": 4
    })
    assert resp.status_code == 400
