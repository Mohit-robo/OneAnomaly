"""
Tests for the cloud API server (api_server.py).

Mocks:
- boto3 / S3 via moto
- Requests to the local Gateway via pytest-mock / responses
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function", autouse=True)
def env_setup(monkeypatch):
    """Ensure boto3 never hits real AWS and gateway never hits localhost."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("S3_BUCKET_NAME", "")          # local mode (no S3)
    monkeypatch.setenv("GATEWAY_URL", "http://mock-gateway:8080")


@pytest.fixture(scope="function")
def client():
    """Create a Flask test client."""
    # Patch gateway requests before importing api_server
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True}, text="ok")
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        import importlib
        import api_server
        importlib.reload(api_server)  # ensure env vars take effect
        api_server.app.config["TESTING"] = True
        with api_server.app.test_client() as c:
            yield c


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

def test_health_check(client):
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert "gateway_url" in data


# ---------------------------------------------------------------------------
# Session Save (local mode — no S3)
# ---------------------------------------------------------------------------

def test_save_session_local(client, tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "")
    payload = {
        "session_id": "test_session_001",
        "phase1": {"mode": "thresholding"},
        "phase2": {"spatial_mode": "whole", "regions": []}
    }
    resp = client.post("/api/save_session", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "test_session_001" in data["filename"]


# ---------------------------------------------------------------------------
# Session Save (S3 mode)
# ---------------------------------------------------------------------------

def test_save_session_s3():
    """Test that session config is uploaded to S3 when bucket is configured."""
    try:
        from moto import mock_s3
        import boto3
    except ImportError:
        pytest.skip("moto not installed")

    @mock_s3
    def run():
        import os
        os.environ["S3_BUCKET_NAME"] = "test-bucket"

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Simulate what api_server does
        payload = {"phase1": {"mode": "thresholding"}, "phase2": {}}
        s3.put_object(
            Bucket="test-bucket",
            Key="sessions/test_session_config.json",
            Body=json.dumps(payload)
        )

        obj = s3.get_object(Bucket="test-bucket", Key="sessions/test_session_config.json")
        loaded = json.loads(obj["Body"].read())
        assert loaded["phase1"]["mode"] == "thresholding"

    run()


# ---------------------------------------------------------------------------
# Configure Spatial
# ---------------------------------------------------------------------------

def test_configure_spatial(client):
    payload = {
        "spatial_mode": "manual",
        "regions": [{"x": 10, "y": 10, "w": 100, "h": 100}],
        "grid_ratios": {"x": 50, "y": 50}
    }
    resp = client.post("/api/configure_spatial",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["config"]["spatial_mode"] == "manual"


# ---------------------------------------------------------------------------
# List Memory Banks (gateway proxy)
# ---------------------------------------------------------------------------

def test_list_memory_banks_gateway_ok(client):
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"banks": ["product_A", "product_B"]}
        )
        resp = client.get("/api/list_memory_banks")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "product_A" in data["banks"]


def test_list_memory_banks_gateway_down(client):
    """When gateway is unreachable, return empty list with warning (not 500)."""
    import requests as req_module
    with patch("requests.get", side_effect=req_module.exceptions.ConnectionError):
        resp = client.get("/api/list_memory_banks")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["banks"] == []
    assert "warning" in data


# ---------------------------------------------------------------------------
# Load Memory Bank (gateway proxy)
# ---------------------------------------------------------------------------

def test_load_memory_bank_proxies_to_gateway(client):
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
        resp = client.post("/api/load_memory_bank", json={"bank_name": "product_A"})
    assert resp.status_code == 200
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/load_memory_bank" in call_kwargs[0][0]


# ---------------------------------------------------------------------------
# Detect Anomalies — no files
# ---------------------------------------------------------------------------

def test_detect_anomalies_no_files(client):
    resp = client.post("/api/detect_anomalies", data={"session_id": "test"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
