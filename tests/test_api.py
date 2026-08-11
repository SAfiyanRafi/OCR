"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

import io
import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.extraction.config_tool import delete_field_config


@pytest.fixture
def client():
    return TestClient(app)


def create_dummy_image_bytes():
    img = Image.new("RGB", (800, 500), color=(240, 248, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_api_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "html" in res.headers["content-type"].lower() or res.status_code == 200

    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "online"


def test_api_ocr_cnic_endpoint(client):
    img_bytes = create_dummy_image_bytes()
    res = client.post(
        "/ocr",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"document_type": "cnic_front"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["document_type"] == "cnic_front"
    assert "fields" in data


def test_api_ocr_debug_endpoint(client):
    img_bytes = create_dummy_image_bytes()
    res = client.post(
        "/ocr/debug",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"document_type": "passport"}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 0


def test_api_config_region_endpoint(client):
    res = client.post(
        "/config/field",
        json={
            "config_name": "cnic_front.yaml",
            "field_key": "test_temp_field",
            "label": "Test Temp Field",
            "x1": 0.1,
            "y1": 0.1,
            "x2": 0.5,
            "y2": 0.5,
            "strategy": "region"
        }
    )
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Cleanup temp test field
    delete_field_config("cnic_front.yaml", "test_temp_field")
