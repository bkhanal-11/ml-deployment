import pytest
from fastapi.testclient import TestClient
from yolo_app import app

client = TestClient(app)

def test_uploadfile():
    response = client.post("/uploadfile/")
    assert response.status_code == 200