
import os

# Ensure the API module can be imported during tests.
os.environ.setdefault("ORTHANC_PASSWORD", "test-password")

import base64
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from medical_imaging_ai import database
from medical_imaging_ai.constants import CLASSES
from medical_imaging_ai.api import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def mock_model(monkeypatch):
    model = Mock()

    probabilities = np.linspace(
        0.01, 0.14, len(CLASSES), dtype=np.float32
    )

    model.predict.return_value = np.array([probabilities])

    monkeypatch.setattr(main, "model", model)

    return model


def test_health_check(client, monkeypatch):
    monkeypatch.setattr(main, "model", Mock())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True
    }


def test_health_check_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_analyze_rejects_unsupported_file(client, mock_model):
    response = client.post(
        "/analyze",
        files={
            "file": (
                "report.pdf",
                b"not an image",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert "Only PNG and JPEG" in response.json()["detail"]


def test_analyze_rejects_empty_file(client, mock_model):
    response = client.post(
        "/analyze",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_analyze_rejects_invalid_disease(
    client, mock_model, monkeypatch
):
    monkeypatch.setattr(
        main,
        "preprocess_uploaded_image",
        lambda image_bytes: (
            np.zeros((1, 224, 224, 3)),
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.png",
                b"fake-image-data",
                "image/png"
            )
        },
        data={"disease": "UnknownDisease"}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown disease name."


def test_analyze_success(
    client, mock_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        main,
        "preprocess_uploaded_image",
        lambda image_bytes: (
            np.zeros((1, 224, 224, 3)),
            np.zeros((224, 224, 3), dtype=np.uint8)
        )
    )

    monkeypatch.setattr(
        main,
        "generate_gradcam",
        lambda model, image, class_index: np.zeros((224, 224))
    )

    overlay = base64.b64encode(b"fake-gradcam").decode()

    monkeypatch.setattr(
        main,
        "create_gradcam_overlay",
        lambda original, heatmap: overlay
    )

    monkeypatch.setattr(
        main,
        "save_analysis",
        lambda **kwargs: 123
    )

    monkeypatch.setattr(
        main,
        "save_study_images",
        lambda **kwargs: (
            tmp_path / "original.png",
            tmp_path / "gradcam.png"
        )
    )

    monkeypatch.setattr(
        main,
        "update_study_image_paths",
        lambda **kwargs: None
    )

    response = client.post(
        "/analyze",
        files={
            "file": (
                "sample.png",
                b"fake-image-data",
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["study_id"] == 123
    assert len(result["predictions"]) == 14
    assert result["explained_disease"] in CLASSES
    assert result["gradcam_overlay"] == overlay
    assert "not for clinical" in result["disclaimer"].lower()

    probabilities = [
        item["probability"]
        for item in result["predictions"]
    ]

    assert probabilities == sorted(probabilities, reverse=True)


def test_list_studies(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_all_studies",
        lambda: [{"id": 1}, {"id": 2}]
    )

    response = client.get("/studies")

    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_retrieve_study(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_study_by_id",
        lambda study_id: {
            "id": study_id,
            "predictions": []
        }
    )

    response = client.get("/studies/1")

    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_retrieve_missing_study(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "get_study_by_id",
        lambda study_id: None
    )

    response = client.get("/studies/99999")

    assert response.status_code == 404
