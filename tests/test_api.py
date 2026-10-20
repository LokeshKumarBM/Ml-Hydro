"""API tests using FastAPI's TestClient. Run from the project root: pytest"""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)

VALID = {
    "material": "4130X",
    "environment": "H2: 45 / RT",
    "pressure_mpa": 45,
    "delta_k_mpa_sqrt_m": 15,
    "r_ratio": 0.1,
    "frequency_hz": 1,
    "orientation": "CL",
    "test_method": "ASTM E647",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["fatigue_model_loaded"] is True


def test_dashboard_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "MLHydro" in response.text


def test_options():
    response = client.get("/model/options")
    assert response.status_code == 200
    assert len(response.json()["tested_conditions"]) == 5


def test_predict_valid():
    response = client.post("/predict/fatigue", json=VALID)
    assert response.status_code == 200
    assert response.json()["predicted_da_dn_m_per_cycle"] > 0


def test_predict_unknown_environment_returns_422():
    response = client.post("/predict/fatigue", json={**VALID, "environment": "H2 45 MPa"})
    assert response.status_code == 422
    assert "Tested combinations" in response.json()["detail"]


def test_predict_bad_number_returns_422():
    response = client.post("/predict/fatigue", json={**VALID, "r_ratio": 5})
    assert response.status_code == 422
