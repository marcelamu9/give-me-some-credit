"""Demo examples against the real champion; skipped where the artifact does not exist (CI)."""

import joblib
import pytest
from fastapi.testclient import TestClient

from credit_risk.api.examples import INVALID_AGE, LOW_RISK, SAME_CLIENT_WITH_DELAYS
from credit_risk.api.main import create_app
from credit_risk.api.schemas import ClientFeatures
from credit_risk.config import load_config

ARTIFACT = load_config().paths.models / "champion.joblib"

pytestmark = pytest.mark.skipif(not ARTIFACT.exists(), reason="real model not available (CI)")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_low_risk_is_approved(client):
    assert client.post("/predict", json=LOW_RISK).json()["decision"] == "approve"


def test_delays_raise_the_decision(client):
    low = client.post("/predict", json=LOW_RISK).json()
    delayed = client.post("/predict", json=SAME_CLIENT_WITH_DELAYS).json()
    assert delayed["decision"] in {"review", "reject"}
    assert delayed["probability"] > low["probability"]


def test_invalid_age_is_rejected(client):
    assert client.post("/predict", json=INVALID_AGE).status_code == 422


def test_probability_matches_artifact(client):
    model = joblib.load(ARTIFACT)["model"]
    row = ClientFeatures.model_validate(LOW_RISK).to_frame()
    expected = round(model.predict_proba(row)[0, 1], 4)
    assert client.post("/predict", json=LOW_RISK).json()["probability"] == expected
