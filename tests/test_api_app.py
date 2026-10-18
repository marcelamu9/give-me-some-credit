import pytest
from api_support import T_STAR, TRAINED_AT, build_model_dir
from fastapi.testclient import TestClient

from credit_risk.api.examples import INVALID_AGE, LOW_RISK, SAME_CLIENT_WITH_DELAYS
from credit_risk.api.main import create_app

RESPONSE_KEYS = {"probability", "risk_level", "decision", "top_reasons", "model_version"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_dir = build_model_dir(tmp_path_factory.mktemp("model"))
    with TestClient(create_app(model_dir)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path):
    with TestClient(create_app(tmp_path)) as test_client:
        yield test_client


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == TRAINED_AT


def test_health_without_model(empty_client):
    response = empty_client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "model_version": None,
    }


def test_predict_low_risk_example(client):
    response = client.post("/predict", json=LOW_RISK)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert len(body["top_reasons"]) == 3
    assert 0 <= body["probability"] <= 1
    assert body["probability"] == round(body["probability"], 4)


def test_predict_delays_example_is_valid(client):
    assert client.post("/predict", json=SAME_CLIENT_WITH_DELAYS).status_code == 200


def test_predict_nulls_accepted(client):
    payload = {**LOW_RISK, "MonthlyIncome": None, "NumberOfDependents": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


def test_predict_special_codes(client):
    payload = {
        **LOW_RISK,
        "NumberOfTime30-59DaysPastDueNotWorse": 98,
        "NumberOfTimes90DaysLate": 98,
        "NumberOfTime60-89DaysPastDueNotWorse": 98,
    }
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_extreme_values(client):
    payload = {
        **LOW_RISK,
        "DebtRatio": 1e9,
        "RevolvingUtilizationOfUnsecuredLines": 5e4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


def test_predict_invalid_age_example(client):
    response = client.post("/predict", json=INVALID_AGE)
    assert response.status_code == 422
    assert "age" in str(response.json()["detail"])


def test_predict_text_in_numeric_field(client):
    assert client.post("/predict", json={**LOW_RISK, "DebtRatio": "abc"}).status_code == 422


def test_predict_without_model(empty_client):
    response = empty_client.post("/predict", json=LOW_RISK)
    assert response.status_code == 503


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["thresholds"]["t_star"] == T_STAR
    assert body["model_version"] == TRAINED_AT
    assert body["champion"] == "lightgbm"


def test_model_info_without_model(empty_client):
    assert empty_client.get("/model-info").status_code == 503


def test_missing_metadata_returns_503(tmp_path):
    source = build_model_dir(tmp_path / "full")
    (source / "champion_metadata.json").unlink()
    with TestClient(create_app(source)) as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json=LOW_RISK).status_code == 503


def test_openapi_has_demo_examples(client):
    schema = client.get("/openapi.json").json()
    content = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]
    assert set(content["examples"]) == {"low_risk", "same_client_with_delays", "invalid_age"}
