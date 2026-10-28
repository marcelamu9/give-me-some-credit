import urllib.error

import pytest

from credit_risk.api.examples import INVALID_AGE, SAME_CLIENT_WITH_DELAYS
from credit_risk.deploy.smoke import main, normalize_url, run_smoke

HEALTH_OK = (200, {"status": "ok", "model_loaded": True, "model_version": "v1"})


def make_fetch(
    health=HEALTH_OK,
    low=(200, {"decision": "approve", "probability": 0.01}),
    delays=(200, {"decision": "reject", "probability": 0.20}),
    invalid=(422, {"detail": [{"loc": ["body", "age"]}]}),
):
    def fetch(method, url, body):
        if url.endswith("/health"):
            return health
        if body == INVALID_AGE:
            return invalid
        if body == SAME_CLIENT_WITH_DELAYS:
            return delays
        return low

    return fetch


def by_name(results):
    return {result.name: result for result in results}


def test_all_checks_pass():
    results = run_smoke("https://api.example", fetch=make_fetch(), repeats=2)
    assert [r.name for r in results] == [
        "health",
        "low_risk_approved",
        "delays_raise_decision",
        "invalid_age_rejected",
        "latency",
    ]
    assert all(r.passed for r in results)
    assert "ms" in by_name(results)["latency"].detail


def test_health_without_model_fails_with_status():
    health = (503, {"status": "unavailable", "model_loaded": False, "model_version": None})
    check = by_name(run_smoke("https://api.example", fetch=make_fetch(health=health)))["health"]
    assert not check.passed
    assert "503" in check.detail


def test_low_risk_rejected_fails():
    low = (200, {"decision": "reject", "probability": 0.5})
    results = by_name(run_smoke("https://api.example", fetch=make_fetch(low=low)))
    assert not results["low_risk_approved"].passed


def test_delays_must_raise_the_probability():
    delays = (200, {"decision": "review", "probability": 0.005})
    results = by_name(run_smoke("https://api.example", fetch=make_fetch(delays=delays)))
    assert not results["delays_raise_decision"].passed


def test_invalid_age_accepted_fails():
    invalid = (200, {"decision": "approve", "probability": 0.01})
    results = by_name(run_smoke("https://api.example", fetch=make_fetch(invalid=invalid)))
    assert not results["invalid_age_rejected"].passed


def test_network_error_is_reported_not_raised():
    def broken(method, url, body):
        raise urllib.error.URLError("connection refused")

    results = run_smoke("https://api.example", fetch=broken)
    assert not any(r.passed for r in results)
    assert all("URLError" in r.detail for r in results)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("x.on.aws/", "https://x.on.aws"),
        ("https://x.on.aws", "https://x.on.aws"),
        ("http://localhost:8000/", "http://localhost:8000"),
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_main_exit_codes(capsys):
    assert main(["--url", "api.example"], fetch=make_fetch()) == 0
    failing = make_fetch(invalid=(200, {}))
    assert main(["--url", "api.example"], fetch=failing) == 1
    assert "invalid_age_rejected" in capsys.readouterr().out
