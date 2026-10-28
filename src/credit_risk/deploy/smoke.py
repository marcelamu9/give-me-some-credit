"""Deployment smoke test: checks a running API end to end (local container or AWS).

It only uses the standard library, so it runs anywhere: on the laptop, against the container
or in GitHub Actions. Usage: ``python -m credit_risk.deploy.smoke --url <URL>``.
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from credit_risk.api.examples import INVALID_AGE, LOW_RISK, SAME_CLIENT_WITH_DELAYS

# (method, url, json body or None) -> (HTTP status, JSON response)
Fetch = Callable[[str, str, dict | None], tuple[int, dict]]

TIMEOUT_SECONDS = 30
LOCAL_HOSTS = ("localhost", "127.0.0.1")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def normalize_url(url: str) -> str:
    """Drop the trailing slash; add https:// (or http:// for local hosts) if there is no scheme."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        host = url.split("/")[0].split(":")[0]
        url = f"{'http' if host in LOCAL_HOSTS else 'https'}://{url}"
    return url


def urllib_fetch(method: str, url: str, body: dict | None) -> tuple[int, dict]:
    """HTTP call with the standard library; error statuses are returned, not raised."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"{}")
        except json.JSONDecodeError:
            return error.code, {}


def _run(name: str, check: Callable[[], tuple[bool, str]]) -> CheckResult:
    try:
        passed, detail = check()
    except Exception as exc:  # network errors, unexpected bodies: report, do not crash
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")
    return CheckResult(name, passed, detail)


def run_smoke(base_url: str, fetch: Fetch = urllib_fetch, repeats: int = 5) -> list[CheckResult]:
    """Health, the three demo examples and the mean latency of ``/predict``."""
    base = normalize_url(base_url)
    predict_url = f"{base}/predict"
    seen: dict[str, float] = {}

    def health() -> tuple[bool, str]:
        status, body = fetch("GET", f"{base}/health", None)
        loaded = body.get("model_loaded")
        return status == 200 and loaded is True, f"HTTP {status}, model_loaded={loaded}"

    def low_risk() -> tuple[bool, str]:
        status, body = fetch("POST", predict_url, LOW_RISK)
        seen["low_risk"] = body.get("probability", float("nan"))
        decision = body.get("decision")
        return status == 200 and decision == "approve", (
            f"HTTP {status}, decision={decision}, p={body.get('probability')}"
        )

    def delays() -> tuple[bool, str]:
        status, body = fetch("POST", predict_url, SAME_CLIENT_WITH_DELAYS)
        decision, probability = body.get("decision"), body.get("probability")
        higher = probability is not None and probability > seen.get("low_risk", float("inf"))
        ok = status == 200 and decision in {"review", "reject"} and higher
        return ok, f"HTTP {status}, decision={decision}, p={probability}"

    def invalid_age() -> tuple[bool, str]:
        status, _ = fetch("POST", predict_url, INVALID_AGE)
        return status == 422, f"HTTP {status} (se espera 422)"

    def latency() -> tuple[bool, str]:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fetch("POST", predict_url, LOW_RISK)
            times.append((time.perf_counter() - start) * 1000)
        return True, f"{statistics.mean(times):.0f} ms de media en {repeats} llamadas"

    checks = [
        ("health", health),
        ("low_risk_approved", low_risk),
        ("delays_raise_decision", delays),
        ("invalid_age_rejected", invalid_age),
        ("latency", latency),
    ]
    return [_run(name, check) for name, check in checks]


def main(argv: list[str] | None = None, fetch: Fetch = urllib_fetch) -> int:
    parser = argparse.ArgumentParser(description="Smoke test de la API desplegada.")
    parser.add_argument("--url", required=True, help="URL base de la API")
    parser.add_argument("--repeats", type=int, default=5, help="llamadas para medir la latencia")
    args = parser.parse_args(argv)

    results = run_smoke(args.url, fetch=fetch, repeats=args.repeats)
    print(f"Smoke test contra {normalize_url(args.url)}")
    for result in results:
        print(f"  [{'OK' if result.passed else 'FALLA'}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
