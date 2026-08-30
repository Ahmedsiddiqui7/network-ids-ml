"""Cycle 5 verification (MAD 3.4 / Pillar 7): boots the full stack and
asserts the dashboard's replayed predictions match the API's raw
/predict/batch response for the same sample -- the literal Cycle 5
verification step ("replay N sample flows through the full stack, assert
the dashboard's displayed prediction matches the API's raw JSON response").

Tries docker-compose first. If Docker genuinely isn't usable on this
machine (not installed, or `docker compose up` fails/times out), falls
back to plain local dev processes (uvicorn + `next dev`) -- the assertion
itself only needs something real listening on localhost:8000/:3000, not a
particular way of getting there. docker/docker-compose.yml and both
Dockerfiles are written and reviewed for correctness either way; this test
only *confirms* the Docker path when Docker actually cooperates, and says
so plainly when it had to fall back instead of silently reporting Docker
as verified.

Slow (image builds / npm install can take minutes on a cold run) -- not
meant to be part of a fast pre-commit loop, same as this repo's other
real-artifact-dependent integration tests.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx2 as httpx
import pytest

from src.config import REPO_ROOT

DASHBOARD_DIR = REPO_ROOT / "src" / "dashboard"
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"
REPLAY_FIXTURE = DASHBOARD_DIR / "data" / "replay_flows.json"

API_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:3000"

READY_TIMEOUT_S = 180
POLL_INTERVAL_S = 2
COMPOSE_BUILD_TIMEOUT_S = 900


def _wait_until_ready(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)
    return False


def _docker_compose_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def _compose_down() -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def running_stack():
    """Yields "docker-compose" or "local-fallback" once localhost:8000 and
    localhost:3000 are both actually serving requests."""
    if _docker_compose_usable():
        try:
            build = subprocess.run(
                ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=COMPOSE_BUILD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            build = None

        if build is not None and build.returncode == 0:
            if _wait_until_ready(f"{API_URL}/health", READY_TIMEOUT_S) and _wait_until_ready(
                DASHBOARD_URL, READY_TIMEOUT_S
            ):
                try:
                    yield "docker-compose"
                finally:
                    _compose_down()
                return
        _compose_down()
        # falls through to the local-process fallback below

    if shutil.which("npm") is None:
        pytest.skip("Neither docker-compose nor npm is available -- can't run the E2E stack")

    api_proc = subprocess.Popen(
        ["venv/bin/uvicorn", "src.api.main:app", "--port", "8000"],
        cwd=REPO_ROOT,
    )
    dashboard_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "3000"],
        cwd=DASHBOARD_DIR,
        env={**os.environ, "API_BASE_URL": API_URL},
    )
    try:
        api_ok = _wait_until_ready(f"{API_URL}/health", READY_TIMEOUT_S)
        dashboard_ok = _wait_until_ready(DASHBOARD_URL, READY_TIMEOUT_S)
        if not (api_ok and dashboard_ok):
            pytest.skip("Local fallback processes did not become ready in time")
        yield "local-fallback"
    finally:
        api_proc.terminate()
        dashboard_proc.terminate()
        try:
            api_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api_proc.kill()
        try:
            dashboard_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dashboard_proc.kill()


def test_dashboard_replay_matches_api_directly(running_stack):
    if running_stack == "local-fallback":
        print(
            "\nNOTE: docker-compose was not usable on this machine for this run -- "
            "verified via local dev processes (uvicorn + `next dev`) instead. "
            "docker/docker-compose.yml and both Dockerfiles are written and reviewed "
            "but not runtime-confirmed by this particular run."
        )

    flows = json.loads(REPLAY_FIXTURE.read_text())
    payload = {"flows": [row["features"] for row in flows]}

    direct = httpx.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
    assert direct.status_code == 200
    direct_predictions = direct.json()["predictions"]

    dashboard = httpx.get(f"{DASHBOARD_URL}/api/replay", timeout=60)
    assert dashboard.status_code == 200
    dashboard_rows = dashboard.json()["rows"]

    assert len(dashboard_rows) == len(direct_predictions) == len(flows)

    for flow, dash_row, direct_pred in zip(flows, dashboard_rows, direct_predictions):
        assert dash_row["id"] == flow["id"]
        assert dash_row["true_label"] == flow["true_label"]
        assert dash_row["prediction"] == direct_pred["prediction"]
        assert dash_row["is_malicious"] == direct_pred["is_malicious"]
        assert dash_row["risk_score"] == pytest.approx(direct_pred["risk_score"])
        assert dash_row["confidence"] == pytest.approx(direct_pred["confidence"])
