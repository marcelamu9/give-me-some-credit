"""Build the API image tagged with the short git SHA (and ``latest``) and the model version.

Python instead of bash so ``uv run poe docker-build`` works the same on Windows and Linux.
"""

import json
import subprocess
from pathlib import Path

from credit_risk.config import PROJECT_ROOT, load_config

IMAGE_NAME = "credit-risk-api"


def git_sha() -> str:
    """Short SHA of HEAD; ``-dirty`` if there are uncommitted changes (image not reproducible)."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{sha}-dirty" if status else sha


def model_version(model_dir: Path) -> str:
    """``trained_at`` of the champion, the same version the API reports."""
    metadata = model_dir / "champion_metadata.json"
    if not metadata.exists():
        return "unknown"
    return json.loads(metadata.read_text(encoding="utf-8"))["trained_at"]


def build_command(tag: str, version: str) -> list[str]:
    return [
        "docker",
        "build",
        "-t",
        f"{IMAGE_NAME}:{tag}",
        "-t",
        f"{IMAGE_NAME}:latest",
        "--build-arg",
        f"MODEL_VERSION={version}",
        ".",
    ]


def main() -> int:
    command = build_command(git_sha(), model_version(load_config().paths.models))
    print(" ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
