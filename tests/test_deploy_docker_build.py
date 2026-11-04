import json

from credit_risk.deploy.docker_build import IMAGE_NAME, build_command, model_version


def test_model_version_reads_trained_at(tmp_path):
    (tmp_path / "champion_metadata.json").write_text(
        json.dumps({"trained_at": "2026-10-18T02:59:33+00:00"}), encoding="utf-8"
    )
    assert model_version(tmp_path) == "2026-10-18T02:59:33+00:00"


def test_model_version_unknown_without_metadata(tmp_path):
    assert model_version(tmp_path) == "unknown"


def test_build_command_tags_sha_and_latest():
    command = build_command("abc1234", "v1")
    assert command[:2] == ["docker", "build"]
    assert f"{IMAGE_NAME}:abc1234" in command
    assert f"{IMAGE_NAME}:latest" in command
    assert command[command.index("--build-arg") + 1] == "MODEL_VERSION=v1"
    assert command[-1] == "."
