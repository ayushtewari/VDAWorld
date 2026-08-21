from pathlib import Path

import pytest
import yaml

from vdaworld.start_api import _service_environment, _vllm_command


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vllm_command_uses_configured_environment_and_tool_calling():
    services = yaml.safe_load(
        (REPO_ROOT / "api_config.yaml").read_text(encoding="utf-8")
    )

    command = _vllm_command(services["vllm"])

    assert command[:8] == [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "vllm",
        "vllm",
        "serve",
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    ]
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--tensor-parallel-size") + 1] == "4"


def test_vllm_command_requires_explicit_conda_environment():
    with pytest.raises(KeyError, match="conda_env"):
        _vllm_command({"model": "example/model"})


def test_service_environment_removes_active_virtualenv_from_path(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/workspace/.venv")
    monkeypatch.setenv("VIRTUAL_ENV_PROMPT", "(.venv)")
    monkeypatch.setenv("PATH", "/workspace/.venv/bin:/usr/local/bin:/usr/bin")

    environment = _service_environment("2")

    assert "VIRTUAL_ENV" not in environment
    assert "VIRTUAL_ENV_PROMPT" not in environment
    assert environment["PATH"] == "/usr/local/bin:/usr/bin"
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
