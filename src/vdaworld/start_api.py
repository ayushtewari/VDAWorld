"""Launch VDAWorld's isolated local model services."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from vdaworld.utils.printname import printname

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "api_config.yaml"


def _service_environment(cvd: str) -> dict[str, str]:
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    env.pop("VIRTUAL_ENV_PROMPT", None)
    env["CUDA_VISIBLE_DEVICES"] = cvd

    # ``uv run`` prepends its virtual environment to PATH. Merely removing
    # VIRTUAL_ENV is insufficient: ``conda run`` may otherwise resolve child
    # commands from the uv environment instead of the requested Conda env.
    if virtual_env and env.get("PATH"):
        virtual_env_bin = Path(virtual_env).resolve() / "bin"
        env["PATH"] = os.pathsep.join(
            entry
            for entry in env["PATH"].split(os.pathsep)
            if Path(entry).resolve() != virtual_env_bin
        )

    source_root = str(_REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, existing_pythonpath) if value
    )
    return env


def _vllm_command(service: dict[str, Any]) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(service["conda_env"]),
        "vllm",
        "serve",
        str(service["model"]),
        "--host",
        str(service.get("host", "127.0.0.1")),
        "--port",
        str(service.get("port", 8000)),
    ]
    for flag, key in (
        ("--gpu-memory-utilization", "gpu_memory_utilization"),
        ("--max-model-len", "max_model_len"),
        ("--tensor-parallel-size", "tensor_parallel_size"),
        ("--dtype", "dtype"),
        ("--tool-call-parser", "tool_call_parser"),
        ("--reasoning-parser", "reasoning_parser"),
        ("--chat-template", "chat_template"),
    ):
        if key in service:
            command.extend((flag, str(service[key])))
    if service.get("enable_auto_tool_choice"):
        command.append("--enable-auto-tool-choice")
    if service.get("trust_remote_code"):
        command.append("--trust-remote-code")
    return command


def _microservice_command(service: dict[str, Any]) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        str(service["conda_env"]),
        "python",
        "-m",
        "uvicorn",
        f"{service['module']}:app",
        "--host",
        str(service.get("host", "127.0.0.1")),
        "--port",
        str(service.get("port", 5000)),
    ]


def main() -> None:
    """Parse configuration, start enabled services, and supervise them."""

    printname()
    parser = argparse.ArgumentParser(description="Launch VDAWorld API services.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="API configuration YAML (default: repository api_config.yaml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        parser.error("API configuration must contain a YAML mapping.")

    processes: list[tuple[subprocess.Popen, str]] = []

    def cleanup(_signum=None, _frame=None) -> None:
        for process, name in processes:
            if process.poll() is not None:
                continue
            logging.info("Terminating %s (PID %d)", name, process.pid)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("%s did not stop; killing it", name)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def handle_signal(_signum, _frame) -> None:
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for service_name, raw_service in config.items():
        if not isinstance(raw_service, dict):
            parser.error(f"Service {service_name!r} must be a mapping.")
        if not raw_service.get("enabled", True):
            logging.info("Skipping disabled service %s", service_name)
            continue

        service_type = raw_service.get("type", "conda")
        if service_type == "postprocess":
            continue
        if service_type == "vllm":
            command = _vllm_command(raw_service)
        elif service_type == "conda":
            command = _microservice_command(raw_service)
        else:
            parser.error(f"Service {service_name!r} has unknown type {service_type!r}.")

        cvd = str(raw_service.get("cvd", "0"))
        env = _service_environment(cvd)
        logging.info("Launching %s on GPU selection %s", service_name, cvd)
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
            )
        except OSError:
            cleanup()
            raise

        time.sleep(2)
        if process.poll() is not None:
            cleanup()
            raise RuntimeError(
                f"Service {service_name!r} exited with code {process.returncode}."
            )
        processes.append((process, service_name))

    if not processes:
        logging.warning("No API services are enabled.")
        return

    logging.info("All enabled services are online. Press Ctrl+C to stop.")
    try:
        while True:
            for process, name in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"Service {name!r} exited unexpectedly with code {exit_code}."
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
