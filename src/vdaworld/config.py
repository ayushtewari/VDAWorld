"""
Pydantic configuration models for VDAWorld.

Runtime settings are loaded from YAML and validated before the pipeline starts.
Paths in a configuration file are resolved relative to the repository root, so
commands do not depend on the caller's working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_StrictModel):
    """Pricing and identity for a model alias."""

    model_name: str
    backend: Literal["anthropic", "gemini", "gemini_api", "openai", "vllm"] = (
        "gemini_api"
    )
    base_url: str | None = None
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)
    cache_price: float = Field(default=0.0, ge=0)


class VLMConfig(_StrictModel):
    """Configuration for the vision-language model interface."""

    model: str = Field(
        default="gemini_flash",
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    temperature: float = Field(default=0.0, ge=0)


class TaskConfig(_StrictModel):
    """Configuration for a generation task.

    Exactly one of `image_path` and `video_path` must be set. In image
    mode, the simulator starts at the supplied image. In video mode, the
    simulator continues from the final input frame.
    """

    image_path: str | None = None
    video_path: str | None = None
    caption: str = Field(min_length=1)
    output_dir: str = Field(default="outputs", min_length=1)
    simulator_class_name: str = Field(
        default="VideoSimulation",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    prompts_path: str = Field(default="prompts/generic", min_length=1)
    fps: int = Field(default=30, gt=0)
    simulation_steps: int = Field(default=150, gt=0)
    frame_size: tuple[int, int] = (1024, 576)
    critic_max_turns: int = Field(default=40, gt=0)
    lint_check: bool = False
    provide_image: bool = True
    restricted_tools: bool = False
    no_api: bool = False
    no_mhi: bool = False
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one_input(self) -> "TaskConfig":
        if bool(self.image_path) == bool(self.video_path):
            raise ValueError(
                "Exactly one of task.image_path or task.video_path must be set "
                f"(got image_path={self.image_path!r}, "
                f"video_path={self.video_path!r})."
            )
        if any(dimension <= 0 for dimension in self.frame_size):
            raise ValueError("task.frame_size dimensions must be positive")
        return self


class AppConfig(_StrictModel):
    """Root configuration object validated by Pydantic."""

    vlm: VLMConfig = Field(default_factory=VLMConfig)
    task: TaskConfig
    model_spec: ModelSpec = Field(
        default_factory=lambda: ModelSpec(model_name="gemini-3.5-flash")
    )


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | None, project_root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def _load_model_spec(model_alias: str, project_root: Path) -> ModelSpec:
    """Load a model specification from the repository's models directory."""

    model_path = project_root / "models" / f"{model_alias}.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model spec not found: {model_path}. Create a model YAML with "
            "model_name, backend, and optional pricing fields."
        )
    data = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Model spec must contain a YAML mapping: {model_path}")
    return ModelSpec(**data)


def load_config(config_path: str) -> AppConfig:
    """Load and validate a pipeline configuration file."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    if "task" not in data or not isinstance(data["task"], dict):
        raise ValueError("Configuration must contain a 'task' mapping.")

    project_root = _find_project_root(path)
    vlm_data = data.get("vlm") or {}
    if not isinstance(vlm_data, dict):
        raise ValueError("The 'vlm' field must be a mapping.")

    task_data = dict(data["task"])
    task_data.setdefault("output_dir", "outputs")
    task_data.setdefault("prompts_path", "prompts/generic")
    for field_name in (
        "image_path",
        "video_path",
        "output_dir",
        "prompts_path",
        "cache_dir",
    ):
        if field_name in task_data:
            task_data[field_name] = _resolve_path(task_data[field_name], project_root)

    vlm_config = VLMConfig(**vlm_data)
    return AppConfig(
        vlm=vlm_config,
        task=TaskConfig(**task_data),
        model_spec=_load_model_spec(vlm_config.model, project_root),
    )
