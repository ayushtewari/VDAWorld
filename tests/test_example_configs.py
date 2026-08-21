from pathlib import Path

import yaml

from vdaworld.config import ModelSpec, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_load_with_existing_inputs_and_unique_outputs():
    config_paths = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    configs = [load_config(str(path)) for path in config_paths]

    assert config_paths
    inputs = [config.task.image_path or config.task.video_path for config in configs]
    assert all(input_path and Path(input_path).is_file() for input_path in inputs)

    output_dirs = [config.task.output_dir for config in configs]
    assert len(output_dirs) == len(set(output_dirs))


def test_model_specs_validate():
    model_paths = sorted((REPO_ROOT / "models").glob("*.yaml"))

    assert model_paths
    for model_path in model_paths:
        model_data = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        ModelSpec.model_validate(model_data)
