from types import SimpleNamespace
from typing import Any, cast

import pytest

from vdaworld.pipeline import (
    WorldGenerationPipeline,
    _static_check,
    _uses_mujoco,
)


def _pipeline_for(output_dir):
    pipeline = object.__new__(WorldGenerationPipeline)
    pipeline.config = cast(
        Any, SimpleNamespace(task=SimpleNamespace(output_dir=str(output_dir)))
    )
    return pipeline


def test_prepare_output_dir_marks_new_directory(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    _pipeline_for(output_dir)._prepare_output_dir(load_existing=False)

    assert {path.name for path in output_dir.iterdir()} == {".vdaworld-output"}


def test_prepare_output_dir_refuses_unmarked_content(tmp_path):
    output_dir = tmp_path / "valuable-data"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("do not remove", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to clear"):
        _pipeline_for(output_dir)._prepare_output_dir(load_existing=False)

    assert sentinel.read_text(encoding="utf-8") == "do not remove"


def test_prepare_output_dir_clears_only_marked_content(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / ".vdaworld-output").touch()
    (output_dir / "old.txt").write_text("stale", encoding="utf-8")
    nested = output_dir / "nested"
    nested.mkdir()
    (nested / "old.txt").write_text("stale", encoding="utf-8")

    _pipeline_for(output_dir)._prepare_output_dir(load_existing=False)

    assert {path.name for path in output_dir.iterdir()} == {".vdaworld-output"}


def test_static_check_reports_syntax_errors(tmp_path):
    source = tmp_path / "simulator.py"
    source.write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = _static_check(str(source))

    assert result is not None
    assert "SyntaxError" in result


def test_static_check_accepts_valid_python(tmp_path):
    source = tmp_path / "simulator.py"
    source.write_text("answer = 42\n", encoding="utf-8")

    assert _static_check(str(source)) is None


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import mujoco", True),
        ("from MUJOCO import Renderer", True),
        ("import pymunk", False),
    ],
)
def test_uses_mujoco(source, expected):
    assert _uses_mujoco(source) is expected
