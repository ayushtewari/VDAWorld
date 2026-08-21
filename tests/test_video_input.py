"""Tests for video-input mode (video-conditioned continuation)."""

from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

import imageio.v3 as iio
from PIL import Image

from vdaworld.config import TaskConfig, VLMConfig
from vdaworld.core.critic_toolbox import CriticSandbox


def _write_video(path: str, n: int = 6, h: int = 16, w: int = 16) -> np.ndarray:
    """Write a tiny synthetic mp4 and return its frames."""
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, 2:6, i : i + 3] = 255  # a moving white square
    iio.imwrite(path, frames, fps=10, codec="libx264")
    return frames


# ---------------------------------------------------------------------------
# Config validation: exactly one of image_path / video_path
# ---------------------------------------------------------------------------


def test_config_accepts_image_only():
    cfg = TaskConfig(image_path="a.png", caption="c")
    assert cfg.image_path == "a.png" and cfg.video_path is None


def test_config_accepts_video_only():
    cfg = TaskConfig(video_path="v.mp4", caption="c")
    assert cfg.video_path == "v.mp4" and cfg.image_path is None


def test_config_rejects_both():
    with pytest.raises(ValidationError):
        TaskConfig(image_path="a.png", video_path="v.mp4", caption="c")


def test_config_rejects_neither():
    with pytest.raises(ValidationError):
        TaskConfig(caption="c")


def test_config_rejects_model_alias_path_traversal():
    with pytest.raises(ValidationError):
        VLMConfig(model="../../private")


# ---------------------------------------------------------------------------
# Input video tools
# ---------------------------------------------------------------------------


def test_view_input_frame_and_mhi(tmp_path):
    vid = str(tmp_path / "in.mp4")
    frames = _write_video(vid)
    sb = CriticSandbox(
        code="# stub\n",
        fps=10,
        n_frames=frames.shape[0],
        frame_size=(16, 16),
        input_image_path="unused.png",
        simulator_class_name="VideoSimulation",
        sandbox_dir=str(tmp_path / "sbox"),
        input_video_path=vid,
    )
    try:
        n = frames.shape[0]
        assert isinstance(sb.view_input_frame(0), Image.Image)
        assert isinstance(sb.view_input_frame(n - 1), Image.Image)
        assert isinstance(sb.view_input_mhi(), Image.Image)
        # Out-of-bounds and invalid indices return an error string, not an image.
        assert isinstance(sb.view_input_frame(n), str)
        assert isinstance(sb.view_input_frame(-1), str)
        assert isinstance(sb.view_input_frame(cast(Any, "x")), str)
    finally:
        sb.cleanup()


def test_run_simulation_passes_video_path(tmp_path, monkeypatch):
    """In video mode, run_simulation must add --video-path to the subprocess."""
    vid = str(tmp_path / "in.mp4")
    _write_video(vid)
    sb = CriticSandbox(
        code="# stub\n",
        fps=10,
        n_frames=3,
        frame_size=(16, 16),
        input_image_path="unused.png",
        simulator_class_name="VideoSimulation",
        sandbox_dir=str(tmp_path / "sbox"),
        input_video_path=vid,
    )
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr("subprocess.run", _fake_run)
    try:
        sb.run_simulation()
    finally:
        sb.cleanup()

    assert "--video-path" in captured["cmd"]
    assert vid in captured["cmd"]
