import asyncio
import io
import re
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image, PngImagePlugin

from demo import server
from demo.server import _file_url, _normalise_uploaded_image


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_normalise_uploaded_image_strips_metadata():
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Author", "private")
    source = io.BytesIO()
    Image.new("RGB", (4, 3), "red").save(
        source,
        format="PNG",
        pnginfo=metadata,
    )

    result = _normalise_uploaded_image(source.getvalue())

    with Image.open(io.BytesIO(result)) as image:
        assert image.mode == "RGB"
        assert image.size == (4, 3)
        assert image.info == {}


def test_normalise_uploaded_image_rejects_non_image():
    with pytest.raises(HTTPException) as exc_info:
        _normalise_uploaded_image(b"not an image")

    assert exc_info.value.status_code == 400


def test_file_url_percent_encodes_filesystem_names():
    assert _file_url("run", 'visualizations/a b".png') == (
        "/api/files/run/visualizations/a%20b%22.png"
    )


def test_media_endpoint_blocks_non_media_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_DIR", tmp_path)
    generation_dir = tmp_path / "run"
    generation_dir.mkdir()
    (generation_dir / "frame.png").write_bytes(b"image")
    (generation_dir / "active.html").write_text("<script></script>")
    (tmp_path / "outside.png").write_bytes(b"outside")

    response = asyncio.run(server.api_file("run", "frame.png"))
    assert response.headers["x-content-type-options"] == "nosniff"

    with pytest.raises(HTTPException) as non_media:
        asyncio.run(server.api_file("run", "active.html"))
    assert non_media.value.status_code == 404

    with pytest.raises(HTTPException) as escaped:
        asyncio.run(server.api_file("run", "../outside.png"))
    assert escaped.value.status_code == 403


def test_demo_model_options_have_model_specs():
    html = (REPO_ROOT / "demo/static/index.html").read_text(encoding="utf-8")
    aliases = re.findall(r'<option value="([A-Za-z0-9_-]+)">', html)

    assert aliases
    assert "claude_fable5" not in aliases
    for alias in aliases:
        assert (REPO_ROOT / "models" / f"{alias}.yaml").is_file()
