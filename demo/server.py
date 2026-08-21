#!/usr/bin/env python3
"""
VDAWorld Demo Server — exposes the world generation pipeline via a web UI.

Run from the repo root:
    .venv/bin/python demo/server.py [--port 7860]
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import multiprocessing
import os
import queue
import re
import sys
import threading
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator
from urllib.parse import quote

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DEMO_DIR = Path(__file__).parent
RESULTS_DIR = REPO_ROOT / "results" / "demo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    queue_task = asyncio.create_task(_process_queue_task())
    try:
        yield
    finally:
        queue_task.cancel()
        with suppress(asyncio.CancelledError):
            await queue_task


app = FastAPI(title="VDAWorld Demo", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

# job_id -> job metadata, event queue, and optional worker process
_JOBS: dict[str, dict] = {}
# Ordered list of job_ids waiting to run (not yet started)
_JOB_QUEUE: list[str] = []
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_msg(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Queue-based log handler
# ---------------------------------------------------------------------------


class _QueueHandler(logging.Handler):
    def __init__(self, q: multiprocessing.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put_nowait(
                {
                    "type": "log",
                    "level": record.levelname,
                    "message": self.format(record),
                }
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tool-call watcher
# ---------------------------------------------------------------------------

_IMG_NAMES = (
    "frame.png",
    "first_frame.png",
    "last_frame.png",
    "first_seg_frame.png",
    "seg_frame.png",
    "mhi.png",
    "debug.png",
    "debug_view.png",
)


def _watch_tool_calls(
    tool_calls_dir: str, q: queue.Queue, stop: threading.Event, gen_name: str
) -> None:
    """Poll the tool_calls directory and emit an event for each new call."""
    seen: set[str] = set()
    while not stop.is_set():
        if os.path.isdir(tool_calls_dir):

            def _sort_key(name: str) -> int:
                try:
                    return int(name.split("_")[0])
                except ValueError:
                    return 9999

            for entry in sorted(os.listdir(tool_calls_dir), key=_sort_key):
                if entry in seen:
                    continue
                meta_path = os.path.join(tool_calls_dir, entry, "metadata.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entry_dir = os.path.join(tool_calls_dir, entry)
                    meta["images"] = [
                        _file_url(gen_name, f"tool_calls/{entry}/{img}")
                        for img in _IMG_NAMES
                        if os.path.isfile(os.path.join(entry_dir, img))
                    ]
                    q.put_nowait({"type": "tool_call", "data": meta})
                    seen.add(entry)
                except Exception:
                    pass
        stop.wait(0.5)


# ---------------------------------------------------------------------------
# Pipeline runner — runs in a child process so it can be cancelled cleanly
# ---------------------------------------------------------------------------


def _pipeline_worker(
    q: multiprocessing.Queue,
    image_path: str,
    caption: str,
    model: str,
    output_dir: str,
    repo_root: str,
) -> None:
    """Entry point for the child process. Must be a module-level function."""
    import sys as _sys

    _sys.path.insert(0, str(Path(repo_root) / "src"))

    handler: logging.Handler | None = None
    stop_watcher = threading.Event()

    try:
        handler = _QueueHandler(q)
        handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
        logging.getLogger("vdaworld").addHandler(handler)
        logging.getLogger("vdaworld").setLevel(logging.INFO)

        q.put_nowait(
            {"type": "status", "status": "running", "message": "Pipeline initialising…"}
        )

        from PIL import Image as _Image

        with _Image.open(image_path) as _img:
            img_w, img_h = _img.size

        config_data = {
            "vlm": {"model": model, "temperature": 1.0},
            "task": {
                "image_path": image_path,
                "caption": caption,
                "output_dir": output_dir,
                "simulator_class_name": "VideoSimulation",
                "prompts_path": "prompts/generic",
                "fps": 30,
                "simulation_steps": 150,
                "frame_size": [img_w, img_h],
                "critic_max_turns": 40,
            },
        }
        Path(output_dir, ".vdaworld-output").touch()
        config_path = os.path.join(output_dir, "_demo_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        gen_name = os.path.basename(output_dir)
        tool_calls_dir = os.path.join(output_dir, "tool_calls")
        watcher = threading.Thread(
            target=_watch_tool_calls,
            args=(tool_calls_dir, q, stop_watcher, gen_name),
            daemon=True,
        )
        watcher.start()

        from vdaworld.config import load_config
        from vdaworld.pipeline import WorldGenerationPipeline

        os.chdir(repo_root)
        try:
            config = load_config(config_path)
            pipeline = WorldGenerationPipeline(config)
            result = pipeline.run()
        finally:
            stop_watcher.set()
            watcher.join(timeout=3)
            # Re-save after pipeline.run() clears the output directory
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

        if result is None:
            q.put_nowait(
                {
                    "type": "status",
                    "status": "failed",
                    "message": "Pipeline returned no result.",
                }
            )
        else:
            q.put_nowait(
                {
                    "type": "status",
                    "status": "complete",
                    "message": "Generation complete!",
                    "tool_calls": result.tool_call_count,
                    "elapsed": round(result.elapsed_seconds, 1),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cached_tokens": result.cached_tokens,
                }
            )

    except Exception:
        import traceback

        traceback.print_exc()
        q.put_nowait(
            {
                "type": "status",
                "status": "failed",
                "message": "Generation failed; see the server log for details.",
            }
        )

    finally:
        if handler is not None:
            logging.getLogger("vdaworld").removeHandler(handler)
        q.put_nowait({"type": "done"})


# ---------------------------------------------------------------------------
# Queue processor
# ---------------------------------------------------------------------------


def _start_job(job_id: str) -> None:
    """Spawn the worker process for a queued job."""

    job = _JOBS.get(job_id)
    if not job or job.get("status") == "cancelled":
        return
    proc = multiprocessing.Process(
        target=_pipeline_worker,
        args=(
            job["events"],
            job["image_path"],
            job["caption"],
            job["model"],
            job["output_dir"],
            str(REPO_ROOT),
        ),
        daemon=True,
    )
    with _JOBS_LOCK:
        _JOBS[job_id]["process"] = proc
        _JOBS[job_id]["status"] = "starting"
    proc.start()


async def _process_queue_task() -> None:
    """Background asyncio task: starts one queued job at a time."""
    while True:
        await asyncio.sleep(1.0)

        next_job_id: str | None = None
        with _JOBS_LOCK:
            # Is any job's process still alive?
            has_active = any(
                j.get("process") is not None and j["process"].is_alive()
                for j in _JOBS.values()
            )
            if has_active or not _JOB_QUEUE:
                continue

            next_job_id = _JOB_QUEUE.pop(0)

            # Notify remaining queued jobs of their updated position
            for i, jid in enumerate(_JOB_QUEUE):
                j = _JOBS.get(jid)
                if j:
                    try:
                        j["events"].put_nowait(
                            {
                                "type": "status",
                                "status": "queued",
                                "position": i + 1,
                                "message": f"Queued — position {i + 1}",
                            }
                        )
                    except Exception:
                        pass

        if next_job_id:
            _start_job(next_job_id)


# ---------------------------------------------------------------------------
# API — generate
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 25_000_000
_MAX_CAPTION_CHARS = 10_000
_ALLOWED_MEDIA_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
}


def _generation_dir(name: str) -> Path:
    if not _SAFE_NAME.fullmatch(name):
        raise HTTPException(400, "Invalid generation name")
    return RESULTS_DIR / name


def _file_url(name: str, relative_path: str) -> str:
    """Build a URL with every filesystem-derived component percent-encoded."""

    return f"/api/files/{quote(name, safe='')}/{quote(relative_path, safe='/')}"


def _normalise_uploaded_image(data: bytes) -> bytes:
    """Validate an uploaded image and return a metadata-free RGB PNG."""

    if not data or len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be between 1 byte and 20 MiB")
    try:
        with Image.open(io.BytesIO(data)) as uploaded:
            width, height = uploaded.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise HTTPException(413, "Image dimensions are too large")
            rgb_image = uploaded.convert("RGB")
            output = io.BytesIO()
            rgb_image.save(output, format="PNG")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Upload is not a valid supported image") from exc
    return output.getvalue()


@app.post("/api/generate")
async def api_generate(
    image: UploadFile = File(...),
    caption: str = Form(...),
    model: str = Form("gemini_flash"),
    name: str = Form(""),
) -> dict:
    """Enqueue a new generation. Returns {job_id, name, queue_position}."""
    if not name:
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    output_path = _generation_dir(name)

    if not _SAFE_NAME.fullmatch(model):
        raise HTTPException(400, "Invalid model alias")
    if not caption or len(caption) > _MAX_CAPTION_CHARS:
        raise HTTPException(400, "Caption must contain 1 to 10,000 characters")

    upload_bytes = await image.read(_MAX_UPLOAD_BYTES + 1)
    normalised_image = _normalise_uploaded_image(upload_bytes)

    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(
            409,
            "A generation with this name already exists. Use Re-run to replace it.",
        ) from exc
    output_dir = str(output_path)

    # Keep the input outside output_dir so the pipeline's preparation step
    # cannot remove it before use. Saving as PNG strips upload metadata.
    uploads_dir = RESULTS_DIR / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_path = uploads_dir / f"{name}.png"
    upload_path.write_bytes(normalised_image)
    image_path = str(upload_path)

    job_id = str(uuid.uuid4())
    q: multiprocessing.Queue = multiprocessing.Queue()

    with _JOBS_LOCK:
        queue_position = len(_JOB_QUEUE) + 1
        _JOBS[job_id] = {
            "status": "queued",
            "name": name,
            "output_dir": output_dir,
            "image_path": image_path,
            "caption": caption,
            "model": model,
            "events": q,
            "process": None,
            "error": None,
        }
        _JOB_QUEUE.append(job_id)

    # Put initial status so the SSE stream has something to read immediately
    q.put_nowait(
        {
            "type": "status",
            "status": "queued",
            "position": queue_position,
            "message": f"Queued — position {queue_position}",
        }
    )

    return {"job_id": job_id, "name": name, "queue_position": queue_position}


# ---------------------------------------------------------------------------
# API — SSE stream
# ---------------------------------------------------------------------------


@app.get("/api/stream/{job_id}")
async def api_stream(job_id: str, request: Request) -> StreamingResponse:
    if job_id not in _JOBS:
        raise HTTPException(404, "Job not found")

    q: multiprocessing.Queue = _JOBS[job_id]["events"]

    async def gen() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await loop.run_in_executor(None, lambda: q.get(timeout=1.0))

                # Keep _JOBS status in sync so the queue processor knows when a slot opens
                if event.get("type") == "status":
                    with _JOBS_LOCK:
                        if job_id in _JOBS:
                            _JOBS[job_id]["status"] = event["status"]

                yield _sse_msg(event)

                if event.get("type") == "done":
                    break

            except queue.Empty:
                # Safety net: if the process died without sending done, recover
                with _JOBS_LOCK:
                    job = _JOBS.get(job_id)
                    proc = job.get("process") if job else None
                    cur_status = job.get("status") if job else None

                if (
                    proc is not None
                    and not proc.is_alive()
                    and cur_status in ("starting", "running")
                ):
                    with _JOBS_LOCK:
                        if job_id in _JOBS:
                            _JOBS[job_id]["status"] = "failed"
                    yield _sse_msg(
                        {
                            "type": "status",
                            "status": "failed",
                            "message": "Process exited unexpectedly.",
                        }
                    )
                    yield _sse_msg({"type": "done"})
                    break

                yield ": heartbeat\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# API — active queue (lets fresh page loads discover in-progress jobs)
# ---------------------------------------------------------------------------


@app.get("/api/queue")
async def api_queue() -> dict:
    """Returns all currently running and queued jobs."""
    with _JOBS_LOCK:
        running = []
        for jid, j in _JOBS.items():
            proc = j.get("process")
            if proc and proc.is_alive():
                running.append(
                    {
                        "job_id": jid,
                        "name": j["name"],
                        "status": j["status"],
                        "caption": j.get("caption", ""),
                    }
                )
        queued = []
        for i, jid in enumerate(_JOB_QUEUE):
            queued_job = _JOBS.get(jid)
            if queued_job:
                queued.append(
                    {
                        "job_id": jid,
                        "name": queued_job["name"],
                        "position": i + 1,
                        "caption": queued_job.get("caption", ""),
                    }
                )
    return {"running": running, "queued": queued}


# ---------------------------------------------------------------------------
# API — job status (for polling / reconnect initialisation)
# ---------------------------------------------------------------------------


@app.get("/api/status/{job_id}")
async def api_status(job_id: str) -> dict:
    """Return current job status without requiring an SSE connection."""

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        position = _JOB_QUEUE.index(job_id) + 1 if job_id in _JOB_QUEUE else None
        proc = job.get("process")
        return {
            "job_id": job_id,
            "name": job["name"],
            "status": job["status"],
            "queue_position": position,
            "process_alive": proc.is_alive() if proc else False,
        }


# ---------------------------------------------------------------------------
# API — rerun an existing generation
# ---------------------------------------------------------------------------


@app.post("/api/rerun/{name}")
async def api_rerun(name: str) -> dict:
    """Re-enqueue a completed generation using its original image and caption."""
    gen_dir = _generation_dir(name)
    if not gen_dir.is_dir():
        raise HTTPException(404, "Generation not found")

    cfg_p = gen_dir / "_demo_config.yaml"
    if not cfg_p.exists():
        raise HTTPException(400, "No config found — cannot determine original inputs")

    try:
        cfg = yaml.safe_load(cfg_p.read_text()) or {}
        caption = cfg.get("task", {}).get("caption", "")
        model = cfg.get("vlm", {}).get("model", "gemini_flash")
    except Exception as exc:
        raise HTTPException(500, f"Failed to read config: {exc}")

    # Find the original uploaded image (any extension)
    uploads_dir = RESULTS_DIR / "_uploads"
    matches = list(uploads_dir.glob(f"{name}.*")) if uploads_dir.is_dir() else []
    if not matches:
        raise HTTPException(400, "Original image not found — it may have been deleted")
    image_path = str(matches[0])

    job_id = str(uuid.uuid4())
    q: multiprocessing.Queue = multiprocessing.Queue()
    output_dir = str(RESULTS_DIR / name)

    with _JOBS_LOCK:
        queue_position = len(_JOB_QUEUE) + 1
        _JOBS[job_id] = {
            "status": "queued",
            "name": name,
            "output_dir": output_dir,
            "image_path": image_path,
            "caption": caption,
            "model": model,
            "events": q,
            "process": None,
            "error": None,
        }
        _JOB_QUEUE.append(job_id)

    q.put_nowait(
        {
            "type": "status",
            "status": "queued",
            "position": queue_position,
            "message": f"Queued — position {queue_position}",
        }
    )

    return {"job_id": job_id, "name": name, "queue_position": queue_position}


# ---------------------------------------------------------------------------
# API — cancel
# ---------------------------------------------------------------------------


@app.post("/api/cancel/{job_id}")
async def api_cancel(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        # Remove from queue if it was waiting (hadn't started yet)
        if job_id in _JOB_QUEUE:
            _JOB_QUEUE.remove(job_id)
            for i, jid in enumerate(_JOB_QUEUE):
                j = _JOBS.get(jid)
                if j:
                    try:
                        j["events"].put_nowait(
                            {
                                "type": "status",
                                "status": "queued",
                                "position": i + 1,
                                "message": f"Queued — position {i + 1}",
                            }
                        )
                    except Exception:
                        pass

        proc: multiprocessing.Process | None = job.get("process")
        q: multiprocessing.Queue = job["events"]
        job["status"] = "cancelled"

    if proc and proc.is_alive():
        proc.terminate()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: proc.join(timeout=5))
        if proc.is_alive():
            proc.kill()
            await loop.run_in_executor(None, lambda: proc.join(timeout=2))

    # Drain any pending events then send cancelled + done so the SSE stream closes.
    while True:
        try:
            q.get_nowait()
        except Exception:
            break
    q.put_nowait(
        {"type": "status", "status": "cancelled", "message": "Cancelled by user."}
    )
    q.put_nowait({"type": "done"})
    return {"ok": True}


# ---------------------------------------------------------------------------
# API — generations list
# ---------------------------------------------------------------------------


def _gen_meta(entry: Path) -> dict | None:
    vis = entry / "visualizations"
    has_video = (vis / "simulation.mp4").exists()
    has_thumb = (vis / "frame_0_sim.png").exists()

    tc_dir = entry / "tool_calls"
    tc_count = sum(1 for d in tc_dir.iterdir() if d.is_dir()) if tc_dir.is_dir() else 0

    if not has_video and not has_thumb and tc_count == 0:
        return None

    caption = ""
    config_p = entry / "_demo_config.yaml"
    if config_p.exists():
        try:
            cfg = yaml.safe_load(config_p.read_text()) or {}
            caption = cfg.get("task", {}).get("caption", "")
        except Exception:
            pass

    mtime = entry.stat().st_mtime
    return {
        "name": entry.name,
        "has_video": has_video,
        "tool_call_count": tc_count,
        "caption": caption,
        "thumbnail": _file_url(entry.name, "visualizations/frame_0_sim.png")
        if has_thumb
        else None,
        "mtime": mtime,
    }


@app.get("/api/generations")
async def api_generations() -> list:
    out = []
    for entry in RESULTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        m = _gen_meta(entry)
        if m:
            out.append(m)
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# API — single generation detail
# ---------------------------------------------------------------------------


@app.get("/api/generation/{name}")
async def api_generation_detail(name: str) -> dict:
    gen_dir = _generation_dir(name)
    if not gen_dir.is_dir():
        raise HTTPException(404, "Generation not found")

    tool_calls = []
    tc_dir = gen_dir / "tool_calls"
    if tc_dir.is_dir():

        def _tc_sort_key(p: Path) -> int:
            try:
                return int(p.name.split("_")[0])
            except ValueError:
                return 9999

        for entry in sorted(tc_dir.iterdir(), key=_tc_sort_key):
            mp = entry / "metadata.json"
            if not mp.exists():
                continue
            meta = json.loads(mp.read_text())
            meta["images"] = [
                _file_url(name, f"tool_calls/{entry.name}/{img}")
                for img in _IMG_NAMES
                if (entry / img).exists()
            ]
            tool_calls.append(meta)

    vis_dir = gen_dir / "visualizations"
    visualizations: dict[str, str] = {}
    if vis_dir.is_dir():
        for f in vis_dir.iterdir():
            if f.suffix in (".mp4", ".png", ".jpg", ".jpeg"):
                visualizations[f.name] = _file_url(name, f"visualizations/{f.name}")

    code = ""
    cp = gen_dir / "simulator_gen.py"
    if cp.exists():
        code = cp.read_text()

    def _read_md(filename: str) -> str:
        p = gen_dir / filename
        return p.read_text(encoding="utf-8") if p.exists() else ""

    caption = model = ""
    cfg_p = gen_dir / "_demo_config.yaml"
    if cfg_p.exists():
        try:
            cfg = yaml.safe_load(cfg_p.read_text()) or {}
            caption = cfg.get("task", {}).get("caption", "")
            model = cfg.get("vlm", {}).get("model", "")
        except Exception:
            pass

    return {
        "name": name,
        "caption": caption,
        "model": model,
        "tool_calls": tool_calls,
        "visualizations": visualizations,
        "code": code,
        "prompt": _read_md("prompt.md"),
        "thinking": _read_md("thinking.md"),
        "response": _read_md("response.md"),
    }


# ---------------------------------------------------------------------------
# API — file serving
# ---------------------------------------------------------------------------


@app.get("/api/files/{name}/{path:path}")
async def api_file(name: str, path: str) -> FileResponse:
    generation_dir = _generation_dir(name).resolve()
    target = (generation_dir / path).resolve()
    try:
        target.relative_to(generation_dir)
    except ValueError as exc:
        raise HTTPException(403, "Path escapes the generation directory") from exc
    if not target.is_file() or target.suffix.lower() not in _ALLOWED_MEDIA_SUFFIXES:
        raise HTTPException(404, "Media file not found")
    return FileResponse(
        str(target),
        headers={"X-Content-Type-Options": "nosniff"},
    )


# ---------------------------------------------------------------------------
# Static files + index
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(DEMO_DIR / "static")), name="static")


@app.get("/{full_path:path}")
async def catch_all(full_path: str) -> FileResponse:
    return FileResponse(str(DEMO_DIR / "static" / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("DEMO_PORT", 7860))
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  VDAWorld Demo  →  http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
