"""
End-to-end world generation pipeline.

The pipeline runs a single agentic generation loop in which the VLM writes,
tests, and validates simulator code in one continuous tool-calling session.
All artefacts are written flat into ``output_dir``::

    <output_dir>/
      _cache/
      prompt.md  turns.md  response.md  thinking.md  state.json
      simulator_gen.py              ← final code (post-loop, re-compiled)
      tool_calls/                   ← per-call logs from the sandbox loop
        0_write_code_from_scratch/
        1_check_compilation/
        2_run_simulation/
        …
      api_calls/                    ← WorldAPI calls from the final re-run
      visualizations/
        simulation.mp4
        frame_0_sim.png             ← initial rendered frame from simulator
        frame_0_gt.png              ← ground-truth input image
"""

from __future__ import annotations

import gc
import importlib.util
import inspect
import linecache
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


import imageio.v3 as iio
import numpy as np
from PIL import Image

from vdaworld.api.vlm import VLMClient
from vdaworld.config import AppConfig
from vdaworld.core.agentic_generator import AgenticGenerator
from vdaworld.core.api import WorldAPI
from vdaworld.core.interfaces import GenerationResult
from vdaworld.core.simulator import SimulatorBase
from vdaworld.utils.error_handling import StaticCheckError

logger = logging.getLogger(__name__)

_OUTPUT_MARKER = ".vdaworld-output"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Encapsulates the output of a successful :meth:`WorldGenerationPipeline.run` call.

    Attributes:
        code: The final simulator source code produced by the agentic loop.
        output_dir: Filesystem path to the pipeline output directory.
        tool_call_count: Number of sandbox tool calls made during generation.
        elapsed_seconds: Wall-clock seconds for the full pipeline run.
        input_tokens: Total input tokens consumed by the VLM.
        output_tokens: Total output tokens produced by the VLM.
        cached_tokens: Total cached tokens used by the VLM.
    """

    code: str
    output_dir: str
    tool_call_count: int
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class WorldGenerationPipeline:
    """Orchestrator for the unified agentic world generation process.

    All configuration is injected via an :class:`~vdaworld.config.AppConfig`
    instance; no magic global state is used.

    Args:
        config: A fully-validated application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        self.vlm = VLMClient(
            model_name=config.model_spec.model_name,
            temperature=config.vlm.temperature,
            backend=config.model_spec.backend,
            base_url=config.model_spec.base_url,
        )
        cache_dir = config.task.cache_dir or os.path.join(
            config.task.output_dir, "_cache"
        )
        self.world_api = WorldAPI(output_dir=None, cache_dir=cache_dir)
        self._available_tools: str = self._collect_api_tool_names()

        # Video vs image mode. In video mode the input is an mp4 the simulation
        # must *continue*; the single still that the simulation's first frame
        # corresponds to is the input video's LAST frame, which we extract to
        # ``reference_image_path`` for all downstream image-conditioning steps.
        self._is_video: bool = config.task.video_path is not None
        self._input_video_path: str | None = config.task.video_path
        if self._is_video:
            if config.task.video_path is None:
                raise ValueError("Video mode requires task.video_path")
            input_frames = np.asarray(iio.imread(config.task.video_path))
            self._n_input_frames: int | None = int(input_frames.shape[0])
            self._input_last_frame: np.ndarray | None = np.asarray(input_frames[-1])
            self._reference_image_path: str = os.path.join(
                config.task.output_dir, "visualizations", "input_last_frame.png"
            )
        else:
            if config.task.image_path is None:
                raise ValueError("Image mode requires task.image_path")
            self._n_input_frames = None
            self._input_last_frame = None
            self._reference_image_path = config.task.image_path

        self.agentic_generator = AgenticGenerator(
            vlm=self.vlm,
            prompts_path=config.task.prompts_path,
            max_turns=config.task.critic_max_turns,
            n_frames=config.task.simulation_steps,
            input_image_path=self._reference_image_path,
            fps=config.task.fps,
            frame_size=config.task.frame_size,
            simulator_class_name=config.task.simulator_class_name,
            output_dir=config.task.output_dir,
            cache_dir=cache_dir,
            available_tools=self._available_tools,
            caption=config.task.caption,
            provide_image=config.task.provide_image,
            restricted_tools=config.task.restricted_tools,
            no_api=config.task.no_api,
            no_mhi=config.task.no_mhi,
            input_video_path=self._input_video_path,
            n_input_frames=self._n_input_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, load_existing: bool = False) -> RunResult | None:
        """Run the unified agentic generation loop.

        Invokes :class:`~vdaworld.core.agentic_generator.AgenticGenerator` to
        generate and validate simulator code in a single tool-calling session,
        then compiles the result and exports the simulation video.

        Args:
            load_existing: If True, skip the VLM call and load an existing
                ``simulator_gen.py`` from the output directory.

        Returns:
            A :class:`RunResult` on success, or ``None`` if generation or
            final compilation fails.
        """
        output_dir = self.config.task.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._prepare_output_dir(load_existing)

        # Video mode: materialise the input video's last frame as the reference
        # still used by fitting comparisons and downstream conditioning. Done
        # after _prepare_output_dir, which clears the output directory.
        if self._is_video:
            if self._input_last_frame is None:
                raise RuntimeError("Input video did not yield a final frame")
            os.makedirs(os.path.dirname(self._reference_image_path), exist_ok=True)
            Image.fromarray(self._input_last_frame).save(self._reference_image_path)

        self.vlm.set_stage_dir(output_dir)

        t_start = time.perf_counter()
        input_tokens = output_tokens = cached_tokens = 0

        if load_existing:
            code = self._load_existing_code(output_dir)
            if code is None:
                return None
            tool_call_count = 0
        else:
            gen_result: GenerationResult = self.agentic_generator.generate(
                target_image_path=self._reference_image_path,
                stage_dir=output_dir,
            )
            code = gen_result.code
            tool_call_count = gen_result.tool_call_count
            input_tokens = gen_result.input_tokens
            output_tokens = gen_result.output_tokens
            cached_tokens = gen_result.cached_tokens

        self._save_code(code, output_dir)
        self.world_api._set_stage_dir(output_dir)

        try:
            simulator, frame = self._compile_and_render(code, output_dir)
        except Exception:
            logger.error("Final compilation failed after agentic generation.")
            return None

        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        Image.fromarray(frame).save(os.path.join(vis_dir, "frame_0_sim.png"))
        shutil.copy(self._reference_image_path, os.path.join(vis_dir, "frame_0_gt.png"))
        logger.info("Initial frames saved to %s", vis_dir)

        self._export_video(simulator, code, output_dir)
        self._run_diffusion_as_shader(output_dir)

        return RunResult(
            code=code,
            output_dir=output_dir,
            tool_call_count=tool_call_count,
            elapsed_seconds=time.perf_counter() - t_start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _prepare_output_dir(self, load_existing: bool) -> None:
        """Prepare a marked output directory without risking unrelated data."""

        if load_existing:
            return

        output_path = Path(self.config.task.output_dir).expanduser().resolve()
        protected_paths = {
            Path(output_path.anchor),
            Path.home().resolve(),
            Path(__file__).resolve().parents[2],
        }
        if output_path in protected_paths:
            raise RuntimeError(
                f"Refusing to use protected path as output_dir: {output_path}"
            )

        output_dir = str(output_path)
        marker_path = os.path.join(output_dir, _OUTPUT_MARKER)
        entries = os.listdir(output_dir)
        if entries and not os.path.isfile(marker_path):
            raise RuntimeError(
                "Refusing to clear a non-empty directory that was not created "
                f"by VDAWorld: {output_dir}. Choose a new output_dir or, after "
                f"checking its contents, create {marker_path!r}."
            )

        for item in entries:
            if item == _OUTPUT_MARKER:
                continue
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        Path(marker_path).touch()

    # ------------------------------------------------------------------
    # Code helpers
    # ------------------------------------------------------------------

    def _collect_api_tool_names(self) -> str:
        """Return a bullet-point list of public WorldAPI method names."""
        return "\n".join(
            f"- `{name}`"
            for name, _ in inspect.getmembers(
                self.world_api, predicate=inspect.ismethod
            )
            if not name.startswith("_")
        )

    def _load_existing_code(self, stage_dir: str) -> str | None:
        """Load ``simulator_gen.py`` from *stage_dir*, or return ``None``."""
        module_path = os.path.join(stage_dir, "simulator_gen.py")
        if os.path.exists(module_path):
            logger.info("Loading existing simulator from %s.", module_path)
            with open(module_path, encoding="utf-8") as f:
                return f.read()
        logger.error(
            "load_existing requested but no simulator found at %s.", module_path
        )
        return None

    @staticmethod
    def _save_code(code: str, stage_dir: str) -> None:
        """Write *code* to ``simulator_gen.py`` inside *stage_dir*."""
        with open(
            os.path.join(stage_dir, "simulator_gen.py"), "w", encoding="utf-8"
        ) as f:
            f.write(code)

    # ------------------------------------------------------------------
    # Compilation and execution
    # ------------------------------------------------------------------

    def _compile_and_render(
        self, code: str, stage_dir: str
    ) -> tuple[SimulatorBase, np.ndarray]:
        """Write *code* to *stage_dir*, load it as a module, and render the first frame.

        Args:
            code: Python source code for the simulator module.
            stage_dir: The current pipeline stage directory.

        Returns:
            ``(simulator, first_frame)`` on success.

        Raises:
            Any exception raised during import or execution of the module.
        """
        module_path = os.path.join(stage_dir, "simulator_gen.py")
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(code)

        static_errors = _static_check(
            module_path, lint_check=self.config.task.lint_check
        )
        if static_errors:
            raise StaticCheckError(
                f"Static checks failed before execution:\n{static_errors}"
            )

        logger.info("Compiling simulator module from %s...", module_path)
        spec = importlib.util.spec_from_file_location("simulator_gen", module_path)
        if not spec or not spec.loader:
            raise RuntimeError("Failed to create module spec from simulator file.")

        module = importlib.util.module_from_spec(spec)
        setattr(module, "SimulatorBase", SimulatorBase)
        setattr(module, "WorldAPI", WorldAPI)
        sys.modules["simulator_gen"] = module
        spec.loader.exec_module(module)

        sim_class = getattr(module, self.config.task.simulator_class_name)
        logger.info(
            "Module compiled. Instantiating %s...",
            self.config.task.simulator_class_name,
        )

        simulator: SimulatorBase = sim_class(
            api=self.world_api,
            fps=self.config.task.fps,
            frame_size=self.config.task.frame_size,
        )

        if self._is_video:
            if self._input_video_path is None:
                raise RuntimeError("Video mode requires an input video path")
            logger.info("Fitting simulator to input video...")
            fit_input = np.asarray(iio.imread(self._input_video_path))
        else:
            logger.info("Fitting simulator to input image...")
            fit_input = np.array(Image.open(self._reference_image_path).convert("RGB"))
        simulator.fit(fit_input)

        if _uses_mujoco(code):
            logger.info("MuJoCo detected — discarding initialisation frame.")
            next(simulator)

        frame: np.ndarray = simulator.render_frame()
        logger.info("Frame rendered: shape=%s dtype=%s", frame.shape, frame.dtype)
        return simulator, frame

    def _export_video(
        self,
        simulator: SimulatorBase,
        code: str,
        output_dir: str,
    ) -> None:
        """Run the full simulation and save the video.

        Args:
            simulator: A compiled and fitted simulator instance.
            code: The simulator source code (used to detect MuJoCo).
            output_dir: Pipeline output directory where visualizations are written.
        """
        logger.info(
            "Running simulation for %d steps...", self.config.task.simulation_steps
        )
        simulator.reset()
        if _uses_mujoco(code):
            logger.info("MuJoCo detected — discarding initialisation frame on reset.")
            next(simulator)

        frames = simulator.run_simulation(self.config.task.simulation_steps)
        frames_np = np.array(frames)

        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        if len(frames_np) > 0:
            video_path = os.path.join(vis_dir, "simulation.mp4")
            iio.imwrite(
                video_path, frames_np, fps=self.config.task.fps, codec="libx264"
            )
            logger.info("Simulation video saved: %s", video_path)

    def _generate_diffusion_caption(self, vis_dir: str) -> str:
        """Ask the VLM to produce a detailed visual caption of the input image.

        The caption describes colours, textures, materials, lighting, and
        camera angle for the optional video diffusion post-processor.

        The generated caption is saved to ``<vis_dir>/diffusion_caption.txt``.

        Args:
            vis_dir: Visualisations directory (used only for saving the caption).

        Returns:
            The generated visual caption string.
        """
        prompt = (
            f'The following caption describes a video: "{self.config.task.caption}"\n\n'
            "You are looking at the first frame of that video. Write a detailed prompt "
            "for a photorealistic video diffusion model that will recreate this video. "
            "Your prompt must incorporate the motion and events described in the caption "
            "above, and enrich them with precise visual detail observed in the image: "
            "the colours, textures, and materials of every object; the background and "
            "surface the objects rest on; the lighting (direction, quality, shadows); "
            "and the camera angle and framing. "
            "Write a single dense paragraph with no headings or bullet points."
        )

        logger.info("Generating visual caption for diffusion post-processing...")
        caption = self.vlm.generate_reply(
            prompt=prompt,
            image_paths=[self._reference_image_path],
            interaction_name="diffusion_visual_caption",
        )

        caption = (
            caption.rstrip()
            + " The video must follow the style of the reference image exactly."
        )

        caption_path = os.path.join(vis_dir, "diffusion_caption.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(caption)
        logger.info("Visual caption saved to %s", caption_path)

        return caption

    def _run_diffusion_as_shader(self, output_dir: str) -> None:
        """Run Diffusion as Shader on the exported simulation video.

        Configuration is read from ``api_config.yaml`` (repo root) under the
        ``diffusion_as_shader`` key.
        """
        import yaml as _yaml

        _pkg_dir = os.path.dirname(__file__)
        api_config_path = os.path.abspath(
            os.path.join(_pkg_dir, "..", "..", "api_config.yaml")
        )
        if not os.path.exists(api_config_path):
            return

        with open(api_config_path, "r") as _f:
            api_cfg = _yaml.safe_load(_f) or {}

        das_cfg = api_cfg.get("diffusion_as_shader")
        if not das_cfg or not das_cfg.get("enabled", False):
            return

        path_env = das_cfg.get("path_env", "DIFFUSION_AS_SHADER_PATH")
        das_path = os.environ.get(path_env)
        if not das_path:
            logger.warning(
                "Diffusion as Shader skipped: environment variable %s is not set",
                path_env,
            )
            return

        das_path = os.path.abspath(os.path.expanduser(das_path))
        if not os.path.isdir(das_path):
            logger.warning(
                "Diffusion as Shader skipped: directory not found at %s", das_path
            )
            return

        cvd = str(das_cfg.get("cvd", "0"))
        tracker = das_cfg.get("tracker", "cotracker")

        vis_dir = os.path.join(output_dir, "visualizations")
        sim_video = os.path.abspath(os.path.join(vis_dir, "simulation.mp4"))

        if not os.path.exists(sim_video):
            logger.warning(
                "Diffusion as Shader skipped: simulation video not found at %s",
                sim_video,
            )
            return

        # Use visual caption if it exists, otherwise generate it
        caption_path = os.path.join(vis_dir, "diffusion_caption.txt")
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                visual_caption = f.read().strip()
        else:
            visual_caption = self._generate_diffusion_caption(vis_dir)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cvd

        lock_path = os.path.join(das_path, ".das_transfer.lock")

        from filelock import FileLock

        with FileLock(lock_path):
            self._run_diffusion_as_shader_locked(
                das_path=das_path,
                sim_video=sim_video,
                visual_caption=visual_caption,
                tracker=tracker,
                env=env,
            )

    def _run_diffusion_as_shader_locked(
        self,
        das_path: str,
        sim_video: str,
        visual_caption: str,
        tracker: str,
        env: dict,
    ) -> None:
        vis_dir = os.path.dirname(sim_video)
        cvd = env.get("CUDA_VISIBLE_DEVICES", "0")
        output_dir = os.path.join(vis_dir, "diffusion_as_shader")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "result.mp4")

        cmd = [
            "uv",
            "run",
            "--no-sync",
            "python",
            "simple_inference.py",
            "--image",
            os.path.abspath(self._reference_image_path),
            "--signal_video",
            sim_video,
            "--prompt",
            visual_caption,
            "--tracker",
            tracker,
            "--output",
            output_path,
        ]

        logger.info(
            "Running Diffusion as Shader (tracker=%s, CVD=%s): %s",
            tracker,
            cvd,
            " ".join(cmd),
        )
        proc = subprocess.Popen(cmd, cwd=das_path, env=env, start_new_session=True)
        try:
            proc.wait()
        except BaseException:
            logger.warning(
                "Diffusion as Shader interrupted — terminating process group %d",
                proc.pid,
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
        if proc.returncode != 0:
            logger.error(
                "Diffusion as Shader (tracker=%s) exited with code %d",
                tracker,
                proc.returncode,
            )
        else:
            logger.info(
                "Diffusion as Shader (tracker=%s) complete. Output in %s",
                tracker,
                output_dir,
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _static_check(module_path: str, lint_check: bool = False) -> str | None:
    """Run zero-cost static checks on *module_path* before execution.

    Performs:

    1. **Syntax check** — Python's built-in :func:`compile`.
    2. **Pyflakes lint (optional)** — ``ruff check --select F --ignore F821``.

    Returns a human-readable error string on failure, or ``None`` if clean.
    """
    with open(module_path, encoding="utf-8") as f:
        source = f.read()

    try:
        compile(source, module_path, "exec")
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}\n  {exc.text}"

    if lint_check:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--select",
                "F",
                "--ignore",
                "F821",
                "--output-format",
                "concise",
                "--no-cache",
                module_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            output = (result.stdout or result.stderr).strip()
            return f"Lint errors (ruff F-codes):\n{output}"

    return None


def _uses_mujoco(code: str) -> bool:
    """Return True if the source code appears to use MuJoCo."""
    return "mujoco" in code.lower()


def _cleanup_module() -> None:
    """Evict the cached simulator module and run garbage collection."""
    sys.modules.pop("simulator_gen", None)
    gc.collect()
    linecache.clearcache()
