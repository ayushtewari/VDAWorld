"""
Unified agentic generation loop.

:class:`AgenticGenerator` drives a single tool-calling session that writes,
compiles, and validates simulator code without external critic stages.  It
replaces the previous ``SceneCritic``, ``DynamicsCritic``, and ``ErrorCritic``
classes.
"""

from __future__ import annotations

import logging
import os

from vdaworld.api.llm.methods.prompt import get_prompt
from vdaworld.core.critic_toolbox import CriticSandbox
from vdaworld.core.interfaces import GenerationResult

logger = logging.getLogger(__name__)

# Minimal placeholder written to the sandbox at construction time so the file
# is never empty.  The agent's first action will be write_code_from_scratch.
_STUB_CODE = "# Placeholder — agent will call write_code_from_scratch\n"

# Per-tool prompt descriptions.  Keyed by the callable's __name__ so the
# prompt always lists exactly the tools that are actually passed to the VLM.
_TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_api_documentation": (
        "- **`get_api_documentation(method_names)`** — Retrieve the full signature and"
        " docstring for one or more WorldAPI methods. Call this before writing any code"
        " that uses the API."
    ),
    "write_code_from_scratch": (
        "- **`write_code_from_scratch(code)`** — Write a complete Python source file."
        " Use this for your initial implementation and for any full restart when the"
        " current code is too broken to patch incrementally."
    ),
    "read_code": (
        "- **`read_code()`** — Return the full current source of the sandbox file. Use"
        " this before `edit_code` when you are unsure of the exact whitespace or"
        " indentation of the block you want to replace."
    ),
    "run_simulation": (
        "- **`run_simulation()`** — Execute the current code and render frames. Returns"
        " combined stdout and stderr, including any `print()` statements in the code."
    ),
    "read_terminal": (
        "- **`read_terminal()`** — Re-read the stdout/stderr from the last"
        " `run_simulation()` without re-executing. Useful for inspecting debug output."
    ),
    "edit_code": (
        "- **`edit_code(search, replace)`** — Apply a targeted SEARCH/REPLACE patch to"
        " the current code. Use this for incremental fixes and tuning. The `search`"
        " block must match exactly (including whitespace and indentation). Call"
        " `read_code()` first if you are not certain of the exact text."
    ),
    "view_rendered_frame": (
        "- **`view_rendered_frame(frame_index)`** — Inspect a specific rendered frame"
        " visually. Frame indices start at 0."
    ),
    "view_debug": (
        "- **`view_debug()`** — Return the last image the simulator passed to"
        " `self.log_debug_view(image)`. Use this to expose auxiliary detail that is"
        " hard to see in normal output frames — for example, a component-only render,"
        " an annotated overlay of detected objects or fitted geometry, or an"
        " intermediate state that needs refinement. Only the last image logged is"
        " retained; call this after `run_simulation()`."
    ),
    "view_motion_history_image": (
        "- **`view_motion_history_image()`** — Compute and return a Motion History"
        " Image (MHI) from the last simulation run. Colour encodes _when_ motion"
        " occurred (blue = early, purple = mid, red = late); saturation encodes _how"
        " often_ a pixel changed (white = frequent, saturated = rare). Use this to"
        " verify dynamics are physically correct."
    ),
    "view_input_frame": (
        "- **`view_input_frame(frame_index)`** — View a specific frame of the"
        " **input video** you must continue. Frame indices start at 0; the last"
        " frame (index N-1) is the one your simulation's first rendered frame must"
        " match. Use this to inspect the input before and while writing your code."
    ),
    "view_input_mhi": (
        "- **`view_input_mhi()`** — Compute and return a Motion History Image (MHI)"
        " of the **input video**, using the same colour/saturation encoding as"
        " `view_motion_history_image()`. Use this to understand the input motion"
        " before designing the continuation."
    ),
    "read_tool_call_metadata": (
        "- **`read_tool_call_metadata(tool_call_index=None)`** — Return metadata from"
        " previous tool calls made during this session. With no argument, lists all"
        " calls so far. Pass a zero-based index to inspect a specific call."
    ),
    "read_world_api_call_metadata": (
        "- **`read_world_api_call_metadata(call_index=None)`** — Inspect what the"
        " WorldAPI returned during the last `run_simulation()`. With no argument, lists"
        ' all API calls made (e.g. `0: "segment"`, `1: "intrinsics"`) with warnings'
        " for unusual results such as errors, poor primitive fits, or empty outputs."
        " Pass a zero-based index to read the full metadata for that call — mask"
        " counts, 3D coordinates, fitted primitive parameters, etc."
    ),
}


# Mode-dependent prompt text, substituted into the generic prompt parts via
# get_prompt kwargs ([INPUT_DESCRIPTION] in part_task.md, [FIT_INPUT_DESC] in
# spec_simulator.md).
_IMAGE_INPUT_DESCRIPTION = (
    'This image is the first frame of a video which has the following caption: "{caption}".'
    " Your task is to design a simulator which faithfully simulates the rest of the"
    " video after this frame. The simulator should match its starting state,"
    " including the camera viewpoint, to the frame of the video provided, and"
    " subsequently be iterated over to produce the simulation. Any required content"
    " which is not visible in that frame or which cannot successfully be derived or"
    " calculated directly from the frame must be inferred using heuristics."
)
_IMAGE_FIT_INPUT_DESC = (
    "Image showing physical process to be simulated. Shape [H, W, 3]."
)

_VIDEO_INPUT_DESCRIPTION = (
    'You are given an input video which has the following caption: "{caption}".'
    " The input video has {n_input_frames} frames at {fps} fps. Your task is to"
    " design a simulator which **continues** this video: the first frame your"
    " simulator renders must match the **last frame of the input video**, so that"
    " when your simulation is concatenated onto the end of the input video they form"
    " a single, continuous video.\n\n"
    "You are not shown the video directly. Instead, inspect it with the"
    " `view_input_frame(i)` tool (for i in 0..{last_index}, where frame"
    " {last_index} is the last frame your simulation must continue from) and the"
    " `view_input_mhi()` tool (to understand its motion). Match your simulator's"
    " starting state, including the camera viewpoint, to the last input frame"
    " (`view_input_frame({last_index})`). Any required content which is not visible"
    " or cannot be derived directly from the input video must be inferred using"
    " heuristics."
)
_VIDEO_FIT_INPUT_DESC = (
    "Input video to be continued, of shape [N, H, W, 3] where N is the number of"
    " input frames. Fit your simulator's initial state to the LAST frame,"
    " image[-1] — your first rendered frame must match it."
)


class AgenticGenerator:
    """Runs a single agentic tool-calling loop that generates, tests, and
    validates simulator code in one continuous session.

    Args:
        vlm: A :class:`~vdaworld.api.vlm.VLMClient` instance.
        prompts_path: Directory containing the prompt ``.md`` files.
        max_turns: Hard cap on the number of model turns (each turn may contain
            unlimited tool calls).
        n_frames: Number of frames rendered per ``run_simulation()`` call.
        input_image_path: Path to the target/reference image.
        fps: Frames per second passed to the simulator.
        frame_size: ``(width, height)`` tuple for the simulator.
        simulator_class_name: Name of the class to instantiate from the
            generated module.
        output_dir: Root output directory (used for stage-level logging).
        cache_dir: Optional WorldAPI result cache directory.
        available_tools: Bullet-point string of available WorldAPI method
            names, injected into the prompt.
        caption: Natural-language task description, injected into the prompt.
        input_video_path: Optional path to an input video (mp4). When set, the
            generator runs in *video mode*: ``fit`` receives the full video, the
            ``view_input_frame``/``view_input_mhi`` tools are exposed, the prompt
            instructs the model to continue from the last input frame, and no
            static image is attached to the conversation.
        n_input_frames: Number of frames in the input video (video mode only),
            injected into the prompt so the model knows the valid frame indices.
    """

    def __init__(
        self,
        vlm,
        prompts_path: str,
        max_turns: int,
        n_frames: int,
        input_image_path: str,
        fps: int,
        frame_size: tuple[int, int],
        simulator_class_name: str,
        output_dir: str,
        cache_dir: str | None,
        available_tools: str,
        caption: str,
        provide_image: bool = True,
        restricted_tools: bool = False,
        no_api: bool = False,
        no_mhi: bool = False,
        input_video_path: str | None = None,
        n_input_frames: int | None = None,
    ) -> None:
        self._vlm = vlm
        self._prompts_path = prompts_path
        self._max_turns = max_turns
        self._n_frames = n_frames
        self._input_image_path = input_image_path
        self._fps = fps
        self._frame_size = frame_size
        self._simulator_class_name = simulator_class_name
        self._output_dir = output_dir
        self._cache_dir = cache_dir
        self._available_tools = available_tools
        self._caption = caption
        self._provide_image = provide_image
        self._restricted_tools = restricted_tools
        self._no_api = no_api
        self._no_mhi = no_mhi
        # Video mode: when input_video_path is set, fit() receives the full
        # video and the view_input_frame / view_input_mhi tools are exposed.
        self._input_video_path = input_video_path
        self._n_input_frames = n_input_frames

    def generate(self, target_image_path: str, stage_dir: str) -> GenerationResult:
        """Run the agentic generation loop and return the final simulator code.

        Creates a :class:`~vdaworld.core.critic_toolbox.CriticSandbox`, builds
        the prompt, and calls
        :meth:`~vdaworld.api.vlm.VLMClient.generate_agentic_reply` with all
        available sandbox tools.  After the loop ends, the contents of the
        sandbox file are returned as the final simulator code.

        Args:
            target_image_path: Path to the ground-truth reference image sent
                to the VLM as the first image in the conversation.
            stage_dir: Pipeline stage directory.  VLM interaction files and
                sandbox tool-call logs are written here.

        Returns:
            A :class:`~vdaworld.core.interfaces.GenerationResult` with the
            final code and metadata.
        """
        tool_calls_log_dir = os.path.join(stage_dir, "tool_calls")
        sandbox_dir = os.path.join(stage_dir, "sandbox")
        world_api_log_dir = os.path.join(sandbox_dir, "api_calls")
        os.makedirs(tool_calls_log_dir, exist_ok=True)

        is_video = self._input_video_path is not None

        sandbox = CriticSandbox(
            code=_STUB_CODE,
            fps=self._fps,
            n_frames=self._n_frames,
            frame_size=self._frame_size,
            input_image_path=self._input_image_path,
            simulator_class_name=self._simulator_class_name,
            sandbox_dir=sandbox_dir,
            tool_calls_log_dir=tool_calls_log_dir,
            cache_dir=self._cache_dir,
            world_api_log_dir=world_api_log_dir,
            no_api=self._no_api,
            input_video_path=self._input_video_path,
        )

        logger.info(
            "AgenticGenerator: starting agentic loop (max_turns=%d)...", self._max_turns
        )
        try:
            # Build tool callables list, optionally restricting to only the two core tools.
            if self._restricted_tools:
                tool_callables = [
                    sandbox.get_api_documentation,
                    sandbox.write_code_from_scratch,
                ]
            else:
                tool_callables = [
                    sandbox.get_api_documentation,
                    sandbox.write_code_from_scratch,
                    sandbox.read_code,
                    sandbox.run_simulation,
                    sandbox.read_terminal,
                    sandbox.edit_code,
                    sandbox.view_rendered_frame,
                    sandbox.view_debug,
                    sandbox.view_motion_history_image,
                    sandbox.read_tool_call_metadata,
                    sandbox.read_world_api_call_metadata,
                ]
                # Video mode: expose tools to inspect the input video. Withheld
                # in image mode so the model isn't offered tools it can't use.
                if is_video:
                    tool_callables.append(sandbox.view_input_frame)
                    tool_callables.append(sandbox.view_input_mhi)

            if self._no_api:
                tool_callables = [
                    t for t in tool_callables if t.__name__ != "get_api_documentation"
                ]

            # Remove motion history image tools if ablation is enabled (both the
            # simulation MHI and, in video mode, the input-video MHI).
            if self._no_mhi:
                tool_callables = [
                    t
                    for t in tool_callables
                    if t.__name__ not in ("view_motion_history_image", "view_input_mhi")
                ]

            sandbox_tools = "\n".join(
                _TOOL_DESCRIPTIONS[fn.__name__]
                for fn in tool_callables
                if fn.__name__ in _TOOL_DESCRIPTIONS
            )
            if is_video:
                last_index = self._n_input_frames - 1 if self._n_input_frames else "N-1"
                input_description = _VIDEO_INPUT_DESCRIPTION.format(
                    caption=self._caption,
                    n_input_frames=self._n_input_frames,
                    fps=self._fps,
                    last_index=last_index,
                )
                fit_input_desc = _VIDEO_FIT_INPUT_DESC
            else:
                input_description = _IMAGE_INPUT_DESCRIPTION.format(
                    caption=self._caption
                )
                fit_input_desc = _IMAGE_FIT_INPUT_DESC

            prompt = get_prompt(
                self._prompts_path,
                "agentic_generation",
                available_tools=self._available_tools,
                caption=self._caption,
                simulator_class_name=self._simulator_class_name,
                max_turns=self._max_turns,
                sandbox_tools=sandbox_tools,
                input_description=input_description,
                fit_input_desc=fit_input_desc,
            )

            # Video mode delivers the input via tools, so no static image is
            # attached to the conversation.
            attach_image = self._provide_image and not is_video
            vlm_result = self._vlm.generate_agentic_reply(
                prompt=prompt,
                image_paths=[target_image_path] if attach_image else None,
                tool_callables=tool_callables,
                max_turns=self._max_turns,
                interaction_name="agentic_generation",
            )
            with open(sandbox._sandbox_path, encoding="utf-8") as f:
                final_code = f.read()
        finally:
            sandbox.cleanup()

        tool_call_count = sandbox._tool_call_idx
        logger.info(
            "AgenticGenerator: completed after %d tool call(s) across %d turn(s).",
            tool_call_count,
            vlm_result.turns_used,
        )

        return GenerationResult(
            code=final_code,
            stage_dir=stage_dir,
            tool_call_count=tool_call_count,
            input_tokens=vlm_result.input_tokens,
            output_tokens=vlm_result.output_tokens,
            cached_tokens=vlm_result.cached_tokens,
        )
