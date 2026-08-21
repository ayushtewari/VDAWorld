"""
Vision-Language Model client supporting Gemini, first-party OpenAI, and VLLM
(OpenAI-compatible local server) backends.

Select the backend via the ``backend`` constructor argument:
  - ``"gemini_api"`` (default): uses the Google GenAI SDK with ``GEMINI_API_KEY``.
  - ``"gemini"``: uses the Google GenAI SDK via Vertex AI.
  - ``"openai"``: uses the OpenAI SDK with ``OPENAI_API_KEY``.
  - ``"vllm"``: uses the OpenAI SDK pointed at a local VLLM server.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from PIL import Image

from vdaworld.api.docs import get_api_documentation

logger = logging.getLogger(__name__)


@dataclass
class AgenticReplyResult:
    """Return value of :meth:`VLMClient.generate_agentic_reply`."""

    text: str
    tool_call_count: int = 0  # total individual tool calls across all turns
    turns_used: int = 0  # number of model turns consumed
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _image_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return (
        f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"
    )


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_openai_reasoning_text(message) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        return ""
    return reasoning if isinstance(reasoning, str) else str(reasoning)


def _callable_to_anthropic_tool(fn) -> dict:
    """Convert a Python callable to an Anthropic tool schema dict."""
    import inspect
    import re
    import types as _types

    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n")[0] if doc else fn.__name__

    _primitive = {int: "integer", float: "number", bool: "boolean", str: "string"}

    def _ann_to_schema(ann) -> dict:
        if isinstance(ann, _types.UnionType):
            non_none = [a for a in ann.__args__ if a is not type(None)]
            return _ann_to_schema(non_none[0]) if non_none else {"type": "string"}
        origin = getattr(ann, "__origin__", None)
        if origin is list:
            item_args = getattr(ann, "__args__", (str,))
            return {"type": "array", "items": _ann_to_schema(item_args[0])}
        return {"type": _primitive.get(ann, "string")}

    param_descs: dict[str, str] = {}
    lines = doc.split("\n")
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(args|arguments)\s*:$", stripped, re.IGNORECASE):
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not line[0].isspace():
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            break
        param_match = re.match(r"^    (\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            rest = param_match.group(2).strip()
            current_desc = [rest] if rest else []
        elif current_param and line.startswith("        "):
            current_desc.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc).strip()

    properties: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        schema = (
            _ann_to_schema(ann)
            if ann is not inspect.Parameter.empty
            else {"type": "string"}
        )
        desc = param_descs.get(name, "")
        if desc:
            schema["description"] = desc
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _callable_to_openai_tool(fn) -> dict:
    """Convert a Python callable to an OpenAI tool schema dict.

    Extracts the function description from the first docstring line, parses
    per-parameter descriptions from the Args/Arguments section, and maps
    Python type annotations to JSON Schema types (including list[T] and X|None).
    """
    import inspect
    import re
    import types as _types

    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or ""
    description = doc.split("\n")[0] if doc else fn.__name__

    _primitive = {int: "integer", float: "number", bool: "boolean", str: "string"}

    def _ann_to_schema(ann) -> dict:
        if isinstance(ann, _types.UnionType):
            non_none = [a for a in ann.__args__ if a is not type(None)]
            return _ann_to_schema(non_none[0]) if non_none else {"type": "string"}
        origin = getattr(ann, "__origin__", None)
        if origin is list:
            item_args = getattr(ann, "__args__", (str,))
            return {"type": "array", "items": _ann_to_schema(item_args[0])}
        return {"type": _primitive.get(ann, "string")}

    # Parse the Args/Arguments section for per-parameter descriptions
    param_descs: dict[str, str] = {}
    lines = doc.split("\n")
    in_args = False
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(args|arguments)\s*:$", stripped, re.IGNORECASE):
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not line[0].isspace():
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            break
        param_match = re.match(r"^    (\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            current_param = param_match.group(1)
            rest = param_match.group(2).strip()
            current_desc = [rest] if rest else []
        elif current_param and line.startswith("        "):
            current_desc.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc).strip()

    properties: dict = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        schema = (
            _ann_to_schema(ann)
            if ann is not inspect.Parameter.empty
            else {"type": "string"}
        )
        desc = param_descs.get(name, "")
        if desc:
            schema["description"] = desc
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------


class VLMClient:
    """Thin wrapper around the Google GenAI, OpenAI, or VLLM chat API.

    All pipeline interactions with the VLM go through this class so that
    model selection, authentication, and tool registration are managed in a
    single place.

    Args:
        model_name: Model identifier (Gemini name or HuggingFace ID).
        temperature: Sampling temperature (0.0 = deterministic).
        backend: ``"gemini_api"`` (default), ``"gemini"``, ``"openai"``,
            ``"anthropic"``, or ``"vllm"``.
        base_url: Base URL for the VLLM OpenAI-compatible server,
            e.g. ``"http://127.0.0.1:8000/v1"``. Required when
            ``backend="vllm"`` and ignored for first-party OpenAI.
        llm_interactions_dir: Directory for logging VLM interactions.
    """

    _SYSTEM_INSTRUCTION = (
        "You are an expert World Building programming agent. Your role is to write "
        "Python physics simulations that faithfully replicate the dynamics seen in a "
        "reference video frame, using a WorldAPI for computer vision and geometry "
        "estimation and a sandboxed tool environment for iterative code development."
        "You are not to anticipate a specific outcome. Your role is to ensure that the scene is set up"
        " with the correct objects and physics, but you should not force a specific physical"
        " outcome which you expect to occur.\n\n"
        "Work iteratively and incrementally: implement one component at a time, verify "
        "it using the available tools, then proceed to the next.\n\n"
        "Robust API usage: after every WorldAPI call inside fit(), check whether the "
        "result is usable before proceeding. An API call may succeed but return an "
        "empty or degenerate result (e.g. an empty array, zero-magnitude values, fewer "
        "items than expected, or None). When this happens, substitute a sensible "
        "geometric prior rather than leaving the simulator uninitialised — your "
        "intuition about the scene is always better than producing nothing. Guard "
        "against degenerate results (e.g. `if result is None or len(result) == 0:`), "
        "but do NOT suppress exceptions from incorrect API usage. Exceptions from "
        "wrong argument types or violated API contracts must propagate — never wrap "
        "an entire API call in a bare `except Exception: pass`."
    )

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
        backend: str = "gemini_api",
        base_url: Optional[str] = None,
        llm_interactions_dir: Optional[str | pathlib.Path] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.backend = backend
        self.base_url = base_url
        self._interaction_count = 0
        self.llm_interactions_dir = (
            pathlib.Path(llm_interactions_dir) if llm_interactions_dir else None
        )
        self._stage_dir: Optional[pathlib.Path] = None
        # SDK response types change frequently and differ between compatible
        # servers. Keep that variability at this adapter boundary.
        self._gemini_client: Any = None
        self._openai_client: Any = None
        self._anthropic_client: Any = None

        if backend == "gemini":
            from google import genai

            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT is required for the Vertex AI backend."
                )
            self._gemini_client = genai.Client(
                vertexai=True,
                project=project,
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        elif backend == "gemini_api":
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required for the Gemini API backend."
                )
            self._gemini_client = genai.Client(
                api_key=api_key,
                vertexai=False,
            )
        elif backend == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI backend.")
            self._openai_client = OpenAI(api_key=api_key)
        elif backend == "anthropic":
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for the Anthropic backend."
                )
            self._anthropic_client = Anthropic(api_key=api_key)
        elif backend == "vllm":
            if not base_url:
                raise ValueError("base_url is required when backend='vllm'")
            from openai import OpenAI

            self._openai_client = OpenAI(base_url=base_url, api_key="EMPTY")
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Choose 'gemini', 'gemini_api', 'openai', 'anthropic', or 'vllm'."
            )

    def set_stage_dir(self, stage_dir: Optional[str | pathlib.Path]) -> None:
        """Set the directory for logging the current pipeline stage's VLM interaction."""
        self._stage_dir = pathlib.Path(stage_dir) if stage_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        use_tools: bool = False,
        interaction_name: Optional[str] = None,
    ) -> str:
        """Send a prompt (and optional images) to the VLM and return the text response."""
        logger.debug(
            "VLMClient.generate_reply: backend=%s model=%s images=%d use_tools=%s",
            self.backend,
            self.model_name,
            len(image_paths) if image_paths else 0,
            use_tools,
        )
        if self.backend in ("gemini", "gemini_api"):
            return self._generate_reply_gemini(
                prompt, image_paths, use_tools, interaction_name
            )
        if self.backend == "anthropic":
            return self._generate_reply_anthropic(prompt, image_paths, interaction_name)
        return self._generate_reply_openai_compatible(
            prompt, image_paths, interaction_name
        )

    def generate_agentic_reply(
        self,
        prompt: str,
        image_paths: Optional[list[str]] = None,
        tool_callables: Optional[list[Callable[..., Any]]] = None,
        max_turns: int = 5,
        interaction_name: Optional[str] = None,
    ) -> AgenticReplyResult:
        """Run a multi-turn agentic loop with the VLM using manual tool dispatch."""
        logger.debug(
            "VLMClient.generate_agentic_reply: backend=%s model=%s images=%d tools=%s max_turns=%d",
            self.backend,
            self.model_name,
            len(image_paths) if image_paths else 0,
            [fn.__name__ for fn in (tool_callables or [])],
            max_turns,
        )
        if self.backend in ("gemini", "gemini_api"):
            return self._generate_agentic_reply_gemini(
                prompt, image_paths, tool_callables, max_turns, interaction_name
            )
        if self.backend == "anthropic":
            return self._generate_agentic_reply_anthropic(
                prompt, image_paths, tool_callables, max_turns, interaction_name
            )
        return self._generate_agentic_reply_openai_compatible(
            prompt, image_paths, tool_callables, max_turns, interaction_name
        )

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _generate_reply_gemini(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        use_tools: bool,
        interaction_name: Optional[str],
    ) -> str:
        from google.genai import types

        contents: list = [prompt]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    contents.append(Image.open(img_path))
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=self._SYSTEM_INSTRUCTION,
            tools=[get_api_documentation] if use_tools else [],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        chat = self._gemini_client.chats.create(model=self.model_name, config=config)
        response = self._gemini_send(chat, contents)

        thinking_texts = []
        final_texts = []

        if (
            response.candidates
            and response.candidates[0].content
            and hasattr(response.candidates[0].content, "parts")
            and response.candidates[0].content.parts
        ):
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    if getattr(p, "thought", False):
                        thinking_texts.append(p.text)
                    else:
                        final_texts.append(p.text)

        final_text = "\n".join(final_texts)

        self._log_interaction(
            prompt=prompt,
            final_text=final_text,
            thinking_texts=thinking_texts,
            image_paths=image_paths,
            extra_state={"use_tools": use_tools},
            interaction_name=interaction_name,
        )

        if not final_text and not thinking_texts:
            logger.warning(
                "VLM response contained no text parts (parts: %s); returning empty string.",
                (
                    [type(p).__name__ for p in response.candidates[0].content.parts]
                    if response.candidates
                    and hasattr(response.candidates[0].content, "parts")
                    else "no candidates"
                ),
            )

        return final_text

    def _generate_agentic_reply_gemini(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        tool_callables: Optional[list[Callable[..., Any]]],
        max_turns: int,
        interaction_name: Optional[str],
    ) -> AgenticReplyResult:
        import time as _time
        from google.genai import types

        tool_callables = tool_callables or []
        tool_map: dict[str, Callable[..., Any]] = {
            fn.__name__: fn for fn in tool_callables
        }

        contents: list = [prompt]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    contents.append(Image.open(img_path))
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=self._SYSTEM_INSTRUCTION,
            # The SDK accepts Python callables here, but its invariant list
            # annotation does not accept a narrower list[Callable].
            tools=cast(Any, tool_callables) if tool_callables else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        chat = self._gemini_client.chats.create(model=self.model_name, config=config)
        _t = _time.perf_counter()
        response = self._gemini_send(chat, contents)
        print(f"[vlm] turn 0 (initial prompt): {_time.perf_counter() - _t:.1f}s")

        all_thinking: list[str] = []
        all_turns: list[str] = []
        tool_call_count = 0
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0

        def _accumulate_usage(r) -> None:
            nonlocal total_input_tokens, total_output_tokens, total_cached_tokens
            if r.usage_metadata:
                total_input_tokens += r.usage_metadata.prompt_token_count or 0
                total_output_tokens += r.usage_metadata.candidates_token_count or 0
                total_cached_tokens += r.usage_metadata.cached_content_token_count or 0

        _accumulate_usage(response)

        for turn_idx in range(max_turns):
            turns_used = turn_idx + 1
            function_calls = []
            text_parts = []
            thinking_parts = []

            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        function_calls.append(part.function_call)
                    elif hasattr(part, "text") and part.text:
                        if getattr(part, "thought", False):
                            thinking_parts.append(part.text)
                        else:
                            text_parts.append(part.text)

            all_thinking.extend(thinking_parts)

            if not function_calls:
                # Model finished — no more tool calls
                all_turns.append(f"=== TURN {turns_used} ===")
                if text_parts:
                    all_turns.append("\n".join(text_parts))
                break

            all_turns.append(f"=== TURN {turns_used} ===")
            if text_parts:
                all_turns.append("\n".join(text_parts))

            response_parts: list = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                fc_id = getattr(fc, "id", None)
                tool_call_count += 1
                logger.info(
                    "Agentic tool call (turn %d): %s(%s)",
                    turns_used,
                    fn_name,
                    ", ".join(f"{k}={v!r}" for k, v in fn_args.items()),
                )
                all_turns.append(f"[tool_call] {fn_name}({fn_args})")

                if fn_name not in tool_map:
                    tool_result = f"[error] Unknown tool: {fn_name}"
                    logger.warning("Unknown tool requested: %s", fn_name)
                else:
                    try:
                        _t_tool = _time.perf_counter()
                        tool_result = tool_map[fn_name](**fn_args)
                        print(
                            f"[vlm] tool {fn_name}: {_time.perf_counter() - _t_tool:.1f}s"
                        )
                    except Exception as exc:
                        tool_result = (
                            f"[error] {fn_name} raised {type(exc).__name__}: {exc}"
                        )
                        logger.warning("Tool %s raised exception: %s", fn_name, exc)

                all_turns.append(f"[tool_result] {fn_name}: {str(tool_result)[:200]}")

                if isinstance(tool_result, Image.Image):
                    if fc_id:
                        response_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=fc_id,
                                    name=fn_name,
                                    response={"output": "Frame image attached below."},
                                )
                            )
                        )
                    else:
                        response_parts.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"output": "Frame image attached below."},
                            )
                        )
                    response_parts.append(tool_result)
                else:
                    if fc_id:
                        response_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=fc_id,
                                    name=fn_name,
                                    response={"output": str(tool_result)},
                                )
                            )
                        )
                    else:
                        response_parts.append(
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"output": str(tool_result)},
                            )
                        )

            # Single budget message at the end of the turn (after all tool calls)
            remaining_turns = max_turns - turns_used
            budget_msg = (
                f"[Turn {turns_used}/{max_turns} complete — "
                f"{remaining_turns} turn(s) remaining.]"
            )
            all_turns.append(f"[budget] {budget_msg}")
            response_parts.append(types.Part.from_text(text=budget_msg))

            _t = _time.perf_counter()
            response = self._gemini_send(chat, response_parts)
            print(
                f"[vlm] turn {turns_used}/{max_turns}"
                f" ({len(function_calls)} tool call(s)): {_time.perf_counter() - _t:.1f}s"
            )
            _accumulate_usage(response)
        else:
            # Budget exhausted — force a final assessment with tools disabled
            logger.warning(
                "Agentic generator hit max_turns=%d without finishing. Requesting forced assessment.",
                max_turns,
            )
            all_turns.append("=== budget exhausted — requesting final assessment ===")
            forced_config = types.GenerateContentConfig(
                temperature=self.temperature,
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE
                    )
                ),
            )
            _t = _time.perf_counter()
            forced_response = self._gemini_send(
                chat,
                types.Part.from_text(
                    text="Turn budget exhausted. You may not call any more tools. "
                    "Provide your final assessment of the simulation's state now."
                ),
                config=forced_config,
            )
            print(f"[vlm] forced final response: {_time.perf_counter() - _t:.1f}s")
            _accumulate_usage(forced_response)
            forced_text_parts = []
            if (
                forced_response.candidates
                and forced_response.candidates[0].content
                and forced_response.candidates[0].content.parts
            ):
                for part in forced_response.candidates[0].content.parts:
                    if (
                        hasattr(part, "text")
                        and part.text
                        and not getattr(part, "thought", False)
                    ):
                        forced_text_parts.append(part.text)
            all_turns.append("\n".join(forced_text_parts))

        print(
            f"[tokens] input={total_input_tokens:,}  cached={total_cached_tokens:,}"
            f"  output={total_output_tokens:,}"
            f"  total={total_input_tokens + total_output_tokens:,}"
            f"  ({turns_used} turn(s), {tool_call_count} tool call(s))"
        )

        final_text = "\n".join(
            t
            for t in all_turns
            if not t.startswith("[tool_")
            and not t.startswith("[budget]")
            and not t.startswith("===")
        )
        result = AgenticReplyResult(
            text=final_text,
            tool_call_count=tool_call_count,
            turns_used=turns_used,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=total_cached_tokens,
        )

        self._log_agentic_interaction(
            prompt=prompt,
            final_text=final_text,
            all_thinking=all_thinking,
            all_turns=all_turns,
            image_paths=image_paths,
            tool_map=tool_map,
            tool_call_count=tool_call_count,
            interaction_name=interaction_name,
        )

        return result

    # ------------------------------------------------------------------
    # Anthropic backend
    # ------------------------------------------------------------------

    def _generate_reply_anthropic(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        interaction_name: Optional[str],
    ) -> str:
        user_content: list = []
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)
        user_content.append({"type": "text", "text": prompt})

        response = self._anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=8192,
            system=self._SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": user_content}],
            temperature=self.temperature,
        )

        thinking_texts = [
            b.thinking
            for b in response.content
            if getattr(b, "type", None) == "thinking"
        ]
        final_text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )

        self._log_interaction(
            prompt=prompt,
            final_text=final_text,
            thinking_texts=thinking_texts,
            image_paths=image_paths,
            extra_state={},
            interaction_name=interaction_name,
        )
        return final_text

    def _generate_agentic_reply_anthropic(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        tool_callables: Optional[list[Callable[..., Any]]],
        max_turns: int,
        interaction_name: Optional[str],
    ) -> AgenticReplyResult:
        import time as _time

        tool_callables = tool_callables or []
        tool_map: dict[str, Callable[..., Any]] = {
            fn.__name__: fn for fn in tool_callables
        }
        anthropic_tools = [
            _callable_to_anthropic_tool(fn) for fn in tool_callables
        ] or None

        user_content: list = []
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)
        user_content.append({"type": "text", "text": prompt})

        messages: list[dict] = [{"role": "user", "content": user_content}]

        all_thinking: list[str] = []
        all_turns: list[str] = []
        tool_call_count = 0
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0

        kwargs = dict(
            model=self.model_name,
            max_tokens=16000,
            system=self._SYSTEM_INSTRUCTION,
            temperature=self.temperature,
        )
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        _t = _time.perf_counter()
        response = self._anthropic_client.messages.create(messages=messages, **kwargs)
        print(f"[vlm] turn 0 (initial prompt): {_time.perf_counter() - _t:.1f}s")

        total_input_tokens += response.usage.input_tokens or 0
        total_output_tokens += response.usage.output_tokens or 0

        for turn_idx in range(max_turns):
            turns_used = turn_idx + 1

            # Collect thinking and text from this response
            turn_thinking = [
                b.thinking
                for b in response.content
                if getattr(b, "type", None) == "thinking"
            ]
            turn_text = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            tool_use_blocks = [
                b for b in response.content if getattr(b, "type", None) == "tool_use"
            ]

            if turn_thinking:
                all_thinking.extend(turn_thinking)

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn" or not tool_use_blocks:
                all_turns.append(f"=== TURN {turns_used} ===")
                all_turns.append(turn_text)
                break

            all_turns.append(f"=== TURN {turns_used} ===")
            if turn_text:
                all_turns.append(turn_text)

            # Dispatch each tool use block
            tool_results: list[dict] = []
            image_tool_results: list[tuple[str, Image.Image]] = []

            for block in tool_use_blocks:
                fn_name = block.name
                fn_args = block.input or {}

                # Coerce primitive types
                if fn_name in tool_map:
                    import typing as _typing

                    _PRIMITIVE = (int, float, bool, str)
                    try:
                        hints = _typing.get_type_hints(tool_map[fn_name])
                    except Exception:
                        hints = {}
                    for pname, ann in hints.items():
                        if pname in fn_args and ann in _PRIMITIVE:
                            try:
                                fn_args[pname] = ann(fn_args[pname])
                            except (ValueError, TypeError):
                                pass

                tool_call_count += 1
                logger.info(
                    "Agentic tool call (turn %d): %s(%s)",
                    turns_used,
                    fn_name,
                    ", ".join(f"{k}={v!r}" for k, v in fn_args.items()),
                )
                all_turns.append(f"[tool_call] {fn_name}({fn_args})")

                if fn_name not in tool_map:
                    tool_result = f"[error] Unknown tool: {fn_name}"
                    logger.warning("Unknown tool requested: %s", fn_name)
                else:
                    try:
                        _t_tool = _time.perf_counter()
                        tool_result = tool_map[fn_name](**fn_args)
                        print(
                            f"[vlm] tool {fn_name}: {_time.perf_counter() - _t_tool:.1f}s"
                        )
                    except Exception as exc:
                        tool_result = (
                            f"[error] {fn_name} raised {type(exc).__name__}: {exc}"
                        )
                        logger.warning("Tool %s raised exception: %s", fn_name, exc)

                all_turns.append(f"[tool_result] {fn_name}: {str(tool_result)[:200]}")

                if isinstance(tool_result, Image.Image):
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Frame rendered. See image attachment below.",
                        }
                    )
                    image_tool_results.append((fn_name, tool_result))
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(tool_result),
                        }
                    )

            # Build user message with tool results and any returned images
            user_reply: list = tool_results
            if image_tool_results:
                user_reply = list(tool_results)  # copy
                user_reply.append({"type": "text", "text": "Images returned by tools:"})
                for fn_name, image in image_tool_results:
                    user_reply.append({"type": "text", "text": f"{fn_name}:"})
                    user_reply.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _pil_to_b64(image),
                            },
                        }
                    )

            remaining_turns = max_turns - turns_used
            budget_msg = (
                f"[Turn {turns_used}/{max_turns} complete — "
                f"{remaining_turns} turn(s) remaining.]"
            )
            all_turns.append(f"[budget] {budget_msg}")
            user_reply.append({"type": "text", "text": budget_msg})
            messages.append({"role": "user", "content": user_reply})

            _t = _time.perf_counter()
            response = self._anthropic_client.messages.create(
                messages=messages, **kwargs
            )
            print(
                f"[vlm] turn {turns_used}/{max_turns}"
                f" ({len(tool_use_blocks)} tool call(s)): {_time.perf_counter() - _t:.1f}s"
            )
            total_input_tokens += response.usage.input_tokens or 0
            total_output_tokens += response.usage.output_tokens or 0
        else:
            logger.warning(
                "Agentic generator hit max_turns=%d without finishing. Extracting partial text.",
                max_turns,
            )
            all_turns.append(f"=== TURN {max_turns} (incomplete) ===")
            final_partial = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            )
            if final_partial:
                all_turns.append(final_partial)

        print(
            f"[tokens] input={total_input_tokens:,}  output={total_output_tokens:,}"
            f"  total={total_input_tokens + total_output_tokens:,}"
            f"  ({turns_used} turn(s), {tool_call_count} tool call(s))"
        )

        final_text = "\n".join(
            t
            for t in all_turns
            if not t.startswith("[tool_")
            and not t.startswith("[budget]")
            and not t.startswith("===")
        )
        result = AgenticReplyResult(
            text=final_text,
            tool_call_count=tool_call_count,
            turns_used=turns_used,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=0,
        )

        self._log_agentic_interaction(
            prompt=prompt,
            final_text=final_text,
            all_thinking=all_thinking,
            all_turns=all_turns,
            image_paths=image_paths,
            tool_map=tool_map,
            tool_call_count=tool_call_count,
            interaction_name=interaction_name,
        )

        return result

    # OpenAI Chat Completions-compatible backends
    # ------------------------------------------------------------------

    def _generate_reply_openai_compatible(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        interaction_name: Optional[str],
    ) -> str:
        user_content: list = [{"type": "text", "text": prompt}]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        messages = [
            {"role": "system", "content": self._SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]

        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )

        message = response.choices[0].message
        reasoning_text = _extract_openai_reasoning_text(message)
        final_text = message.content or ""

        self._log_interaction(
            prompt=prompt,
            final_text=final_text,
            thinking_texts=[reasoning_text] if reasoning_text else [],
            image_paths=image_paths,
            extra_state={},
            interaction_name=interaction_name,
        )

        return final_text

    def _generate_agentic_reply_openai_compatible(
        self,
        prompt: str,
        image_paths: Optional[list[str]],
        tool_callables: Optional[list[Callable[..., Any]]],
        max_turns: int,
        interaction_name: Optional[str],
    ) -> AgenticReplyResult:
        import time as _time

        tool_callables = tool_callables or []
        tool_map: dict[str, Callable[..., Any]] = {
            fn.__name__: fn for fn in tool_callables
        }
        openai_tools = (
            [_callable_to_openai_tool(fn) for fn in tool_callables]
            if tool_callables
            else None
        )

        # Build the initial user message
        user_content: list = [{"type": "text", "text": prompt}]
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    b64 = _encode_image_b64(img_path)
                    mime = _image_mime(img_path)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                else:
                    logger.warning("Image path does not exist, skipping: %s", img_path)

        messages: list[dict] = [
            {"role": "system", "content": self._SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]

        all_thinking: list[str] = []
        all_turns: list[str] = []
        tool_call_count = 0
        turns_used = 0
        total_input_tokens = 0
        total_output_tokens = 0

        _t = _time.perf_counter()
        response = self._openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=openai_tools,
            temperature=self.temperature,
        )
        print(f"[vlm] turn 0 (initial prompt): {_time.perf_counter() - _t:.1f}s")

        if response.usage:
            total_input_tokens += response.usage.prompt_tokens or 0
            total_output_tokens += response.usage.completion_tokens or 0

        for turn_idx in range(max_turns):
            turns_used = turn_idx + 1
            msg = response.choices[0].message
            reasoning_text = _extract_openai_reasoning_text(msg)
            if reasoning_text:
                all_thinking.append(reasoning_text)

            # Append assistant message to history
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )

            if not msg.tool_calls:
                all_turns.append(f"=== TURN {turns_used} ===")
                all_turns.append(msg.content or "")
                break

            all_turns.append(f"=== TURN {turns_used} ===")
            if msg.content:
                all_turns.append(msg.content)

            image_tool_results: list[tuple[str, Image.Image]] = []

            # Dispatch each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Coerce args to match parameter annotations (e.g. "0" → 0 for int
                # params) so the model doesn't need to retry on type mismatches.
                # get_type_hints() is used instead of param.annotation because
                # `from __future__ import annotations` makes annotations lazy strings.
                if fn_name in tool_map:
                    import typing as _typing

                    _PRIMITIVE = (int, float, bool, str)
                    try:
                        hints = _typing.get_type_hints(tool_map[fn_name])
                    except Exception:
                        hints = {}
                    for pname, ann in hints.items():
                        if pname in fn_args and ann in _PRIMITIVE:
                            try:
                                fn_args[pname] = ann(fn_args[pname])
                            except (ValueError, TypeError):
                                pass

                tool_call_count += 1
                logger.info(
                    "Agentic tool call (turn %d): %s(%s)",
                    turns_used,
                    fn_name,
                    ", ".join(f"{k}={v!r}" for k, v in fn_args.items()),
                )
                all_turns.append(f"[tool_call] {fn_name}({fn_args})")

                if fn_name not in tool_map:
                    tool_result = f"[error] Unknown tool: {fn_name}"
                    logger.warning("Unknown tool requested: %s", fn_name)
                else:
                    try:
                        _t_tool = _time.perf_counter()
                        tool_result = tool_map[fn_name](**fn_args)
                        print(
                            f"[vlm] tool {fn_name}: {_time.perf_counter() - _t_tool:.1f}s"
                        )
                    except Exception as exc:
                        tool_result = (
                            f"[error] {fn_name} raised {type(exc).__name__}: {exc}"
                        )
                        logger.warning("Tool %s raised exception: %s", fn_name, exc)

                all_turns.append(f"[tool_result] {fn_name}: {str(tool_result)[:200]}")

                if isinstance(tool_result, Image.Image):
                    # Chat Completions tool-role messages are text-only in
                    # practice for our target backends. Satisfy every tool call
                    # first, then add returned images as a user message below.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Frame rendered. See image attachment below.",
                        }
                    )
                    image_tool_results.append((fn_name, tool_result))
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result),
                        }
                    )

            if image_tool_results:
                content: list[dict[str, Any]] = [
                    {"type": "text", "text": "Images returned by tools:"}
                ]
                for fn_name, image in image_tool_results:
                    content.append({"type": "text", "text": f"{fn_name}:"})
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_pil_to_b64(image)}"
                            },
                        }
                    )
                messages.append({"role": "user", "content": content})

            # Single budget message at the end of the turn (after all tool calls)
            remaining_turns = max_turns - turns_used
            budget_msg = (
                f"[Turn {turns_used}/{max_turns} complete — "
                f"{remaining_turns} turn(s) remaining.]"
            )
            all_turns.append(f"[budget] {budget_msg}")
            messages.append({"role": "user", "content": budget_msg})

            _t = _time.perf_counter()
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=openai_tools,
                temperature=self.temperature,
            )
            print(
                f"[vlm] turn {turns_used}/{max_turns}"
                f" ({len(msg.tool_calls)} tool call(s)): {_time.perf_counter() - _t:.1f}s"
            )

            if response.usage:
                total_input_tokens += response.usage.prompt_tokens or 0
                total_output_tokens += response.usage.completion_tokens or 0
        else:
            logger.warning(
                "Agentic generator hit max_turns=%d without finishing. Extracting partial text.",
                max_turns,
            )
            all_turns.append(f"=== TURN {max_turns} (incomplete) ===")
            final_msg = response.choices[0].message
            if final_msg.content:
                all_turns.append(final_msg.content)

        print(
            f"[tokens] input={total_input_tokens:,}  output={total_output_tokens:,}"
            f"  total={total_input_tokens + total_output_tokens:,}"
            f"  ({turns_used} turn(s), {tool_call_count} tool call(s))"
        )

        final_text = "\n".join(
            t
            for t in all_turns
            if not t.startswith("[tool_")
            and not t.startswith("[budget]")
            and not t.startswith("===")
        )
        result = AgenticReplyResult(
            text=final_text,
            tool_call_count=tool_call_count,
            turns_used=turns_used,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cached_tokens=0,
        )

        self._log_agentic_interaction(
            prompt=prompt,
            final_text=final_text,
            all_thinking=all_thinking,
            all_turns=all_turns,
            image_paths=image_paths,
            tool_map=tool_map,
            tool_call_count=tool_call_count,
            interaction_name=interaction_name,
        )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gemini_send(self, chat, msg, config=None):
        """Wrap chat.send_message with automatic retry on 429 RESOURCE_EXHAUSTED."""
        import time as _time

        for attempt in range(5):
            try:
                return chat.send_message(msg, config=config)
            except Exception as exc:
                is_429 = False
                try:
                    from google.genai.errors import ClientError

                    if isinstance(exc, ClientError):
                        is_429 = getattr(exc, "code", None) == 429 or "429" in str(exc)
                except ImportError:
                    pass
                if is_429 and attempt < 9:
                    print(
                        f"[vlm] Google API quota exhausted (429 RESOURCE_EXHAUSTED) "
                        f"— attempt {attempt + 1}/10. Waiting 2 minutes before retrying..."
                    )
                    _time.sleep(120)
                    continue
                raise

    # ------------------------------------------------------------------
    # Shared logging helpers
    # ------------------------------------------------------------------

    def _resolve_out_dir(
        self, interaction_name: Optional[str]
    ) -> Optional[pathlib.Path]:
        if self._stage_dir:
            out_dir = self._stage_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir
        if self.llm_interactions_dir:
            name = interaction_name or "interaction"
            out_dir = self.llm_interactions_dir / f"{self._interaction_count}_{name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._interaction_count += 1
            return out_dir
        return None

    def _log_interaction(
        self,
        prompt: str,
        final_text: str,
        thinking_texts: list[str],
        image_paths: Optional[list[str]],
        extra_state: dict,
        interaction_name: Optional[str],
    ) -> None:
        out_dir = self._resolve_out_dir(interaction_name)
        if out_dir is None:
            return

        with open(out_dir / "prompt.md", "w") as f:
            f.write(prompt)

        state = {
            "model_name": self.model_name,
            "backend": self.backend,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "interaction_name": interaction_name,
            "image_paths": [
                os.path.abspath(p) for p in (image_paths or []) if os.path.exists(p)
            ],
            **extra_state,
        }
        with open(out_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        if thinking_texts:
            with open(out_dir / "thinking.md", "w") as f:
                f.write("\n\n".join(thinking_texts))

        with open(out_dir / "response.md", "w") as f:
            f.write(final_text)

    def _log_agentic_interaction(
        self,
        prompt: str,
        final_text: str,
        all_thinking: list[str],
        all_turns: list[str],
        image_paths: Optional[list[str]],
        tool_map: dict,
        tool_call_count: int,
        interaction_name: Optional[str],
    ) -> None:
        out_dir = self._resolve_out_dir(interaction_name)
        if out_dir is None:
            return

        with open(out_dir / "prompt.md", "w") as f:
            f.write(prompt)

        state = {
            "model_name": self.model_name,
            "backend": self.backend,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "interaction_name": interaction_name,
            "image_paths": [
                os.path.abspath(p) for p in (image_paths or []) if os.path.exists(p)
            ],
            "tools": list(tool_map.keys()),
            "tool_call_count": tool_call_count,
        }
        with open(out_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        if all_thinking:
            with open(out_dir / "thinking.md", "w") as f:
                f.write("\n\n".join(all_thinking))

        with open(out_dir / "turns.md", "w") as f:
            f.write("\n\n".join(all_turns))

        with open(out_dir / "response.md", "w") as f:
            f.write(final_text)
