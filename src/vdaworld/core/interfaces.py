"""
Core abstract interfaces and shared data structures for the vdaworld pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of a single :class:`~vdaworld.core.agentic_generator.AgenticGenerator` run.

    Attributes:
        code: The final simulator source code produced by the agentic loop.
        stage_dir: Filesystem path to the agentic generation stage directory.
        tool_call_count: Number of sandbox tool calls made during generation.
        input_tokens: Total input tokens consumed by the VLM.
        output_tokens: Total output tokens produced by the VLM.
        cached_tokens: Total cached tokens used by the VLM.
    """

    code: str
    stage_dir: str
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
