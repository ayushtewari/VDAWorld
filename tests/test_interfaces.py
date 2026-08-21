from vdaworld.core.interfaces import GenerationResult


def test_generation_result_initialization():
    result = GenerationResult(
        code="class VideoSimulation: pass",
        stage_dir="results/test",
        tool_call_count=3,
        input_tokens=100,
        output_tokens=50,
        cached_tokens=25,
    )
    assert result.code == "class VideoSimulation: pass"
    assert result.stage_dir == "results/test"
    assert result.tool_call_count == 3
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cached_tokens == 25
