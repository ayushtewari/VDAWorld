from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vdaworld.api.geometry import DummyGeometryAPI
from vdaworld.api.physics import DummyPhysicsEngine
from vdaworld.api.vlm import VLMClient


def test_geometry_api():
    geom = DummyGeometryAPI()
    assert geom.extract_point_cloud("img") == "mock_point_cloud"
    assert geom.fit_primitive("pc") == {
        "shape": "box",
        "size": [1.0, 1.0, 1.0],
    }


def test_physics_api():
    phys = DummyPhysicsEngine()
    phys.step(dt=0.1)
    assert phys is not None


@patch.dict(
    "os.environ",
    {
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "test-location",
    },
    clear=True,
)
@patch("google.genai.Client")
def test_vlm_client_reads_vertex_configuration_from_environment(mock_client):
    client = VLMClient(
        model_name="test-model",
        temperature=0.0,
        backend="gemini",
    )

    assert client.backend == "gemini"
    mock_client.assert_called_once_with(
        vertexai=True,
        project="test-project",
        location="test-location",
    )


@patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True)
@patch("google.genai.Client")
def test_vlm_client_initializes_gemini_api_backend(mock_client):
    client = VLMClient(model_name="test-model", backend="gemini_api")

    assert client.backend == "gemini_api"
    mock_client.assert_called_once_with(api_key="test-key", vertexai=False)


def test_vlm_client_requires_vllm_base_url():
    try:
        VLMClient(model_name="test-model", backend="vllm")
    except ValueError as exc:
        assert "base_url is required" in str(exc)
    else:
        raise AssertionError("VLMClient should require base_url for vllm backend")


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
@patch("openai.OpenAI")
def test_vlm_client_initializes_openai_backend(mock_openai):
    client = VLMClient(model_name="gpt-test", backend="openai")

    assert client.model_name == "gpt-test"
    assert client.backend == "openai"
    mock_openai.assert_called_once_with(api_key="test-key")


@patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
@patch("openai.OpenAI")
def test_openai_agentic_reply_dispatches_tool_calls(mock_openai):
    def echo(value: str) -> str:
        """Echo a value.

        Args:
            value: Value to echo.
        """
        return f"echo:{value}"

    first_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="echo", arguments='{"value": "abc"}'
                            ),
                        )
                    ],
                )
            )
        ],
    )
    second_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4),
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None))
        ],
    )

    create = MagicMock(side_effect=[first_response, second_response])
    mock_openai.return_value.chat.completions.create = create

    client = VLMClient(model_name="gpt-test", backend="openai")
    result = client.generate_agentic_reply(
        prompt="Use tools",
        tool_callables=[echo],
        max_turns=3,
    )

    assert result.text == "done"
    assert result.tool_call_count == 1
    assert result.input_tokens == 30
    assert result.output_tokens == 6

    first_request = create.call_args_list[0].kwargs
    assert first_request["model"] == "gpt-test"
    assert first_request["tools"][0]["function"]["name"] == "echo"

    second_messages = create.call_args_list[1].kwargs["messages"]
    assert any(
        message.get("role") == "tool"
        and message.get("tool_call_id") == "call_1"
        and message.get("content") == "echo:abc"
        for message in second_messages
    )
