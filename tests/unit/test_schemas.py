"""Unit tests for Pydantic schemas."""
import pytest
from src.schemas.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatRole,
    UsageInfo,
    CompletionRequest,
    EmbeddingRequest,
    BatchInferenceRequest,
)
from src.schemas.models import (
    ModelRegisterRequest,
    ModelFormat,
    ModelCapability,
    ModelInfo,
    ModelStatus,
    QuantizationConfig,
    ModelHardwareRequirements,
)
from src.schemas.auth import APIKeyCreate, UserRole


class TestChatCompletionRequest:
    def test_minimal_request(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello")]
        )
        assert req.model == "default"
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.stream is False
        assert len(req.messages) == 1

    def test_full_request(self):
        req = ChatCompletionRequest(
            model="llama-3.1-8b-instruct",
            messages=[
                ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
                ChatMessage(role=ChatRole.USER, content="Hi"),
            ],
            temperature=0.5,
            top_p=0.9,
            max_tokens=4096,
            stream=True,
            stop=["\n"],
            seed=42,
            engine="vllm",
        )
        assert req.model == "llama-3.1-8b-instruct"
        assert req.temperature == 0.5
        assert req.stream is True
        assert req.engine == "vllm"

    def test_temperature_bounds(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                temperature=3.0,
            )

    def test_max_tokens_bounds(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="Hi")],
                max_tokens=0,
            )


class TestChatCompletionResponse:
    def test_response_creation(self):
        resp = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role=ChatRole.ASSISTANT, content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        assert resp.model == "test-model"
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage.total_tokens == 8
        assert resp.id.startswith("chatcmpl-")


class TestModelSchemas:
    def test_register_request(self):
        req = ModelRegisterRequest(
            model_id="test-model",
            source="org/test-model",
            format=ModelFormat.SAFETENSORS,
            capabilities=[ModelCapability.CHAT, ModelCapability.CODE_GENERATION],
            compatible_engines=["vllm", "sglang"],
            context_length=32768,
            parameters_billion=7.0,
        )
        assert req.model_id == "test-model"
        assert ModelCapability.CHAT in req.capabilities
        assert req.context_length == 32768

    def test_quantization_config(self):
        qc = QuantizationConfig(method="awq", bits=4, group_size=128)
        assert qc.method == "awq"
        assert qc.bits == 4

    def test_hardware_requirements(self):
        hw = ModelHardwareRequirements(
            min_gpu_memory_gb=24.0,
            recommended_gpu_memory_gb=40.0,
            min_gpu_count=1,
        )
        assert hw.min_gpu_memory_gb == 24.0


class TestAuthSchemas:
    def test_api_key_create(self):
        key = APIKeyCreate(
            name="test-key",
            role=UserRole.DEVELOPER,
            rate_limit_per_minute=100,
        )
        assert key.name == "test-key"
        assert key.role == UserRole.DEVELOPER

    def test_default_role(self):
        key = APIKeyCreate(name="default-key")
        assert key.role == UserRole.DEVELOPER


class TestBatchRequest:
    def test_batch_creation(self):
        batch = BatchInferenceRequest(
            requests=[
                ChatCompletionRequest(
                    messages=[ChatMessage(role=ChatRole.USER, content=f"Q{i}")]
                )
                for i in range(5)
            ]
        )
        assert len(batch.requests) == 5
        assert batch.batch_id.startswith("batch-")


class TestEmbeddingRequest:
    def test_single_input(self):
        req = EmbeddingRequest(input="Hello world")
        assert req.input == "Hello world"
        assert req.encoding_format == "float"

    def test_list_input(self):
        req = EmbeddingRequest(input=["Hello", "World"])
        assert len(req.input) == 2