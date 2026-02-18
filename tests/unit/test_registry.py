"""Unit tests for Model Registry."""
import pytest
from datetime import datetime, timezone

from src.core.registry import ModelRegistry
from src.schemas.models import (
    ModelCapability,
    ModelFormat,
    ModelRegisterRequest,
    ModelStatus,
)


@pytest.fixture
def registry():
    return ModelRegistry()


class TestModelRegistry:
    @pytest.mark.asyncio
    async def test_default_models_loaded(self, registry):
        models = await registry.list_models()
        assert len(models) >= 5
        model_ids = [m.model_id for m in models]
        assert "llama-3.1-8b-instruct" in model_ids
        assert "qwen2.5-72b-instruct" in model_ids

    @pytest.mark.asyncio
    async def test_register_model(self, registry):
        req = ModelRegisterRequest(
            model_id="test-custom-model",
            source="org/custom-model",
            format=ModelFormat.SAFETENSORS,
            capabilities=[ModelCapability.CHAT],
            compatible_engines=["vllm"],
            context_length=8192,
            parameters_billion=3.0,
        )
        info = await registry.register(req)
        assert info.model_id == "test-custom-model"
        assert info.status == ModelStatus.REGISTERED
        assert info.registered_at is not None

    @pytest.mark.asyncio
    async def test_register_duplicate_fails(self, registry):
        req = ModelRegisterRequest(
            model_id="dup-model",
            source="org/dup",
            capabilities=[ModelCapability.CHAT],
            compatible_engines=["vllm"],
            context_length=4096,
        )
        await registry.register(req)
        with pytest.raises(ValueError, match="already registered"):
            await registry.register(req)

    @pytest.mark.asyncio
    async def test_get_model(self, registry):
        model = await registry.get("llama-3.1-8b-instruct")
        assert model is not None
        assert model.source == "meta-llama/Llama-3.1-8B-Instruct"
        assert ModelCapability.CHAT in model.capabilities

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, registry):
        model = await registry.get("nonexistent-model")
        assert model is None

    @pytest.mark.asyncio
    async def test_filter_by_capability(self, registry):
        code_models = await registry.list_models(capability=ModelCapability.CODE_GENERATION)
        for m in code_models:
            assert ModelCapability.CODE_GENERATION in m.capabilities

    @pytest.mark.asyncio
    async def test_filter_by_engine(self, registry):
        vllm_models = await registry.list_models(engine="vllm")
        for m in vllm_models:
            assert "vllm" in m.compatible_engines

    @pytest.mark.asyncio
    async def test_update_status(self, registry):
        await registry.update_status("llama-3.1-8b-instruct", ModelStatus.LOADING, "vllm")
        model = await registry.get("llama-3.1-8b-instruct")
        assert model.status == ModelStatus.LOADING

        await registry.update_status("llama-3.1-8b-instruct", ModelStatus.READY, "vllm")
        model = await registry.get("llama-3.1-8b-instruct")
        assert model.status == ModelStatus.READY
        assert "vllm" in model.loaded_on_engines

    @pytest.mark.asyncio
    async def test_resolve_model_default(self, registry):
        model = await registry.resolve_model("default")
        assert model is not None

    @pytest.mark.asyncio
    async def test_resolve_model_by_source(self, registry):
        model = await registry.resolve_model("meta-llama/Llama-3.1-8B-Instruct")
        assert model is not None
        assert model.model_id == "llama-3.1-8b-instruct"

    @pytest.mark.asyncio
    async def test_resolve_model_partial(self, registry):
        model = await registry.resolve_model("llama-3.1-8b")
        assert model is not None

    @pytest.mark.asyncio
    async def test_delete_model(self, registry):
        req = ModelRegisterRequest(
            model_id="to-delete",
            source="org/delete-me",
            capabilities=[ModelCapability.CHAT],
            compatible_engines=["vllm"],
            context_length=4096,
        )
        await registry.register(req)
        deleted = await registry.delete("to-delete")
        assert deleted is True
        model = await registry.get("to-delete")
        assert model is None

    @pytest.mark.asyncio
    async def test_delete_loaded_model_fails(self, registry):
        await registry.update_status("llama-3.1-8b-instruct", ModelStatus.READY, "vllm")
        with pytest.raises(ValueError, match="still loaded"):
            await registry.delete("llama-3.1-8b-instruct")

    @pytest.mark.asyncio
    async def test_count(self, registry):
        initial = registry.count
        req = ModelRegisterRequest(
            model_id="count-test",
            source="org/count",
            capabilities=[ModelCapability.CHAT],
            compatible_engines=["vllm"],
            context_length=4096,
        )
        await registry.register(req)
        assert registry.count == initial + 1