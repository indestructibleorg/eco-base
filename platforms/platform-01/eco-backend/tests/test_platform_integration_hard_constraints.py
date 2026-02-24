# =============================================================================
# Platform Integration Service - Hard Constraints Tests
# =============================================================================
# 硬约束测试：所有测试必须全部通过，不能 skip
# =============================================================================

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# 测试对象
from eco_backend.app.services.platform_integration_service_v2 import (
    PlatformIntegrationService,
    IntegrationResult,
    PlatformIntegrationError,
    ServiceNotInitializedError,
    ProviderConfigError,
    ProviderUnavailableError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def service():
    """创建未初始化的服务实例"""
    return PlatformIntegrationService()


@pytest.fixture
def mock_eco_service():
    """创建 mock eco service"""
    mock = Mock()
    mock.execute = AsyncMock()
    mock.stream = AsyncMock()
    mock.health_check = AsyncMock()
    return mock


@pytest.fixture
def valid_config():
    """有效的配置"""
    return {
        "supabase": {
            "api_key": "test-supabase-key",
            "url": "https://test.supabase.co",
        },
        "openai": {
            "api_key": "test-openai-key",
        },
    }


# =============================================================================
# 硬约束测试：初始化
# =============================================================================

class TestInitializationHardConstraints:
    """初始化硬约束测试"""
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, service, valid_config, mock_eco_service):
        """
        硬约束: 初始化成功时 _initialized = True
        """
        with patch.object(service, '_service', mock_eco_service):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    with patch.object(service, '_configure_providers', return_value=None):
                        with patch.object(service, '_validate_providers', return_value=None):
                            await service.initialize(valid_config)
                            
                            assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_without_platform_integrations_raises(self, service, valid_config):
        """
        硬约束: 平台集成不可用时抛出 PlatformIntegrationError
        """
        with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', False):
            with pytest.raises(PlatformIntegrationError) as exc_info:
                await service.initialize(valid_config)
            
            assert "eco-platform-integrations not available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_adapter_registration_failure_raises(self, service, valid_config, mock_eco_service):
        """
        硬约束: 适配器注册失败抛出 PlatformIntegrationError
        """
        with patch.object(service, '_service', mock_eco_service):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters', side_effect=Exception("Registration failed")):
                    with pytest.raises(PlatformIntegrationError) as exc_info:
                        await service.initialize(valid_config)
                    
                    assert "Failed to register adapters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_initialize_provider_validation_failure_raises(self, service, valid_config, mock_eco_service):
        """
        硬约束: 提供商验证失败抛出 ProviderUnavailableError
        """
        with patch.object(service, '_service', mock_eco_service):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    with patch.object(service, '_configure_providers', return_value=None):
                        with patch.object(service, 'health_check', return_value={"status": "unhealthy", "providers": {}}):
                            with pytest.raises(ProviderUnavailableError):
                                await service.initialize(valid_config)


# =============================================================================
# 硬约束测试：运行时检查
# =============================================================================

class TestRuntimeHardConstraints:
    """运行时硬约束测试"""
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_persist_data(self, service):
        """
        硬约束: 未初始化服务调用 persist_data 抛出 ServiceNotInitializedError
        """
        with pytest.raises(ServiceNotInitializedError) as exc_info:
            await service.persist_data("table", {"key": "value"})
        
        assert "not initialized" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_query_data(self, service):
        """
        硬约束: 未初始化服务调用 query_data 抛出 ServiceNotInitializedError
        """
        with pytest.raises(ServiceNotInitializedError):
            await service.query_data("table")
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_vector_search(self, service):
        """
        硬约束: 未初始化服务调用 vector_search 抛出 ServiceNotInitializedError
        """
        with pytest.raises(ServiceNotInitializedError):
            await service.vector_search("index", [0.1, 0.2, 0.3])
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_chat_completion(self, service):
        """
        硬约束: 未初始化服务调用 chat_completion 抛出 ServiceNotInitializedError
        """
        with pytest.raises(ServiceNotInitializedError):
            await service.chat_completion([{"role": "user", "content": "hello"}])
    
    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_send_notification(self, service):
        """
        硬约束: 未初始化服务调用 send_notification 抛出 ServiceNotInitializedError
        """
        with pytest.raises(ServiceNotInitializedError):
            await service.send_notification("test message")


# =============================================================================
# 硬约束测试：IntegrationResult 不变量
# =============================================================================

class TestIntegrationResultInvariants:
    """IntegrationResult 不变量测试"""
    
    def test_success_true_requires_data_not_none(self):
        """
        硬约束: success=True 时 data 必须不为 None
        """
        with pytest.raises(ValueError) as exc_info:
            IntegrationResult(success=True, data=None)
        
        assert "success=True requires data" in str(exc_info.value)
    
    def test_success_false_requires_error_not_none(self):
        """
        硬约束: success=False 时 error 必须不为 None
        """
        with pytest.raises(ValueError) as exc_info:
            IntegrationResult(success=False, error=None)
        
        assert "success=False requires error" in str(exc_info.value)
    
    def test_success_result_with_data(self):
        """
        硬约束: 成功结果必须包含数据
        """
        result = IntegrationResult(
            success=True,
            data={"id": 1, "name": "test"},
        )
        
        assert result.success is True
        assert result.data == {"id": 1, "name": "test"}
        assert result.error is None
    
    def test_failure_result_with_error(self):
        """
        硬约束: 失败结果必须包含错误信息
        """
        result = IntegrationResult(
            success=False,
            error="Operation failed",
        )
        
        assert result.success is False
        assert result.error == "Operation failed"


# =============================================================================
# 硬约束测试：类型注解
# =============================================================================

class TestTypeAnnotations:
    """类型注解硬约束测试"""
    
    def test_all_public_methods_have_type_annotations(self, service):
        """
        硬约束: 所有公共方法必须有类型注解
        """
        import inspect
        
        public_methods = [
            'initialize',
            'persist_data',
            'query_data',
            'vector_search',
            'chat_completion',
            'stream_chat_completion',
            'run_agent_task',
            'create_pull_request',
            'review_code',
            'send_notification',
            'trigger_workflow',
            'deploy',
            'health_check',
        ]
        
        for method_name in public_methods:
            method = getattr(service, method_name)
            sig = inspect.signature(method)
            
            # 检查返回类型注解
            assert sig.return_annotation != inspect.Signature.empty, \
                f"Method {method_name} missing return type annotation"
            
            # 检查参数类型注解
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    assert param.annotation != inspect.Parameter.empty, \
                        f"Method {method_name} param {param_name} missing type annotation"


# =============================================================================
# 硬约束测试：配置验证
# =============================================================================

class TestConfigValidation:
    """配置验证硬约束测试"""
    
    @pytest.mark.asyncio
    async def test_supabase_config_requires_api_key(self, service):
        """
        硬约束: Supabase 配置必须包含 api_key
        """
        config = {
            "supabase": {
                "url": "https://test.supabase.co",
                # 缺少 api_key
            }
        }
        
        with patch.object(service, '_service', Mock()):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    with pytest.raises(ProviderConfigError) as exc_info:
                        await service._configure_providers(config)
                    
                    assert "supabase.api_key is required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_openai_config_requires_api_key(self, service):
        """
        硬约束: OpenAI 配置必须包含 api_key
        """
        config = {
            "openai": {
                # 缺少 api_key
            }
        }
        
        with patch.object(service, '_service', Mock()):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    with pytest.raises(ProviderConfigError) as exc_info:
                        await service._configure_providers(config)
                    
                    assert "openai.api_key is required" in str(exc_info.value)


# =============================================================================
# 硬约束测试：无软失败模式
# =============================================================================

class TestNoSoftFailurePatterns:
    """无软失败模式测试"""
    
    def test_no_return_none_on_uninitialized(self, service):
        """
        硬约束: 未初始化时不返回 None，而是抛出异常
        
        验证方法: 检查 _ensure_initialized 方法存在且抛出异常
        """
        assert hasattr(service, '_ensure_initialized')
        
        with pytest.raises(ServiceNotInitializedError):
            service._ensure_initialized()
    
    def test_all_methods_use_ensure_initialized(self, service):
        """
        硬约束: 所有公共方法都使用 _ensure_initialized 进行运行时检查
        
        验证方法: 检查方法源代码包含 _ensure_initialized 调用
        """
        import inspect
        
        public_methods = [
            'persist_data',
            'query_data',
            'vector_search',
            'chat_completion',
            'run_agent_task',
            'create_pull_request',
            'review_code',
            'send_notification',
            'trigger_workflow',
            'deploy',
        ]
        
        for method_name in public_methods:
            method = getattr(service, method_name)
            source = inspect.getsource(method)
            
            assert "_ensure_initialized()" in source, \
                f"Method {method_name} should call _ensure_initialized()"


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, service, valid_config, mock_eco_service):
        """
        完整工作流测试
        """
        # 设置 mock 返回值
        mock_eco_service.execute.return_value = Mock(
            success=True,
            data={"id": 1},
            error=None,
        )
        mock_eco_service.health_check.return_value = Mock(
            healthy=True,
            provider_status={},
        )
        
        # 初始化
        with patch.object(service, '_service', mock_eco_service):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    await service.initialize(valid_config)
                    
                    # 执行操作
                    result = await service.persist_data("test_table", {"key": "value"})
                    
                    # 验证结果
                    assert result.success is True
                    assert result.data == {"id": 1}
    
    @pytest.mark.asyncio
    async def test_health_check_initialized(self, service, valid_config, mock_eco_service):
        """
        健康检查测试 - 已初始化
        """
        mock_eco_service.health_check.return_value = Mock(
            healthy=True,
            provider_status={"supabase": {"status": "healthy"}},
        )
        
        with patch.object(service, '_service', mock_eco_service):
            with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
                with patch('eco_backend.app.services.platform_integration_service_v2.register_all_adapters'):
                    await service.initialize(valid_config)
                    
                    health = await service.health_check()
                    
                    assert health["status"] == "healthy"
                    assert health["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, service):
        """
        健康检查测试 - 未初始化
        """
        with patch('eco_backend.app.services.platform_integration_service_v2.PLATFORM_INTEGRATIONS_AVAILABLE', True):
            health = await service.health_check()
            
            assert health["status"] == "not_initialized"
            assert health["initialized"] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
