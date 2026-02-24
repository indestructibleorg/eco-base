#!/usr/bin/env python3
# =============================================================================
# 端到端可執行適配器測試
# =============================================================================
# 測試所有適配器的具體可執行代碼實現
# =============================================================================

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import (
    CapabilityContext, InferenceRequest, QuerySpec, MutationSpec,
    CodeContext, ReviewRequest, MessagePayload, DocumentSpec,
    KnowledgeQuery, DeploySpec, BuildSpec, ExerciseSubmission
)
from registry.provider_registry import ProviderRegistry

# 導入所有適配器
from adapters.cognitive_compute import GammaCognitiveAdapter, DeltaCognitiveAdapter, EpsilonCognitiveAdapter
from adapters.data_persistence import AlphaPersistenceAdapter, BetaPersistenceAdapter, validate_sql_query, SQLSecurityError
from adapters.code_engineering import ZetaCodeAdapter, EtaCodeAdapter, ThetaCodeAdapter
from adapters.collaboration import IotaCollaborationAdapter, KappaCollaborationAdapter
from adapters.visual_design import LambdaVisualAdapter, MuVisualAdapter
from adapters.knowledge_mgmt import NuKnowledgeAdapter, XiKnowledgeAdapter
from adapters.deployment import OmicronDeploymentAdapter, PiDeploymentAdapter, RhoDeploymentAdapter
from adapters.learning import SigmaLearningAdapter, TauLearningAdapter, UpsilonLearningAdapter


# =============================================================================
# 認知計算適配器測試
# =============================================================================

class TestGammaCognitiveAdapter:
    """Gamma 認知計算適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return GammaCognitiveAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-001', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'generate' in capabilities
        assert 'stream' in capabilities
        assert 'embed' in capabilities
    
    @pytest.mark.asyncio
    async def test_generate_method_exists(self, adapter, ctx):
        """測試 generate 方法存在且可調用"""
        request = InferenceRequest(prompt="Test prompt", parameters={})
        # 使用 mock 避免實際網絡請求
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate(request, ctx)
            # 預期會失敗（無法連接），但方法應該可執行
            assert hasattr(result, 'success')
            assert hasattr(result, 'error_message')
    
    @pytest.mark.asyncio
    async def test_embed_method_exists(self, adapter, ctx):
        """測試 embed 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.embed(["test text"], ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_function_call_method_exists(self, adapter, ctx):
        """測試 function_call 方法存在且可調用"""
        functions = [{"name": "test_func", "parameters": {}}]
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.function_call("Test", functions, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_multimodal_process_method_exists(self, adapter, ctx):
        """測試 multimodal_process 方法存在且可調用"""
        inputs = {"text": "Describe this image", "image_url": "http://example.com/image.jpg"}
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.multimodal_process(inputs, ctx)
            assert hasattr(result, 'success')


class TestDeltaCognitiveAdapter:
    """Delta 認知計算適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return DeltaCognitiveAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-002', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'generate' in capabilities
        assert 'long_context' in capabilities
    
    @pytest.mark.asyncio
    async def test_generate_method_exists(self, adapter, ctx):
        """測試 generate 方法存在且可調用"""
        request = InferenceRequest(prompt="Test prompt", parameters={})
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate(request, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_tool_call_method_exists(self, adapter, ctx):
        """測試 function_call（工具調用）方法存在且可調用"""
        tools = [{"name": "test_tool", "description": "Test tool"}]
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.function_call("Test", tools, ctx)
            assert hasattr(result, 'success')


class TestEpsilonCognitiveAdapter:
    """Epsilon 認知計算適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return EpsilonCognitiveAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-003', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'reasoning' in capabilities
    
    @pytest.mark.asyncio
    async def test_generate_method_exists(self, adapter, ctx):
        """測試 generate 方法存在且可調用"""
        request = InferenceRequest(prompt="Test prompt", parameters={})
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate(request, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_generate_stream_method_exists(self, adapter, ctx):
        """測試 generate_stream 方法存在且可調用"""
        request = InferenceRequest(prompt="Test prompt", parameters={})
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            chunks = []
            async for chunk in adapter.generate_stream(request, ctx):
                chunks.append(chunk)
                if chunk.is_final:
                    break
            assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_execute_agent_task_method_exists(self, adapter, ctx):
        """測試 execute_agent_task 方法存在且可調用"""
        from core.interfaces import AgentTask
        task = AgentTask(
            task_type="analysis",
            description="Analyze this data",
            inputs={"data": "test"},
            expected_output_format="json"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.execute_agent_task(task, ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 數據持久化適配器測試
# =============================================================================

class TestAlphaPersistenceAdapter:
    """Alpha 數據持久化適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return AlphaPersistenceAdapter({
            'url': 'http://localhost:9999',
            'anon_key': 'test-key',
            'service_key': 'test-service-key'
        })
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-004', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'query' in capabilities
        assert 'vector_search' in capabilities
    
    @pytest.mark.asyncio
    async def test_query_method_exists(self, adapter, ctx):
        """測試 query 方法存在且可調用"""
        spec = QuerySpec(table="users", filters={"status": "active"}, limit=10)
        result = await adapter.query(spec, ctx)
        assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_mutate_method_exists(self, adapter, ctx):
        """測試 mutate 方法存在且可調用"""
        spec = MutationSpec(
            table="users",
            operation="insert",
            data={"name": "Test", "email": "test@example.com"}
        )
        result = await adapter.mutate(spec, ctx)
        assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_vector_search_method_exists(self, adapter, ctx):
        """測試 vector_search 方法存在且可調用"""
        result = await adapter.vector_search("items", [0.1, 0.2, 0.3], 5, ctx)
        assert hasattr(result, 'success')
    
    def test_sql_injection_protection(self):
        """測試 SQL 注入防護"""
        malicious_sql = "SELECT * FROM users; DROP TABLE users;"
        with pytest.raises(SQLSecurityError):
            validate_sql_query(malicious_sql)
    
    def test_sql_injection_protection_union(self):
        """測試 UNION 注入防護"""
        malicious_sql = "SELECT * FROM users UNION SELECT * FROM passwords"
        with pytest.raises(SQLSecurityError):
            validate_sql_query(malicious_sql)


class TestBetaPersistenceAdapter:
    """Beta 數據持久化適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return BetaPersistenceAdapter({
            'host': 'localhost:9999',
            'username': 'test',
            'password': 'test',
            'database': 'test'
        })
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-005', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'sharding' in capabilities
    
    @pytest.mark.asyncio
    async def test_query_method_exists(self, adapter, ctx):
        """測試 query 方法存在且可調用"""
        spec = QuerySpec(table="users", filters={})
        result = await adapter.query(spec, ctx)
        assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_execute_sql_method_exists(self, adapter, ctx):
        """測試 execute_sql 方法存在且可調用"""
        result = await adapter.execute_sql("SELECT * FROM users", [], ctx)
        assert hasattr(result, 'success')


# =============================================================================
# 代碼工程適配器測試
# =============================================================================

class TestZetaCodeAdapter:
    """Zeta 代碼工程適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return ZetaCodeAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-006', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'complete' in capabilities
        assert 'refactor' in capabilities
    
    @pytest.mark.asyncio
    async def test_complete_method_exists(self, adapter, ctx):
        """測試 complete 方法存在且可調用"""
        context = CodeContext(
            content="def hello():",
            language="python",
            cursor_position=13
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.complete(context, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_explain_method_exists(self, adapter, ctx):
        """測試 explain 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.explain("print('hello')", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_refactor_method_exists(self, adapter, ctx):
        """測試 refactor 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.refactor("x = 1 + 2", "python", "Simplify", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_review_method_exists(self, adapter, ctx):
        """測試 review 方法存在且可調用"""
        request = ReviewRequest(
            code="def test(): pass",
            language="python",
            review_type="general"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.review(request, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_generate_tests_method_exists(self, adapter, ctx):
        """測試 generate_tests 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate_tests("def add(a, b): return a + b", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_translate_language_method_exists(self, adapter, ctx):
        """測試 translate_language 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.translate_language("console.log('hi')", "javascript", "python", ctx)
            assert hasattr(result, 'success')


class TestEtaCodeAdapter:
    """Eta 代碼工程適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return EtaCodeAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-007', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'review' in capabilities
    
    @pytest.mark.asyncio
    async def test_review_method_exists(self, adapter, ctx):
        """測試 review 方法存在且可調用（核心功能）"""
        request = ReviewRequest(
            code="def test(): pass",
            language="python",
            review_type="security"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.review(request, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_explain_method_exists(self, adapter, ctx):
        """測試 explain 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.explain("print('hello')", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_generate_tests_method_exists(self, adapter, ctx):
        """測試 generate_tests 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate_tests("def add(a, b): return a + b", "python", ctx)
            assert hasattr(result, 'success')


class TestThetaCodeAdapter:
    """Theta 代碼工程適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return ThetaCodeAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-008', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'collaborate' in capabilities
    
    @pytest.mark.asyncio
    async def test_complete_method_exists(self, adapter, ctx):
        """測試 complete 方法存在且可調用"""
        context = CodeContext(content="def hello():", language="python", cursor_position=13)
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.complete(context, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_explain_method_exists(self, adapter, ctx):
        """測試 explain 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.explain("print('hello')", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_refactor_method_exists(self, adapter, ctx):
        """測試 refactor 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.refactor("x = 1 + 2", "python", "Simplify", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_review_method_exists(self, adapter, ctx):
        """測試 review 方法存在且可調用"""
        request = ReviewRequest(
            code="def test(): pass",
            language="python",
            review_type="general"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.review(request, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_generate_tests_method_exists(self, adapter, ctx):
        """測試 generate_tests 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate_tests("def add(a, b): return a + b", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_translate_language_method_exists(self, adapter, ctx):
        """測試 translate_language 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.translate_language("console.log('hi')", "javascript", "python", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_search_repository_method_exists(self, adapter, ctx):
        """測試 search_repository 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.search_repository("test query", None, ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 協作通信適配器測試
# =============================================================================

class TestIotaCollaborationAdapter:
    """Iota 協作通信適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return IotaCollaborationAdapter({'bot_token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-009', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'message' in capabilities
        assert 'workflow' in capabilities
    
    @pytest.mark.asyncio
    async def test_send_message_method_exists(self, adapter, ctx):
        """測試 send_message 方法存在且可調用"""
        payload = MessagePayload(channel="general", content="Hello")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.send_message(payload, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_create_channel_method_exists(self, adapter, ctx):
        """測試 create_channel 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.create_channel("test-channel", ["user-001"], ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_setup_workflow_method_exists(self, adapter, ctx):
        """測試 setup_workflow 方法存在且可調用"""
        from core.interfaces import WorkflowTrigger
        trigger = WorkflowTrigger(
            event_type="message",
            conditions={"channel": "general"},
            actions=[{"type": "notify"}]
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.setup_workflow(trigger, ctx)
            assert hasattr(result, 'success')


class TestKappaCollaborationAdapter:
    """Kappa 協作通信適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return KappaCollaborationAdapter({'token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-010', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'pr' in capabilities
    
    @pytest.mark.asyncio
    async def test_send_message_method_exists(self, adapter, ctx):
        """測試 send_message 方法存在且可調用"""
        payload = MessagePayload(channel="owner/repo/issue/123", content="Test comment")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.send_message(payload, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_create_channel_method_exists(self, adapter, ctx):
        """測試 create_channel 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.create_channel("owner/repo", ["user-001"], ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_search_knowledge_method_exists(self, adapter, ctx):
        """測試 search_knowledge 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.search_knowledge("test query", ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 視覺設計適配器測試
# =============================================================================

class TestLambdaVisualAdapter:
    """Lambda 視覺設計適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return LambdaVisualAdapter({'token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-011', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'components' in capabilities
        assert 'generate' in capabilities
    
    @pytest.mark.asyncio
    async def test_get_components_method_exists(self, adapter, ctx):
        """測試 get_components 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_components(None, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_export_asset_method_exists(self, adapter, ctx):
        """測試 export_asset 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.export_asset("asset-123", "png", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_generate_from_description_method_exists(self, adapter, ctx):
        """測試 generate_from_description 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.generate_from_description("A button", "component", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_inspect_design_method_exists(self, adapter, ctx):
        """測試 inspect_design 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.inspect_design("design-123", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_create_prototype_method_exists(self, adapter, ctx):
        """測試 create_prototype 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.create_prototype(["screen-1"], [], ctx)
            assert hasattr(result, 'success')


class TestMuVisualAdapter:
    """Mu 視覺設計適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return MuVisualAdapter({'token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-012', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'export' in capabilities
    
    @pytest.mark.asyncio
    async def test_get_components_method_exists(self, adapter, ctx):
        """測試 get_components 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_components("doc-123", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_export_asset_method_exists(self, adapter, ctx):
        """測試 export_asset 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.export_asset("layer-123", "svg", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_inspect_design_method_exists(self, adapter, ctx):
        """測試 inspect_design 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.inspect_design("doc-123", ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 知識管理適配器測試
# =============================================================================

class TestNuKnowledgeAdapter:
    """Nu 知識管理適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return NuKnowledgeAdapter({'token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-013', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'create' in capabilities
        assert 'query' in capabilities
    
    @pytest.mark.asyncio
    async def test_create_document_method_exists(self, adapter, ctx):
        """測試 create_document 方法存在且可調用"""
        spec = DocumentSpec(title="Test", content="# Test Doc")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.create_document(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_update_document_method_exists(self, adapter, ctx):
        """測試 update_document 方法存在且可調用"""
        spec = DocumentSpec(title="Updated", content="# Updated")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.update_document("doc-123", spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_query_knowledge_method_exists(self, adapter, ctx):
        """測試 query_knowledge 方法存在且可調用"""
        query = KnowledgeQuery(query="test", top_k=5)
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.query_knowledge(query, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_export_to_format_method_exists(self, adapter, ctx):
        """測試 export_to_format 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.export_to_format("doc-123", "markdown", ctx)
            assert hasattr(result, 'success')


class TestXiKnowledgeAdapter:
    """Xi 知識管理適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return XiKnowledgeAdapter({'token': 'test-token'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-014', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'sync' in capabilities
    
    @pytest.mark.asyncio
    async def test_create_document_method_exists(self, adapter, ctx):
        """測試 create_document 方法存在且可調用"""
        spec = DocumentSpec(title="Test", content="# Test")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.create_document(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_sync_from_git_method_exists(self, adapter, ctx):
        """測試 sync_from_git 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.sync_from_git("https://github.com/test/repo", "main", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_export_to_format_method_exists(self, adapter, ctx):
        """測試 export_to_format 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.export_to_format("doc-123", "pdf", ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 部署交付適配器測試
# =============================================================================

class TestOmicronDeploymentAdapter:
    """Omicron 部署交付適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return OmicronDeploymentAdapter({'token': 'test-token', 'team_id': 'test-team'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-015', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'deploy' in capabilities
        assert 'preview' in capabilities
    
    @pytest.mark.asyncio
    async def test_build_method_exists(self, adapter, ctx):
        """測試 build 方法存在且可調用"""
        spec = BuildSpec(source_path="./src", target_platform="static")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.build(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_deploy_method_exists(self, adapter, ctx):
        """測試 deploy 方法存在且可調用"""
        spec = DeploySpec(artifact_path="build", environment="prod", version="1.0.0")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.deploy(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_get_deployment_status_method_exists(self, adapter, ctx):
        """測試 get_deployment_status 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_deployment_status("deploy-123", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_rollback_method_exists(self, adapter, ctx):
        """測試 rollback 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.rollback("deploy-123", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_preview_deployment_method_exists(self, adapter, ctx):
        """測試 preview_deployment 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.preview_deployment("feature/test", ctx)
            assert hasattr(result, 'success')


class TestPiDeploymentAdapter:
    """Pi 部署交付適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return PiDeploymentAdapter({'token': 'test-token', 'org_id': 'test-org'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-016', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'cache' in capabilities
    
    @pytest.mark.asyncio
    async def test_build_method_exists(self, adapter, ctx):
        """測試 build 方法存在且可調用"""
        spec = BuildSpec(source_path="./src", target_platform="docker")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.build(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_get_deployment_status_method_exists(self, adapter, ctx):
        """測試 get_deployment_status 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_deployment_status("build-123", ctx)
            assert hasattr(result, 'success')


class TestRhoDeploymentAdapter:
    """Rho 部署交付適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return RhoDeploymentAdapter({'token': 'test-token', 'org_id': 'test-org'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-017', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'plan' in capabilities
        assert 'state' in capabilities
    
    @pytest.mark.asyncio
    async def test_build_method_exists(self, adapter, ctx):
        """測試 build 方法存在且可調用（Terraform Plan）"""
        spec = BuildSpec(source_path="workspace-123", target_platform="terraform")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.build(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_deploy_method_exists(self, adapter, ctx):
        """測試 deploy 方法存在且可調用（Terraform Apply）"""
        spec = DeploySpec(artifact_path="run-123", environment="prod", version="1.0.0")
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.deploy(spec, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_rollback_method_exists(self, adapter, ctx):
        """測試 rollback 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.rollback("workspace-123", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_preview_deployment_method_exists(self, adapter, ctx):
        """測試 preview_deployment 方法存在且可調用（Speculative Plan）"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.preview_deployment("workspace-123", ctx)
            assert hasattr(result, 'success')


# =============================================================================
# 學習教育適配器測試
# =============================================================================

class TestSigmaLearningAdapter:
    """Sigma 學習教育適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return SigmaLearningAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-018', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'path' in capabilities
        assert 'hint' in capabilities
    
    @pytest.mark.asyncio
    async def test_get_learning_path_method_exists(self, adapter, ctx):
        """測試 get_learning_path 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_learning_path("python", "beginner", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_submit_exercise_method_exists(self, adapter, ctx):
        """測試 submit_exercise 方法存在且可調用"""
        submission = ExerciseSubmission(
            exercise_id="ex-001",
            code="print('hello')",
            language="python"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.submit_exercise(submission, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_get_hint_method_exists(self, adapter, ctx):
        """測試 get_hint 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_hint("ex-001", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_track_progress_method_exists(self, adapter, ctx):
        """測試 track_progress 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.track_progress("user-001", ctx)
            assert hasattr(result, 'success')


class TestTauLearningAdapter:
    """Tau 學習教育適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return TauLearningAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-019', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'collaborate' in capabilities
    
    @pytest.mark.asyncio
    async def test_get_learning_path_method_exists(self, adapter, ctx):
        """測試 get_learning_path 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.get_learning_path("javascript", "intermediate", ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_submit_exercise_method_exists(self, adapter, ctx):
        """測試 submit_exercise 方法存在且可調用"""
        submission = ExerciseSubmission(
            exercise_id="ex-002",
            code="console.log('hello')",
            language="javascript"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.submit_exercise(submission, ctx)
            assert hasattr(result, 'success')


class TestUpsilonLearningAdapter:
    """Upsilon 學習教育適配器可執行測試"""
    
    @pytest.fixture
    def adapter(self):
        return UpsilonLearningAdapter({'api_key': 'test-key'})
    
    @pytest.fixture
    def ctx(self):
        return CapabilityContext(request_id='test-020', user_id='user-001')
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter):
        """測試獲取能力列表"""
        capabilities = await adapter.get_capabilities()
        assert isinstance(capabilities, list)
        assert 'submit' in capabilities
    
    @pytest.mark.asyncio
    async def test_submit_exercise_method_exists(self, adapter, ctx):
        """測試 submit_exercise 方法存在且可調用（創建 Pen）"""
        submission = ExerciseSubmission(
            exercise_id="ex-003",
            code="<div>Hello</div>",
            language="html"
        )
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.submit_exercise(submission, ctx)
            assert hasattr(result, 'success')
    
    @pytest.mark.asyncio
    async def test_track_progress_method_exists(self, adapter, ctx):
        """測試 track_progress 方法存在且可調用"""
        with patch.object(adapter, '_base_url', 'http://localhost:9999'):
            result = await adapter.track_progress("user-001", ctx)
            assert hasattr(result, 'success')


# =============================================================================
# Provider Registry 測試
# =============================================================================

class TestProviderRegistry:
    """提供者註冊中心可執行測試"""
    
    @pytest.fixture
    def registry(self):
        return ProviderRegistry()
    
    def test_singleton_pattern(self):
        """測試單例模式"""
        reg1 = ProviderRegistry()
        reg2 = ProviderRegistry()
        assert reg1 is reg2
    
    def test_register_adapter_class(self, registry):
        """測試註冊適配器類"""
        registry.register_adapter_class('test-adapter', GammaCognitiveAdapter)
        assert 'test-adapter' in registry._adapter_classes
    
    def test_create_provider(self, registry):
        """測試創建提供者實例"""
        registry.register_adapter_class('gamma-test', GammaCognitiveAdapter)
        provider = registry.create_provider('gamma-test', {'api_key': 'test'})
        assert provider is not None
        assert provider.provider_id == 'gamma-cognitive'


# =============================================================================
# 整合測試
# =============================================================================

class TestIntegration:
    """端到端整合測試"""
    
    @pytest.mark.asyncio
    async def test_all_adapters_have_required_methods(self):
        """測試所有適配器都有必需的方法"""
        adapters = [
            GammaCognitiveAdapter({'api_key': 'test'}),
            DeltaCognitiveAdapter({'api_key': 'test'}),
            EpsilonCognitiveAdapter({'api_key': 'test'}),
            AlphaPersistenceAdapter({'url': 'http://test', 'anon_key': 'test', 'service_key': 'test'}),
            BetaPersistenceAdapter({'host': 'test', 'username': 'test', 'password': 'test', 'database': 'test'}),
            ZetaCodeAdapter({'api_key': 'test'}),
            EtaCodeAdapter({'api_key': 'test'}),
            ThetaCodeAdapter({'api_key': 'test'}),
            IotaCollaborationAdapter({'bot_token': 'test'}),
            KappaCollaborationAdapter({'token': 'test'}),
            LambdaVisualAdapter({'token': 'test'}),
            MuVisualAdapter({'token': 'test'}),
            NuKnowledgeAdapter({'token': 'test'}),
            XiKnowledgeAdapter({'token': 'test'}),
            OmicronDeploymentAdapter({'token': 'test', 'team_id': 'test'}),
            PiDeploymentAdapter({'token': 'test', 'org_id': 'test'}),
            RhoDeploymentAdapter({'token': 'test', 'org_id': 'test'}),
            SigmaLearningAdapter({'api_key': 'test'}),
            TauLearningAdapter({'api_key': 'test'}),
            UpsilonLearningAdapter({'api_key': 'test'}),
        ]
        
        for adapter in adapters:
            # 所有適配器都應該有這些基本方法
            assert hasattr(adapter, 'provider_id')
            assert hasattr(adapter, 'domain')
            assert hasattr(adapter, 'health_check')
            assert hasattr(adapter, 'get_capabilities')
            
            # 測試 get_capabilities 返回列表
            capabilities = await adapter.get_capabilities()
            assert isinstance(capabilities, list)
            assert len(capabilities) > 0
        
        print(f"\n✅ 所有 {len(adapters)} 個適配器都有必需的方法")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
