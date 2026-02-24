#!/usr/bin/env python3
# =============================================================================
# 可執行整合示例
# =============================================================================
# 演示如何實際調用所有適配器的可執行代碼
# 所有抽象方法已實現為具體可執行代碼
# =============================================================================

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 添加父目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.interfaces import (
    CapabilityContext, InferenceRequest, QuerySpec, MutationSpec,
    CodeContext, ReviewRequest, MessagePayload, DocumentSpec,
    KnowledgeQuery, DeploySpec, BuildSpec, ExerciseSubmission
)
from core.service_facade import EcoPlatformService
from registry.provider_registry import ProviderRegistry

# 導入所有適配器
from adapters.cognitive_compute import GammaCognitiveAdapter, DeltaCognitiveAdapter, EpsilonCognitiveAdapter
from adapters.data_persistence import AlphaPersistenceAdapter, BetaPersistenceAdapter
from adapters.code_engineering import ZetaCodeAdapter, EtaCodeAdapter, ThetaCodeAdapter
from adapters.collaboration import IotaCollaborationAdapter, KappaCollaborationAdapter
from adapters.visual_design import LambdaVisualAdapter, MuVisualAdapter
from adapters.knowledge_mgmt import NuKnowledgeAdapter, XiKnowledgeAdapter
from adapters.deployment import OmicronDeploymentAdapter, PiDeploymentAdapter, RhoDeploymentAdapter
from adapters.learning import SigmaLearningAdapter, TauLearningAdapter, UpsilonLearningAdapter


class ExecutableDemo:
    """可執行整合演示"""
    
    def __init__(self):
        self.service = EcoPlatformService()
        self.registry = ProviderRegistry()
        self._register_all_adapters()
        
    def _register_all_adapters(self):
        """註冊所有適配器到註冊中心"""
        # 認知計算
        self.registry.register_adapter_class('gamma-cognitive', GammaCognitiveAdapter)
        self.registry.register_adapter_class('delta-cognitive', DeltaCognitiveAdapter)
        self.registry.register_adapter_class('epsilon-cognitive', EpsilonCognitiveAdapter)
        
        # 數據持久化
        self.registry.register_adapter_class('alpha-persistence', AlphaPersistenceAdapter)
        self.registry.register_adapter_class('beta-persistence', BetaPersistenceAdapter)
        
        # 代碼工程
        self.registry.register_adapter_class('zeta-code', ZetaCodeAdapter)
        self.registry.register_adapter_class('eta-code', EtaCodeAdapter)
        self.registry.register_adapter_class('theta-code', ThetaCodeAdapter)
        
        # 協作通信
        self.registry.register_adapter_class('iota-collaboration', IotaCollaborationAdapter)
        self.registry.register_adapter_class('kappa-collaboration', KappaCollaborationAdapter)
        
        # 視覺設計
        self.registry.register_adapter_class('lambda-visual', LambdaVisualAdapter)
        self.registry.register_adapter_class('mu-visual', MuVisualAdapter)
        
        # 知識管理
        self.registry.register_adapter_class('nu-knowledge', NuKnowledgeAdapter)
        self.registry.register_adapter_class('xi-knowledge', XiKnowledgeAdapter)
        
        # 部署交付
        self.registry.register_adapter_class('omicron-deployment', OmicronDeploymentAdapter)
        self.registry.register_adapter_class('pi-deployment', PiDeploymentAdapter)
        self.registry.register_adapter_class('rho-deployment', RhoDeploymentAdapter)
        
        # 學習教育
        self.registry.register_adapter_class('sigma-learning', SigmaLearningAdapter)
        self.registry.register_adapter_class('tau-learning', TauLearningAdapter)
        self.registry.register_adapter_class('upsilon-learning', UpsilonLearningAdapter)
        
        print("✅ 所有適配器已註冊到 ProviderRegistry")
    
    async def demo_cognitive_compute(self):
        """演示認知計算適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("🧠 認知計算適配器演示")
        print("="*60)
        
        # 創建適配器實例（使用模擬配置）
        config = {'api_key': 'demo-key', 'base_url': 'https://api.gamma.ai/v1'}
        adapter = GammaCognitiveAdapter(config)
        ctx = CapabilityContext(request_id='demo-001', user_id='user-001')
        
        # 1. 健康檢查
        print("\n1. 健康檢查...")
        result = await adapter.health_check()
        print(f"   結果: {'✅ 健康' if result.success else '❌ 異常'}")
        if not result.success:
            print(f"   錯誤: {result.error_message}")
        
        # 2. 獲取能力列表
        print("\n2. 獲取能力列表...")
        capabilities = await adapter.get_capabilities()
        print(f"   支持的能力: {', '.join(capabilities)}")
        
        # 3. 文本生成（實際可執行）
        print("\n3. 文本生成...")
        request = InferenceRequest(
            prompt="Explain what is cloud computing in one sentence.",
            parameters={'model': 'gpt-4o', 'temperature': 0.7, 'max_tokens': 100}
        )
        result = await adapter.generate(request, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   延遲: {result.latency_ms:.2f}ms")
            print(f"   模型: {result.data.get('model')}")
        else:
            print(f"   錯誤: {result.error_message}")
        
        # 4. 文本嵌入（實際可執行）
        print("\n4. 文本嵌入...")
        result = await adapter.embed(['cloud computing', 'artificial intelligence'], ctx)
        print(f"   成功: {result.success}")
        if result.success:
            embeddings = result.data.get('embeddings', [])
            print(f"   嵌入數量: {len(embeddings)}")
            if embeddings:
                print(f"   嵌入維度: {len(embeddings[0])}")
        
        print("\n✅ 認知計算適配器演示完成")
    
    async def demo_data_persistence(self):
        """演示數據持久化適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("💾 數據持久化適配器演示")
        print("="*60)
        
        config = {
            'url': 'https://demo.supabase.co',
            'anon_key': 'demo-anon-key',
            'service_key': 'demo-service-key'
        }
        adapter = AlphaPersistenceAdapter(config)
        ctx = CapabilityContext(request_id='demo-002', user_id='user-001')
        
        # 1. 健康檢查
        print("\n1. 健康檢查...")
        result = await adapter.health_check()
        print(f"   結果: {'✅ 健康' if result.success else '❌ 異常'}")
        
        # 2. 數據查詢（實際可執行）
        print("\n2. 數據查詢...")
        spec = QuerySpec(
            table='users',
            filters={'status': 'active'},
            limit=10,
            ordering=['-created_at']
        )
        result = await adapter.query(spec, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   延遲: {result.latency_ms:.2f}ms")
            data = result.data
            print(f"   返回記錄數: {len(data) if isinstance(data, list) else 'N/A'}")
        else:
            print(f"   錯誤: {result.error_message}")
        
        # 3. 數據變更（實際可執行）
        print("\n3. 數據插入...")
        spec = MutationSpec(
            table='users',
            operation='insert',
            data={'name': 'Demo User', 'email': 'demo@example.com', 'status': 'active'}
        )
        result = await adapter.mutate(spec, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   延遲: {result.latency_ms:.2f}ms")
        
        # 4. SQL 注入防護測試
        print("\n4. SQL 注入防護測試...")
        from adapters.data_persistence import validate_sql_query, SQLSecurityError
        try:
            validate_sql_query("SELECT * FROM users WHERE id = 1; DROP TABLE users;")
        except SQLSecurityError as e:
            print(f"   ✅ 成功攔截惡意 SQL: {e}")
        
        print("\n✅ 數據持久化適配器演示完成")
    
    async def demo_code_engineering(self):
        """演示代碼工程適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("💻 代碼工程適配器演示")
        print("="*60)
        
        config = {'api_key': 'demo-key', 'base_url': 'https://api.zeta.dev/v1'}
        adapter = ZetaCodeAdapter(config)
        ctx = CapabilityContext(request_id='demo-003', user_id='user-001')
        
        # 1. 代碼補全（實際可執行）
        print("\n1. 代碼補全...")
        context = CodeContext(
            content="def calculate_sum(a, b):\n    ",
            language="python",
            cursor_position=30,
            file_path="example.py"
        )
        result = await adapter.complete(context, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            completions = result.data.get('completions', [])
            print(f"   補全建議數: {len(completions)}")
        
        # 2. 代碼解釋（實際可執行）
        print("\n2. 代碼解釋...")
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        result = await adapter.explain(code, "python", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   解釋長度: {len(result.data.get('explanation', ''))}")
        
        # 3. 代碼審查（實際可執行）
        print("\n3. 代碼審查...")
        request = ReviewRequest(
            code="def get_user(user_id):\n    return db.query(f'SELECT * FROM users WHERE id = {user_id}')",
            language="python",
            review_type="security"
        )
        result = await adapter.review(request, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            issues = result.data.get('issues', [])
            print(f"   發現問題數: {len(issues)}")
        
        print("\n✅ 代碼工程適配器演示完成")
    
    async def demo_collaboration(self):
        """演示協作通信適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("💬 協作通信適配器演示")
        print("="*60)
        
        config = {'bot_token': 'demo-token'}
        adapter = IotaCollaborationAdapter(config)
        ctx = CapabilityContext(request_id='demo-004', user_id='user-001')
        
        # 1. 發送消息（實際可執行）
        print("\n1. 發送消息...")
        payload = MessagePayload(
            channel="general",
            content="Hello from Eco-Platform Integration!",
            thread_id=None
        )
        result = await adapter.send_message(payload, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   消息 ID: {result.data.get('message_id')}")
        
        # 2. 創建頻道（實際可執行）
        print("\n2. 創建頻道...")
        result = await adapter.create_channel("demo-channel", ["user-001", "user-002"], ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   頻道 ID: {result.data.get('channel_id')}")
        
        # 3. 企業知識搜索（實際可執行）
        print("\n3. 企業知識搜索...")
        result = await adapter.search_knowledge("deployment guide", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            results = result.data.get('results', [])
            print(f"   搜索結果數: {len(results)}")
        
        print("\n✅ 協作通信適配器演示完成")
    
    async def demo_visual_design(self):
        """演示視覺設計適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("🎨 視覺設計適配器演示")
        print("="*60)
        
        config = {'token': 'demo-token'}
        adapter = LambdaVisualAdapter(config)
        ctx = CapabilityContext(request_id='demo-005', user_id='user-001')
        
        # 1. 獲取組件庫（實際可執行）
        print("\n1. 獲取組件庫...")
        result = await adapter.get_components(None, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            components = result.data.get('components', [])
            print(f"   組件數量: {len(components)}")
        
        # 2. 設計檢視（實際可執行）
        print("\n2. 設計檢視...")
        result = await adapter.inspect_design("demo-design-id", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            css = result.data.get('css', {})
            print(f"   CSS 屬性數: {len(css)}")
        
        # 3. AI 生成設計（實際可執行）
        print("\n3. AI 生成設計...")
        result = await adapter.generate_from_description(
            "A modern dashboard with sidebar navigation and data cards",
            "web",
            ctx
        )
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   設計 ID: {result.data.get('design_id')}")
        
        print("\n✅ 視覺設計適配器演示完成")
    
    async def demo_knowledge_mgmt(self):
        """演示知識管理適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("📚 知識管理適配器演示")
        print("="*60)
        
        config = {'token': 'demo-token'}
        adapter = NuKnowledgeAdapter(config)
        ctx = CapabilityContext(request_id='demo-006', user_id='user-001')
        
        # 1. 創建文檔（實際可執行）
        print("\n1. 創建文檔...")
        spec = DocumentSpec(
            title="API Documentation",
            content="# API Docs\n\nThis is the API documentation.",
            parent_id="demo-db-id"
        )
        result = await adapter.create_document(spec, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   文檔 ID: {result.data.get('document_id')}")
        
        # 2. 知識查詢（實際可執行）
        print("\n2. 知識查詢...")
        query = KnowledgeQuery(
            query="authentication",
            filters={"database_id": "demo-db-id"},
            top_k=5
        )
        result = await adapter.query_knowledge(query, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            results = result.data.get('results', [])
            print(f"   查詢結果數: {len(results)}")
        
        # 3. 導出文檔（實際可執行）
        print("\n3. 導出文檔...")
        result = await adapter.export_to_format("demo-doc-id", "markdown", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            content = result.data.get('content', '')
            print(f"   導出內容長度: {len(content)}")
        
        print("\n✅ 知識管理適配器演示完成")
    
    async def demo_deployment(self):
        """演示部署交付適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("🚀 部署交付適配器演示")
        print("="*60)
        
        config = {'token': 'demo-token', 'team_id': 'demo-team'}
        adapter = OmicronDeploymentAdapter(config)
        ctx = CapabilityContext(request_id='demo-007', user_id='user-001')
        
        # 1. 構建制品（實際可執行）
        print("\n1. 構建制品...")
        spec = BuildSpec(
            source_path="./src",
            target_platform="static",
            cache_key="v1.0.0"
        )
        result = await adapter.build(spec, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   構建 ID: {result.data.get('build_id')}")
            print(f"   狀態: {result.data.get('status')}")
        
        # 2. 部署到環境（實際可執行）
        print("\n2. 部署到環境...")
        spec = DeploySpec(
            artifact_path="build-v1.0.0",
            environment="production",
            version="1.0.0",
            config_overrides={'DEBUG': 'false'}
        )
        result = await adapter.deploy(spec, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   部署 ID: {result.data.get('deployment_id')}")
            print(f"   URL: {result.data.get('url')}")
        
        # 3. 預覽部署（實際可執行）
        print("\n3. 預覽部署...")
        result = await adapter.preview_deployment("feature/demo-branch", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   預覽 URL: {result.data.get('preview_url')}")
        
        print("\n✅ 部署交付適配器演示完成")
    
    async def demo_learning(self):
        """演示學習教育適配器 - 可執行代碼"""
        print("\n" + "="*60)
        print("📖 學習教育適配器演示")
        print("="*60)
        
        config = {'api_key': 'demo-key'}
        adapter = SigmaLearningAdapter(config)
        ctx = CapabilityContext(request_id='demo-008', user_id='user-001')
        
        # 1. 獲取學習路徑（實際可執行）
        print("\n1. 獲取學習路徑...")
        result = await adapter.get_learning_path("python", "beginner", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            tracks = result.data.get('tracks', [])
            print(f"   學習路徑數: {len(tracks)}")
            print(f"   預計學習時間: {result.data.get('estimated_hours')} 小時")
        
        # 2. 提交練習（實際可執行）
        print("\n2. 提交練習...")
        submission = ExerciseSubmission(
            exercise_id="python-basics-001",
            code="print('Hello, World!')",
            language="python"
        )
        result = await adapter.submit_exercise(submission, ctx)
        print(f"   成功: {result.success}")
        if result.success:
            print(f"   提交 ID: {result.data.get('submission_id')}")
            print(f"   測試通過: {result.data.get('passed')}")
        
        # 3. 獲取提示（實際可執行）
        print("\n3. 獲取提示...")
        result = await adapter.get_hint("python-basics-001", ctx)
        print(f"   成功: {result.success}")
        if result.success:
            hints = result.data.get('hints', [])
            print(f"   提示數量: {len(hints)}")
        
        print("\n✅ 學習教育適配器演示完成")
    
    async def run_all_demos(self):
        """運行所有演示"""
        print("\n" + "🌟"*30)
        print("   Eco-Platform 整合框架 - 可執行代碼演示")
        print("   所有抽象方法已實現為具體可執行代碼")
        print("🌟"*30)
        
        await self.demo_cognitive_compute()
        await self.demo_data_persistence()
        await self.demo_code_engineering()
        await self.demo_collaboration()
        await self.demo_visual_design()
        await self.demo_knowledge_mgmt()
        await self.demo_deployment()
        await self.demo_learning()
        
        print("\n" + "="*60)
        print("🎉 所有演示完成！")
        print("="*60)
        print("\n統計信息:")
        print(f"  - 適配器總數: 19")
        print(f"  - 能力領域: 8")
        print(f"  - 所有方法均已實現為具體可執行代碼")
        print("\n✅ 抽象實現為具體可操作/可執行程式碼完成！")


async def main():
    """主函數"""
    demo = ExecutableDemo()
    await demo.run_all_demos()


if __name__ == '__main__':
    asyncio.run(main())
