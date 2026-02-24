# Eco-Platform Integration Framework

企業級第三方平台能力集成框架

## 架構概述

本框架提供一個統一的抽象層，將30+第三方平台的技術能力整合到您的後端架構中。通過**依賴倒置原則**和**適配器模式**，實現：

- **零耦合**: 核心業務邏輯不依賴任何第三方平台
- **可替換**: 隨時切換底層實現而不影響業務代碼
- **可擴展**: 輕鬆添加新的平台適配器

## 能力領域分類

| 領域 | 抽象接口 | 適配器 |
|------|---------|--------|
| 數據持久化 | `IDataPersistenceProvider` | Alpha, Beta |
| 認知計算 | `ICognitiveComputeProvider` | Gamma, Delta, Epsilon |
| 代碼工程 | `ICodeEngineeringProvider` | Zeta, Eta, Theta |
| 協作通信 | `ICollaborationProvider` | Iota, Kappa |
| 視覺設計 | `IVisualDesignProvider` | Lambda, Mu |
| 知識管理 | `IKnowledgeManagementProvider` | Nu, Xi |
| 部署交付 | `IDeploymentProvider` | Omicron, Pi, Rho |
| 學習教育 | `ILearningProvider` | Sigma, Tau, Upsilon |

## 快速開始

### 1. 安裝依賴

```bash
pip install httpx websockets
```

### 2. 配置環境變數

```bash
# 數據持久化
export ALPHA_URL="https://your-project.supabase.co"
export ALPHA_ANON_KEY="your-anon-key"
export ALPHA_SERVICE_KEY="your-service-key"

# 認知計算
export GAMMA_API_KEY="your-openai-key"
export DELTA_API_KEY="your-anthropic-key"
export EPSILON_API_KEY="your-deepseek-key"

# 代碼工程
export ZETA_API_KEY="your-cursor-key"
export ETA_API_KEY="your-sider-key"
export THETA_API_KEY="your-replit-key"

# 協作通信
export IOTA_BOT_TOKEN="your-slack-bot-token"
export KAPPA_TOKEN="your-github-token"

# 視覺設計
export LAMBDA_TOKEN="your-figma-token"
export MU_TOKEN="your-sketch-token"

# 知識管理
export NU_TOKEN="your-notion-token"
export XI_TOKEN="your-gitbook-token"

# 部署交付
export OMICRON_TOKEN="your-vercel-token"
export PI_TOKEN="your-depot-token"
export RHO_TOKEN="your-terraform-token"

# 學習教育
export SIGMA_API_KEY="your-mimo-key"
export TAU_API_KEY="your-replit-key"
export UPSILON_API_KEY="your-codepen-key"
```

### 3. 使用服務門面

```python
import asyncio
from core.service_facade import eco_service

async def main():
    # 初始化服務
    await eco_service.initialize()
    
    # 使用認知計算
    result = await eco_service.generate_text(
        prompt="Explain quantum computing",
        provider='gamma-cognitive'  # 或 'delta-cognitive', 'epsilon-cognitive'
    )
    print(result.data)
    
    # 使用數據持久化
    result = await eco_service.query_data(
        table='users',
        filters={'status': 'active'},
        provider='alpha-persistence'
    )
    print(result.data)
    
    # 使用代碼工程
    result = await eco_service.complete_code(
        code="def fibonacci(n):",
        language='python',
        provider='zeta-code'
    )
    print(result.data)

asyncio.run(main())
```

## 核心接口

### 數據持久化

```python
# 查詢
result = await eco_service.query_data(
    table='users',
    filters={'status': 'active'},
    provider='alpha-persistence'
)

# 插入
result = await eco_service.mutate_data(
    operation='insert',
    table='users',
    data={'name': 'John', 'email': 'john@example.com'},
    provider='alpha-persistence'
)

# 向量搜索
result = await eco_service.vector_search(
    table='documents',
    vector=[0.1, 0.2, 0.3],
    top_k=10,
    provider='alpha-persistence'
)
```

### 認知計算

```python
# 文本生成
result = await eco_service.generate_text(
    prompt="Write a story about AI",
    parameters={'temperature': 0.7, 'max_tokens': 500},
    provider='gamma-cognitive'
)

# 流式生成
async for chunk in eco_service.generate_text_stream(
    prompt="Write a poem",
    provider='gamma-cognitive'
):
    print(chunk.content, end='')

# 文本嵌入
result = await eco_service.embed_texts(
    texts=["Hello", "World"],
    provider='gamma-cognitive'
)

# 函數調用
result = await eco_service.function_call(
    prompt="What's the weather in Tokyo?",
    functions=[{
        'name': 'get_weather',
        'description': 'Get weather info',
        'parameters': {...}
    }],
    provider='gamma-cognitive'
)
```

### 代碼工程

```python
# 代碼補全
result = await eco_service.complete_code(
    code="def quicksort(arr):",
    language='python',
    provider='zeta-code'
)

# 代碼解釋
result = await eco_service.explain_code(
    code="...",
    language='python',
    provider='zeta-code'
)

# 代碼審查
result = await eco_service.review_code(
    code="...",
    language='python',
    review_type='security',  # general, security, performance, style
    provider='zeta-code'
)
```

### 協作通信

```python
# 發送消息
result = await eco_service.send_message(
    channel='#general',
    content='Hello team!',
    provider='iota-collaboration'
)

# 頻道摘要
result = await eco_service.summarize_channel(
    channel='#engineering',
    provider='iota-collaboration'
)
```

### 部署交付

```python
# 部署應用
result = await eco_service.deploy(
    artifact_path='my-app',
    environment='production',
    version='v1.0.0',
    provider='omicron-deployment'
)
```

## 目錄結構

```
eco-platform-integrations/
├── core/
│   ├── interfaces.py          # 抽象接口定義
│   └── service_facade.py      # 統一服務門面
├── adapters/
│   ├── data_persistence.py    # 數據持久化適配器
│   ├── cognitive_compute.py   # 認知計算適配器
│   ├── code_engineering.py    # 代碼工程適配器
│   ├── collaboration.py       # 協作通信適配器
│   ├── visual_design.py       # 視覺設計適配器
│   ├── knowledge_mgmt.py      # 知識管理適配器
│   ├── deployment.py          # 部署交付適配器
│   └── learning.py            # 學習教育適配器
├── registry/
│   └── provider_registry.py   # 提供者註冊中心
├── config/
│   └── platform_configs.py    # 平台配置模板
├── examples/
│   └── usage_example.py       # 使用示例
└── README.md
```

## 添加新適配器

```python
# 1. 創建適配器類
from core.interfaces import ICapabilityProvider, CapabilityDomain, OperationResult

class MyCustomAdapter(ICapabilityProvider):
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COGNITIVE_COMPUTE
    
    @property
    def provider_id(self) -> str:
        return "my-custom-provider"
    
    async def health_check(self) -> OperationResult:
        # 實現健康檢查
        pass
    
    async def get_capabilities(self) -> List[str]:
        return ['generate', 'embed']
    
    # 實現其他接口方法...

# 2. 註冊適配器
from registry.provider_registry import registry

registry.register_adapter_class('my-custom-provider', MyCustomAdapter)

# 3. 創建實例
config = {'api_key': 'your-key'}
registry.create_provider('my-custom-provider', config)

# 4. 使用
result = await eco_service.generate_text(
    prompt="Hello",
    provider='my-custom-provider'
)
```

## 健康檢查

```python
# 檢查所有提供者狀態
health = await eco_service.health_check()

for provider_id, status in health.items():
    print(f"{provider_id}: {'✅' if status['healthy'] else '❌'}")
```

## 錯誤處理

```python
result = await eco_service.generate_text(
    prompt="Hello",
    provider='gamma-cognitive'
)

if result.success:
    print(result.data)
else:
    print(f"Error: {result.error_code} - {result.error_message}")
```

## 最佳實踐

1. **使用環境變數**: 所有敏感配置通過環境變數注入
2. **初始化時指定提供者**: 只初始化需要的提供者以減少資源消耗
3. **處理超時**: 設置合理的超時時間
4. **熔斷機制**: 對失敗的提供者實施熔斷
5. **日誌記錄**: 啟用詳細的日誌以便排查問題

## 平台映射對照表

| 原始平台 | 抽象提供者ID | 領域 |
|---------|-------------|------|
| Supabase | alpha-persistence | 數據持久化 |
| PlanetScale | beta-persistence | 數據持久化 |
| OpenAI/GPT | gamma-cognitive | 認知計算 |
| Claude | delta-cognitive | 認知計算 |
| DeepSeek | epsilon-cognitive | 認知計算 |
| Cursor | zeta-code | 代碼工程 |
| Sider | eta-code | 代碼工程 |
| Replit | theta-code | 代碼工程 |
| Slack | iota-collaboration | 協作通信 |
| GitHub | kappa-collaboration | 協作通信 |
| Figma | lambda-visual | 視覺設計 |
| Sketch | mu-visual | 視覺設計 |
| Notion | nu-knowledge | 知識管理 |
| GitBook | xi-knowledge | 知識管理 |
| Vercel | omicron-deployment | 部署交付 |
| Depot | pi-deployment | 部署交付 |
| Terraform | rho-deployment | 部署交付 |
| Mimo | sigma-learning | 學習教育 |
| Replit | tau-learning | 學習教育 |
| CodePen | upsilon-learning | 學習教育 |

## 許可證

MIT License
