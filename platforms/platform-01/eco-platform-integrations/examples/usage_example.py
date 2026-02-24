# =============================================================================
# Eco-Platform Integration Framework - Usage Examples
# =============================================================================
# 使用示例
# =============================================================================

import asyncio
import os
from typing import List

# 設置環境變數 (實際使用時應從 Secret Manager 獲取)
os.environ['ALPHA_URL'] = 'https://your-project.supabase.co'
os.environ['ALPHA_ANON_KEY'] = 'your-anon-key'
os.environ['GAMMA_API_KEY'] = 'your-openai-key'
os.environ['ZETA_API_KEY'] = 'your-cursor-key'

from core.service_facade import eco_service, EcoPlatformService
from core.interfaces import CapabilityContext


async def example_data_persistence():
    """
    數據持久化示例
    """
    print("=" * 50)
    print("數據持久化示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize(['alpha-persistence'])
    
    # 插入數據
    result = await eco_service.mutate_data(
        operation='insert',
        table='users',
        data={'name': 'John Doe', 'email': 'john@example.com'},
        provider='alpha-persistence'
    )
    print(f"Insert result: {result.success}")
    
    # 查詢數據
    result = await eco_service.query_data(
        table='users',
        filters={'email': 'john@example.com'},
        provider='alpha-persistence'
    )
    print(f"Query result: {result.data}")
    
    # 向量搜索
    query_vector = [0.1, 0.2, 0.3, 0.4]  # 示例向量
    result = await eco_service.vector_search(
        table='documents',
        vector=query_vector,
        top_k=5,
        provider='alpha-persistence'
    )
    print(f"Vector search result: {result.data}")


async def example_cognitive_compute():
    """
    認知計算示例
    """
    print("\n" + "=" * 50)
    print("認知計算示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize(['gamma-cognitive'])
    
    # 文本生成
    result = await eco_service.generate_text(
        prompt="Explain quantum computing in simple terms",
        parameters={'temperature': 0.7, 'max_tokens': 500},
        provider='gamma-cognitive'
    )
    print(f"Generated text: {result.data}")
    
    # 流式生成
    print("\n流式生成:")
    async for chunk in eco_service.generate_text_stream(
        prompt="Write a short poem about AI",
        provider='gamma-cognitive'
    ):
        if isinstance(chunk.content, str):
            print(chunk.content, end='', flush=True)
        if chunk.is_final:
            print()
    
    # 文本嵌入
    texts = ["Hello world", "Machine learning", "Cloud computing"]
    result = await eco_service.embed_texts(
        texts=texts,
        provider='gamma-cognitive'
    )
    print(f"\nEmbeddings count: {len(result.data.get('embeddings', []))}")
    
    # 函數調用
    functions = [
        {
            'name': 'get_weather',
            'description': 'Get weather for a location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {'type': 'string'},
                    'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}
                },
                'required': ['location']
            }
        }
    ]
    result = await eco_service.function_call(
        prompt="What's the weather in Tokyo?",
        functions=functions,
        provider='gamma-cognitive'
    )
    print(f"Function call result: {result.data}")


async def example_code_engineering():
    """
    代碼工程示例
    """
    print("\n" + "=" * 50)
    print("代碼工程示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize(['zeta-code'])
    
    # 代碼補全
    code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    """
    result = await eco_service.complete_code(
        code=code,
        language='python',
        cursor_position=len(code),
        provider='zeta-code'
    )
    print(f"Code completion: {result.data}")
    
    # 代碼解釋
    code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
    result = await eco_service.explain_code(
        code=code,
        language='python',
        provider='zeta-code'
    )
    print(f"Code explanation: {result.data}")
    
    # 代碼審查
    code = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""
    result = await eco_service.review_code(
        code=code,
        language='python',
        review_type='security',
        provider='zeta-code'
    )
    print(f"Code review: {result.data}")


async def example_collaboration():
    """
    協作通信示例
    """
    print("\n" + "=" * 50)
    print("協作通信示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize(['iota-collaboration'])
    
    # 發送消息
    result = await eco_service.send_message(
        channel='#general',
        content='Hello team! Deployment completed successfully.',
        provider='iota-collaboration'
    )
    print(f"Message sent: {result.success}")
    
    # 頻道摘要
    result = await eco_service.summarize_channel(
        channel='#engineering',
        provider='iota-collaboration'
    )
    print(f"Channel summary: {result.data}")


async def example_deployment():
    """
    部署交付示例
    """
    print("\n" + "=" * 50)
    print("部署交付示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize(['omicron-deployment'])
    
    # 部署應用
    result = await eco_service.deploy(
        artifact_path='my-app',
        environment='production',
        version='v1.2.3',
        config_overrides={'DEBUG': 'false'},
        provider='omicron-deployment'
    )
    print(f"Deployment result: {result.data}")


async def example_multi_provider():
    """
    多提供者組合使用示例
    """
    print("\n" + "=" * 50)
    print("多提供者組合示例")
    print("=" * 50)
    
    # 初始化所有服務
    await eco_service.initialize()
    
    # 1. 從數據庫獲取用戶查詢
    result = await eco_service.query_data(
        table='user_queries',
        filters={'status': 'pending'},
        provider='alpha-persistence'
    )
    queries = result.data or []
    
    # 2. 使用 AI 生成回覆
    for query in queries[:3]:  # 處理前3個
        result = await eco_service.generate_text(
            prompt=f"User asked: {query.get('question')}\nProvide a helpful response:",
            provider='gamma-cognitive'
        )
        response = result.data.get('content') if result.success else "Error generating response"
        
        # 3. 更新數據庫
        await eco_service.mutate_data(
            operation='update',
            table='user_queries',
            data={'response': response, 'status': 'answered'},
            conditions={'id': query.get('id')},
            provider='alpha-persistence'
        )
        
        print(f"Processed query {query.get('id')}: {response[:100]}...")
    
    # 4. 發送通知
    await eco_service.send_message(
        channel='#support',
        content=f'Processed {len(queries)} user queries',
        provider='iota-collaboration'
    )


async def example_health_check():
    """
    健康檢查示例
    """
    print("\n" + "=" * 50)
    print("健康檢查示例")
    print("=" * 50)
    
    # 初始化服務
    await eco_service.initialize()
    
    # 執行健康檢查
    health = await eco_service.health_check()
    
    for provider_id, status in health.items():
        healthy = status.get('healthy', False)
        error = status.get('error')
        print(f"{provider_id}: {'✅' if healthy else '❌'} {error or 'OK'}")


async def main():
    """
    主函數
    """
    print("\n" + "=" * 60)
    print("Eco-Platform Integration Framework - Examples")
    print("=" * 60 + "\n")
    
    try:
        # 運行示例
        # await example_data_persistence()
        # await example_cognitive_compute()
        # await example_code_engineering()
        # await example_collaboration()
        # await example_deployment()
        # await example_multi_provider()
        await example_health_check()
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
