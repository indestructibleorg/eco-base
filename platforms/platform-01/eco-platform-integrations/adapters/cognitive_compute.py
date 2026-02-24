# =============================================================================
# Cognitive Computing Adapters
# =============================================================================
# 認知計算適配器實現
# 封裝 LLM/AI 服務調用
# =============================================================================

import json
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime
import httpx
import asyncio

from core.interfaces import (
    ICognitiveComputeProvider, CapabilityContext, OperationResult,
    StreamChunk, InferenceRequest, AgentTask, CapabilityDomain
)


class GammaCognitiveAdapter(ICognitiveComputeProvider):
    """
    Gamma 認知計算適配器
    多模型聚合平台，支持模型切換
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = 'https://api.gamma.ai/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COGNITIVE_COMPUTE
    
    @property
    def provider_id(self) -> str:
        return "gamma-cognitive"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/models',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['generate', 'stream', 'function_call', 'embed', 'multimodal', 'bot_creation']
    
    async def generate(self, request: InferenceRequest, ctx: CapabilityContext) -> OperationResult:
        """單次生成"""
        start_time = datetime.utcnow()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': request.parameters.get('model', 'gpt-4o'),
                        'messages': self._build_messages(request),
                        'temperature': request.parameters.get('temperature', 0.7),
                        'max_tokens': request.parameters.get('max_tokens', 2000)
                    },
                    timeout=60.0
                )
                
                data = response.json()
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': data['choices'][0]['message']['content'],
                        'model': data.get('model'),
                        'usage': data.get('usage')
                    },
                    latency_ms=latency
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_stream(self, request: InferenceRequest, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """流式生成"""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    'POST',
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': request.parameters.get('model', 'gpt-4o'),
                        'messages': self._build_messages(request),
                        'stream': True
                    },
                    timeout=60.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                yield StreamChunk(content='', is_final=True)
                                break
                            try:
                                chunk = json.loads(data)
                                content = chunk['choices'][0]['delta'].get('content', '')
                                if content:
                                    yield StreamChunk(content=content, is_final=False)
                            except:
                                pass
        except Exception as e:
            yield StreamChunk(content={'error': str(e)}, is_final=True)
    
    async def function_call(self, prompt: str, functions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """函數調用"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'gpt-4o',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'functions': functions,
                        'function_call': 'auto'
                    }
                )
                
                data = response.json()
                message = data['choices'][0]['message']
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': message.get('content'),
                        'function_call': message.get('function_call')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def execute_agent_task(self, task: AgentTask, ctx: CapabilityContext) -> OperationResult:
        """執行代理任務"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/agent/execute',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'task_type': task.task_type,
                        'description': task.description,
                        'inputs': task.inputs,
                        'output_format': task.expected_output_format
                    }
                )
                return OperationResult(
                    success=response.status_code == 200,
                    data=response.json()
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def embed(self, texts: List[str], ctx: CapabilityContext) -> OperationResult:
        """文本嵌入"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/embeddings',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'text-embedding-3-small',
                        'input': texts
                    }
                )
                
                data = response.json()
                embeddings = [item['embedding'] for item in data['data']]
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={'embeddings': embeddings}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def multimodal_process(self, inputs: Dict[str, Any], ctx: CapabilityContext) -> OperationResult:
        """多模態處理"""
        try:
            messages = []
            
            if 'text' in inputs:
                messages.append({
                    'type': 'text',
                    'text': inputs['text']
                })
            
            if 'image_url' in inputs:
                messages.append({
                    'type': 'image_url',
                    'image_url': {'url': inputs['image_url']}
                })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'gpt-4o-vision',
                        'messages': [{'role': 'user', 'content': messages}]
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'content': data['choices'][0]['message']['content']}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    def _build_messages(self, request: InferenceRequest) -> List[Dict]:
        """構建消息列表"""
        messages = []
        
        if request.context:
            for msg in request.context:
                messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content')
                })
        
        messages.append({
            'role': 'user',
            'content': request.prompt
        })
        
        return messages


class DeltaCognitiveAdapter(ICognitiveComputeProvider):
    """
    Delta 認知計算適配器
    企業級推理模型，支持長上下文
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = 'https://api.delta.ai/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COGNITIVE_COMPUTE
    
    @property
    def provider_id(self) -> str:
        return "delta-cognitive"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/health',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['generate', 'stream', 'long_context', 'coding', 'reasoning']
    
    async def generate(self, request: InferenceRequest, ctx: CapabilityContext) -> OperationResult:
        """單次生成"""
        try:
            # 支持 200K 上下文
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/messages',
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-sonnet',
                        'max_tokens': request.parameters.get('max_tokens', 4096),
                        'messages': [{'role': 'user', 'content': request.prompt}],
                        'temperature': request.parameters.get('temperature', 0.7)
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': data['content'][0]['text'],
                        'usage': data.get('usage')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_stream(self, request: InferenceRequest, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """流式生成"""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    'POST',
                    f'{self._base_url}/messages',
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-sonnet',
                        'max_tokens': 4096,
                        'messages': [{'role': 'user', 'content': request.prompt}],
                        'stream': True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                yield StreamChunk(content='', is_final=True)
                                break
                            try:
                                event = json.loads(data)
                                if event.get('type') == 'content_block_delta':
                                    yield StreamChunk(
                                        content=event['delta'].get('text', ''),
                                        is_final=False
                                    )
                            except:
                                pass
        except Exception as e:
            yield StreamChunk(content={'error': str(e)}, is_final=True)
    
    async def function_call(self, prompt: str, functions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """工具調用"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/messages',
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-sonnet',
                        'max_tokens': 4096,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'tools': [{'type': 'function', 'function': f} for f in functions]
                    }
                )
                
                data = response.json()
                content = data['content']
                
                # 檢查是否有工具調用
                tool_calls = [c for c in content if c['type'] == 'tool_use']
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': [c for c in content if c['type'] == 'text'][0].get('text') if not tool_calls else None,
                        'tool_calls': tool_calls
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def execute_agent_task(self, task: AgentTask, ctx: CapabilityContext) -> OperationResult:
        """執行代理任務"""
        # Delta 支持複雜多步驟任務
        return await self.generate(
            InferenceRequest(
                prompt=f"Task: {task.description}\nInputs: {json.dumps(task.inputs)}\nPlease complete this task.",
                parameters={'max_tokens': 8000}
            ),
            ctx
        )
    
    async def embed(self, texts: List[str], ctx: CapabilityContext) -> OperationResult:
        """文本嵌入"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/embeddings',
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-embedding',
                        'input': texts
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'embeddings': data.get('embeddings', [])}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def multimodal_process(self, inputs: Dict[str, Any], ctx: CapabilityContext) -> OperationResult:
        """多模態處理"""
        # Delta 支持圖像理解
        content = []
        
        if 'text' in inputs:
            content.append({'type': 'text', 'text': inputs['text']})
        
        if 'image_url' in inputs:
            content.append({
                'type': 'image',
                'source': {
                    'type': 'url',
                    'url': inputs['image_url']
                }
            })
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/messages',
                    headers={
                        'Authorization': f'Bearer {self._api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-sonnet',
                        'max_tokens': 4096,
                        'messages': [{'role': 'user', 'content': content}]
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'content': data['content'][0]['text']}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class EpsilonCognitiveAdapter(ICognitiveComputeProvider):
    """
    Epsilon 認知計算適配器
    開源推理模型，成本效益高
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = config.get('base_url', 'https://api.epsilon.ai/v1')
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.COGNITIVE_COMPUTE
    
    @property
    def provider_id(self) -> str:
        return "epsilon-cognitive"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/models',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['generate', 'stream', 'long_context', 'reasoning', 'coding']
    
    async def generate(self, request: InferenceRequest, ctx: CapabilityContext) -> OperationResult:
        """單次生成"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'deepseek-reasoner',
                        'messages': [{'role': 'user', 'content': request.prompt}],
                        'max_tokens': request.parameters.get('max_tokens', 4096)
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': data['choices'][0]['message']['content'],
                        'reasoning_content': data['choices'][0]['message'].get('reasoning_content')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_stream(self, request: InferenceRequest, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """流式生成"""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    'POST',
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'deepseek-reasoner',
                        'messages': [{'role': 'user', 'content': request.prompt}],
                        'stream': True,
                        'max_tokens': request.parameters.get('max_tokens', 4096)
                    },
                    timeout=60.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                yield StreamChunk(content='', is_final=True)
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                reasoning = delta.get('reasoning_content', '')
                                if content or reasoning:
                                    yield StreamChunk(
                                        content={'content': content, 'reasoning': reasoning},
                                        is_final=False
                                    )
                            except:
                                pass
        except Exception as e:
            yield StreamChunk(content={'error': str(e)}, is_final=True)
    
    async def function_call(self, prompt: str, functions: List[Dict], ctx: CapabilityContext) -> OperationResult:
        """函數調用"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'deepseek-chat',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'functions': functions,
                        'function_call': 'auto'
                    }
                )
                
                data = response.json()
                message = data['choices'][0]['message']
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'content': message.get('content'),
                        'function_call': message.get('function_call')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def execute_agent_task(self, task: AgentTask, ctx: CapabilityContext) -> OperationResult:
        """執行代理任務"""
        try:
            # Epsilon 使用推理模型執行複雜任務
            system_prompt = f"""You are an AI agent. Complete the following task:
Task Type: {task.task_type}
Description: {task.description}
Expected Output Format: {task.expected_output_format}

Please provide a detailed response."""
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/chat/completions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'deepseek-reasoner',
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': json.dumps(task.inputs)}
                        ],
                        'max_tokens': 8000
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'result': data['choices'][0]['message']['content'],
                        'reasoning': data['choices'][0]['message'].get('reasoning_content')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def embed(self, texts: List[str], ctx: CapabilityContext) -> OperationResult:
        """文本嵌入"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/embeddings',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'model': 'deepseek-embedding',
                        'input': texts
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'embeddings': [item['embedding'] for item in data['data']]}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def multimodal_process(self, inputs: Dict[str, Any], ctx: CapabilityContext) -> OperationResult:
        """多模態處理"""
        return OperationResult(
            success=False,
            error_message='Multimodal not supported in Epsilon adapter'
        )
