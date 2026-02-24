# =============================================================================
# Code Engineering Adapters
# =============================================================================
# 代碼工程適配器實現
# 封裝代碼編輯、審查、協作服務調用
# =============================================================================

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    ICodeEngineeringProvider, CapabilityContext, OperationResult,
    CodeContext, ReviewRequest, CapabilityDomain
)


class ZetaCodeAdapter(ICodeEngineeringProvider):
    """
    Zeta 代碼工程適配器
    AI 驅動的代碼編輯器服務
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = config.get('base_url', 'https://api.zeta.dev/v1')
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.CODE_ENGINEERING
    
    @property
    def provider_id(self) -> str:
        return "zeta-code"
    
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
        return ['complete', 'explain', 'refactor', 'review', 'generate_tests', 'translate']
    
    async def complete(self, context: CodeContext, ctx: CapabilityContext) -> OperationResult:
        """代碼補全"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/complete',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': context.content,
                        'language': context.language,
                        'cursor_position': context.cursor_position,
                        'file_path': context.file_path
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'completions': data.get('completions', []),
                        'suggested_code': data.get('suggested_code')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def explain(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """代碼解釋"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/explain',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'explanation': data.get('explanation'),
                        'key_points': data.get('key_points', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def refactor(self, code: str, language: str, instruction: str, ctx: CapabilityContext) -> OperationResult:
        """代碼重構"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/refactor',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language,
                        'instruction': instruction
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'refactored_code': data.get('refactored_code'),
                        'changes': data.get('changes', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def review(self, request: ReviewRequest, ctx: CapabilityContext) -> OperationResult:
        """代碼審查"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/review',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': request.code,
                        'language': request.language,
                        'review_type': request.review_type
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'issues': data.get('issues', []),
                        'suggestions': data.get('suggestions', []),
                        'score': data.get('score')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_tests(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """生成測試"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/generate-tests',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'test_code': data.get('test_code'),
                        'test_cases': data.get('test_cases', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def translate_language(self, code: str, source_lang: str, target_lang: str, ctx: CapabilityContext) -> OperationResult:
        """跨語言轉換"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/translate',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'source_language': source_lang,
                        'target_language': target_lang
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'translated_code': data.get('translated_code'),
                        'notes': data.get('notes', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def search_repository(self, query: str, repo_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """倉庫搜索"""
        try:
            async with httpx.AsyncClient() as client:
                params = {'q': query}
                if repo_id:
                    params['repo'] = repo_id
                
                response = await client.get(
                    f'{self._base_url}/search',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    params=params
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'results': data.get('results', []),
                        'total_count': data.get('total_count', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class EtaCodeAdapter(ICodeEngineeringProvider):
    """
    Eta 代碼工程適配器
    代碼審查與 CI 自動化服務
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = config.get('base_url', 'https://api.eta.dev/v1')
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.CODE_ENGINEERING
    
    @property
    def provider_id(self) -> str:
        return "eta-code"
    
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
        return ['review', 'complete', 'explain']
    
    async def complete(self, context: CodeContext, ctx: CapabilityContext) -> OperationResult:
        """代碼補全 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Completion not supported in Eta adapter'
        )
    
    async def explain(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """代碼解釋"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/explain',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'explanation': data.get('explanation'),
                        'key_points': data.get('key_points', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def refactor(self, code: str, language: str, instruction: str, ctx: CapabilityContext) -> OperationResult:
        """代碼重構 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Refactor not supported in Eta adapter'
        )
    
    async def review(self, request: ReviewRequest, ctx: CapabilityContext) -> OperationResult:
        """代碼審查 (核心功能)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/analyze',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': request.code,
                        'language': request.language,
                        'rules': self._get_rules_for_type(request.review_type)
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'issues': data.get('issues', []),
                        'violations': data.get('violations', []),
                        'metrics': data.get('metrics', {})
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_tests(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """生成測試"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/generate-tests',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'test_code': data.get('test_code'),
                        'test_cases': data.get('test_cases', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def translate_language(self, code: str, source_lang: str, target_lang: str, ctx: CapabilityContext) -> OperationResult:
        """跨語言轉換 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Language translation not supported in Eta adapter'
        )
    
    async def search_repository(self, query: str, repo_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """倉庫搜索 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Repository search not supported in Eta adapter'
        )
    
    def _get_rules_for_type(self, review_type: str) -> List[str]:
        """根據審查類型獲取規則"""
        rules_map = {
            'general': ['complexity', 'naming', 'documentation'],
            'security': ['sql-injection', 'xss', 'csrf', 'secrets'],
            'performance': ['n+1', 'memory-leak', 'inefficient-loop'],
            'style': ['indentation', 'naming-convention', 'max-line-length']
        }
        return rules_map.get(review_type, rules_map['general'])


class ThetaCodeAdapter(ICodeEngineeringProvider):
    """
    Theta 代碼工程適配器
    雲端 IDE 與協作編程服務
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = config.get('base_url', 'https://api.theta.dev/v1')
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.CODE_ENGINEERING
    
    @property
    def provider_id(self) -> str:
        return "theta-code"
    
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
        return ['complete', 'explain', 'generate_tests', 'collaborate']
    
    async def complete(self, context: CodeContext, ctx: CapabilityContext) -> OperationResult:
        """代碼補全"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/complete',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': context.content,
                        'language': context.language,
                        'cursor_line': context.cursor_position
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'completion': data.get('completion')}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def explain(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """代碼解釋"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/explain',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'explanation': data.get('explanation'),
                        'key_points': data.get('key_points', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def refactor(self, code: str, language: str, instruction: str, ctx: CapabilityContext) -> OperationResult:
        """代碼重構"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/refactor',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language,
                        'instruction': instruction
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'refactored_code': data.get('refactored_code'),
                        'changes': data.get('changes', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def review(self, request: ReviewRequest, ctx: CapabilityContext) -> OperationResult:
        """代碼審查"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/review',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': request.code,
                        'language': request.language,
                        'review_type': request.review_type
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'issues': data.get('issues', []),
                        'suggestions': data.get('suggestions', []),
                        'score': data.get('score')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def generate_tests(self, code: str, language: str, ctx: CapabilityContext) -> OperationResult:
        """生成測試"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/tests',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'language': language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'tests': data.get('tests', []),
                        'test_file': data.get('test_file')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def translate_language(self, code: str, source_lang: str, target_lang: str, ctx: CapabilityContext) -> OperationResult:
        """跨語言轉換"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/ghostwriter/translate',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'code': code,
                        'source_language': source_lang,
                        'target_language': target_lang
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'translated_code': data.get('translated_code'),
                        'notes': data.get('notes', [])
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def search_repository(self, query: str, repo_id: Optional[str], ctx: CapabilityContext) -> OperationResult:
        """倉庫搜索"""
        try:
            async with httpx.AsyncClient() as client:
                params = {'q': query}
                if repo_id:
                    params['repl_id'] = repo_id
                
                response = await client.get(
                    f'{self._base_url}/search',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    params=params
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'results': data.get('results', []),
                        'total_count': data.get('total_count', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
