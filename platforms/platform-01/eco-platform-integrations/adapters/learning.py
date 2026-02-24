# =============================================================================
# Learning Adapters
# =============================================================================
# 學習教育適配器實現
# 封裝學習平台服務調用
# =============================================================================

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    ILearningProvider, CapabilityContext, OperationResult,
    LearningPath, ExerciseSubmission, CapabilityDomain
)


class SigmaLearningAdapter(ILearningProvider):
    """
    Sigma 學習教育適配器
    互動式程式學習平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = 'https://api.sigma.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.LEARNING
    
    @property
    def provider_id(self) -> str:
        return "sigma-learning"
    
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
        return ['path', 'submit', 'hint', 'progress']
    
    async def get_learning_path(self, topic: str, skill_level: str, ctx: CapabilityContext) -> OperationResult:
        """獲取學習路徑"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/tracks',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    params={'topic': topic, 'level': skill_level}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'tracks': data.get('tracks', []),
                        'estimated_hours': sum(t.get('duration', 0) for t in data.get('tracks', []))
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def submit_exercise(self, submission: ExerciseSubmission, ctx: CapabilityContext) -> OperationResult:
        """提交練習"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/submissions',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'exercise_id': submission.exercise_id,
                        'code': submission.code,
                        'language': submission.language
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'submission_id': data.get('id'),
                        'status': data.get('status'),
                        'test_results': data.get('test_results', []),
                        'passed': data.get('passed', False)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_hint(self, exercise_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取提示"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/exercises/{exercise_id}/hints',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'hints': data.get('hints', []),
                        'ai_hint': data.get('ai_hint')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def track_progress(self, user_id: str, ctx: CapabilityContext) -> OperationResult:
        """追蹤學習進度"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/users/{user_id}/progress',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'completed_exercises': data.get('completed', 0),
                        'streak_days': data.get('streak', 0),
                        'certificates': data.get('certificates', []),
                        'skills': data.get('skills', {})
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class TauLearningAdapter(ILearningProvider):
    """
    Tau 學習教育適配器
    雲端 IDE 與協作學習平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = 'https://api.tau.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.LEARNING
    
    @property
    def provider_id(self) -> str:
        return "tau-learning"
    
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
        return ['path', 'submit', 'collaborate']
    
    async def get_learning_path(self, topic: str, skill_level: str, ctx: CapabilityContext) -> OperationResult:
        """獲取學習路徑"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/courses',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    params={'topic': topic, 'level': skill_level}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={'courses': data.get('courses', [])}
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def submit_exercise(self, submission: ExerciseSubmission, ctx: CapabilityContext) -> OperationResult:
        """提交練習"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/repls/run',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'language': submission.language,
                        'code': submission.code
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'output': data.get('output'),
                        'error': data.get('error'),
                        'success': data.get('success')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_hint(self, exercise_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取提示 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Hints not supported in Tau adapter'
        )
    
    async def track_progress(self, user_id: str, ctx: CapabilityContext) -> OperationResult:
        """追蹤學習進度"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/users/{user_id}',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'repls_count': len(data.get('repls', [])),
                        'followers': data.get('followers', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class UpsilonLearningAdapter(ILearningProvider):
    """
    Upsilon 學習教育適配器
    前端實驗與展示平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._api_key = config.get('api_key')
        self._base_url = 'https://api.upsilon.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.LEARNING
    
    @property
    def provider_id(self) -> str:
        return "upsilon-learning"
    
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
        return ['submit', 'collaborate']
    
    async def get_learning_path(self, topic: str, skill_level: str, ctx: CapabilityContext) -> OperationResult:
        """獲取學習路徑 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Learning paths not supported in Upsilon adapter'
        )
    
    async def submit_exercise(self, submission: ExerciseSubmission, ctx: CapabilityContext) -> OperationResult:
        """提交練習 (創建 Pen)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/pen',
                    headers={'Authorization': f'Bearer {self._api_key}'},
                    json={
                        'title': f'Exercise {submission.exercise_id}',
                        'html': submission.code if submission.language == 'html' else '',
                        'css': submission.code if submission.language == 'css' else '',
                        'js': submission.code if submission.language == 'javascript' else ''
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'pen_id': data.get('id'),
                        'url': data.get('link')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_hint(self, exercise_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取提示 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Hints not supported in Upsilon adapter'
        )
    
    async def track_progress(self, user_id: str, ctx: CapabilityContext) -> OperationResult:
        """追蹤學習進度"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/user/{user_id}',
                    headers={'Authorization': f'Bearer {self._api_key}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'pens_count': data.get('pens_count', 0),
                        'followers': data.get('followers', 0)
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
