# =============================================================================
# Deployment Adapters
# =============================================================================
# 部署交付適配器實現
# 封裝 CI/CD、部署服務調用
# =============================================================================

import json
import base64
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from core.interfaces import (
    IDeploymentProvider, CapabilityContext, OperationResult,
    DeploySpec, BuildSpec, CapabilityDomain
)


class OmicronDeploymentAdapter(IDeploymentProvider):
    """
    Omicron 部署交付適配器
    前端部署與 Serverless 平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._team_id = config.get('team_id')
        self._base_url = 'https://api.omicron.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.DEPLOYMENT
    
    @property
    def provider_id(self) -> str:
        return "omicron-deployment"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/user',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['build', 'deploy', 'preview', 'rollback']
    
    async def build(self, spec: BuildSpec, ctx: CapabilityContext) -> OperationResult:
        """構建制品"""
        try:
            async with httpx.AsyncClient() as client:
                # 上傳源代碼並觸發構建
                response = await client.post(
                    f'{self._base_url}/v13/deployments',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'name': spec.source_path,
                        'target': spec.target_platform
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'build_id': data.get('id'),
                        'status': data.get('state')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def deploy(self, spec: DeploySpec, ctx: CapabilityContext) -> OperationResult:
        """部署到環境"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/v13/deployments',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'name': spec.artifact_path,
                        'target': spec.environment,
                        'env': spec.config_overrides or {}
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'deployment_id': data.get('id'),
                        'url': data.get('url'),
                        'status': data.get('state')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_deployment_status(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取部署狀態"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/v13/deployments/{deployment_id}',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'status': data.get('state'),
                        'url': data.get('url'),
                        'created_at': data.get('created'),
                        'ready_state': data.get('readyState')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def rollback(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """回滾部署"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/v13/deployments/{deployment_id}/rollback',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def preview_deployment(self, branch: str, ctx: CapabilityContext) -> OperationResult:
        """預覽部署 (Preview URL)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/v13/deployments',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'gitSource': {
                            'ref': branch,
                            'type': 'github'
                        },
                        'target': 'preview'
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'preview_url': data.get('url'),
                        'deployment_id': data.get('id')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class PiDeploymentAdapter(IDeploymentProvider):
    """
    Pi 部署交付適配器
    構建加速與 CI/CD 優化平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._org_id = config.get('org_id')
        self._base_url = 'https://api.pi.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.DEPLOYMENT
    
    @property
    def provider_id(self) -> str:
        return "pi-deployment"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/health',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['build', 'cache', 'parallel']
    
    async def build(self, spec: BuildSpec, ctx: CapabilityContext) -> OperationResult:
        """構建制品 (遠程構建加速)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/builds',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'context': spec.source_path,
                        'platform': spec.target_platform,
                        'cache_key': spec.cache_key,
                        'parallel_jobs': spec.parallel_jobs
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'build_id': data.get('buildId'),
                        'status': data.get('status')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def deploy(self, spec: DeploySpec, ctx: CapabilityContext) -> OperationResult:
        """部署到環境 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Direct deployment not supported in Pi adapter'
        )
    
    async def get_deployment_status(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取部署狀態"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/builds/{deployment_id}',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'status': data.get('status'),
                        'duration': data.get('duration'),
                        'cache_hit': data.get('cacheHit')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def rollback(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """回滾部署 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Rollback not supported in Pi adapter'
        )
    
    async def preview_deployment(self, branch: str, ctx: CapabilityContext) -> OperationResult:
        """預覽部署 (不支持)"""
        return OperationResult(
            success=False,
            error_message='Preview not supported in Pi adapter'
        )


class RhoDeploymentAdapter(IDeploymentProvider):
    """
    Rho 部署交付適配器
    基礎設施即代碼 (IaC) 平台
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._token = config.get('token')
        self._org_id = config.get('org_id')
        self._base_url = 'https://api.rho.dev/v1'
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.DEPLOYMENT
    
    @property
    def provider_id(self) -> str:
        return "rho-deployment"
    
    async def health_check(self) -> OperationResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/account/details',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                return OperationResult(success=response.status_code == 200)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['build', 'deploy', 'plan', 'state']
    
    async def build(self, spec: BuildSpec, ctx: CapabilityContext) -> OperationResult:
        """構建制品 (Terraform Plan)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/runs',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'workspace_id': spec.source_path,
                        'message': 'Triggered via API'
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'run_id': data.get('data', {}).get('id'),
                        'status': data.get('data', {}).get('attributes', {}).get('status')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def deploy(self, spec: DeploySpec, ctx: CapabilityContext) -> OperationResult:
        """部署到環境 (Terraform Apply)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/runs/{spec.artifact_path}/actions/apply',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                return OperationResult(success=response.status_code == 202)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_deployment_status(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """獲取部署狀態"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f'{self._base_url}/runs/{deployment_id}',
                    headers={'Authorization': f'Bearer {self._token}'}
                )
                
                data = response.json()
                attrs = data.get('data', {}).get('attributes', {})
                
                return OperationResult(
                    success=response.status_code == 200,
                    data={
                        'status': attrs.get('status'),
                        'plan_status': attrs.get('plan-only'),
                        'has_changes': attrs.get('has-changes')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def rollback(self, deployment_id: str, ctx: CapabilityContext) -> OperationResult:
        """回滾部署 (Terraform Destroy 或 State 恢復)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/runs',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'workspace_id': deployment_id,
                        'is-destroy': True
                    }
                )
                
                return OperationResult(success=response.status_code == 201)
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def preview_deployment(self, branch: str, ctx: CapabilityContext) -> OperationResult:
        """預覽部署 (Speculative Plan)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'{self._base_url}/runs',
                    headers={'Authorization': f'Bearer {self._token}'},
                    json={
                        'workspace_id': branch,
                        'plan-only': True
                    }
                )
                
                data = response.json()
                return OperationResult(
                    success=response.status_code == 201,
                    data={
                        'plan_id': data.get('data', {}).get('id')
                    }
                )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
