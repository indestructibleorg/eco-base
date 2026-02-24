# =============================================================================
# Platform Configuration Templates
# =============================================================================
# 第三方平台配置模板
# 所有敏感信息應通過環境變數或 Secret Manager 注入
# =============================================================================

from typing import Any, Dict
import os


# =============================================================================
# 配置模板
# =============================================================================

DATA_PERSISTENCE_CONFIGS = {
    'alpha-persistence': {
        'url': os.getenv('SUPABASE_URL') or os.getenv('ALPHA_URL'),
        'anon_key': os.getenv('SUPABASE_ANON_KEY') or os.getenv('ALPHA_ANON_KEY'),
        'service_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_API_KEY') or os.getenv('ALPHA_SERVICE_KEY'),
    },
    'beta-persistence': {
        'host': os.getenv('BETA_HOST'),
        'username': os.getenv('BETA_USERNAME'),
        'password': os.getenv('BETA_PASSWORD'),
        'database': os.getenv('BETA_DATABASE'),
    },
}

COGNITIVE_COMPUTE_CONFIGS = {
    'gamma-cognitive': {
        'api_key': os.getenv('OPENAI_API_KEY') or os.getenv('GAMMA_API_KEY'),
    },
    'delta-cognitive': {
        'api_key': os.getenv('DELTA_API_KEY'),
    },
    'epsilon-cognitive': {
        'api_key': os.getenv('EPSILON_API_KEY'),
        'base_url': os.getenv('EPSILON_BASE_URL', 'https://api.epsilon.ai/v1'),
    },
}

CODE_ENGINEERING_CONFIGS = {
    'zeta-code': {
        'api_key': os.getenv('ZETA_API_KEY'),
        'base_url': os.getenv('ZETA_BASE_URL', 'https://api.zeta.dev/v1'),
    },
    'eta-code': {
        'api_key': os.getenv('ETA_API_KEY'),
        'base_url': os.getenv('ETA_BASE_URL', 'https://api.eta.dev/v1'),
    },
    'theta-code': {
        'api_key': os.getenv('THETA_API_KEY'),
        'base_url': os.getenv('THETA_BASE_URL', 'https://api.theta.dev/v1'),
    },
}

COLLABORATION_CONFIGS = {
    'iota-collaboration': {
        'bot_token': os.getenv('SLACK_API_KEY') or os.getenv('IOTA_BOT_TOKEN'),
        'channel': os.getenv('SLACK_CHANNEL'),
    },
    'kappa-collaboration': {
        'token': os.getenv('GITHUB_TOKEN') or os.getenv('KAPPA_TOKEN'),
        'owner': os.getenv('GITHUB_OWNER'),
        'repo': os.getenv('GITHUB_REPO'),
    },
}

VISUAL_DESIGN_CONFIGS = {
    'lambda-visual': {
        'token': os.getenv('LAMBDA_TOKEN'),
    },
    'mu-visual': {
        'token': os.getenv('MU_TOKEN'),
    },
}

KNOWLEDGE_MGMT_CONFIGS = {
    'nu-knowledge': {
        'token': os.getenv('NU_TOKEN'),
    },
    'xi-knowledge': {
        'token': os.getenv('XI_TOKEN'),
    },
}

DEPLOYMENT_CONFIGS = {
    'omicron-deployment': {
        'token': os.getenv('VERCEL_API_KEY') or os.getenv('OMICRON_TOKEN'),
        'team_id': os.getenv('VERCEL_TEAM_ID') or os.getenv('OMICRON_TEAM_ID'),
    },
    'pi-deployment': {
        'token': os.getenv('PI_TOKEN'),
        'org_id': os.getenv('PI_ORG_ID'),
    },
    'rho-deployment': {
        'token': os.getenv('RHO_TOKEN'),
        'org_id': os.getenv('RHO_ORG_ID'),
    },
}

LEARNING_CONFIGS = {
    'sigma-learning': {
        'api_key': os.getenv('SIGMA_API_KEY'),
    },
    'tau-learning': {
        'api_key': os.getenv('TAU_API_KEY'),
    },
    'upsilon-learning': {
        'api_key': os.getenv('UPSILON_API_KEY'),
    },
}


# =============================================================================
# 配置獲取函數
# =============================================================================

def get_provider_config(provider_id: str) -> Dict[str, Any]:
    """
    獲取指定提供者的配置
    
    Args:
        provider_id: 提供者ID
        
    Returns:
        配置字典
    """
    all_configs = {
        **DATA_PERSISTENCE_CONFIGS,
        **COGNITIVE_COMPUTE_CONFIGS,
        **CODE_ENGINEERING_CONFIGS,
        **COLLABORATION_CONFIGS,
        **VISUAL_DESIGN_CONFIGS,
        **KNOWLEDGE_MGMT_CONFIGS,
        **DEPLOYMENT_CONFIGS,
        **LEARNING_CONFIGS,
    }
    
    config = all_configs.get(provider_id, {})
    
    # 過濾掉 None 值
    return {k: v for k, v in config.items() if v is not None}


def get_all_configs() -> Dict[str, Dict[str, Any]]:
    """
    獲取所有配置
    
    Returns:
        所有配置字典
    """
    return {
        **DATA_PERSISTENCE_CONFIGS,
        **COGNITIVE_COMPUTE_CONFIGS,
        **CODE_ENGINEERING_CONFIGS,
        **COLLABORATION_CONFIGS,
        **VISUAL_DESIGN_CONFIGS,
        **KNOWLEDGE_MGMT_CONFIGS,
        **DEPLOYMENT_CONFIGS,
        **LEARNING_CONFIGS,
    }


def validate_config(provider_id: str, config: Dict[str, Any]) -> bool:
    """
    驗證配置是否完整
    
    Args:
        provider_id: 提供者ID
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_fields = {
        'alpha-persistence': ['url', 'anon_key'],
        'beta-persistence': ['host', 'username', 'password'],
        'gamma-cognitive': ['api_key'],
        'delta-cognitive': ['api_key'],
        'epsilon-cognitive': ['api_key'],
        'zeta-code': ['api_key'],
        'eta-code': ['api_key'],
        'theta-code': ['api_key'],
        'iota-collaboration': ['bot_token'],
        'kappa-collaboration': ['token'],
        'lambda-visual': ['token'],
        'mu-visual': ['token'],
        'nu-knowledge': ['token'],
        'xi-knowledge': ['token'],
        'omicron-deployment': ['token'],
        'pi-deployment': ['token'],
        'rho-deployment': ['token'],
        'sigma-learning': ['api_key'],
        'tau-learning': ['api_key'],
        'upsilon-learning': ['api_key'],
    }
    
    required = required_fields.get(provider_id, [])
    return all(field in config and config[field] for field in required)
