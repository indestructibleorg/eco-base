# =============================================================================
# Eco-Backend Configuration
# =============================================================================
# 生產級後端配置管理
# =============================================================================

from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """應用配置類"""
    
    # =================================================================
    # 應用基礎配置
    # =================================================================
    APP_NAME: str = Field(default="eco-backend", description="應用名稱")
    APP_VERSION: str = Field(default="1.0.0", description="應用版本")
    DEBUG: bool = Field(default=False, description="調試模式")
    ENVIRONMENT: str = Field(default="production", description="運行環境")
    
    # =================================================================
    # 服務器配置
    # =================================================================
    HOST: str = Field(default="0.0.0.0", description="服務器監聽地址")
    PORT: int = Field(default=8000, description="服務器端口")
    WORKERS: int = Field(default=4, description="工作進程數")
    
    # =================================================================
    # 數據庫配置
    # =================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:pass@localhost/eco_db",
        description="數據庫連接URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="連接池大小")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="連接池溢出限制")
    DATABASE_ECHO: bool = Field(default=False, description="SQL日誌輸出")
    
    # =================================================================
    # Redis配置
    # =================================================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis連接URL"
    )
    REDIS_POOL_SIZE: int = Field(default=50, description="Redis連接池大小")
    
    # =================================================================
    # 認證配置
    # =================================================================
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT密鑰"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT算法")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="訪問令牌過期時間(分鐘)"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        description="刷新令牌過期時間(天)"
    )
    
    # =================================================================
    # CORS配置
    # =================================================================
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="允許的CORS來源"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # =================================================================
    # 限流配置
    # =================================================================
    RATE_LIMIT_DEFAULT: str = Field(
        default="100/minute",
        description="默認限流規則"
    )
    RATE_LIMIT_AUTH: str = Field(
        default="5/minute",
        description="認證端點限流"
    )
    RATE_LIMIT_AI: str = Field(
        default="10/minute",
        description="AI端點限流"
    )
    
    # =================================================================
    # 熔斷配置
    # =================================================================
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5)
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60)
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = Field(default="Exception")
    
    # =================================================================
    # 日誌配置
    # =================================================================
    LOG_LEVEL: str = Field(default="INFO", description="日誌級別")
    LOG_FORMAT: str = Field(default="json", description="日誌格式")
    LOG_FILE: Optional[str] = Field(default=None, description="日誌文件路徑")
    
    # =================================================================
    # 監控配置
    # =================================================================
    METRICS_ENABLED: bool = Field(default=True, description="啟用指標收集")
    METRICS_PORT: int = Field(default=9090, description="指標端口")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="健康檢查間隔")
    
    # =================================================================
    # Celery配置
    # =================================================================
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=False)
    CELERY_WORKER_CONCURRENCY: int = Field(default=4)
    
    # =================================================================
    # 第三方平台配置 (通過環境變數注入)
    # =================================================================
    # 數據持久化
    ALPHA_URL: Optional[str] = Field(default=None)
    ALPHA_ANON_KEY: Optional[str] = Field(default=None)
    ALPHA_SERVICE_KEY: Optional[str] = Field(default=None)
    
    # 認知計算
    GAMMA_API_KEY: Optional[str] = Field(default=None)
    DELTA_API_KEY: Optional[str] = Field(default=None)
    EPSILON_API_KEY: Optional[str] = Field(default=None)
    
    # 代碼工程
    ZETA_API_KEY: Optional[str] = Field(default=None)
    ETA_API_KEY: Optional[str] = Field(default=None)
    THETA_API_KEY: Optional[str] = Field(default=None)
    
    # 協作通信
    IOTA_BOT_TOKEN: Optional[str] = Field(default=None)
    KAPPA_TOKEN: Optional[str] = Field(default=None)
    
    # 視覺設計
    LAMBDA_TOKEN: Optional[str] = Field(default=None)
    MU_TOKEN: Optional[str] = Field(default=None)
    
    # 知識管理
    NU_TOKEN: Optional[str] = Field(default=None)
    XI_TOKEN: Optional[str] = Field(default=None)
    
    # 部署交付
    OMICRON_TOKEN: Optional[str] = Field(default=None)
    PI_TOKEN: Optional[str] = Field(default=None)
    RHO_TOKEN: Optional[str] = Field(default=None)
    
    # 學習教育
    SIGMA_API_KEY: Optional[str] = Field(default=None)
    TAU_API_KEY: Optional[str] = Field(default=None)
    UPSILON_API_KEY: Optional[str] = Field(default=None)
    
    # =================================================================
    # 平台集成配置 (Platform Integration)
    # =================================================================
    # Supabase
    SUPABASE_API_KEY: Optional[str] = Field(default=None, description="Supabase API Key")
    SUPABASE_URL: Optional[str] = Field(default=None, description="Supabase URL")
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENAI_MODEL: str = Field(default="gpt-4", description="OpenAI Model")
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API Key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone Environment")
    
    # GitHub
    GITHUB_API_KEY: Optional[str] = Field(default=None, description="GitHub API Key")
    GITHUB_OWNER: Optional[str] = Field(default=None, description="GitHub Repository Owner")
    GITHUB_REPO: Optional[str] = Field(default=None, description="GitHub Repository Name")
    
    # Slack
    SLACK_API_KEY: Optional[str] = Field(default=None, description="Slack API Key")
    SLACK_CHANNEL: Optional[str] = Field(default=None, description="Slack Channel")
    
    # Vercel
    VERCEL_API_KEY: Optional[str] = Field(default=None, description="Vercel API Key")
    VERCEL_TEAM_ID: Optional[str] = Field(default=None, description="Vercel Team ID")
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """解析CORS來源列表"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def database_async_url(self) -> str:
        """獲取異步數據庫URL"""
        return self.DATABASE_URL
    
    @property
    def is_development(self) -> bool:
        """是否為開發環境"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """是否為生產環境"""
        return self.ENVIRONMENT.lower() == "production"
    
    def get_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有第三方平台配置"""
        return {
            "alpha-persistence": {
                "url": self.ALPHA_URL,
                "anon_key": self.ALPHA_ANON_KEY,
                "service_key": self.ALPHA_SERVICE_KEY,
            },
            "gamma-cognitive": {
                "api_key": self.GAMMA_API_KEY,
            },
            "delta-cognitive": {
                "api_key": self.DELTA_API_KEY,
            },
            "epsilon-cognitive": {
                "api_key": self.EPSILON_API_KEY,
            },
            "zeta-code": {
                "api_key": self.ZETA_API_KEY,
            },
            "iota-collaboration": {
                "bot_token": self.IOTA_BOT_TOKEN,
            },
            "kappa-collaboration": {
                "token": self.KAPPA_TOKEN,
            },
            "omicron-deployment": {
                "token": self.OMICRON_TOKEN,
            },
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置實例
settings = Settings()
