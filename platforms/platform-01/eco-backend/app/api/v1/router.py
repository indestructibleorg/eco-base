# =============================================================================
# API Router
# =============================================================================

from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, providers, cognitive, data, code, collaboration, deployment, health

api_router = APIRouter()

# 認證相關
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# 用戶相關
api_router.include_router(users.router, prefix="/users", tags=["Users"])

# 提供者相關
api_router.include_router(providers.router, prefix="/providers", tags=["Providers"])

# 能力領域相關
api_router.include_router(cognitive.router, prefix="/cognitive", tags=["Cognitive Computing"])
api_router.include_router(data.router, prefix="/data", tags=["Data Persistence"])
api_router.include_router(code.router, prefix="/code", tags=["Code Engineering"])
api_router.include_router(collaboration.router, prefix="/collaboration", tags=["Collaboration"])
api_router.include_router(deployment.router, prefix="/deployment", tags=["Deployment"])

# 健康檢查
api_router.include_router(health.router, prefix="/health", tags=["System Health"])
