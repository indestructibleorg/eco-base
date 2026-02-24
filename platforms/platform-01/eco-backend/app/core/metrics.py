# =============================================================================
# Prometheus Metrics
# =============================================================================
# 指標收集與暴露
# =============================================================================

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
from functools import wraps
from typing import Callable, Optional
import time

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("metrics")

# 創建註冊表
registry = CollectorRegistry()

# =============================================================================
# HTTP 請求指標
# =============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=registry
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=registry
)

# =============================================================================
# 業務指標
# =============================================================================

# 活躍用戶數
active_users = Gauge(
    'active_users',
    'Number of active users',
    registry=registry
)

# API 密鑰使用
api_key_usage_total = Counter(
    'api_key_usage_total',
    'Total API key usage',
    ['key_id'],
    registry=registry
)

# 月度配額使用率
monthly_quota_usage = Gauge(
    'monthly_quota_usage',
    'Monthly quota usage percentage',
    ['user_id'],
    registry=registry
)

# =============================================================================
# 提供者調用指標
# =============================================================================

provider_calls_total = Counter(
    'provider_calls_total',
    'Total provider calls',
    ['provider', 'operation', 'status'],
    registry=registry
)

provider_call_duration_seconds = Histogram(
    'provider_call_duration_seconds',
    'Provider call duration in seconds',
    ['provider', 'operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry
)

provider_call_errors_total = Counter(
    'provider_call_errors_total',
    'Total provider call errors',
    ['provider', 'operation', 'error_type'],
    registry=registry
)

# 熔斷器狀態
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['provider'],
    registry=registry
)

# =============================================================================
# 數據庫指標
# =============================================================================

db_connections_active = Gauge(
    'db_connections_active',
    'Number of active database connections',
    registry=registry
)

db_connections_idle = Gauge(
    'db_connections_idle',
    'Number of idle database connections',
    registry=registry
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)

# =============================================================================
# 緩存指標
# =============================================================================

cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_name'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_name'],
    registry=registry
)

cache_size = Gauge(
    'cache_size',
    'Current cache size',
    ['cache_name'],
    registry=registry
)

# =============================================================================
# 系統指標
# =============================================================================

app_info = Info(
    'app',
    'Application information',
    registry=registry
)

# 設置應用信息
app_info.info({
    'name': settings.APP_NAME,
    'version': settings.APP_VERSION,
    'environment': settings.ENVIRONMENT,
})

# =============================================================================
# 裝飾器
# =============================================================================

def track_request_duration(endpoint: str):
    """追蹤請求處理時間的裝飾器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=kwargs.get('method', 'UNKNOWN'),
                    endpoint=endpoint
                ).observe(duration)
        return wrapper
    return decorator


def track_provider_call(provider: str, operation: str):
    """追蹤提供者調用的裝飾器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                provider_call_errors_total.labels(
                    provider=provider,
                    operation=operation,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                provider_calls_total.labels(
                    provider=provider,
                    operation=operation,
                    status=status
                ).inc()
                provider_call_duration_seconds.labels(
                    provider=provider,
                    operation=operation
                ).observe(duration)
        return wrapper
    return decorator


def track_db_query(operation: str):
    """追蹤數據庫查詢時間的裝飾器"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                db_query_duration_seconds.labels(
                    operation=operation
                ).observe(duration)
        return wrapper
    return decorator


# =============================================================================
# 指標收集函數
# =============================================================================

def record_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
):
    """記錄 HTTP 請求指標"""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()
    
    if request_size is not None:
        http_request_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(request_size)
    
    if response_size is not None:
        http_response_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(response_size)


def record_provider_call(
    provider: str,
    operation: str,
    duration: float,
    success: bool = True,
    error_type: Optional[str] = None,
):
    """記錄提供者調用指標"""
    status = 'success' if success else 'error'
    
    provider_calls_total.labels(
        provider=provider,
        operation=operation,
        status=status
    ).inc()
    
    provider_call_duration_seconds.labels(
        provider=provider,
        operation=operation
    ).observe(duration)
    
    if not success and error_type:
        provider_call_errors_total.labels(
            provider=provider,
            operation=operation,
            error_type=error_type
        ).inc()


def update_circuit_breaker_state(provider: str, state: str):
    """更新熔斷器狀態指標"""
    state_map = {
        'closed': 0,
        'open': 1,
        'half-open': 2,
    }
    circuit_breaker_state.labels(provider=provider).set(state_map.get(state, 0))


def record_cache_operation(cache_name: str, hit: bool):
    """記錄緩存操作"""
    if hit:
        cache_hits_total.labels(cache_name=cache_name).inc()
    else:
        cache_misses_total.labels(cache_name=cache_name).inc()


# =============================================================================
# 指標暴露
# =============================================================================

def get_metrics() -> tuple:
    """獲取 Prometheus 格式的指標"""
    return generate_latest(registry), CONTENT_TYPE_LATEST
