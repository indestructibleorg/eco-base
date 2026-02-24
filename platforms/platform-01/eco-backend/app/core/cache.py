# =============================================================================
# Cache System
# =============================================================================
# 緩存策略實現
# =============================================================================

import json
import pickle
from functools import wraps
from typing import Any, Optional, Callable, Union
from datetime import timedelta
import hashlib

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import record_cache_operation

logger = get_logger("cache")


class CacheManager:
    """緩存管理器"""
    
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._default_ttl = 300  # 5 分鐘
    
    async def connect(self):
        """連接 Redis"""
        if self._redis is None:
            self._redis = redis.from_url(
                settings.REDIS_URL,
                encoding='utf-8',
                decode_responses=True
            )
    
    async def disconnect(self):
        """斷開連接"""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """生成緩存鍵"""
        key_parts = [prefix]
        
        if args:
            key_parts.append(str(args))
        if kwargs:
            # 排序以確保一致性
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(str(sorted_kwargs))
        
        raw_key = ':'.join(key_parts)
        # 使用哈希縮短鍵名
        return f"eco:{prefix}:{hashlib.md5(raw_key.encode()).hexdigest()[:16]}"
    
    async def get(self, key: str) -> Optional[Any]:
        """獲取緩存值"""
        await self.connect()
        
        try:
            value = await self._redis.get(key)
            
            if value is not None:
                record_cache_operation("default", hit=True)
                return json.loads(value)
            
            record_cache_operation("default", hit=False)
            return None
            
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """設置緩存值"""
        await self.connect()
        
        try:
            serialized = json.dumps(value, default=str)
            await self._redis.setex(
                key,
                ttl or self._default_ttl,
                serialized
            )
            return True
            
        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """刪除緩存值"""
        await self.connect()
        
        try:
            await self._redis.delete(key)
            return True
            
        except Exception as e:
            logger.error("cache_delete_error", key=key, error=str(e))
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """刪除匹配模式的緩存"""
        await self.connect()
        
        try:
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error("cache_delete_pattern_error", pattern=pattern, error=str(e))
            return 0
    
    async def exists(self, key: str) -> bool:
        """檢查鍵是否存在"""
        await self.connect()
        
        try:
            return await self._redis.exists(key) > 0
            
        except Exception as e:
            logger.error("cache_exists_error", key=key, error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """原子遞增"""
        await self.connect()
        
        try:
            return await self._redis.incrby(key, amount)
            
        except Exception as e:
            logger.error("cache_increment_error", key=key, error=str(e))
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """設置過期時間"""
        await self.connect()
        
        try:
            return await self._redis.expire(key, ttl)
            
        except Exception as e:
            logger.error("cache_expire_error", key=key, error=str(e))
            return False


# 全局緩存管理器
cache_manager = CacheManager()


def cached(
    prefix: str,
    ttl: int = 300,
    key_func: Optional[Callable] = None
):
    """
    緩存裝飾器
    
    Args:
        prefix: 緩存鍵前綴
        ttl: 過期時間（秒）
        key_func: 自定義鍵生成函數
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成緩存鍵
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._make_key(prefix, *args[1:], **kwargs)
            
            # 嘗試從緩存獲取
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug("cache_hit", key=cache_key)
                return cached_value
            
            # 執行函數
            result = await func(*args, **kwargs)
            
            # 存入緩存
            await cache_manager.set(cache_key, result, ttl)
            logger.debug("cache_set", key=cache_key, ttl=ttl)
            
            return result
        
        # 添加清除緩存的方法
        async def invalidate(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._make_key(prefix, *args[1:], **kwargs)
            await cache_manager.delete(cache_key)
        
        wrapper.invalidate = invalidate
        return wrapper
    return decorator


def cache_evict(prefix: str, key_func: Optional[Callable] = None):
    """
    緩存清除裝飾器
    
    在函數執行後清除緩存
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 清除緩存
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager._make_key(prefix, *args[1:], **kwargs)
            
            await cache_manager.delete(cache_key)
            logger.debug("cache_evicted", key=cache_key)
            
            return result
        return wrapper
    return decorator


class CacheAsidePattern:
    """
    Cache-Aside 模式實現
    
    讀取時：先查緩存，未命中則查數據庫並寫入緩存
    寫入時：先寫數據庫，再刪除緩存
    """
    
    def __init__(self, cache: CacheManager):
        self.cache = cache
    
    async def get(
        self,
        key: str,
        loader: Callable,
        ttl: int = 300
    ) -> Any:
        """
        獲取數據（帶緩存）
        
        Args:
            key: 緩存鍵
            loader: 數據加載函數
            ttl: 緩存過期時間
            
        Returns:
            數據
        """
        # 嘗試從緩存獲取
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # 從數據源加載
        value = await loader()
        
        # 寫入緩存
        if value is not None:
            await self.cache.set(key, value, ttl)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """設置緩存"""
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """刪除緩存"""
        return await self.cache.delete(key)


# 常用緩存鍵前綴
CACHE_KEYS = {
    'user': 'user',
    'user_by_email': 'user:email',
    'api_key': 'api_key',
    'provider_list': 'provider:list',
    'provider_health': 'provider:health',
}
