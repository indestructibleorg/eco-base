# =============================================================================
# Plugin System
# =============================================================================
# 插件架構實現
# =============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum, auto

from app.core.logging import get_logger
from app.core.events import Event, EventType, event_bus

logger = get_logger("plugins")


class HookPoint(Enum):
    """插件掛載點"""
    # 請求生命周期
    BEFORE_REQUEST = auto()
    AFTER_REQUEST = auto()
    ON_ERROR = auto()
    
    # 認證相關
    BEFORE_AUTH = auto()
    AFTER_AUTH = auto()
    
    # 提供者調用
    BEFORE_PROVIDER_CALL = auto()
    AFTER_PROVIDER_CALL = auto()
    ON_PROVIDER_ERROR = auto()
    
    # 數據操作
    BEFORE_DB_QUERY = auto()
    AFTER_DB_QUERY = auto()
    
    # 系統事件
    ON_STARTUP = auto()
    ON_SHUTDOWN = auto()


@dataclass
class PluginContext:
    """插件上下文"""
    request: Optional[Any] = None
    response: Optional[Any] = None
    user_id: Optional[str] = None
    provider_id: Optional[str] = None
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class Plugin(ABC):
    """
    插件基類
    
    所有插件必須繼承此類
    """
    
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    
    def __init__(self):
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化插件
        
        Args:
            config: 插件配置
            
        Returns:
            是否初始化成功
        """
        pass
    
    async def shutdown(self) -> None:
        """關閉插件"""
        pass
    
    # 鉤子方法（可選實現）
    async def before_request(self, context: PluginContext) -> PluginContext:
        """請求前處理"""
        return context
    
    async def after_request(self, context: PluginContext) -> PluginContext:
        """請求後處理"""
        return context
    
    async def on_error(self, context: PluginContext, error: Exception) -> None:
        """錯誤處理"""
        pass
    
    async def before_provider_call(
        self,
        provider_id: str,
        operation: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提供者調用前處理"""
        return payload
    
    async def after_provider_call(
        self,
        provider_id: str,
        operation: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提供者調用後處理"""
        return result
    
    async def on_startup(self) -> None:
        """應用啟動時調用"""
        pass
    
    async def on_shutdown(self) -> None:
        """應用關閉時調用"""
        pass


class PluginManager:
    """
    插件管理器
    
    管理插件的生命周期和掛載點
    """
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[HookPoint, List[str]] = {
            hook: [] for hook in HookPoint
        }
    
    def register(self, plugin_class: Type[Plugin]) -> bool:
        """
        註冊插件
        
        Args:
            plugin_class: 插件類
            
        Returns:
            是否註冊成功
        """
        try:
            plugin = plugin_class()
            
            if plugin.name in self._plugins:
                logger.warning("plugin_already_registered", name=plugin.name)
                return False
            
            self._plugins[plugin.name] = plugin
            
            logger.info(
                "plugin_registered",
                name=plugin.name,
                version=plugin.version
            )
            
            return True
            
        except Exception as e:
            logger.error("plugin_registration_failed", error=str(e))
            return False
    
    async def initialize_plugin(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        初始化插件
        
        Args:
            name: 插件名稱
            config: 插件配置
            
        Returns:
            是否初始化成功
        """
        plugin = self._plugins.get(name)
        if not plugin:
            logger.error("plugin_not_found", name=name)
            return False
        
        try:
            success = await plugin.initialize(config)
            
            if success:
                logger.info("plugin_initialized", name=name)
            else:
                logger.warning("plugin_initialization_failed", name=name)
            
            return success
            
        except Exception as e:
            logger.error("plugin_initialization_error", name=name, error=str(e))
            return False
    
    def unregister(self, name: str) -> bool:
        """
        註銷插件
        
        Args:
            name: 插件名稱
            
        Returns:
            是否註銷成功
        """
        if name not in self._plugins:
            return False
        
        del self._plugins[name]
        
        # 從所有掛載點移除
        for hook_list in self._hooks.values():
            if name in hook_list:
                hook_list.remove(name)
        
        logger.info("plugin_unregistered", name=name)
        return True
    
    def enable(self, name: str) -> bool:
        """啟用插件"""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = True
            logger.info("plugin_enabled", name=name)
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """禁用插件"""
        plugin = self._plugins.get(name)
        if plugin:
            plugin.enabled = False
            logger.info("plugin_disabled", name=name)
            return True
        return False
    
    async def execute_hook(
        self,
        hook_point: HookPoint,
        context: PluginContext
    ) -> PluginContext:
        """
        執行掛載點
        
        Args:
            hook_point: 掛載點
            context: 上下文
            
        Returns:
            處理後的上下文
        """
        plugin_names = self._hooks.get(hook_point, [])
        
        for name in plugin_names:
            plugin = self._plugins.get(name)
            if not plugin or not plugin.enabled:
                continue
            
            try:
                if hook_point == HookPoint.BEFORE_REQUEST:
                    context = await plugin.before_request(context)
                elif hook_point == HookPoint.AFTER_REQUEST:
                    context = await plugin.after_request(context)
                elif hook_point == HookPoint.ON_ERROR:
                    error = context.data.get("error")
                    if error:
                        await plugin.on_error(context, error)
                        
            except Exception as e:
                logger.error(
                    "plugin_hook_execution_failed",
                    plugin=name,
                    hook=hook_point.name,
                    error=str(e)
                )
        
        return context
    
    async def execute_provider_hook(
        self,
        hook_point: HookPoint,
        provider_id: str,
        operation: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行提供者相關掛載點"""
        plugin_names = self._hooks.get(hook_point, [])
        
        for name in plugin_names:
            plugin = self._plugins.get(name)
            if not plugin or not plugin.enabled:
                continue
            
            try:
                if hook_point == HookPoint.BEFORE_PROVIDER_CALL:
                    data = await plugin.before_provider_call(
                        provider_id, operation, data
                    )
                elif hook_point == HookPoint.AFTER_PROVIDER_CALL:
                    data = await plugin.after_provider_call(
                        provider_id, operation, data
                    )
                    
            except Exception as e:
                logger.error(
                    "plugin_provider_hook_failed",
                    plugin=name,
                    hook=hook_point.name,
                    error=str(e)
                )
        
        return data
    
    async def startup_all(self) -> None:
        """啟動所有插件"""
        for name, plugin in self._plugins.items():
            if plugin.enabled:
                try:
                    await plugin.on_startup()
                    logger.info("plugin_started", name=name)
                except Exception as e:
                    logger.error("plugin_startup_failed", name=name, error=str(e))
    
    async def shutdown_all(self) -> None:
        """關閉所有插件"""
        for name, plugin in self._plugins.items():
            try:
                await plugin.on_shutdown()
                await plugin.shutdown()
                logger.info("plugin_shutdown", name=name)
            except Exception as e:
                logger.error("plugin_shutdown_failed", name=name, error=str(e))
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有插件"""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "enabled": p.enabled,
            }
            for p in self._plugins.values()
        ]


# 全局插件管理器實例
plugin_manager = PluginManager()


# =============================================================================
# 示例插件
# =============================================================================

class LoggingPlugin(Plugin):
    """日誌插件示例"""
    
    name = "logging"
    version = "1.0.0"
    description = "記錄所有請求和響應"
    author = "eco-team"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        return True
    
    async def before_request(self, context: PluginContext) -> PluginContext:
        logger.info(
            "plugin_request_started",
            path=context.request.url.path if context.request else None,
            user_id=context.user_id
        )
        return context
    
    async def after_request(self, context: PluginContext) -> PluginContext:
        logger.info(
            "plugin_request_completed",
            path=context.request.url.path if context.request else None,
            status=context.response.status_code if context.response else None
        )
        return context


class RateLimitPlugin(Plugin):
    """限流插件示例"""
    
    name = "rate_limit"
    version = "1.0.0"
    description = "基於用戶的請求限流"
    author = "eco-team"
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        self.requests: Dict[str, List[float]] = {}
        return True
    
    async def before_request(self, context: PluginContext) -> PluginContext:
        # 簡單的限流邏輯
        user_id = context.user_id
        if user_id:
            now = __import__('time').time()
            user_requests = self.requests.get(user_id, [])
            
            # 清理過期請求
            user_requests = [t for t in user_requests if now - t < 60]
            
            if len(user_requests) >= self.config.get("limit", 100):
                raise Exception("Rate limit exceeded")
            
            user_requests.append(now)
            self.requests[user_id] = user_requests
        
        return context
