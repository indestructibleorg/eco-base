# =============================================================================
# Event System
# =============================================================================
# 事件總線實現
# =============================================================================

import asyncio
from typing import Dict, List, Callable, Any, Coroutine
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

from app.core.logging import get_logger

logger = get_logger("events")


class EventType(Enum):
    """事件類型"""
    # 用戶相關
    USER_CREATED = auto()
    USER_UPDATED = auto()
    USER_DELETED = auto()
    USER_LOGIN = auto()
    USER_LOGOUT = auto()
    
    # API 密鑰相關
    API_KEY_CREATED = auto()
    API_KEY_DELETED = auto()
    API_KEY_REVOKED = auto()
    
    # 提供者相關
    PROVIDER_CALLED = auto()
    PROVIDER_ERROR = auto()
    PROVIDER_TIMEOUT = auto()
    
    # 系統相關
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    CONFIG_RELOADED = auto()


@dataclass
class Event:
    """事件數據類"""
    type: EventType
    payload: Dict[str, Any]
    timestamp: datetime
    source: str
    
    def __init__(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        source: str = "system"
    ):
        self.type = event_type
        self.payload = payload
        self.timestamp = datetime.utcnow()
        self.source = source


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    事件總線
    
    用於組件間的異步通信
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._middleware: List[Callable[[Event], Coroutine[Any, Any, Event]]] = []
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> Callable[[], None]:
        """
        訂閱事件
        
        Args:
            event_type: 事件類型
            handler: 事件處理函數
            
        Returns:
            取消訂閱函數
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        
        logger.info(
            "event_subscribed",
            event_type=event_type.name,
            handler=handler.__name__
        )
        
        def unsubscribe():
            self._handlers[event_type].remove(handler)
            logger.info(
                "event_unsubscribed",
                event_type=event_type.name,
                handler=handler.__name__
            )
        
        return unsubscribe
    
    def add_middleware(
        self,
        middleware: Callable[[Event], Coroutine[Any, Any, Event]]
    ):
        """添加中間件"""
        self._middleware.append(middleware)
    
    async def publish(self, event: Event) -> None:
        """
        發布事件
        
        Args:
            event: 事件對象
        """
        # 應用中間件
        for middleware in self._middleware:
            event = await middleware(event)
        
        handlers = self._handlers.get(event.type, [])
        
        if not handlers:
            logger.debug(
                "no_handlers_for_event",
                event_type=event.type.name
            )
            return
        
        # 異步執行所有處理器
        tasks = [handler(event) for handler in handlers]
        
        logger.info(
            "event_published",
            event_type=event.type.name,
            handler_count=len(handlers),
            source=event.source
        )
        
        # 等待所有處理器完成，但忽略錯誤
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "event_handler_failed",
                    event_type=event.type.name,
                    handler=handlers[i].__name__,
                    error=str(result)
                )
    
    async def publish_simple(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        source: str = "system"
    ) -> None:
        """簡化版發布事件"""
        event = Event(event_type, payload, source)
        await self.publish(event)


# 全局事件總線實例
event_bus = EventBus()


# =============================================================================
# 常用事件處理器
# =============================================================================

async def log_event_handler(event: Event) -> None:
    """日誌事件處理器"""
    logger.info(
        "event_processed",
        event_type=event.type.name,
        payload_keys=list(event.payload.keys()),
        source=event.source
    )


async def metrics_event_handler(event: Event) -> None:
    """指標收集事件處理器"""
    from app.core.metrics import provider_calls, provider_errors
    
    if event.type == EventType.PROVIDER_CALLED:
        # 記錄提供者調用指標
        pass
    elif event.type == EventType.PROVIDER_ERROR:
        # 記錄提供者錯誤指標
        pass


async def notification_event_handler(event: Event) -> None:
    """通知事件處理器"""
    if event.type == EventType.USER_LOGIN:
        # 發送登入通知
        user_id = event.payload.get("user_id")
        logger.info("login_notification", user_id=user_id)
    
    elif event.type == EventType.API_KEY_REVOKED:
        # 發送 API 密鑰撤銷通知
        user_id = event.payload.get("user_id")
        key_id = event.payload.get("key_id")
        logger.info("api_key_revoked_notification", user_id=user_id, key_id=key_id)


# =============================================================================
# 事件裝飾器
# =============================================================================

def on_event(event_type: EventType):
    """
    事件訂閱裝飾器
    
    用法:
        @on_event(EventType.USER_CREATED)
        async def handle_user_created(event: Event):
            ...
    """
    def decorator(func: EventHandler) -> EventHandler:
        event_bus.subscribe(event_type, func)
        return func
    return decorator


def emit_event(
    event_type: EventType,
    payload: Dict[str, Any],
    source: str = "system"
) -> asyncio.Task:
    """
    發射事件（異步）
    
    Args:
        event_type: 事件類型
        payload: 事件數據
        source: 事件來源
        
    Returns:
        異步任務
    """
    event = Event(event_type, payload, source)
    return asyncio.create_task(event_bus.publish(event))


# =============================================================================
# 初始化
# =============================================================================

def init_event_handlers():
    """初始化事件處理器"""
    # 註冊日誌處理器
    event_bus.subscribe(EventType.USER_CREATED, log_event_handler)
    event_bus.subscribe(EventType.USER_LOGIN, log_event_handler)
    event_bus.subscribe(EventType.PROVIDER_CALLED, log_event_handler)
    event_bus.subscribe(EventType.PROVIDER_ERROR, log_event_handler)
    
    # 註冊通知處理器
    event_bus.subscribe(EventType.USER_LOGIN, notification_event_handler)
    event_bus.subscribe(EventType.API_KEY_REVOKED, notification_event_handler)
    
    logger.info("event_handlers_initialized")
