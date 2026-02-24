# =============================================================================
# Distributed Tracing
# =============================================================================
# 分布式追踪實現
# 支持 OpenTelemetry 標準
# =============================================================================

import uuid
import time
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass, field
from contextvars import ContextVar
from functools import wraps

from app.core.logging import get_logger

logger = get_logger("tracing")

# =============================================================================
# 追踪上下文
# =============================================================================

# 當前追踪上下文
current_trace_context: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace_context", default=None
)


@dataclass
class Span:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"  # ok, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.attributes:
            self.attributes = {}
        if not self.events:
            self.events = []
    
    def set_attribute(self, key: str, value: Any) -> None:
        """設置屬性"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """設置狀態"""
        self.status = status
        if description:
            self.attributes["error.description"] = description
    
    def end(self) -> None:
        """結束跨度"""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        """獲取持續時間（毫秒）"""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


@dataclass
class TraceContext:
    """追踪上下文"""
    trace_id: str
    span_id: str
    spans: List[Span] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.spans:
            self.spans = []
        if not self.baggage:
            self.baggage = {}
    
    def create_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None
    ) -> Span:
        """創建新的跨度"""
        span = Span(
            trace_id=self.trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id or self.span_id,
            name=name,
            start_time=time.time()
        )
        self.spans.append(span)
        return span
    
    def _generate_span_id(self) -> str:
        """生成跨度 ID"""
        return uuid.uuid4().hex[:16]
    
    def set_baggage(self, key: str, value: str) -> None:
        """設置 baggage"""
        self.baggage[key] = value
    
    def get_baggage(self, key: str) -> Optional[str]:
        """獲取 baggage"""
        return self.baggage.get(key)
    
    def to_headers(self) -> Dict[str, str]:
        """轉換為 HTTP 頭"""
        return {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
            "X-Baggage": ",".join([f"{k}={v}" for k, v in self.baggage.items()])
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["TraceContext"]:
        """從 HTTP 頭創建"""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        ctx = cls(trace_id=trace_id, span_id=span_id)
        
        # 解析 baggage
        baggage_str = headers.get("X-Baggage", "")
        if baggage_str:
            for item in baggage_str.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    ctx.baggage[k] = v
        
        return ctx


# =============================================================================
# 追踪器
# =============================================================================

class Tracer:
    """追踪器"""
    
    def __init__(self, service_name: str = "eco-backend"):
        self.service_name = service_name
        self._exporters: List[Callable[[List[Span]], None]] = []
    
    def add_exporter(self, exporter: Callable[[List[Span]], None]) -> None:
        """添加導出器"""
        self._exporters.append(exporter)
    
    def start_trace(self, name: str) -> TraceContext:
        """開始新的追踪"""
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        
        ctx = TraceContext(trace_id=trace_id, span_id=span_id)
        
        # 創建根跨度
        root_span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            name=name,
            start_time=time.time(),
            attributes={"service.name": self.service_name}
        )
        ctx.spans.append(root_span)
        
        # 設置當前上下文
        current_trace_context.set(ctx)
        
        logger.debug("trace_started", trace_id=trace_id, name=name)
        
        return ctx
    
    def get_current_context(self) -> Optional[TraceContext]:
        """獲取當前追踪上下文"""
        return current_trace_context.get()
    
    def start_span(
        self,
        name: str,
        context: Optional[TraceContext] = None
    ) -> Span:
        """開始新的跨度"""
        ctx = context or self.get_current_context()
        
        if not ctx:
            # 沒有上下文，創建新的追踪
            ctx = self.start_trace(name)
            return ctx.spans[0]
        
        return ctx.create_span(name)
    
    def end_span(self, span: Span) -> None:
        """結束跨度"""
        span.end()
        logger.debug(
            "span_ended",
            trace_id=span.trace_id,
            span_id=span.span_id,
            name=span.name,
            duration_ms=span.duration_ms
        )
    
    def end_trace(self, context: Optional[TraceContext] = None) -> None:
        """結束追踪"""
        ctx = context or self.get_current_context()
        
        if not ctx:
            return
        
        # 結束所有未結束的跨度
        for span in ctx.spans:
            if span.end_time is None:
                span.end()
        
        # 導出追踪數據
        self._export_spans(ctx.spans)
        
        logger.debug(
            "trace_ended",
            trace_id=ctx.trace_id,
            span_count=len(ctx.spans)
        )
        
        # 清除上下文
        current_trace_context.set(None)
    
    def _export_spans(self, spans: List[Span]) -> None:
        """導出跨度數據"""
        for exporter in self._exporters:
            try:
                exporter(spans)
            except Exception as e:
                logger.error("span_export_failed", error=str(e))


# 全局追踪器實例
tracer = Tracer()


# =============================================================================
# 裝飾器
# =============================================================================

def trace_span(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    追踪跨度裝飾器
    
    Args:
        name: 跨度名稱（默認使用函數名）
        attributes: 跨度屬性
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span = tracer.start_span(span_name)
            
            # 設置屬性
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            
            # 設置函數參數
            span.set_attribute("function.args_count", len(args))
            span.set_attribute("function.kwargs_keys", list(kwargs.keys()))
            
            try:
                result = await func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
            finally:
                tracer.end_span(span)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span = tracer.start_span(span_name)
            
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            
            try:
                result = func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
            finally:
                tracer.end_span(span)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def trace_method(name: Optional[str] = None):
    """類方法追踪裝飾器"""
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            span = tracer.start_span(f"{self.__class__.__name__}.{span_name}")
            span.set_attribute("class.name", self.__class__.__name__)
            
            try:
                result = await func(self, *args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                raise
            finally:
                tracer.end_span(span)
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            span = tracer.start_span(f"{self.__class__.__name__}.{span_name}")
            span.set_attribute("class.name", self.__class__.__name__)
            
            try:
                result = func(self, *args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error", str(e))
                raise
            finally:
                tracer.end_span(span)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# =============================================================================
# 導出器
# =============================================================================

def console_exporter(spans: List[Span]) -> None:
    """控制台導出器（用於開發調試）"""
    for span in spans:
        logger.info(
            "span_exported",
            trace_id=span.trace_id,
            span_id=span.span_id,
            name=span.name,
            duration_ms=round(span.duration_ms, 2),
            status=span.status
        )


def json_exporter(spans: List[Span]) -> List[Dict[str, Any]]:
    """JSON 導出器"""
    return [
        {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "status": span.status,
            "attributes": span.attributes,
            "events": span.events,
        }
        for span in spans
    ]


# =============================================================================
# FastAPI 中間件
# =============================================================================

class TracingMiddleware:
    """追踪中間件"""
    
    def __init__(self, app, tracer_instance: Optional[Tracer] = None):
        self.app = app
        self.tracer = tracer_instance or tracer
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # 從請求頭獲取追踪上下文
        headers = dict(scope.get("headers", []))
        headers = {k.decode(): v.decode() for k, v in headers.items()}
        
        existing_ctx = TraceContext.from_headers(headers)
        
        if existing_ctx:
            # 繼續現有追踪
            current_trace_context.set(existing_ctx)
            span = existing_ctx.create_span(f"{scope['method']} {scope['path']}")
        else:
            # 開始新追踪
            ctx = self.tracer.start_trace(f"{scope['method']} {scope['path']}")
            span = ctx.spans[0]
        
        # 設置屬性
        span.set_attribute("http.method", scope["method"])
        span.set_attribute("http.path", scope["path"])
        span.set_attribute("http.host", headers.get("host", "unknown"))
        span.set_attribute("http.user_agent", headers.get("user-agent", "unknown"))
        
        # 包裝 send 以捕獲響應狀態
        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                span.set_attribute("http.status_code", status_code)
                
                if status_code >= 400:
                    span.set_status("error", f"HTTP {status_code}")
                else:
                    span.set_status("ok")
            
            await send(message)
        
        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as e:
            span.set_status("error", str(e))
            raise
        finally:
            self.tracer.end_span(span)


# =============================================================================
# 初始化
# =============================================================================

def init_tracing(service_name: str = "eco-backend", exporters: Optional[List[Callable]] = None):
    """
    初始化追踪系統
    
    Args:
        service_name: 服務名稱
        exporters: 導出器列表
    """
    global tracer
    tracer = Tracer(service_name)
    
    # 添加默認導出器
    tracer.add_exporter(console_exporter)
    
    # 添加自定義導出器
    if exporters:
        for exporter in exporters:
            tracer.add_exporter(exporter)
    
    logger.info("tracing_initialized", service_name=service_name)


# 導入 asyncio 用於檢查協程函數
import asyncio
