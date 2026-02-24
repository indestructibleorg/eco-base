"""
自动修复引擎
执行自动修复动作
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)


class RemediationStatus(Enum):
    """修复状态"""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class RemediationType(Enum):
    """修复类型"""
    RESTART = "restart"
    SCALE = "scale"
    ROLLBACK = "rollback"
    CLEAR_CACHE = "clear_cache"
    SWITCH_OVER = "switch_over"
    CUSTOM = "custom"


@dataclass
class RemediationAction:
    """修复动作"""
    action_id: str
    action_type: RemediationType
    target: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    retry_delay: int = 10


@dataclass
class RemediationResult:
    """修复结果"""
    action_id: str
    status: RemediationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    output: str = ""
    error: Optional[str] = None
    retry_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoRemediator:
    """自动修复引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 动作处理器
        self.action_handlers: Dict[RemediationType, Callable] = {}
        
        # 执行历史
        self.execution_history: List[RemediationResult] = []
        
        # 正在执行的动作
        self.running_actions: Dict[str, asyncio.Task] = {}
        
        # 注册默认处理器
        self._register_default_handlers()
        
        logger.info("自动修复引擎初始化完成")
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self.action_handlers[RemediationType.RESTART] = self._handle_restart
        self.action_handlers[RemediationType.CLEAR_CACHE] = self._handle_clear_cache
        logger.info("默认修复处理器注册完成")
    
    def register_action_handler(self, action_type: RemediationType, 
                                handler: Callable):
        """注册动作处理器"""
        self.action_handlers[action_type] = handler
        logger.info(f"修复处理器注册: {action_type.value}")
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        """执行修复动作"""
        handler = self.action_handlers.get(action.action_type)
        if not handler:
            return RemediationResult(
                action_id=action.action_id,
                status=RemediationStatus.FAILED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error=f"未找到处理器: {action.action_type.value}"
            )
        
        result = RemediationResult(
            action_id=action.action_id,
            status=RemediationStatus.RUNNING,
            started_at=datetime.now()
        )
        
        # 执行
        for attempt in range(action.retry_count):
            try:
                result.retry_attempts = attempt + 1
                
                # 设置超时
                output = await asyncio.wait_for(
                    handler(action),
                    timeout=action.timeout_seconds
                )
                
                result.status = RemediationStatus.SUCCESS
                result.output = str(output)
                result.completed_at = datetime.now()
                
                self.execution_history.append(result)
                logger.info(f"修复成功: {action.action_id}")
                return result
            
            except asyncio.TimeoutError:
                result.status = RemediationStatus.TIMEOUT
                result.error = f"执行超时 ({action.timeout_seconds}s)"
                logger.warning(f"修复超时: {action.action_id}")
                
                if attempt < action.retry_count - 1:
                    await asyncio.sleep(action.retry_delay)
            
            except Exception as e:
                result.error = str(e)
                logger.warning(f"修复失败 (尝试 {attempt+1}/{action.retry_count}): {e}")
                
                if attempt < action.retry_count - 1:
                    await asyncio.sleep(action.retry_delay)
        
        # 所有重试失败
        result.status = RemediationStatus.FAILED
        result.completed_at = datetime.now()
        self.execution_history.append(result)
        
        logger.error(f"修复最终失败: {action.action_id}")
        return result
    
    async def execute_batch(self, 
                            actions: List[RemediationAction],
                            sequential: bool = True) -> List[RemediationResult]:
        """批量执行修复动作"""
        if sequential:
            results = []
            for action in actions:
                result = await self.execute(action)
                results.append(result)
                if result.status != RemediationStatus.SUCCESS:
                    logger.warning(f"批量执行中断: {action.action_id} 失败")
                    break
            return results
        else:
            tasks = [self.execute(action) for action in actions]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_restart(self, action: RemediationAction) -> str:
        """处理重启"""
        target = action.target
        logger.info(f"执行重启: {target}")
        
        # 模拟重启操作
        await asyncio.sleep(2)
        
        return f"服务 {target} 重启完成"
    
    async def _handle_clear_cache(self, action: RemediationAction) -> str:
        """处理清理缓存"""
        target = action.target
        cache_type = action.params.get('cache_type', 'all')
        
        logger.info(f"清理缓存: {target}, 类型: {cache_type}")
        
        # 模拟清理操作
        await asyncio.sleep(1)
        
        return f"缓存清理完成: {target}"
    
    def execute_action(self, action_type: str, target: str, 
                       params: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行修复动作 (同步接口)"""
        import asyncio
        
        params = params or {}
        action = RemediationAction(
            action_id=f"action_{datetime.now().timestamp()}",
            action_type=RemediationType(action_type),
            target=target,
            params=params,
            timeout_seconds=params.get('timeout', 300)
        )
        
        try:
            # 运行异步执行
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.execute(action))
            
            return {
                'status': result.status.name,
                'action_id': result.action_id,
                'output': result.output,
                'error': result.error
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def get_execution_history(self, 
                              action_type: RemediationType = None,
                              limit: int = 100) -> List[RemediationResult]:
        """获取执行历史"""
        history = self.execution_history
        if action_type:
            history = [h for h in history if h.action_type == action_type]
        return history[-limit:]
    
    def get_success_rate(self, 
                         action_type: RemediationType = None,
                         hours: int = 24) -> float:
        """获取成功率"""
        cutoff = datetime.now() - timedelta(hours=hours)
        history = [h for h in self.execution_history if h.started_at > cutoff]
        
        if action_type:
            history = [h for h in history if h.action_type == action_type]
        
        if not history:
            return 0.0
        
        success = sum(1 for h in history if h.status == RemediationStatus.SUCCESS)
        return success / len(history)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.execution_history)
        if total == 0:
            return {"total": 0}
        
        by_status = {}
        for result in self.execution_history:
            status = result.status.name
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total": total,
            "by_status": by_status,
            "success_rate": self.get_success_rate(),
            "handlers_registered": len(self.action_handlers)
        }
