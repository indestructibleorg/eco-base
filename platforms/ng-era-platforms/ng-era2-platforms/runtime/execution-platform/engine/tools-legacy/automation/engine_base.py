# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: engine_base
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Engine Base - 引擎基礎架構
提供所有自動化引擎的基礎類和介面定義。
支援 100% 自主執行，無需人類審核。
核心特性：
- 自我修復 (Self-Healing)
- 自動重試 (Auto-Retry)
- 狀態持久化 (State Persistence)
- 生命週期管理 (Lifecycle Management)
- 事件驅動 (Event-Driven)
- 可觀測性 (Observability)
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)
# ============================================================================
# 類型定義
# ============================================================================
T = TypeVar("T")
EngineResult = TypeVar("EngineResult")
# ============================================================================
# 枚舉定義
# ============================================================================
class EngineState(Enum):
    """引擎狀態"""
    UNINITIALIZED = auto()  # 未初始化
    INITIALIZING = auto()  # 初始化中
    IDLE = auto()  # 閒置
    RUNNING = auto()  # 運行中
    PAUSED = auto()  # 暫停
    STOPPING = auto()  # 停止中
    STOPPED = auto()  # 已停止
    ERROR = auto()  # 錯誤
    RECOVERING = auto()  # 恢復中
    TERMINATED = auto()  # 已終止
class EngineType(Enum):
    """引擎類型"""
    COGNITIVE = "cognitive"  # 認知推理
    EXECUTION = "execution"  # 執行操作
    VALIDATION = "validation"  # 驗證檢查
    TRANSFORM = "transform"  # 轉換處理
    MONITORING = "monitoring"  # 監控觀測
    ORCHESTRATION = "orchestration"  # 編排調度
    INTEGRATION = "integration"  # 整合連接
    GENERATION = "generation"  # 生成創建
class ExecutionMode(Enum):
    """執行模式"""
    AUTONOMOUS = "autonomous"  # 全自主 (100% 機器)
    SUPERVISED = "supervised"  # 監督式 (有人類監控)
    INTERACTIVE = "interactive"  # 互動式 (需人類確認)
    DRY_RUN = "dry_run"  # 模擬執行
class Priority(Enum):
    """優先級"""
    CRITICAL = 0  # 最高優先
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # 背景執行
# ============================================================================
# 配置資料結構
# ============================================================================
@dataclass
class RetryConfig:
    """重試配置"""
    max_attempts: int = 3
    initial_delay: float = 1.0  # 秒
    max_delay: float = 60.0  # 秒
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
@dataclass
class TimeoutConfig:
    """超時配置"""
    initialization: float = 30.0  # 初始化超時
    execution: float = 300.0  # 執行超時
    shutdown: float = 30.0  # 關閉超時
    heartbeat: float = 10.0  # 心跳間隔
@dataclass
class ResourceConfig:
    """資源配置"""
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_concurrent_tasks: int = 10
    max_queue_size: int = 1000
@dataclass
class PersistenceConfig:
    """持久化配置"""
    enabled: bool = True
    state_dir: str = ".engine_state"
    checkpoint_interval: float = 60.0  # 秒
    max_checkpoints: int = 10
@dataclass
class EngineConfig:
    """引擎主配置"""
    # 基本資訊
    engine_id: str = ""
    engine_name: str = ""
    engine_type: EngineType = EngineType.EXECUTION
    version: str = "1.0.0"
    # 執行模式
    execution_mode: ExecutionMode = ExecutionMode.AUTONOMOUS
    priority: Priority = Priority.NORMAL
    # 子配置
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    # 自動化設定
    auto_start: bool = True
    auto_recover: bool = True
    auto_scale: bool = False
    # 依賴
    dependencies: List[str] = field(default_factory=list)
    # 觸發器
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    # 自定義參數
    params: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        if not self.engine_id:
            self.engine_id = str(uuid.uuid4())[:8]
# ============================================================================
# 事件資料結構
# ============================================================================
@dataclass
class EngineEvent:
    """引擎事件"""
    event_id: str
    event_type: str
    engine_id: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    @classmethod
    def create(cls, event_type: str, engine_id: str, payload: Dict = None):
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            engine_id=engine_id,
            timestamp=datetime.now().isoformat(),
            payload=payload or {},
        )
@dataclass
class TaskResult:
    """任務結果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class HealthStatus:
    """健康狀態"""
    healthy: bool
    state: EngineState
    uptime_seconds: float
    tasks_completed: int
    tasks_failed: int
    last_activity: str
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    error_rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
# ============================================================================
# 引擎基礎類
# ============================================================================
class BaseEngine(ABC):
    """
    引擎基礎抽象類
    所有自動化引擎必須繼承此類並實現抽象方法。
    提供完整的生命週期管理和自動化執行能力。
    """
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._state = EngineState.UNINITIALIZED
        self._start_time: Optional[datetime] = None
        self._last_activity: Optional[datetime] = None
        # 統計
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_execution_time = 0.0
        # 事件處理
        self._event_handlers: Dict[str, List[Callable]] = {}
        # 任務隊列
        self._task_queue: asyncio.Queue = None
        self._active_tasks: Set[str] = set()
        # 控制標誌
        self._running = False
        self._shutdown_event: asyncio.Event = None
        # 日誌
        self._logger = logging.getLogger(
            f"engine.{self.config.engine_name or self.config.engine_id}"
        )
        # 檢查點
        self._checkpoint_data: Dict[str, Any] = {}
    # ========================================================================
    # 屬性
    # ========================================================================
    @property
    def engine_id(self) -> str:
        return self.config.engine_id
    @property
    def engine_name(self) -> str:
        return self.config.engine_name or self.__class__.__name__
    @property
    def state(self) -> EngineState:
        return self._state
    @property
    def is_running(self) -> bool:
        return self._state == EngineState.RUNNING
    @property
    def uptime(self) -> timedelta:
        if self._start_time:
            return datetime.now() - self._start_time
        return timedelta(0)
    # ========================================================================
    # 抽象方法 (子類必須實現)
    # ========================================================================
    @abstractmethod
    async def _initialize(self) -> bool:
        """
        初始化引擎
        Returns:
            bool: 初始化是否成功
        """
        pass
    @abstractmethod
    async def _execute(self, task: Dict[str, Any]) -> TaskResult:
        """
        執行單一任務
        Args:
            task: 任務定義
        Returns:
            TaskResult: 任務結果
        """
        pass
    @abstractmethod
    async def _shutdown(self) -> bool:
        """
        關閉引擎
        Returns:
            bool: 關閉是否成功
        """
        pass
    @abstractmethod
    def _get_capabilities(self) -> Dict[str, Any]:
        """
        獲取引擎能力描述
        Returns:
            Dict: 能力描述
        """
        pass
    # ========================================================================
    # 生命週期管理
    # ========================================================================
    async def start(self) -> bool:
        """啟動引擎"""
        if self._state not in [EngineState.UNINITIALIZED, EngineState.STOPPED]:
            self._logger.warning(f"無法從 {self._state} 狀態啟動")
            return False
        self._logger.info(f"啟動引擎: {self.engine_name}")
        self._state = EngineState.INITIALIZING
        try:
            # 初始化組件
            self._task_queue = asyncio.Queue(
                maxsize=self.config.resource.max_queue_size
            )
            self._shutdown_event = asyncio.Event()
            # 載入檢查點
            if self.config.persistence.enabled:
                await self._load_checkpoint()
            # 執行子類初始化
            success = await asyncio.wait_for(
                self._initialize(), timeout=self.config.timeout.initialization
            )
            if not success:
                self._state = EngineState.ERROR
                return False
            self._state = EngineState.RUNNING
            self._running = True
            self._start_time = datetime.now()
            # 發送啟動事件
            await self._emit_event("engine.started", {"config": asdict(self.config)})
            # 啟動背景任務
            asyncio.create_task(self._main_loop())
            asyncio.create_task(self._heartbeat_loop())
            if self.config.persistence.enabled:
                asyncio.create_task(self._checkpoint_loop())
            self._logger.info(f"引擎啟動成功: {self.engine_name}")
            return True
        except asyncio.TimeoutError:
            self._logger.error("初始化超時")
            self._state = EngineState.ERROR
            return False
        except Exception as e:
            self._logger.error(f"初始化失敗: {e}")
            self._state = EngineState.ERROR
            return False
    async def stop(self, force: bool = False) -> bool:
        """停止引擎"""
        if self._state == EngineState.STOPPED:
            return True
        self._logger.info(f"停止引擎: {self.engine_name}")
        self._state = EngineState.STOPPING
        self._running = False
        try:
            # 設置關閉信號
            if self._shutdown_event:
                self._shutdown_event.set()
            # 等待活動任務完成
            if not force and self._active_tasks:
                self._logger.info(f"等待 {len(self._active_tasks)} 個任務完成...")
                await asyncio.sleep(2)
            # 保存檢查點
            if self.config.persistence.enabled:
                await self._save_checkpoint()
            # 執行子類關閉
            success = await asyncio.wait_for(
                self._shutdown(), timeout=self.config.timeout.shutdown
            )
            self._state = EngineState.STOPPED
            await self._emit_event("engine.stopped", {})
            self._logger.info(f"引擎已停止: {self.engine_name}")
            return success
        except asyncio.TimeoutError:
            self._logger.warning("關閉超時，強制停止")
            self._state = EngineState.TERMINATED
            return False
        except Exception as e:
            self._logger.error(f"關閉失敗: {e}")
            self._state = EngineState.ERROR
            return False
    async def restart(self) -> bool:
        """重啟引擎"""
        await self.stop()
        await asyncio.sleep(1)
        return await self.start()
    async def pause(self) -> bool:
        """暫停引擎"""
        if self._state != EngineState.RUNNING:
            return False
        self._state = EngineState.PAUSED
        await self._emit_event("engine.paused", {})
        return True
    async def resume(self) -> bool:
        """恢復引擎"""
        if self._state != EngineState.PAUSED:
            return False
        self._state = EngineState.RUNNING
        await self._emit_event("engine.resumed", {})
        return True
    # ========================================================================
    # 任務處理
    # ========================================================================
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """提交任務"""
        task_id = task.get("task_id") or str(uuid.uuid4())
        task["task_id"] = task_id
        task["submitted_at"] = datetime.now().isoformat()
        await self._task_queue.put(task)
        self._logger.debug(f"任務已提交: {task_id}")
        return task_id
    async def execute_now(self, task: Dict[str, Any]) -> TaskResult:
        """立即執行任務 (繞過隊列)"""
        task_id = task.get("task_id") or str(uuid.uuid4())
        task["task_id"] = task_id
        return await self._execute_with_retry(task)
    async def _main_loop(self):
        """主執行循環"""
        while self._running:
            try:
                # 檢查狀態
                if self._state == EngineState.PAUSED:
                    await asyncio.sleep(0.5)
                    continue
                # 獲取任務
                try:
                    task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                # 檢查並發限制
                while (
                    len(self._active_tasks) >= self.config.resource.max_concurrent_tasks
                ):
                    await asyncio.sleep(0.1)
                # 執行任務
                asyncio.create_task(self._process_task(task))
            except Exception as e:
                self._logger.error(f"主循環錯誤: {e}")
                await asyncio.sleep(1)
    async def _process_task(self, task: Dict[str, Any]):
        """處理單一任務"""
        task_id = task["task_id"]
        self._active_tasks.add(task_id)
        try:
            result = await self._execute_with_retry(task)
            if result.success:
                self._tasks_completed += 1
            else:
                self._tasks_failed += 1
            self._total_execution_time += result.duration_ms
            self._last_activity = datetime.now()
            await self._emit_event(
                "task.completed",
                {
                    "task_id": task_id,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                },
            )
        except Exception as e:
            self._tasks_failed += 1
            self._logger.error(f"任務處理失敗 {task_id}: {e}")
            await self._emit_event(
                "task.failed",
                {
                    "task_id": task_id,
                    "error": str(e),
                },
            )
        finally:
            self._active_tasks.discard(task_id)
    async def _execute_with_retry(self, task: Dict[str, Any]) -> TaskResult:
        """帶重試的執行"""
        task_id = task["task_id"]
        retry_config = self.config.retry
        last_error = None
        delay = retry_config.initial_delay
        for attempt in range(retry_config.max_attempts):
            try:
                start_time = datetime.now()
                result = await asyncio.wait_for(
                    self._execute(task), timeout=self.config.timeout.execution
                )
                result.duration_ms = (
                    datetime.now() - start_time
                ).total_seconds() * 1000
                return result
            except tuple(retry_config.retry_on) as e:
                last_error = e
                self._logger.warning(f"任務 {task_id} 第 {attempt + 1} 次嘗試失敗: {e}")
                if attempt < retry_config.max_attempts - 1:
                    # 指數退避
                    if retry_config.jitter:
                        import random
                        delay *= retry_config.exponential_base + random.uniform(
                            -0.1, 0.1
                        )
                    else:
                        delay *= retry_config.exponential_base
                    delay = min(delay, retry_config.max_delay)
                    await asyncio.sleep(delay)
            except asyncio.TimeoutError:
                last_error = TimeoutError("執行超時")
                self._logger.warning(f"任務 {task_id} 執行超時")
                break
        return TaskResult(
            task_id=task_id,
            success=False,
            error=str(last_error),
        )
    # ========================================================================
    # 事件系統
    # ========================================================================
    def on_event(self, event_type: str, handler: Callable):
        """註冊事件處理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    async def _emit_event(self, event_type: str, payload: Dict[str, Any]):
        """發送事件"""
        event = EngineEvent.create(event_type, self.engine_id, payload)
        handlers = self._event_handlers.get(event_type, [])
        handlers.extend(self._event_handlers.get("*", []))  # 全局處理器
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"事件處理器錯誤: {e}")
    # ========================================================================
    # 健康監控
    # ========================================================================
    async def _heartbeat_loop(self):
        """心跳循環"""
        while self._running:
            try:
                await self._emit_event(
                    "engine.heartbeat",
                    {
                        "state": self._state.name,
                        "active_tasks": len(self._active_tasks),
                        "queue_size": (
                            self._task_queue.qsize() if self._task_queue else 0
                        ),
                    },
                )
                await asyncio.sleep(self.config.timeout.heartbeat)
            except Exception as e:
                self._logger.error(f"心跳錯誤: {e}")
    def get_health(self) -> HealthStatus:
        """獲取健康狀態"""
        total_tasks = self._tasks_completed + self._tasks_failed
        error_rate = self._tasks_failed / total_tasks if total_tasks > 0 else 0.0
        return HealthStatus(
            healthy=self._state == EngineState.RUNNING and error_rate < 0.5,
            state=self._state,
            uptime_seconds=self.uptime.total_seconds(),
            tasks_completed=self._tasks_completed,
            tasks_failed=self._tasks_failed,
            last_activity=(
                self._last_activity.isoformat() if self._last_activity else ""
            ),
            error_rate=error_rate,
            details={
                "active_tasks": len(self._active_tasks),
                "queue_size": self._task_queue.qsize() if self._task_queue else 0,
                "avg_task_time_ms": (
                    self._total_execution_time / self._tasks_completed
                    if self._tasks_completed > 0
                    else 0
                ),
            },
        )
    # ========================================================================
    # 檢查點 (持久化)
    # ========================================================================
    async def _checkpoint_loop(self):
        """檢查點循環"""
        while self._running:
            try:
                await asyncio.sleep(self.config.persistence.checkpoint_interval)
                await self._save_checkpoint()
            except Exception as e:
                self._logger.error(f"檢查點錯誤: {e}")
    async def _save_checkpoint(self):
        """保存檢查點"""
        state_dir = Path(self.config.persistence.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "engine_id": self.engine_id,
            "timestamp": datetime.now().isoformat(),
            "state": self._state.name,
            "statistics": {
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                "total_execution_time": self._total_execution_time,
            },
            "custom_data": self._checkpoint_data,
        }
        checkpoint_file = state_dir / f"{self.engine_id}_checkpoint.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    async def _load_checkpoint(self):
        """載入檢查點"""
        state_dir = Path(self.config.persistence.state_dir)
        checkpoint_file = state_dir / f"{self.engine_id}_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                self._tasks_completed = checkpoint.get("statistics", {}).get(
                    "tasks_completed", 0
                )
                self._tasks_failed = checkpoint.get("statistics", {}).get(
                    "tasks_failed", 0
                )
                self._checkpoint_data = checkpoint.get("custom_data", {})
                self._logger.info(f"已載入檢查點: {checkpoint_file}")
            except Exception as e:
                self._logger.warning(f"載入檢查點失敗: {e}")
    # ========================================================================
    # 自我修復
    # ========================================================================
    async def _recover(self) -> bool:
        """自我修復"""
        if not self.config.auto_recover:
            return False
        self._logger.info("嘗試自我修復...")
        self._state = EngineState.RECOVERING
        try:
            # 重置錯誤計數
            error_threshold = 10
            if self._tasks_failed > error_threshold:
                self._tasks_failed = 0
            # 清理任務隊列
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except BaseException:
                    break
            # 重新初始化
            await self._initialize()
            self._state = EngineState.RUNNING
            await self._emit_event("engine.recovered", {})
            self._logger.info("自我修復成功")
            return True
        except Exception as e:
            self._logger.error(f"自我修復失敗: {e}")
            self._state = EngineState.ERROR
            return False
    # ========================================================================
    # 工具方法
    # ========================================================================
    def get_info(self) -> Dict[str, Any]:
        """獲取引擎資訊"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "engine_type": self.config.engine_type.value,
            "version": self.config.version,
            "state": self._state.name,
            "execution_mode": self.config.execution_mode.value,
            "capabilities": self._get_capabilities(),
            "health": asdict(self.get_health()),
        }
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.engine_id}, state={self._state.name})>"
# ============================================================================
# 專用引擎基類
# ============================================================================
class CognitiveEngineBase(BaseEngine):
    """認知引擎基類 - 提供高階推理能力"""
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig()
        config.engine_type = EngineType.COGNITIVE
        super().__init__(config)
        self._reasoning_trace: List[Dict] = []
        self._knowledge_base: Dict[str, Any] = {}
    @abstractmethod
    async def understand(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """理解輸入"""
        pass
    @abstractmethod
    async def reason(self, understanding: Dict[str, Any]) -> Dict[str, Any]:
        """推理分析"""
        pass
    @abstractmethod
    async def decide(self, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """做出決策"""
        pass
class ExecutionEngineBase(BaseEngine):
    """執行引擎基類 - 提供操作執行能力"""
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig()
        config.engine_type = EngineType.EXECUTION
        super().__init__(config)
        self._executed_operations: List[Dict] = []
        self._rollback_stack: List[Dict] = []
    @abstractmethod
    async def execute_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """執行操作"""
        pass
    @abstractmethod
    async def rollback(self, steps: int = 1) -> bool:
        """回滾操作"""
        pass
class ValidationEngineBase(BaseEngine):
    """驗證引擎基類 - 提供檢查驗證能力"""
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig()
        config.engine_type = EngineType.VALIDATION
        super().__init__(config)
        self._validation_rules: List[Dict] = []
        self._validation_results: List[Dict] = []
    @abstractmethod
    async def validate(self, target: Any, rules: List[str] = None) -> Dict[str, Any]:
        """執行驗證"""
        pass
    @abstractmethod
    async def fix(self, issues: List[Dict]) -> Dict[str, Any]:
        """自動修復問題"""
        pass
class TransformEngineBase(BaseEngine):
    """轉換引擎基類 - 提供數據轉換能力"""
    def __init__(self, config: Optional[EngineConfig] = None):
        config = config or EngineConfig()
        config.engine_type = EngineType.TRANSFORM
        super().__init__(config)
        self._transformers: Dict[str, Callable] = {}
    @abstractmethod
    async def transform(self, data: Any, target_format: str) -> Any:
        """轉換數據"""
        pass
    def register_transformer(self, name: str, transformer: Callable):
        """註冊轉換器"""
        self._transformers[name] = transformer
