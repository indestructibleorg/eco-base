"""
闭环验证门檻系统 (Verification Gate)

强制治理规范核心组件
验证不通过 = 自动回滚或升级人工，禁止「预设成功」
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """验证状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class VerificationStrategy(Enum):
    """验证失败处理策略"""
    ROLLBACK = "rollback"      # 自动回滚
    ESCALATE = "escalate"      # 升级人工
    ALERT = "alert"            # 仅告警
    IGNORE = "ignore"          # 忽略


@dataclass
class MetricCheck:
    """指标检查项"""
    metric: str
    expected: str  # 表达式，如 "< 70%"
    actual: Optional[str] = None
    passed: bool = False
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "timestamp": self.timestamp
        }


@dataclass
class VerificationResult:
    """验证结果"""
    verification_id: str
    action_id: str
    status: VerificationStatus
    checks: List[MetricCheck]
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    trigger_rollback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_passed(self) -> bool:
        """是否所有检查通过"""
        return all(c.passed for c in self.checks)
    
    @property
    def pass_ratio(self) -> float:
        """通过比例"""
        if not self.checks:
            return 0.0
        passed = sum(1 for c in self.checks if c.passed)
        return passed / len(self.checks)


@dataclass
class VerificationConfig:
    """验证配置"""
    enabled: bool = True
    timeout_seconds: int = 600
    check_interval_seconds: int = 10
    consecutive_success_period_seconds: int = 180
    min_pass_ratio: float = 1.0
    all_checks_required: bool = True
    on_failure_strategy: VerificationStrategy = VerificationStrategy.ROLLBACK
    auto_rollback: bool = True
    escalation_delay_seconds: int = 300


class MetricCollector:
    """指标收集器"""
    
    def __init__(self):
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._prometheus_url = os.environ.get('PROMETHEUS_URL', 'http://prometheus:9090')
    
    async def collect(self, metric_name: str, labels: Optional[Dict] = None) -> Optional[float]:
        """
        收集指标值
        
        从 Prometheus 或其他监控系统获取指标
        """
        # 首先尝试从 Prometheus 获取
        value = await self._fetch_from_prometheus(metric_name, labels)
        if value is not None:
            return value
        
        # 如果 Prometheus 不可用，尝试从缓存获取
        cache_key = f"{metric_name}:{json.dumps(labels, sort_keys=True) if labels else ''}"
        if cache_key in self._metrics_cache:
            cached = self._metrics_cache[cache_key]
            # 缓存有效期 5 分钟
            if datetime.now() - cached['timestamp'] < timedelta(minutes=5):
                return cached['value']
        
        # 返回 None 表示无法获取指标
        return None
    
    async def _fetch_from_prometheus(
        self,
        metric_name: str,
        labels: Optional[Dict] = None
    ) -> Optional[float]:
        """从 Prometheus 获取指标"""
        try:
            import aiohttp
            
            # 构建查询
            query = metric_name
            if labels:
                label_selectors = ','.join([f'{k}="{v}"' for k, v in labels.items()])
                query = f'{metric_name}{{{label_selectors}}}'
            
            url = f"{self._prometheus_url}/api/v1/query"
            params = {"query": query}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if data.get('status') != 'success':
                        return None
                    
                    results = data.get('data', {}).get('result', [])
                    if not results:
                        return None
                    
                    # 获取最新值
                    value = results[0].get('value', [None, None])
                    if len(value) >= 2:
                        return float(value[1])
                    
                    return None
                    
        except ImportError:
            # aiohttp 未安装
            return None
        except Exception:
            # 其他错误
            return None
    
    def cache_metric(self, metric_name: str, value: float, labels: Optional[Dict] = None) -> None:
        """缓存指标值"""
        cache_key = f"{metric_name}:{json.dumps(labels, sort_keys=True) if labels else ''}"
        self._metrics_cache[cache_key] = {
            'value': value,
            'timestamp': datetime.now(),
        }
    
    async def collect_batch(
        self,
        metrics: List[str],
        labels: Optional[Dict] = None
    ) -> Dict[str, Optional[float]]:
        """批量收集指标"""
        results = {}
        for metric in metrics:
            results[metric] = await self.collect(metric, labels)
        return results


class ExpressionEvaluator:
    """表达式求值器"""
    
    # 支持的比较操作符
    OPERATORS = {
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }
    
    def evaluate(self, actual_value: float, expected_expression: str) -> bool:
        """
        评估表达式
        
        Args:
            actual_value: 实际值
            expected_expression: 期望表达式，如 "< 70%"
        
        Returns:
            是否满足条件
        """
        # 解析表达式
        expression = expected_expression.strip()
        
        # 移除百分号并转换
        if "%" in expression:
            expression = expression.replace("%", "")
            actual_value = actual_value  # 已经是百分比形式
        
        # 提取操作符和阈值
        for op_str, op_func in self.OPERATORS.items():
            if expression.startswith(op_str):
                try:
                    threshold = float(expression[len(op_str):].strip())
                    return op_func(actual_value, threshold)
                except ValueError:
                    logger.error(f"Invalid threshold in expression: {expected_expression}")
                    return False
        
        # 尝试直接比较
        try:
            threshold = float(expression)
            return actual_value == threshold
        except ValueError:
            logger.error(f"Cannot evaluate expression: {expected_expression}")
            return False


class VerificationGate:
    """
    闭环验证门檻
    
    核心规则:
    - 验证必须启用
    - 验证不通过 = 自动回滚或升级人工
    - 禁止「预设成功」
    """
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()
        self.metric_collector = MetricCollector()
        self.expression_evaluator = ExpressionEvaluator()
        
        # 验证结果存储
        self._verification_results: Dict[str, VerificationResult] = {}
        
        # 回调函数
        self._success_callbacks: List[Callable[[VerificationResult], None]] = []
        self._failure_callbacks: List[Callable[[VerificationResult], None]] = []
    
    async def verify(
        self,
        action_id: str,
        checks: List[Dict[str, str]],
        labels: Optional[Dict] = None
    ) -> VerificationResult:
        """
        执行验证
        
        Args:
            action_id: 动作ID
            checks: 检查项列表
            labels: 指标标签
        
        Returns:
            验证结果
        """
        if not self.config.enabled:
            logger.warning("Verification is disabled, returning skipped result")
            return VerificationResult(
                verification_id=f"vrf_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                action_id=action_id,
                status=VerificationStatus.SKIPPED,
                checks=[],
                message="Verification is disabled",
                started_at=datetime.now()
            )
        
        verification_id = f"vrf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{action_id[-8:]}"
        start_time = datetime.now()
        
        logger.info(f"Starting verification for action {action_id}")
        
        # 创建验证结果对象
        result = VerificationResult(
            verification_id=verification_id,
            action_id=action_id,
            status=VerificationStatus.IN_PROGRESS,
            checks=[MetricCheck(**c) for c in checks],
            message="Verification in progress",
            started_at=start_time
        )
        
        self._verification_results[verification_id] = result
        
        try:
            # 执行验证循环
            verification_passed = await self._run_verification_loop(result, labels)
            
            # 计算持续时间
            end_time = datetime.now()
            result.duration_ms = (end_time - start_time).total_seconds() * 1000
            result.completed_at = end_time
            
            if verification_passed:
                result.status = VerificationStatus.PASSED
                result.message = "All verification checks passed"
                result.trigger_rollback = False
                
                logger.info(f"Verification {verification_id} passed for action {action_id}")
                
                # 触发成功回调
                for callback in self._success_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as e:
                        logger.warning(f"Verification success callback failed: {e}")
                
            else:
                result.status = VerificationStatus.FAILED
                result.message = "Verification checks failed"
                result.trigger_rollback = self.config.auto_rollback
                
                logger.warning(f"Verification {verification_id} failed for action {action_id}")
                
                # 触发失败回调
                for callback in self._failure_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(result)
                        else:
                            callback(result)
                    except Exception as e:
                        logger.warning(f"Verification failure callback failed: {e}")
            
            return result
            
        except asyncio.TimeoutError:
            result.status = VerificationStatus.TIMEOUT
            result.message = f"Verification timeout after {self.config.timeout_seconds}s"
            result.completed_at = datetime.now()
            result.duration_ms = (result.completed_at - start_time).total_seconds() * 1000
            
            logger.warning(f"Verification {verification_id} timeout")
            return result
        
        except Exception as e:
            result.status = VerificationStatus.FAILED
            result.message = f"Verification error: {str(e)}"
            result.completed_at = datetime.now()
            result.duration_ms = (result.completed_at - start_time).total_seconds() * 1000
            
            logger.exception(f"Verification {verification_id} error: {e}")
            return result
    
    async def _run_verification_loop(
        self,
        result: VerificationResult,
        labels: Optional[Dict]
    ) -> bool:
        """
        运行验证循环
        
        Returns:
            是否验证通过
        """
        timeout = self.config.timeout_seconds
        check_interval = self.config.check_interval_seconds
        consecutive_period = self.config.consecutive_success_period_seconds
        
        start_time = datetime.now()
        consecutive_success_start: Optional[datetime] = None
        
        while True:
            # 检查是否超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise asyncio.TimeoutError()
            
            # 执行一次检查
            all_passed = await self._execute_checks(result.checks, labels)
            
            if all_passed:
                # 记录连续成功开始时间
                if consecutive_success_start is None:
                    consecutive_success_start = datetime.now()
                
                # 检查是否满足连续成功时间
                success_duration = (datetime.now() - consecutive_success_start).total_seconds()
                if success_duration >= consecutive_period:
                    logger.info(
                        f"Verification passed - all checks passed for {success_duration}s"
                    )
                    return True
            else:
                # 重置连续成功计时
                consecutive_success_start = None
            
            # 等待下一次检查
            await asyncio.sleep(check_interval)
    
    async def _execute_checks(
        self,
        checks: List[MetricCheck],
        labels: Optional[Dict]
    ) -> bool:
        """
        执行检查项
        
        Returns:
            是否全部通过
        """
        all_passed = True
        
        for check in checks:
            try:
                # 收集指标
                actual_value = await self.metric_collector.collect(check.metric, labels)
                
                if actual_value is None:
                    logger.warning(f"Metric {check.metric} not available")
                    check.passed = False
                    check.actual = "N/A"
                    all_passed = False
                    continue
                
                # 评估表达式
                passed = self.expression_evaluator.evaluate(actual_value, check.expected)
                
                check.actual = str(actual_value)
                check.passed = passed
                check.timestamp = datetime.now().isoformat()
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                logger.exception(f"Check {check.metric} failed: {e}")
                check.passed = False
                all_passed = False
        
        return all_passed
    
    def get_verification_result(self, verification_id: str) -> Optional[VerificationResult]:
        """获取验证结果"""
        return self._verification_results.get(verification_id)
    
    def get_action_verifications(self, action_id: str) -> List[VerificationResult]:
        """获取动作的所有验证结果"""
        return [
            r for r in self._verification_results.values()
            if r.action_id == action_id
        ]
    
    def register_success_callback(self, callback: Callable):
        """注册验证成功回调"""
        self._success_callbacks.append(callback)
    
    def register_failure_callback(self, callback: Callable):
        """注册验证失败回调"""
        self._failure_callbacks.append(callback)
    
    def should_trigger_rollback(self, verification_id: str) -> bool:
        """检查是否应该触发回滚"""
        result = self._verification_results.get(verification_id)
        if not result:
            return False
        
        if result.status == VerificationStatus.FAILED:
            return self.config.auto_rollback
        
        return False


class RollbackTrigger:
    """
    回滚触发器
    
    根据验证结果自动触发回滚
    """
    
    def __init__(self, verification_gate: VerificationGate):
        self.verification_gate = verification_gate
        self._rollback_handlers: Dict[str, Callable] = {}
    
    def register_rollback_handler(self, action_type: str, handler: Callable):
        """注册回滚处理器"""
        self._rollback_handlers[action_type] = handler
    
    async def check_and_trigger(self, verification_id: str) -> bool:
        """
        检查并触发回滚
        
        Returns:
            是否触发了回滚
        """
        result = self.verification_gate.get_verification_result(verification_id)
        if not result:
            logger.error(f"Verification result not found: {verification_id}")
            return False
        
        if not self.verification_gate.should_trigger_rollback(verification_id):
            logger.info(f"Rollback not triggered for verification {verification_id}")
            return False
        
        logger.warning(
            f"Triggering rollback for action {result.action_id} "
            f"due to verification failure"
        )
        
        # 触发回滚
        # 实际实现中应调用相应的回滚处理器
        
        return True


class VerificationReporter:
    """
    验证报告生成器
    """
    
    def __init__(self, verification_gate: VerificationGate):
        self.verification_gate = verification_gate
    
    def generate_report(self, verification_id: str) -> Dict[str, Any]:
        """生成验证报告"""
        result = self.verification_gate.get_verification_result(verification_id)
        if not result:
            return {"error": "Verification result not found"}
        
        return {
            "verification_id": result.verification_id,
            "action_id": result.action_id,
            "status": result.status.value,
            "message": result.message,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration_ms": result.duration_ms,
            "pass_ratio": result.pass_ratio,
            "all_passed": result.all_passed,
            "checks": [c.to_dict() for c in result.checks],
            "trigger_rollback": result.trigger_rollback
        }
    
    def generate_summary(self, action_id: str) -> Dict[str, Any]:
        """生成验证摘要"""
        verifications = self.verification_gate.get_action_verifications(action_id)
        
        if not verifications:
            return {"error": "No verification results found"}
        
        total = len(verifications)
        passed = sum(1 for v in verifications if v.status == VerificationStatus.PASSED)
        failed = sum(1 for v in verifications if v.status == VerificationStatus.FAILED)
        
        return {
            "action_id": action_id,
            "total_verifications": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "latest_status": verifications[-1].status.value if verifications else None
        }
