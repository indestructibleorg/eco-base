"""
级联控制器
故障隔离、优雅降级、恢复协调
"""

import time
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class CascadeActionType(Enum):
    """级联动作类型"""
    ISOLATE = "isolate"
    DEGRADE = "degrade"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    RESTORE = "restore"


class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ISOLATED = "isolated"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class CascadeAction:
    """级联动作"""
    action_id: str
    action_type: CascadeActionType
    target_service: str
    parameters: Dict[str, Any]
    triggered_at: datetime
    executed_at: Optional[datetime]
    status: str
    result: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'action_type': self.action_type.value,
            'target_service': self.target_service,
            'parameters': self.parameters,
            'triggered_at': self.triggered_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'status': self.status,
            'result': self.result
        }


@dataclass
class ServiceState:
    """服务状态"""
    service_id: str
    status: ServiceStatus
    health_score: float
    last_health_check: datetime
    circuit_breaker_state: str  # closed, open, half_open
    failure_count: int
    last_failure: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_id': self.service_id,
            'status': self.status.value,
            'health_score': self.health_score,
            'last_health_check': self.last_health_check.isoformat(),
            'circuit_breaker_state': self.circuit_breaker_state,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure.isoformat() if self.last_failure else None
        }


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: int = 60,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = 'closed'  # closed, open, half_open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == 'closed':
            return True
        elif self.state == 'open':
            # 检查是否超过恢复超时
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed > self.recovery_timeout:
                    self.state = 'half_open'
                    self.half_open_calls = 0
                    return True
            return False
        elif self.state == 'half_open':
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self):
        """记录成功"""
        if self.state == 'half_open':
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = 'closed'
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == 'half_open':
            self.state = 'open'
        elif self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class FaultIsolator:
    """故障隔离器"""
    
    def __init__(self, topology: Dict[str, List[str]] = None):
        self.topology = topology or {}
        self.isolated_services: Set[str] = set()
        
    def isolate_service(self, service_id: str, 
                       isolate_upstream: bool = True,
                       isolate_downstream: bool = False) -> List[str]:
        """隔离服务"""
        isolated = [service_id]
        self.isolated_services.add(service_id)
        
        if isolate_upstream:
            # 找到依赖该服务的服务
            upstream = self._find_upstream(service_id)
            for svc in upstream:
                self.isolated_services.add(svc)
                isolated.append(svc)
        
        if isolate_downstream:
            # 找到该服务依赖的服务
            downstream = self.topology.get(service_id, [])
            for svc in downstream:
                self.isolated_services.add(svc)
                isolated.append(svc)
        
        logger.info(f"Isolated services: {isolated}")
        return isolated
    
    def _find_upstream(self, service_id: str) -> List[str]:
        """查找上游服务"""
        upstream = []
        for svc, deps in self.topology.items():
            if service_id in deps:
                upstream.append(svc)
        return upstream
    
    def restore_service(self, service_id: str) -> bool:
        """恢复服务"""
        if service_id in self.isolated_services:
            self.isolated_services.remove(service_id)
            logger.info(f"Restored service: {service_id}")
            return True
        return False
    
    def is_isolated(self, service_id: str) -> bool:
        """检查服务是否被隔离"""
        return service_id in self.isolated_services


class GracefulDegradation:
    """优雅降级"""
    
    def __init__(self):
        self.degradation_strategies: Dict[str, Dict[str, Any]] = {
            'reduce_features': {
                'description': 'Reduce non-critical features',
                'priority': 1
            },
            'limit_requests': {
                'description': 'Limit request rate',
                'priority': 2
            },
            'serve_stale_data': {
                'description': 'Serve cached/stale data',
                'priority': 3
            },
            'disable_non_critical': {
                'description': 'Disable non-critical services',
                'priority': 4
            }
        }
        
        self.active_degradations: Dict[str, List[str]] = {}
        
    def apply_degradation(self, service_id: str, 
                         strategy: str,
                         parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """应用降级策略"""
        if strategy not in self.degradation_strategies:
            return {'error': f'Unknown degradation strategy: {strategy}'}
        
        if service_id not in self.active_degradations:
            self.active_degradations[service_id] = []
        
        if strategy not in self.active_degradations[service_id]:
            self.active_degradations[service_id].append(strategy)
        
        logger.info(f"Applied degradation {strategy} to {service_id}")
        
        return {
            'service': service_id,
            'strategy': strategy,
            'status': 'applied',
            'parameters': parameters or {}
        }
    
    def remove_degradation(self, service_id: str, 
                          strategy: str = None) -> Dict[str, Any]:
        """移除降级策略"""
        if service_id not in self.active_degradations:
            return {'error': 'No active degradations for service'}
        
        if strategy:
            if strategy in self.active_degradations[service_id]:
                self.active_degradations[service_id].remove(strategy)
                logger.info(f"Removed degradation {strategy} from {service_id}")
        else:
            # 移除所有降级
            self.active_degradations[service_id] = []
            logger.info(f"Removed all degradations from {service_id}")
        
        return {'service': service_id, 'status': 'restored'}
    
    def get_degradation_status(self, service_id: str) -> Dict[str, Any]:
        """获取降级状态"""
        active = self.active_degradations.get(service_id, [])
        return {
            'service': service_id,
            'active_degradations': active,
            'degradation_level': len(active)
        }


class RecoveryCoordinator:
    """恢复协调器"""
    
    def __init__(self):
        self.recovery_queue: deque = deque()
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_concurrent_recoveries = 2
        self.active_recoveries: Set[str] = set()
        
    def schedule_recovery(self, service_id: str, 
                         recovery_action: str,
                         priority: int = 5) -> str:
        """调度恢复"""
        recovery_id = f"rec_{service_id}_{int(time.time())}"
        
        recovery_task = {
            'recovery_id': recovery_id,
            'service_id': service_id,
            'action': recovery_action,
            'priority': priority,
            'scheduled_at': datetime.now(),
            'status': 'queued'
        }
        
        # 按优先级插入队列
        inserted = False
        for i, task in enumerate(self.recovery_queue):
            if task['priority'] > priority:
                self.recovery_queue.insert(i, recovery_task)
                inserted = True
                break
        
        if not inserted:
            self.recovery_queue.append(recovery_task)
        
        logger.info(f"Scheduled recovery {recovery_id} for {service_id}")
        
        return recovery_id
    
    def execute_next_recovery(self, executor: Callable = None) -> Optional[Dict[str, Any]]:
        """执行下一个恢复任务"""
        if len(self.active_recoveries) >= self.max_concurrent_recoveries:
            return None
        
        if not self.recovery_queue:
            return None
        
        task = self.recovery_queue.popleft()
        service_id = task['service_id']
        
        if service_id in self.active_recoveries:
            # 服务已在恢复中，跳过
            return self.execute_next_recovery(executor)
        
        self.active_recoveries.add(service_id)
        task['status'] = 'executing'
        task['started_at'] = datetime.now()
        
        try:
            if executor:
                result = executor(task)
            else:
                # 模拟恢复
                result = self._simulate_recovery(task)
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.now()
            task['result'] = result
            
            self.recovery_history.append(task)
            
            logger.info(f"Recovery completed for {service_id}")
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            self.recovery_history.append(task)
            
            logger.error(f"Recovery failed for {service_id}: {e}")
        
        finally:
            self.active_recoveries.discard(service_id)
        
        return task
    
    def _simulate_recovery(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """模拟恢复"""
        time.sleep(0.1)  # 模拟耗时
        return {
            'service': task['service_id'],
            'action': task['action'],
            'status': 'recovered'
        }
    
    def get_recovery_status(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """获取恢复状态"""
        # 检查队列
        for task in self.recovery_queue:
            if task['recovery_id'] == recovery_id:
                return task
        
        # 检查历史
        for task in self.recovery_history:
            if task['recovery_id'] == recovery_id:
                return task
        
        return None


class CascadeController:
    """
    级联控制器主类
    整合故障隔离、优雅降级、恢复协调
    """
    
    def __init__(self, topology: Dict[str, List[str]] = None):
        self.topology = topology or {}
        
        self.fault_isolator = FaultIsolator(topology)
        self.degradation = GracefulDegradation()
        self.recovery_coordinator = RecoveryCoordinator()
        
        # 服务状态
        self.service_states: Dict[str, ServiceState] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 级联动作历史
        self.action_history: List[CascadeAction] = []
        
    def register_service(self, service_id: str):
        """注册服务"""
        self.service_states[service_id] = ServiceState(
            service_id=service_id,
            status=ServiceStatus.HEALTHY,
            health_score=1.0,
            last_health_check=datetime.now(),
            circuit_breaker_state='closed',
            failure_count=0,
            last_failure=None
        )
        
        self.circuit_breakers[service_id] = CircuitBreaker()
        
    def handle_failure(self, service_id: str, 
                      failure_type: str,
                      parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理故障"""
        parameters = parameters or {}
        
        # 更新服务状态
        state = self.service_states.get(service_id)
        if state:
            state.failure_count += 1
            state.last_failure = datetime.now()
            state.status = ServiceStatus.FAILED
        
        # 更新熔断器
        cb = self.circuit_breakers.get(service_id)
        if cb:
            cb.record_failure()
        
        # 执行级联控制
        actions = []
        
        # 1. 故障隔离
        if parameters.get('isolate', True):
            isolated = self.fault_isolator.isolate_service(
                service_id,
                isolate_upstream=parameters.get('isolate_upstream', True),
                isolate_downstream=parameters.get('isolate_downstream', False)
            )
            
            action = CascadeAction(
                action_id=f"iso_{service_id}_{int(time.time())}",
                action_type=CascadeActionType.ISOLATE,
                target_service=service_id,
                parameters={'isolated_services': isolated},
                triggered_at=datetime.now(),
                executed_at=datetime.now(),
                status='completed',
                result={'isolated_count': len(isolated)}
            )
            actions.append(action)
        
        # 2. 优雅降级
        if parameters.get('degrade', True):
            degradation_result = self.degradation.apply_degradation(
                service_id,
                'reduce_features',
                parameters.get('degradation_params', {})
            )
            
            action = CascadeAction(
                action_id=f"deg_{service_id}_{int(time.time())}",
                action_type=CascadeActionType.DEGRADE,
                target_service=service_id,
                parameters=degradation_result,
                triggered_at=datetime.now(),
                executed_at=datetime.now(),
                status='completed',
                result=degradation_result
            )
            actions.append(action)
        
        # 3. 熔断
        if parameters.get('circuit_break', True):
            action = CascadeAction(
                action_id=f"cb_{service_id}_{int(time.time())}",
                action_type=CascadeActionType.CIRCUIT_BREAK,
                target_service=service_id,
                parameters={'circuit_state': 'open'},
                triggered_at=datetime.now(),
                executed_at=datetime.now(),
                status='completed',
                result={'circuit_breaker': 'opened'}
            )
            actions.append(action)
        
        # 存储动作
        self.action_history.extend(actions)
        
        return {
            'service_id': service_id,
            'failure_type': failure_type,
            'actions_taken': [a.to_dict() for a in actions],
            'isolated_services': self.fault_isolator.isolated_services
        }
    
    def recover_service(self, service_id: str,
                       recovery_action: str = 'restart') -> Dict[str, Any]:
        """恢复服务"""
        # 调度恢复
        recovery_id = self.recovery_coordinator.schedule_recovery(
            service_id, recovery_action, priority=1
        )
        
        # 执行恢复
        result = self.recovery_coordinator.execute_next_recovery()
        
        if result and result.get('status') == 'completed':
            # 恢复隔离
            self.fault_isolator.restore_service(service_id)
            
            # 移除降级
            self.degradation.remove_degradation(service_id)
            
            # 更新状态
            state = self.service_states.get(service_id)
            if state:
                state.status = ServiceStatus.HEALTHY
                state.health_score = 1.0
                state.failure_count = 0
            
            # 重置熔断器
            cb = self.circuit_breakers.get(service_id)
            if cb:
                cb.state = 'closed'
                cb.failure_count = 0
            
            logger.info(f"Service {service_id} recovered")
        
        return {
            'service_id': service_id,
            'recovery_id': recovery_id,
            'status': result.get('status') if result else 'failed',
            'result': result
        }
    
    def get_service_status(self, service_id: str) -> Dict[str, Any]:
        """获取服务状态"""
        state = self.service_states.get(service_id)
        cb = self.circuit_breakers.get(service_id)
        degradation = self.degradation.get_degradation_status(service_id)
        
        return {
            'service_id': service_id,
            'state': state.to_dict() if state else None,
            'circuit_breaker': {
                'state': cb.state if cb else 'unknown',
                'failure_count': cb.failure_count if cb else 0
            },
            'degradation': degradation,
            'isolated': self.fault_isolator.is_isolated(service_id)
        }
    
    def get_cascade_status(self) -> Dict[str, Any]:
        """获取级联状态"""
        total_services = len(self.service_states)
        healthy = sum(1 for s in self.service_states.values() if s.status == ServiceStatus.HEALTHY)
        failed = sum(1 for s in self.service_states.values() if s.status == ServiceStatus.FAILED)
        isolated = len(self.fault_isolator.isolated_services)
        
        return {
            'total_services': total_services,
            'healthy': healthy,
            'failed': failed,
            'isolated': isolated,
            'active_degradations': len(self.degradation.active_degradations),
            'pending_recoveries': len(self.recovery_coordinator.recovery_queue),
            'circuit_breakers_open': sum(
                1 for cb in self.circuit_breakers.values() if cb.state == 'open'
            )
        }
