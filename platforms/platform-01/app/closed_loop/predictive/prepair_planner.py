"""
预修复规划器
规划预测性修复策略和维护窗口
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MaintenanceWindowType(Enum):
    """维护窗口类型"""
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"
    ROLLING = "rolling"
    CANARY = "canary"


@dataclass
class MaintenanceWindow:
    """维护窗口"""
    window_id: str
    start_time: datetime
    end_time: datetime
    window_type: MaintenanceWindowType
    affected_services: List[str]
    estimated_duration_minutes: float
    risk_level: str
    approval_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'window_id': self.window_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_minutes': (self.end_time - self.start_time).total_seconds() / 60,
            'window_type': self.window_type.value,
            'affected_services': self.affected_services,
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'risk_level': self.risk_level,
            'approval_required': self.approval_required
        }


@dataclass
class PrepairAction:
    """预修复动作"""
    action_id: str
    action_type: str
    target_service: str
    scheduled_time: datetime
    estimated_duration_minutes: float
    rollback_plan: Dict[str, Any]
    prerequisites: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'action_type': self.action_type,
            'target_service': self.target_service,
            'scheduled_time': self.scheduled_time.isoformat(),
            'estimated_duration_minutes': self.estimated_duration_minutes,
            'rollback_plan': self.rollback_plan,
            'prerequisites': self.prerequisites
        }


@dataclass
class PrepairPlan:
    """预修复计划"""
    plan_id: str
    created_at: datetime
    predictions: List[Dict[str, Any]]
    maintenance_windows: List[MaintenanceWindow]
    prepair_actions: List[PrepairAction]
    overall_risk: str
    estimated_success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'created_at': self.created_at.isoformat(),
            'predictions': self.predictions,
            'maintenance_windows': [w.to_dict() for w in self.maintenance_windows],
            'prepair_actions': [a.to_dict() for a in self.prepair_actions],
            'overall_risk': self.overall_risk,
            'estimated_success_rate': self.estimated_success_rate
        }


class MaintenanceWindowOptimizer:
    """维护窗口优化器"""
    
    def __init__(self):
        # 业务低峰时段配置
        self.low_traffic_windows = [
            # 工作日凌晨
            {'start_hour': 2, 'end_hour': 5, 'days': [0, 1, 2, 3, 4], 'weight': 1.0},
            # 周末凌晨
            {'start_hour': 1, 'end_hour': 6, 'days': [5, 6], 'weight': 0.9},
        ]
        
        # 节假日配置
        self.holidays = []
        
    def find_optimal_windows(self, predictions: List[Dict[str, Any]],
                            constraints: Dict[str, Any]) -> List[MaintenanceWindow]:
        """查找最优维护窗口"""
        windows = []
        
        # 分析预测结果
        high_risk_predictions = [p for p in predictions if p.get('failure_probability', 0) > 0.7]
        
        for prediction in high_risk_predictions:
            service_id = prediction.get('service_id', '')
            horizon = prediction.get('horizon_hours', 24)
            
            # 计算建议的维护时间
            suggested_time = self._calculate_optimal_time(
                horizon, prediction.get('predicted_at'), constraints
            )
            
            window = MaintenanceWindow(
                window_id=f"mw_{service_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                start_time=suggested_time,
                end_time=suggested_time + timedelta(hours=2),
                window_type=MaintenanceWindowType.SCHEDULED,
                affected_services=[service_id],
                estimated_duration_minutes=60,
                risk_level='medium',
                approval_required=True
            )
            
            windows.append(window)
        
        # 合并重叠的窗口
        windows = self._merge_overlapping_windows(windows)
        
        return windows
    
    def _calculate_optimal_time(self, horizon_hours: int, 
                               predicted_at: datetime,
                               constraints: Dict[str, Any]) -> datetime:
        """计算最优维护时间"""
        # 基于预测时间和约束计算
        base_time = predicted_at if predicted_at else datetime.now()
        
        # 在预测故障前进行维护
        maintenance_time = base_time + timedelta(hours=horizon_hours * 0.7)
        
        # 调整到最近的低峰时段
        maintenance_time = self._adjust_to_low_traffic(maintenance_time)
        
        return maintenance_time
    
    def _adjust_to_low_traffic(self, dt: datetime) -> datetime:
        """调整到低峰时段"""
        # 找到最近的低峰时段
        for window in self.low_traffic_windows:
            if dt.weekday() in window['days']:
                if dt.hour < window['start_hour']:
                    return dt.replace(hour=window['start_hour'], minute=0)
                elif dt.hour > window['end_hour']:
                    # 找下一个窗口
                    next_day = dt + timedelta(days=1)
                    return next_day.replace(hour=window['start_hour'], minute=0)
        
        return dt
    
    def _merge_overlapping_windows(self, windows: List[MaintenanceWindow]) -> List[MaintenanceWindow]:
        """合并重叠的维护窗口"""
        if not windows:
            return []
        
        # 按开始时间排序
        sorted_windows = sorted(windows, key=lambda w: w.start_time)
        
        merged = [sorted_windows[0]]
        
        for current in sorted_windows[1:]:
            last = merged[-1]
            
            # 检查是否重叠
            if current.start_time <= last.end_time + timedelta(hours=1):
                # 合并
                last.end_time = max(last.end_time, current.end_time)
                last.affected_services.extend(current.affected_services)
                last.affected_services = list(set(last.affected_services))
            else:
                merged.append(current)
        
        return merged


class PrepairStrategyGenerator:
    """预修复策略生成器"""
    
    def __init__(self):
        # 故障类型到修复策略的映射
        self.failure_strategies = {
            'cpu_exhaustion': [
                {'action': 'scale_up', 'priority': 1, 'estimated_minutes': 10},
                {'action': 'optimize_code', 'priority': 2, 'estimated_minutes': 60},
            ],
            'memory_leak': [
                {'action': 'restart', 'priority': 1, 'estimated_minutes': 5},
                {'action': 'deploy_fix', 'priority': 2, 'estimated_minutes': 30},
            ],
            'disk_full': [
                {'action': 'cleanup', 'priority': 1, 'estimated_minutes': 15},
                {'action': 'expand_storage', 'priority': 2, 'estimated_minutes': 30},
            ],
            'network_partition': [
                {'action': 'failover', 'priority': 1, 'estimated_minutes': 5},
                {'action': 'network_repair', 'priority': 2, 'estimated_minutes': 45},
            ],
            'dependency_failure': [
                {'action': 'circuit_breaker', 'priority': 1, 'estimated_minutes': 2},
                {'action': 'dependency_rollback', 'priority': 2, 'estimated_minutes': 20},
            ],
        }
        
    def generate_strategy(self, prediction: Dict[str, Any],
                         impact_analysis: Dict[str, Any]) -> List[PrepairAction]:
        """生成预修复策略"""
        actions = []
        
        failure_type = prediction.get('failure_type', 'unknown')
        service_id = prediction.get('service_id', '')
        
        # 获取策略模板
        strategies = self.failure_strategies.get(failure_type, [
            {'action': 'restart', 'priority': 1, 'estimated_minutes': 5}
        ])
        
        # 根据影响分析调整策略
        overall_severity = impact_analysis.get('overall_severity', 'medium')
        
        for i, strategy in enumerate(strategies):
            # 根据严重程度调整时间估计
            time_multiplier = 1.0
            if overall_severity == 'critical':
                time_multiplier = 0.5  # 紧急情况下加快
            elif overall_severity == 'low':
                time_multiplier = 1.5
            
            estimated_time = strategy['estimated_minutes'] * time_multiplier
            
            action = PrepairAction(
                action_id=f"pa_{service_id}_{strategy['action']}_{i}",
                action_type=strategy['action'],
                target_service=service_id,
                scheduled_time=datetime.now(),  # 将由窗口优化器调整
                estimated_duration_minutes=estimated_time,
                rollback_plan=self._generate_rollback_plan(strategy['action'], service_id),
                prerequisites=self._get_prerequisites(strategy['action'], service_id)
            )
            
            actions.append(action)
        
        return actions
    
    def _generate_rollback_plan(self, action: str, service: str) -> Dict[str, Any]:
        """生成回滚计划"""
        rollback_actions = {
            'scale_up': {'action': 'scale_down', 'target': service},
            'restart': {'action': 'start_previous_version', 'target': service},
            'cleanup': {'action': 'restore_deleted', 'target': service},
            'failover': {'action': 'failback', 'target': service},
            'circuit_breaker': {'action': 'close_circuit', 'target': service},
        }
        
        return {
            'rollback_action': rollback_actions.get(action, {'action': 'manual_rollback'}),
            'estimated_rollback_time': 5,
            'automation_level': 'automatic' if action in rollback_actions else 'manual'
        }
    
    def _get_prerequisites(self, action: str, service: str) -> List[str]:
        """获取前置条件"""
        prerequisites = {
            'scale_up': ['sufficient_quota', 'load_balancer_ready'],
            'restart': ['backup_completed', 'health_check_passed'],
            'cleanup': ['data_backup', 'archive_verified'],
            'failover': ['standby_ready', 'data_synced'],
            'circuit_breaker': ['monitoring_enabled'],
        }
        
        return prerequisites.get(action, [])


class BatchRepairPlanner:
    """批量修复规划器"""
    
    def __init__(self):
        self.max_parallel = 3  # 最大并行修复数
        
    def plan_batch_repair(self, predictions: List[Dict[str, Any]],
                         windows: List[MaintenanceWindow]) -> Dict[str, Any]:
        """规划批量修复"""
        # 按服务分组
        service_groups = {}
        for pred in predictions:
            service = pred.get('service_id')
            if service not in service_groups:
                service_groups[service] = []
            service_groups[service].append(pred)
        
        # 创建批次
        batches = []
        current_batch = []
        
        for service, preds in service_groups.items():
            if len(current_batch) >= self.max_parallel:
                batches.append(current_batch)
                current_batch = []
            
            current_batch.append({
                'service': service,
                'predictions': preds
            })
        
        if current_batch:
            batches.append(current_batch)
        
        # 分配维护窗口
        batch_plan = []
        for i, batch in enumerate(batches):
            if i < len(windows):
                window = windows[i]
                batch_plan.append({
                    'batch_id': f"batch_{i+1}",
                    'services': [b['service'] for b in batch],
                    'maintenance_window': window.to_dict(),
                    'estimated_total_duration': sum(
                        p.get('estimated_duration_minutes', 30) 
                        for b in batch for p in b['predictions']
                    )
                })
        
        return {
            'total_batches': len(batches),
            'batches': batch_plan,
            'parallelism': self.max_parallel
        }


class PrepairPlanner:
    """
    预修复规划器主类
    整合维护窗口优化、策略生成、批量规划
    """
    
    def __init__(self):
        self.window_optimizer = MaintenanceWindowOptimizer()
        self.strategy_generator = PrepairStrategyGenerator()
        self.batch_planner = BatchRepairPlanner()
        
    def create_plan(self, predictions: List[Dict[str, Any]],
                   impact_analyses: Dict[str, Dict[str, Any]],
                   constraints: Dict[str, Any] = None) -> PrepairPlan:
        """
        创建预修复计划
        
        Args:
            predictions: 故障预测列表
            impact_analyses: 影响分析结果
            constraints: 约束条件
        
        Returns:
            预修复计划
        """
        constraints = constraints or {}
        
        # 1. 优化维护窗口
        windows = self.window_optimizer.find_optimal_windows(predictions, constraints)
        
        # 2. 生成修复策略
        all_actions = []
        for prediction in predictions:
            service_id = prediction.get('service_id', '')
            impact = impact_analyses.get(service_id, {})
            
            actions = self.strategy_generator.generate_strategy(prediction, impact)
            all_actions.extend(actions)
        
        # 3. 规划批量修复
        batch_plan = self.batch_planner.plan_batch_repair(predictions, windows)
        
        # 4. 计算整体风险
        overall_risk = self._calculate_overall_risk(predictions, impact_analyses)
        
        # 5. 估算成功率
        success_rate = self._estimate_success_rate(predictions, all_actions)
        
        plan = PrepairPlan(
            plan_id=f"plan_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created_at=datetime.now(),
            predictions=predictions,
            maintenance_windows=windows,
            prepair_actions=all_actions,
            overall_risk=overall_risk,
            estimated_success_rate=success_rate
        )
        
        return plan
    
    def _calculate_overall_risk(self, predictions: List[Dict[str, Any]],
                               impact_analyses: Dict[str, Dict]) -> str:
        """计算整体风险"""
        max_probability = max(
            (p.get('failure_probability', 0) for p in predictions),
            default=0
        )
        
        max_impact = max(
            (i.get('overall_impact_score', 0) for i in impact_analyses.values()),
            default=0
        )
        
        combined_risk = max_probability * 0.5 + max_impact * 0.5
        
        if combined_risk >= 0.7:
            return 'critical'
        elif combined_risk >= 0.5:
            return 'high'
        elif combined_risk >= 0.3:
            return 'medium'
        return 'low'
    
    def _estimate_success_rate(self, predictions: List[Dict[str, Any]],
                              actions: List[PrepairAction]) -> float:
        """估算成功率"""
        if not predictions:
            return 1.0
        
        # 基于预测置信度和动作复杂度
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in predictions])
        
        # 动作复杂度影响
        complex_actions = sum(1 for a in actions if a.estimated_duration_minutes > 30)
        complexity_penalty = complex_actions * 0.05
        
        success_rate = avg_confidence - complexity_penalty
        
        return max(0.5, min(0.95, success_rate))
    
    def schedule_plan(self, plan: PrepairPlan, 
                     approval_callback: callable = None) -> Dict[str, Any]:
        """调度计划"""
        scheduled = []
        pending_approval = []
        
        for window in plan.maintenance_windows:
            if window.approval_required:
                pending_approval.append(window)
            else:
                scheduled.append(window)
        
        return {
            'plan_id': plan.plan_id,
            'scheduled_windows': [w.to_dict() for w in scheduled],
            'pending_approval': [w.to_dict() for w in pending_approval],
            'total_windows': len(plan.maintenance_windows),
            'actions_count': len(plan.prepair_actions)
        }
