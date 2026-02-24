# =============================================================================
# Capacity Planner
# =============================================================================
# 容量规划器 - Phase 2 核心组件
# 基于预测结果制定扩缩容计划
# =============================================================================

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from app.closed_loop.capacity.forecast_engine import ForecastResult


class ScalingActionType(Enum):
    """扩缩容动作类型"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingAction:
    """扩缩容动作"""
    action_type: ScalingActionType
    deployment: str
    namespace: str
    current_replicas: int
    target_replicas: int
    reason: str
    scheduled_time: datetime
    confidence: float


@dataclass
class ScalingPlan:
    """扩缩容计划"""
    id: str
    created_at: datetime
    metric_name: str
    forecast_result: ForecastResult
    actions: List[ScalingAction]
    estimated_cost: float
    expected_improvement: str


@dataclass
class CapacityConstraints:
    """容量约束条件"""
    min_replicas: int = 2
    max_replicas: int = 20
    scale_up_threshold: float = 0.8  # CPU/内存使用率阈值
    scale_down_threshold: float = 0.3
    scale_up_step: int = 2
    scale_down_step: int = 1
    cooldown_minutes: int = 5
    max_cost_per_hour: Optional[float] = None


class CapacityPlanner:
    """
    容量规划器
    
    基于预测结果制定扩缩容计划
    """
    
    def __init__(self):
        self._plans: List[ScalingPlan] = []
        self._last_scaling: Dict[str, datetime] = {}  # 记录上次扩缩容时间
    
    def plan(
        self,
        forecast: ForecastResult,
        current_replicas: int,
        current_utilization: float,
        constraints: CapacityConstraints,
        deployment: str = "eco-backend",
        namespace: str = "default",
    ) -> ScalingPlan:
        """
        制定扩缩容计划
        
        Args:
            forecast: 预测结果
            current_replicas: 当前副本数
            current_utilization: 当前资源使用率
            constraints: 容量约束
            deployment: Deployment 名称
            namespace: 命名空间
        
        Returns:
            扩缩容计划
        """
        actions = []
        
        # 分析预测结果
        predicted_values = forecast.values
        predicted_max = max(predicted_values) if predicted_values else 0
        predicted_min = min(predicted_values) if predicted_values else 0
        predicted_avg = sum(predicted_values) / len(predicted_values) if predicted_values else 0
        
        # 决策逻辑
        action = self._decide_action(
            current_replicas,
            current_utilization,
            predicted_max,
            predicted_avg,
            constraints,
        )
        
        if action:
            actions.append(action)
        
        # 计算预估成本
        estimated_cost = self._estimate_cost(
            current_replicas,
            [a.target_replicas for a in actions],
            forecast.timestamps,
        )
        
        # 生成计划
        plan = ScalingPlan(
            id=f"plan-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            created_at=datetime.utcnow(),
            metric_name=forecast.metric_name,
            forecast_result=forecast,
            actions=actions,
            estimated_cost=estimated_cost,
            expected_improvement=self._generate_improvement_description(actions),
        )
        
        self._plans.append(plan)
        
        return plan
    
    def _decide_action(
        self,
        current_replicas: int,
        current_utilization: float,
        predicted_max: float,
        predicted_avg: float,
        constraints: CapacityConstraints,
    ) -> Optional[ScalingAction]:
        """决定扩缩容动作"""
        # 检查冷却时间
        last_scale = self._last_scaling.get("default")
        if last_scale:
            time_since_last = datetime.utcnow() - last_scale
            if time_since_last < timedelta(minutes=constraints.cooldown_minutes):
                return None
        
        # 扩容决策
        if predicted_max > constraints.scale_up_threshold or current_utilization > constraints.scale_up_threshold:
            target = min(
                current_replicas + constraints.scale_up_step,
                constraints.max_replicas
            )
            
            if target > current_replicas:
                self._last_scaling["default"] = datetime.utcnow()
                return ScalingAction(
                    action_type=ScalingActionType.SCALE_UP,
                    deployment="eco-backend",
                    namespace="default",
                    current_replicas=current_replicas,
                    target_replicas=target,
                    reason=f"Predicted utilization will reach {predicted_max*100:.1f}%, scaling up to handle load",
                    scheduled_time=datetime.utcnow(),
                    confidence=0.8,
                )
        
        # 缩容决策
        elif predicted_avg < constraints.scale_down_threshold and current_utilization < constraints.scale_down_threshold:
            target = max(
                current_replicas - constraints.scale_down_step,
                constraints.min_replicas
            )
            
            if target < current_replicas:
                self._last_scaling["default"] = datetime.utcnow()
                return ScalingAction(
                    action_type=ScalingActionType.SCALE_DOWN,
                    deployment="eco-backend",
                    namespace="default",
                    current_replicas=current_replicas,
                    target_replicas=target,
                    reason=f"Predicted utilization will drop to {predicted_avg*100:.1f}%, scaling down to save cost",
                    scheduled_time=datetime.utcnow() + timedelta(minutes=10),  # 延迟缩容
                    confidence=0.7,
                )
        
        return None
    
    def _estimate_cost(
        self,
        current_replicas: int,
        target_replicas_list: List[int],
        timestamps: List[datetime],
    ) -> float:
        """估算成本"""
        # 简化的成本估算：假设每个副本每小时 $0.1
        cost_per_replica_hour = 0.1
        
        if not timestamps or not target_replicas_list:
            return current_replicas * cost_per_replica_hour
        
        # 计算时间跨度（小时）
        duration_hours = len(timestamps) * 5 / 60  # 假设 5 分钟一个点
        
        # 平均副本数
        avg_replicas = sum(target_replicas_list) / len(target_replicas_list)
        
        return avg_replicas * cost_per_replica_hour * duration_hours
    
    def _generate_improvement_description(
        self,
        actions: List[ScalingAction]
    ) -> str:
        """生成改进描述"""
        if not actions:
            return "No scaling action needed at this time"
        
        action = actions[0]
        
        if action.action_type == ScalingActionType.SCALE_UP:
            return (
                f"Scaling up from {action.current_replicas} to {action.target_replicas} "
                f"replicas will improve response time and prevent service degradation"
            )
        elif action.action_type == ScalingActionType.SCALE_DOWN:
            return (
                f"Scaling down from {action.current_replicas} to {action.target_replicas} "
                f"replicas will reduce cost while maintaining service quality"
            )
        
        return "No improvement expected"
    
    def get_recent_plans(self, limit: int = 10) -> List[ScalingPlan]:
        """获取最近的计划"""
        return sorted(
            self._plans,
            key=lambda p: p.created_at,
            reverse=True
        )[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._plans:
            return {"total_plans": 0}
        
        scale_up_count = sum(
            1 for p in self._plans
            for a in p.actions
            if a.action_type == ScalingActionType.SCALE_UP
        )
        
        scale_down_count = sum(
            1 for p in self._plans
            for a in p.actions
            if a.action_type == ScalingActionType.SCALE_DOWN
        )
        
        return {
            "total_plans": len(self._plans),
            "scale_up_actions": scale_up_count,
            "scale_down_actions": scale_down_count,
            "total_estimated_cost": sum(p.estimated_cost for p in self._plans),
        }


# 全局容量规划器实例
capacity_planner = CapacityPlanner()


__all__ = [
    "ScalingActionType",
    "ScalingAction",
    "ScalingPlan",
    "CapacityConstraints",
    "CapacityPlanner",
    "capacity_planner",
]
