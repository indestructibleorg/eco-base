"""
容量规划器
基于预测结果制定容量调整计划
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from .forecast_engine import ForecastEngine, ForecastResult

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """动作类型"""
    SCALE_UP = "scale_up"          # 扩容
    SCALE_DOWN = "scale_down"      # 缩容
    MAINTAIN = "maintain"          # 保持
    ALERT = "alert"                # 告警


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    INSTANCES = "instances"


@dataclass
class CapacityPlan:
    """容量计划"""
    plan_id: str
    service: str
    action: ActionType
    resource_type: ResourceType
    current_value: float
    target_value: float
    reason: str
    priority: int  # 1-10
    execute_after: datetime
    execute_before: datetime
    estimated_cost: Optional[float] = None
    rollback_plan: Optional[str] = None
    approved: bool = False
    executed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'service': self.service,
            'action': self.action.value,
            'resource_type': self.resource_type.value,
            'current_value': self.current_value,
            'target_value': self.target_value,
            'reason': self.reason,
            'priority': self.priority,
            'execute_after': self.execute_after.isoformat(),
            'execute_before': self.execute_before.isoformat(),
            'approved': self.approved,
            'executed': self.executed
        }


@dataclass
class ScalingRecommendation:
    """扩缩容建议"""
    service: str
    current_instances: int
    recommended_instances: int
    confidence: float
    reason: str
    predicted_cpu: Optional[float] = None
    predicted_memory: Optional[float] = None
    cost_impact: Optional[str] = None


class CapacityPlanner:
    """容量规划器"""
    
    def __init__(self, forecast_engine: ForecastEngine = None, config: Optional[Dict] = None):
        self.forecast_engine = forecast_engine
        self.config = config or {}
        
        # 配置
        self.cpu_threshold = self.config.get('cpu_threshold', 70)
        self.memory_threshold = self.config.get('memory_threshold', 80)
        self.scale_up_cooldown = self.config.get('scale_up_cooldown_minutes', 10)
        self.scale_down_cooldown = self.config.get('scale_down_cooldown_minutes', 30)
        self.max_scale_up = self.config.get('max_scale_up_percent', 50)
        self.max_scale_down = self.config.get('max_scale_down_percent', 30)
        
        # 存储
        self.capacity_plans: Dict[str, CapacityPlan] = {}
        self.thresholds: Dict[str, Dict[ResourceType, float]] = {}
    
    def generate_plans(self, service: str, forecast: Dict[str, Any], 
                       constraints: Dict[str, Any] = None) -> List[CapacityPlan]:
        """生成容量计划 (简化接口)"""
        plans = []
        
        values = forecast.get('values', [])
        upper = forecast.get('upper', values)
        
        if not values:
            return plans
        
        # 检查是否需要扩容
        latest = values[-1] if values else 0
        predicted_max = max(upper) if upper else latest
        
        if predicted_max > 80:
            plan = CapacityPlan(
                plan_id=f"plan_{service}_{datetime.now().timestamp()}",
                service=service,
                action=ActionType.SCALE_UP,
                resource_type=ResourceType.CPU,
                current_value=latest,
                target_value=predicted_max * 1.2,
                reason=f"预测负载将达到 {predicted_max:.0f}%",
                priority=7,
                execute_after=datetime.now(),
                execute_before=datetime.now() + timedelta(hours=1)
            )
            plans.append(plan)
        
        return plans
        self.scaling_history: List[Dict] = []
        
        # 执行器
        self.executors: Dict[ActionType, Callable] = {}
        
        logger.info("容量规划器初始化完成")
    
    def register_executor(self, action: ActionType, executor: Callable):
        """注册执行器"""
        self.executors[action] = executor
        logger.info(f"执行器注册: {action.value}")
    
    def set_threshold(self, service: str, resource: ResourceType, threshold: float):
        """设置阈值"""
        if service not in self.thresholds:
            self.thresholds[service] = {}
        self.thresholds[service][resource] = threshold
        logger.info(f"阈值设置: {service}/{resource.value} = {threshold}")
    
    async def analyze_and_plan(self, service: str) -> List[CapacityPlan]:
        """分析并制定容量计划"""
        plans = []
        
        # 获取预测
        cpu_forecast = self.forecast_engine.get_forecast(f"{service}.cpu")
        memory_forecast = self.forecast_engine.get_forecast(f"{service}.memory")
        
        # CPU 分析
        if cpu_forecast:
            cpu_plan = self._analyze_cpu_forecast(service, cpu_forecast)
            if cpu_plan:
                plans.append(cpu_plan)
        
        # 内存分析
        if memory_forecast:
            memory_plan = self._analyze_memory_forecast(service, memory_forecast)
            if memory_plan:
                plans.append(memory_plan)
        
        # 存储计划
        for plan in plans:
            self.capacity_plans[plan.plan_id] = plan
        
        logger.info(f"容量计划生成: {service} ({len(plans)} 个计划)")
        return plans
    
    def _analyze_cpu_forecast(self, service: str, 
                              forecast: ForecastResult) -> Optional[CapacityPlan]:
        """分析 CPU 预测"""
        threshold = self.thresholds.get(service, {}).get(ResourceType.CPU, self.cpu_threshold)
        
        # 获取峰值
        peak_value, peak_time = forecast.get_peak_value()
        avg_value = forecast.get_average_value()
        
        # 获取当前值
        stats = self.forecast_engine.get_metric_stats(f"{service}.cpu")
        current_value = stats.get('latest', 0)
        
        # 决策逻辑
        if peak_value > threshold * 1.2:  # 超过阈值20%
            # 需要扩容
            target_value = current_value * 1.3  # 增加30%
            
            return CapacityPlan(
                plan_id=f"plan_{service}_cpu_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                service=service,
                action=ActionType.SCALE_UP,
                resource_type=ResourceType.CPU,
                current_value=current_value,
                target_value=target_value,
                reason=f"预测CPU峰值 {peak_value:.1f}% 超过阈值 {threshold}%",
                priority=8 if peak_value > threshold * 1.5 else 5,
                execute_after=datetime.now(),
                execute_before=peak_time - timedelta(minutes=30)
            )
        
        elif avg_value < threshold * 0.3 and current_value < threshold * 0.3:
            # 可以缩容
            target_value = current_value * 0.8  # 减少20%
            
            return CapacityPlan(
                plan_id=f"plan_{service}_cpu_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                service=service,
                action=ActionType.SCALE_DOWN,
                resource_type=ResourceType.CPU,
                current_value=current_value,
                target_value=target_value,
                reason=f"预测CPU平均使用率 {avg_value:.1f}% 远低于阈值",
                priority=3,
                execute_after=datetime.now() + timedelta(hours=1),
                execute_before=datetime.now() + timedelta(hours=24)
            )
        
        return None
    
    def _analyze_memory_forecast(self, service: str,
                                  forecast: ForecastResult) -> Optional[CapacityPlan]:
        """分析内存预测"""
        threshold = self.thresholds.get(service, {}).get(ResourceType.MEMORY, self.memory_threshold)
        
        peak_value, peak_time = forecast.get_peak_value()
        avg_value = forecast.get_average_value()
        
        stats = self.forecast_engine.get_metric_stats(f"{service}.memory")
        current_value = stats.get('latest', 0)
        
        if peak_value > threshold:
            target_value = current_value * 1.25
            
            return CapacityPlan(
                plan_id=f"plan_{service}_memory_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                service=service,
                action=ActionType.SCALE_UP,
                resource_type=ResourceType.MEMORY,
                current_value=current_value,
                target_value=target_value,
                reason=f"预测内存峰值 {peak_value:.1f}% 超过阈值 {threshold}%",
                priority=9,
                execute_after=datetime.now(),
                execute_before=peak_time - timedelta(minutes=15)
            )
        
        return None
    
    async def recommend_scaling(self, service: str,
                                 current_instances: int) -> Optional[ScalingRecommendation]:
        """推荐扩缩容"""
        # 获取预测
        cpu_forecast = self.forecast_engine.get_forecast(f"{service}.cpu")
        memory_forecast = self.forecast_engine.get_forecast(f"{service}.memory")
        
        if not cpu_forecast:
            return None
        
        peak_cpu, _ = cpu_forecast.get_peak_value()
        avg_cpu = cpu_forecast.get_average_value()
        
        cpu_threshold = self.thresholds.get(service, {}).get(ResourceType.CPU, self.cpu_threshold)
        
        # 计算推荐实例数
        recommended = current_instances
        reason = ""
        confidence = 0.5
        
        if peak_cpu > cpu_threshold * 1.3:
            # 需要扩容
            scale_factor = peak_cpu / cpu_threshold
            recommended = int(current_instances * min(scale_factor, 1 + self.max_scale_up/100))
            reason = f"预测CPU峰值 {peak_cpu:.1f}% 远超阈值，建议扩容"
            confidence = min(0.9, 0.6 + (peak_cpu - cpu_threshold) / 100)
        
        elif avg_cpu < cpu_threshold * 0.25 and current_instances > 2:
            # 可以缩容
            recommended = max(2, int(current_instances * 0.8))
            reason = f"CPU使用率持续较低，建议缩容节省成本"
            confidence = 0.7
        
        else:
            return None
        
        if recommended == current_instances:
            return None
        
        # 预测新配置下的资源使用
        predicted_cpu = avg_cpu * current_instances / recommended if recommended > 0 else 0
        
        return ScalingRecommendation(
            service=service,
            current_instances=current_instances,
            recommended_instances=recommended,
            confidence=round(confidence, 2),
            reason=reason,
            predicted_cpu=round(predicted_cpu, 1),
            cost_impact=f"预计成本变化: {(recommended/current_instances - 1) * 100:+.0f}%"
        )
    
    async def approve_plan(self, plan_id: str) -> bool:
        """批准计划"""
        plan = self.capacity_plans.get(plan_id)
        if not plan:
            return False
        
        plan.approved = True
        logger.info(f"计划已批准: {plan_id}")
        return True
    
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """执行计划"""
        plan = self.capacity_plans.get(plan_id)
        if not plan:
            return {'success': False, 'error': '计划不存在'}
        
        if not plan.approved:
            return {'success': False, 'error': '计划未批准'}
        
        if plan.executed:
            return {'success': False, 'error': '计划已执行'}
        
        # 检查时间窗口
        now = datetime.now()
        if now < plan.execute_after:
            return {'success': False, 'error': '未到执行时间'}
        if now > plan.execute_before:
            return {'success': False, 'error': '已超过执行截止时间'}
        
        # 获取执行器
        executor = self.executors.get(plan.action)
        if not executor:
            return {'success': False, 'error': f'未找到执行器: {plan.action.value}'}
        
        # 执行
        try:
            result = await executor(plan)
            plan.executed = True
            
            # 记录历史
            self.scaling_history.append({
                'plan_id': plan_id,
                'service': plan.service,
                'action': plan.action.value,
                'executed_at': datetime.now().isoformat(),
                'result': result
            })
            
            logger.info(f"计划执行成功: {plan_id}")
            return {'success': True, 'result': result}
        
        except Exception as e:
            logger.exception(f"计划执行失败: {plan_id}")
            return {'success': False, 'error': str(e)}
    
    def get_pending_plans(self) -> List[CapacityPlan]:
        """获取待执行计划"""
        return [
            p for p in self.capacity_plans.values()
            if p.approved and not p.executed and datetime.now() <= p.execute_before
        ]
    
    def get_plan(self, plan_id: str) -> Optional[CapacityPlan]:
        """获取计划"""
        return self.capacity_plans.get(plan_id)
    
    def get_plans_for_service(self, service: str) -> List[CapacityPlan]:
        """获取服务的所有计划"""
        return [
            p for p in self.capacity_plans.values()
            if p.service == service
        ]
    
    def get_scaling_history(self, service: str = None, 
                            limit: int = 100) -> List[Dict]:
        """获取扩缩容历史"""
        history = self.scaling_history
        if service:
            history = [h for h in history if h['service'] == service]
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.capacity_plans)
        approved = sum(1 for p in self.capacity_plans.values() if p.approved)
        executed = sum(1 for p in self.capacity_plans.values() if p.executed)
        pending = sum(1 for p in self.capacity_plans.values() 
                     if p.approved and not p.executed)
        
        by_action = {}
        for plan in self.capacity_plans.values():
            action = plan.action.value
            by_action[action] = by_action.get(action, 0) + 1
        
        return {
            'total_plans': total,
            'approved': approved,
            'executed': executed,
            'pending': pending,
            'by_action': by_action,
            'scaling_history': len(self.scaling_history)
        }
