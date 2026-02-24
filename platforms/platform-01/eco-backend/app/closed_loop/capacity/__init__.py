# =============================================================================
# Capacity Manager - Phase 2
# =============================================================================
# 容量管理器 - Phase 2 核心组件
# 
# 功能：
# - 负载预测 (Holt-Winters, Prophet, LSTM)
# - 容量规划
# - 成本优化
# - KEDA 集成
# =============================================================================

from app.closed_loop.capacity.forecast_engine import (
    ForecastResult,
    ForecastModel,
    HoltWintersModel,
    SimpleLinearModel,
    MovingAverageModel,
    ForecastEngine,
    forecast_engine,
)

from app.closed_loop.capacity.planner import (
    ScalingActionType,
    ScalingAction,
    ScalingPlan,
    CapacityConstraints,
    CapacityPlanner,
    capacity_planner,
)


class CapacityManager:
    """
    容量管理器
    
    整合预测、规划和执行，实现智能容量管理
    """
    
    def __init__(
        self,
        forecast_engine: ForecastEngine = None,
        capacity_planner: CapacityPlanner = None,
    ):
        self.forecast_engine = forecast_engine or ForecastEngine()
        self.capacity_planner = capacity_planner or CapacityPlanner()
        
        # 注册默认模型
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """注册默认预测模型"""
        self.forecast_engine.register_model("holt_winters", HoltWintersModel())
        self.forecast_engine.register_model("linear", SimpleLinearModel())
        self.forecast_engine.register_model("moving_average", MovingAverageModel())
    
    def record_metric(
        self,
        metric_name: str,
        timestamp: datetime,
        value: float
    ) -> None:
        """记录指标值"""
        self.forecast_engine.record_metric(metric_name, timestamp, value)
    
    def predict_and_plan(
        self,
        metric_name: str,
        current_replicas: int,
        current_utilization: float,
        constraints: CapacityConstraints = None,
        horizon_minutes: int = 60,
    ) -> Optional[ScalingPlan]:
        """
        预测并制定扩缩容计划
        
        Args:
            metric_name: 指标名称
            current_replicas: 当前副本数
            current_utilization: 当前资源使用率
            constraints: 容量约束
            horizon_minutes: 预测时间范围
        
        Returns:
            扩缩容计划
        """
        from datetime import datetime
        
        # 预测
        forecast = self.forecast_engine.forecast(
            metric_name,
            horizon_minutes=horizon_minutes,
        )
        
        if not forecast:
            return None
        
        # 规划
        constraints = constraints or CapacityConstraints()
        
        plan = self.capacity_planner.plan(
            forecast=forecast,
            current_replicas=current_replicas,
            current_utilization=current_utilization,
            constraints=constraints,
        )
        
        return plan
    
    def get_forecast(
        self,
        metric_name: str,
        horizon_minutes: int = 60,
    ) -> Optional[ForecastResult]:
        """获取预测结果"""
        return self.forecast_engine.forecast(
            metric_name,
            horizon_minutes=horizon_minutes,
        )
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """获取指标统计信息"""
        return self.forecast_engine.get_metric_stats(metric_name)
    
    def get_recent_plans(self, limit: int = 10) -> List[ScalingPlan]:
        """获取最近的扩缩容计划"""
        return self.capacity_planner.get_recent_plans(limit)


# 全局容量管理器实例
capacity_manager = CapacityManager()


__all__ = [
    # 预测引擎
    "ForecastResult",
    "ForecastModel",
    "HoltWintersModel",
    "SimpleLinearModel",
    "MovingAverageModel",
    "ForecastEngine",
    "forecast_engine",
    
    # 容量规划
    "ScalingActionType",
    "ScalingAction",
    "ScalingPlan",
    "CapacityConstraints",
    "CapacityPlanner",
    "capacity_planner",
    
    # 容量管理器
    "CapacityManager",
    "capacity_manager",
]
