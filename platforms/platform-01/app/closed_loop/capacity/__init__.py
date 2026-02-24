"""
容量预测与自动调整入口

提供容量需求预测和自动扩缩容功能
"""

from .forecast_engine import (
    ForecastEngine,
    ForecastResult,
    ForecastModel,
)

from .planner import (
    CapacityPlanner,
    CapacityPlan,
    ScalingRecommendation,
    ActionType,
    ResourceType,
)

__all__ = [
    # 预测引擎
    'ForecastEngine',
    'ForecastResult',
    'ForecastModel',
    
    # 容量规划器
    'CapacityPlanner',
    'CapacityPlan',
    'ScalingRecommendation',
    'ActionType',
    'ResourceType',
]
