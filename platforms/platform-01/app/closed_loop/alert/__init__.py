"""
智能告警路由入口

提供告警路由、聚合和升级功能
"""

from .router import (
    SmartAlertRouter,
    AlertAggregator,
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel,
    RoutingRule,
    OnCallSchedule,
    create_default_router,
)

__all__ = [
    'SmartAlertRouter',
    'AlertAggregator',
    'Alert',
    'AlertSeverity',
    'AlertStatus',
    'NotificationChannel',
    'RoutingRule',
    'OnCallSchedule',
    'create_default_router',
]
