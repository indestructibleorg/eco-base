# =============================================================================
# Smart Alert Router
# =============================================================================
# 智能告警路由 - Phase 2 核心组件
# 智能分类、聚合和路由告警
# =============================================================================

import hashlib
import re
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import uuid


class AlertSeverity(Enum):
    """告警严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertCategory(Enum):
    """告警分类"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"


class NotificationChannel(Enum):
    """通知渠道"""
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class Alert:
    """告警对象"""
    id: str
    timestamp: datetime
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    source: str
    service: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """生成告警指纹"""
        key_parts = [
            self.category.value,
            self.source,
            self.service,
            self.title,
        ]
        key = "|".join(filter(None, key_parts))
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "source": self.source,
            "service": self.service,
            "attributes": self.attributes,
            "fingerprint": self.fingerprint,
        }


@dataclass
class AlertGroup:
    """告警组"""
    id: str
    fingerprint: str
    alerts: List[Alert]
    pattern: str
    first_seen: datetime
    last_seen: datetime
    count: int = 0
    
    def __post_init__(self):
        self.count = len(self.alerts)
        if self.alerts:
            self.first_seen = min(a.timestamp for a in self.alerts)
            self.last_seen = max(a.timestamp for a in self.alerts)
    
    def add_alert(self, alert: Alert) -> None:
        """添加告警到组"""
        self.alerts.append(alert)
        self.count += 1
        self.last_seen = alert.timestamp


@dataclass
class NotificationTarget:
    """通知目标"""
    channel: NotificationChannel
    recipient: str
    priority: int = 100
    
    def __hash__(self):
        return hash((self.channel.value, self.recipient))
    
    def __eq__(self, other):
        return (
            self.channel == other.channel and
            self.recipient == other.recipient
        )


@dataclass
class RoutingRule:
    """路由规则"""
    name: str
    conditions: List[Dict[str, Any]]
    targets: List[NotificationTarget]
    priority: int = 100
    enabled: bool = True
    cooldown_minutes: int = 5
    suppress_duplicates: bool = True
    
    def matches(self, alert: Alert) -> bool:
        """检查告警是否匹配规则"""
        for condition in self.conditions:
            field = condition.get("field")
            operator = condition.get("operator", "==")
            value = condition.get("value")
            
            # 获取字段值
            if field == "severity":
                alert_value = alert.severity.value
            elif field == "category":
                alert_value = alert.category.value
            elif field == "source":
                alert_value = alert.source
            elif field == "service":
                alert_value = alert.service
            elif field.startswith("attributes."):
                attr_name = field.split(".")[1]
                alert_value = alert.attributes.get(attr_name)
            else:
                continue
            
            # 评估条件
            if operator == "==" and alert_value != value:
                return False
            elif operator == "!=" and alert_value == value:
                return False
            elif operator == "in" and alert_value not in value:
                return False
            elif operator == "not_in" and alert_value in value:
                return False
            elif operator == "contains" and value not in str(alert_value):
                return False
            elif operator == "matches" and not re.match(value, str(alert_value)):
                return False
        
        return True


@dataclass
class RoutingResult:
    """路由结果"""
    alert_id: str
    targets: List[NotificationTarget]
    rules_matched: List[str]
    suppressed: bool
    reason: str


class AlertClassifier:
    """告警分类器"""
    
    # 关键词映射到分类
    CATEGORY_KEYWORDS = {
        AlertCategory.INFRASTRUCTURE: [
            "pod", "node", "container", "kubernetes", "k8s", "disk", "cpu", "memory"
        ],
        AlertCategory.DATABASE: [
            "database", "db", "sql", "query", "connection", "pool", "postgres", "mysql"
        ],
        AlertCategory.NETWORK: [
            "network", "connection", "timeout", "dns", "latency", "packet", "ingress"
        ],
        AlertCategory.SECURITY: [
            "security", "auth", "authentication", "unauthorized", "forbidden", "cve"
        ],
        AlertCategory.PERFORMANCE: [
            "performance", "slow", "latency", "throughput", "response_time", "p99"
        ],
        AlertCategory.AVAILABILITY: [
            "availability", "down", "unavailable", "health", "crash", "failure"
        ],
    }
    
    def classify(self, alert: Alert) -> AlertCategory:
        """分类告警"""
        text = f"{alert.title} {alert.message}".lower()
        
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return AlertCategory.APPLICATION


class AlertAggregator:
    """告警聚合器"""
    
    def __init__(self, aggregation_window_minutes: int = 5):
        self.aggregation_window = timedelta(minutes=aggregation_window_minutes)
        self._groups: Dict[str, AlertGroup] = {}
        self._fingerprint_index: Dict[str, AlertGroup] = {}
    
    def aggregate(self, alerts: List[Alert]) -> List[AlertGroup]:
        """聚合告警"""
        for alert in alerts:
            self._add_to_group(alert)
        
        return list(self._groups.values())
    
    def _add_to_group(self, alert: Alert) -> None:
        """将告警添加到组"""
        # 检查是否已有匹配的组
        existing_group = self._fingerprint_index.get(alert.fingerprint)
        
        if existing_group:
            # 检查时间窗口
            time_diff = alert.timestamp - existing_group.last_seen
            if time_diff <= self.aggregation_window:
                existing_group.add_alert(alert)
                return
        
        # 创建新组
        group_id = f"group-{uuid.uuid4().hex[:8]}"
        group = AlertGroup(
            id=group_id,
            fingerprint=alert.fingerprint,
            alerts=[alert],
            pattern=self._extract_pattern(alert),
            first_seen=alert.timestamp,
            last_seen=alert.timestamp,
        )
        
        self._groups[group_id] = group
        self._fingerprint_index[alert.fingerprint] = group
    
    def _extract_pattern(self, alert: Alert) -> str:
        """提取告警模式"""
        # 简化标题作为模式
        pattern = re.sub(r'\d+', '{N}', alert.title)
        pattern = re.sub(r'[a-f0-9]{8,}', '{ID}', pattern)
        return pattern
    
    def get_group(self, group_id: str) -> Optional[AlertGroup]:
        """获取告警组"""
        return self._groups.get(group_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取聚合统计"""
        return {
            "total_groups": len(self._groups),
            "total_alerts": sum(g.count for g in self._groups.values()),
            "avg_group_size": (
                sum(g.count for g in self._groups.values()) / len(self._groups)
                if self._groups else 0
            ),
        }


class AlertRouter:
    """告警路由器"""
    
    def __init__(self):
        self.rules: List[RoutingRule] = []
        self.classifier = AlertClassifier()
        self.aggregator = AlertAggregator()
        self._last_routed: Dict[str, datetime] = {}  # 记录上次路由时间
    
    def add_rule(self, rule: RoutingRule) -> None:
        """添加路由规则"""
        self.rules.append(rule)
        # 按优先级排序
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除路由规则"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False
    
    def route(self, alert: Alert) -> RoutingResult:
        """
        路由单个告警
        
        Returns:
            路由结果
        """
        # 分类告警
        if not alert.category:
            alert.category = self.classifier.classify(alert)
        
        # 检查是否需要抑制
        suppressed, reason = self._should_suppress(alert)
        if suppressed:
            return RoutingResult(
                alert_id=alert.id,
                targets=[],
                rules_matched=[],
                suppressed=True,
                reason=reason,
            )
        
        # 匹配规则
        matched_rules = []
        targets: Set[NotificationTarget] = set()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.matches(alert):
                matched_rules.append(rule.name)
                targets.update(rule.targets)
        
        # 记录路由时间
        self._last_routed[alert.fingerprint] = datetime.utcnow()
        
        return RoutingResult(
            alert_id=alert.id,
            targets=list(targets),
            rules_matched=matched_rules,
            suppressed=False,
            reason="Routed successfully",
        )
    
    def route_batch(self, alerts: List[Alert]) -> List[RoutingResult]:
        """批量路由告警"""
        # 先聚合
        groups = self.aggregator.aggregate(alerts)
        
        results = []
        for group in groups:
            # 使用组内最严重的告警代表整个组
            representative = self._get_representative_alert(group.alerts)
            result = self.route(representative)
            results.append(result)
        
        return results
    
    def _should_suppress(self, alert: Alert) -> tuple[bool, str]:
        """检查是否应该抑制告警"""
        # 检查重复
        last_routed = self._last_routed.get(alert.fingerprint)
        if last_routed:
            # 找到匹配的规则检查冷却时间
            for rule in self.rules:
                if rule.matches(alert) and rule.suppress_duplicates:
                    time_diff = datetime.utcnow() - last_routed
                    if time_diff < timedelta(minutes=rule.cooldown_minutes):
                        return True, f"Suppressed by cooldown ({rule.cooldown_minutes} min)"
        
        return False, ""
    
    def _get_representative_alert(self, alerts: List[Alert]) -> Alert:
        """获取代表性告警（最严重或最新的）"""
        # 按严重级别排序
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        ]
        
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (
                severity_order.index(a.severity),
                a.timestamp
            ),
            reverse=True
        )
        
        return sorted_alerts[0]
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """获取所有规则"""
        return [
            {
                "name": rule.name,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "target_count": len(rule.targets),
            }
            for rule in self.rules
        ]
    
    def setup_default_rules(self) -> None:
        """设置默认路由规则"""
        # 关键告警 -> PagerDuty
        self.add_rule(RoutingRule(
            name="critical_to_pagerduty",
            conditions=[
                {"field": "severity", "operator": "==", "value": "critical"},
            ],
            targets=[
                NotificationTarget(NotificationChannel.PAGERDUTY, "oncall"),
            ],
            priority=10,
            cooldown_minutes=1,
        ))
        
        # 数据库告警 -> DBA 团队
        self.add_rule(RoutingRule(
            name="database_to_dba",
            conditions=[
                {"field": "category", "operator": "==", "value": "database"},
            ],
            targets=[
                NotificationTarget(NotificationChannel.SLACK, "#dba-alerts"),
                NotificationTarget(NotificationChannel.EMAIL, "dba@company.com"),
            ],
            priority=20,
            cooldown_minutes=5,
        ))
        
        # 性能告警 -> 开发团队
        self.add_rule(RoutingRule(
            name="performance_to_dev",
            conditions=[
                {"field": "category", "operator": "==", "value": "performance"},
            ],
            targets=[
                NotificationTarget(NotificationChannel.SLACK, "#dev-alerts"),
            ],
            priority=30,
            cooldown_minutes=10,
        ))
        
        # 基础设施告警 -> SRE 团队
        self.add_rule(RoutingRule(
            name="infrastructure_to_sre",
            conditions=[
                {"field": "category", "operator": "==", "value": "infrastructure"},
            ],
            targets=[
                NotificationTarget(NotificationChannel.SLACK, "#sre-alerts"),
            ],
            priority=25,
            cooldown_minutes=5,
        ))
        
        # 默认规则 -> 通用频道
        self.add_rule(RoutingRule(
            name="default_to_general",
            conditions=[],
            targets=[
                NotificationTarget(NotificationChannel.SLACK, "#alerts"),
            ],
            priority=100,
            cooldown_minutes=15,
        ))


# 全局告警路由器实例
alert_router = AlertRouter()


__all__ = [
    "AlertSeverity",
    "AlertCategory",
    "NotificationChannel",
    "Alert",
    "AlertGroup",
    "NotificationTarget",
    "RoutingRule",
    "RoutingResult",
    "AlertClassifier",
    "AlertAggregator",
    "AlertRouter",
    "alert_router",
]
