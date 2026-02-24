# =============================================================================
# Anomaly Detection Engine
# =============================================================================
# 異常檢測引擎 - Phase 1 基礎閉環核心組件
# 支持多種檢測算法: 統計方法、閾值方法、趨勢方法
# =============================================================================

import asyncio
import statistics
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import numpy as np

from app.core.logging import get_logger
from app.core.events import event_bus, EventType, emit_event

logger = get_logger("anomaly_detector")


class AnomalyType(Enum):
    """異常類型"""
    STATISTICAL = auto()      # 統計異常
    THRESHOLD = auto()        # 閾值異常
    TREND = auto()            # 趨勢異常
    PATTERN = auto()          # 模式異常


class Severity(Enum):
    """嚴重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """異常事件"""
    id: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: Severity
    value: float
    expected_range: tuple
    timestamp: datetime
    duration: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type.name,
            "severity": self.severity.value,
            "value": self.value,
            "expected_min": self.expected_range[0],
            "expected_max": self.expected_range[1],
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration.total_seconds(),
            "metadata": self.metadata
        }


@dataclass
class DetectionRule:
    """檢測規則"""
    name: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: Severity
    
    # 閾值配置
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    
    # 統計配置
    sigma_multiplier: float = 3.0  # 3-sigma
    iqr_multiplier: float = 1.5    # IQR 乘數
    
    # 趨勢配置
    trend_window: int = 10         # 趨勢窗口大小
    trend_threshold: float = 0.1   # 趨勢閾值
    
    # 通用配置
    window_size: int = 100         # 歷史數據窗口
    cooldown_minutes: int = 5      # 冷卻時間
    
    # 狀態
    last_triggered: Optional[datetime] = None
    history: deque = field(default_factory=lambda: deque(maxlen=100))


class StatisticalDetector:
    """統計異常檢測器"""
    
    @staticmethod
    def detect_sigma(values: List[float], sigma_multiplier: float = 3.0) -> List[bool]:
        """
        3-sigma 異常檢測
        
        基於正態分布假設，超過 3 個標準差視為異常
        """
        if len(values) < 3:
            return [False] * len(values)
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        if std == 0:
            return [False] * len(values)
        
        return [abs(x - mean) > sigma_multiplier * std for x in values]
    
    @staticmethod
    def detect_iqr(values: List[float], multiplier: float = 1.5) -> List[bool]:
        """
        IQR (四分位距) 異常檢測
        
        對異常值不敏感，適合有離群點的數據
        """
        if len(values) < 4:
            return [False] * len(values)
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return [x < lower_bound or x > upper_bound for x in values]
    
    @staticmethod
    def detect_mad(values: List[float], threshold: float = 3.5) -> List[bool]:
        """
        MAD (中位數絕對偏差) 異常檢測
        
        對異常值非常穩健
        """
        if len(values) < 3:
            return [False] * len(values)
        
        median = statistics.median(values)
        abs_deviations = [abs(x - median) for x in values]
        mad = statistics.median(abs_deviations)
        
        if mad == 0:
            return [False] * len(values)
        
        modified_z_scores = [0.6745 * (x - median) / mad for x in values]
        return [abs(z) > threshold for z in modified_z_scores]


class TrendDetector:
    """趨勢異常檢測器"""
    
    @staticmethod
    def detect_trend_change(values: List[float], window_size: int = 10) -> List[bool]:
        """
        檢測趨勢變化
        
        使用線性回歸檢測趨勢變化點
        """
        if len(values) < window_size * 2:
            return [False] * len(values)
        
        results = [False] * len(values)
        
        for i in range(window_size, len(values) - window_size):
            # 前半段趨勢
            x1 = list(range(window_size))
            y1 = values[i - window_size:i]
            slope1 = TrendDetector._calculate_slope(x1, y1)
            
            # 後半段趨勢
            x2 = list(range(window_size))
            y2 = values[i:i + window_size]
            slope2 = TrendDetector._calculate_slope(x2, y2)
            
            # 趨勢變化顯著
            if abs(slope2 - slope1) > abs(slope1) * 0.5:
                results[i] = True
        
        return results
    
    @staticmethod
    def _calculate_slope(x: List[int], y: List[float]) -> float:
        """計算線性回歸斜率"""
        n = len(x)
        if n == 0:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    @staticmethod
    def detect_seasonal_anomaly(
        values: List[float],
        season_length: int = 24
    ) -> List[bool]:
        """
        季節性異常檢測
        
        檢測偏離季節性模式的異常
        """
        if len(values) < season_length * 2:
            return [False] * len(values)
        
        results = [False] * len(values)
        
        for i in range(season_length, len(values)):
            # 獲取同一季節位置的歷史值
            seasonal_indices = list(range(i % season_length, i, season_length))
            seasonal_values = [values[j] for j in seasonal_indices]
            
            if len(seasonal_values) >= 3:
                mean = statistics.mean(seasonal_values)
                std = statistics.stdev(seasonal_values)
                
                if std > 0 and abs(values[i] - mean) > 3 * std:
                    results[i] = True
        
        return results


class AnomalyDetector:
    """
    異常檢測引擎
    
    統一管理所有檢測規則和算法
    """
    
    def __init__(self):
        self.rules: Dict[str, DetectionRule] = {}
        self._running = False
        self._check_interval = 30  # 秒
        self._metric_fetcher: Optional[Callable] = None
    
    def register_rule(self, rule: DetectionRule) -> None:
        """註冊檢測規則"""
        self.rules[rule.name] = rule
        logger.info(
            "detection_rule_registered",
            rule_name=rule.name,
            metric=rule.metric_name,
            anomaly_type=rule.anomaly_type.name
        )
    
    def unregister_rule(self, rule_name: str) -> bool:
        """註銷檢測規則"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info("detection_rule_unregistered", rule_name=rule_name)
            return True
        return False
    
    def set_metric_fetcher(self, fetcher: Callable[[str], List[float]]) -> None:
        """設置指標獲取函數"""
        self._metric_fetcher = fetcher
    
    async def detect(self, rule_name: str, current_value: float) -> Optional[AnomalyEvent]:
        """
        執行單次檢測
        
        Args:
            rule_name: 規則名稱
            current_value: 當前值
            
        Returns:
            異常事件或 None
        """
        rule = self.rules.get(rule_name)
        if not rule:
            logger.warning("rule_not_found", rule_name=rule_name)
            return None
        
        # 檢查冷卻時間
        if rule.last_triggered:
            cooldown = timedelta(minutes=rule.cooldown_minutes)
            if datetime.utcnow() - rule.last_triggered < cooldown:
                return None
        
        # 添加到歷史
        rule.history.append(current_value)
        
        # 執行檢測
        is_anomaly = False
        expected_range = (0.0, 0.0)
        
        if rule.anomaly_type == AnomalyType.THRESHOLD:
            is_anomaly, expected_range = self._check_threshold(rule, current_value)
        
        elif rule.anomaly_type == AnomalyType.STATISTICAL:
            is_anomaly, expected_range = self._check_statistical(rule, current_value)
        
        elif rule.anomaly_type == AnomalyType.TREND:
            is_anomaly, expected_range = self._check_trend(rule, current_value)
        
        if is_anomaly:
            rule.last_triggered = datetime.utcnow()
            
            anomaly = AnomalyEvent(
                id=self._generate_anomaly_id(),
                metric_name=rule.metric_name,
                anomaly_type=rule.anomaly_type,
                severity=rule.severity,
                value=current_value,
                expected_range=expected_range,
                timestamp=datetime.utcnow(),
                duration=timedelta(seconds=0),
                metadata={"rule_name": rule_name}
            )
            
            # 發布異常事件
            await self._emit_anomaly_event(anomaly)
            
            logger.warning(
                "anomaly_detected",
                rule_name=rule_name,
                metric=rule.metric_name,
                value=current_value,
                severity=rule.severity.value
            )
            
            return anomaly
        
        return None
    
    def _check_threshold(
        self,
        rule: DetectionRule,
        value: float
    ) -> tuple[bool, tuple]:
        """閾值檢測"""
        is_anomaly = False
        
        if rule.threshold_min is not None and value < rule.threshold_min:
            is_anomaly = True
        
        if rule.threshold_max is not None and value > rule.threshold_max:
            is_anomaly = True
        
        expected_range = (
            rule.threshold_min if rule.threshold_min is not None else float('-inf'),
            rule.threshold_max if rule.threshold_max is not None else float('inf')
        )
        
        return is_anomaly, expected_range
    
    def _check_statistical(
        self,
        rule: DetectionRule,
        value: float
    ) -> tuple[bool, tuple]:
        """統計檢測"""
        if len(rule.history) < 3:
            return False, (0.0, 0.0)
        
        history_list = list(rule.history)
        
        # 使用 3-sigma 檢測
        anomalies = StatisticalDetector.detect_sigma(
            history_list + [value],
            rule.sigma_multiplier
        )
        
        is_anomaly = anomalies[-1]
        
        # 計算預期範圍
        mean = statistics.mean(history_list)
        std = statistics.stdev(history_list) if len(history_list) > 1 else 0
        expected_range = (
            mean - rule.sigma_multiplier * std,
            mean + rule.sigma_multiplier * std
        )
        
        return is_anomaly, expected_range
    
    def _check_trend(
        self,
        rule: DetectionRule,
        value: float
    ) -> tuple[bool, tuple]:
        """趨勢檢測"""
        if len(rule.history) < rule.trend_window * 2:
            return False, (0.0, 0.0)
        
        history_list = list(rule.history) + [value]
        
        anomalies = TrendDetector.detect_trend_change(
            history_list,
            rule.trend_window
        )
        
        is_anomaly = anomalies[-1]
        
        # 趨勢檢測的預期範圍較寬
        expected_range = (float('-inf'), float('inf'))
        
        return is_anomaly, expected_range
    
    async def _emit_anomaly_event(self, anomaly: AnomalyEvent) -> None:
        """發布異常事件"""
        await emit_event(
            EventType.PROVIDER_ERROR,  # 復用 PROVIDER_ERROR 或創建新的 ANOMALY_DETECTED
            {
                "anomaly_id": anomaly.id,
                "metric_name": anomaly.metric_name,
                "anomaly_type": anomaly.anomaly_type.name,
                "severity": anomaly.severity.value,
                "value": anomaly.value,
                "timestamp": anomaly.timestamp.isoformat()
            },
            source="anomaly_detector"
        )
    
    def _generate_anomaly_id(self) -> str:
        """生成異常 ID"""
        import uuid
        return f"anomaly-{uuid.uuid4().hex[:12]}"
    
    async def start_monitoring(self) -> None:
        """啟動持續監控"""
        self._running = True
        logger.info("anomaly_detector_started")
        
        while self._running:
            try:
                await self._check_all_rules()
                await asyncio.sleep(self._check_interval)
            except Exception as e:
                logger.error("monitoring_error", error=str(e))
                await asyncio.sleep(self._check_interval)
    
    async def _check_all_rules(self) -> None:
        """檢查所有規則"""
        if not self._metric_fetcher:
            return
        
        for rule_name, rule in self.rules.items():
            try:
                # 獲取指標數據
                values = self._metric_fetcher(rule.metric_name)
                
                if values:
                    # 使用最新值進行檢測
                    current_value = values[-1]
                    await self.detect(rule_name, current_value)
                    
            except Exception as e:
                logger.error(
                    "rule_check_failed",
                    rule_name=rule_name,
                    error=str(e)
                )
    
    def stop_monitoring(self) -> None:
        """停止監控"""
        self._running = False
        logger.info("anomaly_detector_stopped")
    
    def get_rule_status(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """獲取規則狀態"""
        rule = self.rules.get(rule_name)
        if not rule:
            return None
        
        return {
            "name": rule.name,
            "metric_name": rule.metric_name,
            "anomaly_type": rule.anomaly_type.name,
            "severity": rule.severity.value,
            "history_size": len(rule.history),
            "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
            "current_stats": self._calculate_stats(list(rule.history)) if rule.history else None
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """計算統計信息"""
        if not values:
            return {}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values)
        }
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """列出所有規則"""
        return [
            {
                "name": rule.name,
                "metric_name": rule.metric_name,
                "anomaly_type": rule.anomaly_type.name,
                "severity": rule.severity.value
            }
            for rule in self.rules.values()
        ]


# 全局異常檢測器實例
anomaly_detector = AnomalyDetector()
