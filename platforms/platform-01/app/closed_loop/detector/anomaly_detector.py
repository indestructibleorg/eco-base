"""
异常检测引擎
实现多种异常检测算法
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import Enum
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"              # 尖峰
    DROP = "drop"                # 骤降
    TREND_UP = "trend_up"        # 上升趋势
    TREND_DOWN = "trend_down"    # 下降趋势
    SEASONAL = "seasonal"        # 季节性异常
    NOISE = "noise"              # 噪声
    UNKNOWN = "unknown"          # 未知


class DetectionAlgorithm(Enum):
    """检测算法"""
    THREE_SIGMA = "3sigma"           # 3-sigma
    IQR = "iqr"                      # 四分位距
    MAD = "mad"                      # 中位数绝对偏差
    ZSCORE = "zscore"                # Z-score
    PERCENTAGE_CHANGE = "pct_change" # 百分比变化
    THRESHOLD = "threshold"          # 阈值


@dataclass
class Anomaly:
    """异常检测结果"""
    metric_name: str
    timestamp: datetime
    value: float
    expected_value: float
    anomaly_type: AnomalyType
    severity: float  # 0-1
    confidence: float  # 0-1
    algorithm: DetectionAlgorithm
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """指标时间序列"""
    name: str
    values: deque
    timestamps: deque
    window_size: int = 100
    
    def __post_init__(self):
        if not isinstance(self.values, deque):
            self.values = deque(maxlen=self.window_size)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(maxlen=self.window_size)
    
    def add(self, timestamp: datetime, value: float):
        """添加数据点"""
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_recent(self, n: int = None) -> Tuple[List[datetime], List[float]]:
        """获取最近n个数据点"""
        n = n or len(self.values)
        return (
            list(self.timestamps)[-n:],
            list(self.values)[-n:]
        )
    
    def is_ready(self, min_points: int = 10) -> bool:
        """检查是否有足够数据"""
        return len(self.values) >= min_points


class AnomalyDetector:
    """异常检测引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 指标存储
        self.metrics: Dict[str, MetricSeries] = {}
        
        # 检测配置
        self.algorithms = self.config.get('algorithms', [
            DetectionAlgorithm.THREE_SIGMA,
            DetectionAlgorithm.IQR
        ])
        self.sensitivity = self.config.get('sensitivity', 0.95)
        self.min_points = self.config.get('min_points', 10)
        
        # 回调函数
        self.anomaly_handlers: List[Callable[[Anomaly], None]] = []
        
        logger.info("异常检测引擎初始化完成")
    
    def register_metric(self, metric_name: str, window_size: int = 100):
        """注册指标"""
        self.metrics[metric_name] = MetricSeries(
            name=metric_name,
            values=deque(maxlen=window_size),
            timestamps=deque(maxlen=window_size),
            window_size=window_size
        )
        logger.info(f"指标注册: {metric_name}")
    
    def add_data_point(self, metric_name: str, 
                       timestamp: datetime, 
                       value: float) -> Optional[Anomaly]:
        """添加数据点并检测异常"""
        if metric_name not in self.metrics:
            self.register_metric(metric_name)
        
        series = self.metrics[metric_name]
        series.add(timestamp, value)
        
        # 检测异常
        if series.is_ready(self.min_points):
            anomaly = self._detect_anomaly(series, value, timestamp)
            if anomaly:
                self._trigger_anomaly_handlers(anomaly)
                return anomaly
        
        return None
    
    def _detect_anomaly(self, series: MetricSeries, 
                        current_value: float,
                        timestamp: datetime) -> Optional[Anomaly]:
        """检测异常"""
        _, values = series.get_recent()
        
        for algorithm in self.algorithms:
            try:
                result = self._apply_algorithm(
                    algorithm, values, current_value
                )
                if result and result['is_anomaly']:
                    return Anomaly(
                        metric_name=series.name,
                        timestamp=timestamp,
                        value=current_value,
                        expected_value=result.get('expected', mean(values)),
                        anomaly_type=result.get('type', AnomalyType.UNKNOWN),
                        severity=result.get('severity', 0.5),
                        confidence=result.get('confidence', 0.5),
                        algorithm=algorithm,
                        details=result.get('details', {})
                    )
            except Exception as e:
                logger.warning(f"算法 {algorithm.value} 检测失败: {e}")
        
        return None
    
    def detect(self, data: List[float], algorithm_name: str = 'spike') -> Dict[str, Any]:
        """检测异常 (简化接口)"""
        if not data:
            return {'type': 'unknown', 'is_anomaly': False}
        
        # 注册临时指标
        metric_name = 'temp_metric'
        self.register_metric(metric_name, window_size=len(data) + 10)
        
        # 添加数据点
        now = datetime.now()
        for i, value in enumerate(data[:-1]):
            self.add_data_point(metric_name, now + timedelta(seconds=i), value)
        
        # 检测最后一个点
        anomaly = self.add_data_point(metric_name, now + timedelta(seconds=len(data)), data[-1])
        
        if anomaly:
            return {
                'type': anomaly.anomaly_type.value,
                'is_anomaly': True,
                'severity': anomaly.severity,
                'confidence': anomaly.confidence,
                'value': anomaly.value,
                'expected': anomaly.expected_value
            }
        
        return {'type': 'normal', 'is_anomaly': False}
    
    def _apply_algorithm(self, algorithm: DetectionAlgorithm,
                         values: List[float], 
                         current_value: float) -> Optional[Dict]:
        """应用检测算法"""
        if algorithm == DetectionAlgorithm.THREE_SIGMA:
            return self._three_sigma_detect(values, current_value)
        elif algorithm == DetectionAlgorithm.IQR:
            return self._iqr_detect(values, current_value)
        elif algorithm == DetectionAlgorithm.MAD:
            return self._mad_detect(values, current_value)
        elif algorithm == DetectionAlgorithm.ZSCORE:
            return self._zscore_detect(values, current_value)
        elif algorithm == DetectionAlgorithm.PERCENTAGE_CHANGE:
            return self._pct_change_detect(values, current_value)
        elif algorithm == DetectionAlgorithm.THRESHOLD:
            return self._threshold_detect(values, current_value)
        return None
    
    def _three_sigma_detect(self, values: List[float], 
                            current: float) -> Optional[Dict]:
        """3-sigma 检测"""
        if len(values) < 10:
            return None
        
        m = mean(values)
        try:
            s = stdev(values)
        except:
            return None
        
        if s == 0:
            return None
        
        z_score = abs(current - m) / s
        
        if z_score > 3:
            return {
                'is_anomaly': True,
                'expected': m,
                'type': AnomalyType.SPIKE if current > m else AnomalyType.DROP,
                'severity': min(z_score / 5, 1.0),
                'confidence': min(z_score / 4, 0.99),
                'details': {'z_score': z_score, 'mean': m, 'std': s}
            }
        return {'is_anomaly': False}
    
    def _iqr_detect(self, values: List[float], 
                    current: float) -> Optional[Dict]:
        """IQR 检测"""
        if len(values) < 10:
            return None
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        if current < lower_bound or current > upper_bound:
            distance = min(abs(current - lower_bound), abs(current - upper_bound))
            severity = min(distance / iqr if iqr > 0 else 0.5, 1.0)
            
            return {
                'is_anomaly': True,
                'expected': (q1 + q3) / 2,
                'type': AnomalyType.SPIKE if current > upper_bound else AnomalyType.DROP,
                'severity': severity,
                'confidence': min(severity * 1.2, 0.99),
                'details': {'q1': q1, 'q3': q3, 'iqr': iqr, 
                           'bounds': (lower_bound, upper_bound)}
            }
        return {'is_anomaly': False}
    
    def _mad_detect(self, values: List[float], 
                    current: float) -> Optional[Dict]:
        """MAD 检测"""
        if len(values) < 10:
            return None
        
        median = np.median(values)
        mad = np.median([abs(v - median) for v in values])
        
        if mad == 0:
            return None
        
        modified_z = 0.6745 * (current - median) / mad
        
        if abs(modified_z) > 3.5:
            return {
                'is_anomaly': True,
                'expected': median,
                'type': AnomalyType.SPIKE if current > median else AnomalyType.DROP,
                'severity': min(abs(modified_z) / 5, 1.0),
                'confidence': min(abs(modified_z) / 4, 0.99),
                'details': {'modified_z': modified_z, 'median': median, 'mad': mad}
            }
        return {'is_anomaly': False}
    
    def _zscore_detect(self, values: List[float], 
                       current: float) -> Optional[Dict]:
        """Z-score 检测"""
        return self._three_sigma_detect(values, current)
    
    def _pct_change_detect(self, values: List[float], 
                           current: float) -> Optional[Dict]:
        """百分比变化检测"""
        if len(values) < 2:
            return None
        
        last_value = values[-1]
        if last_value == 0:
            return None
        
        pct_change = (current - last_value) / last_value * 100
        threshold = self.config.get('pct_change_threshold', 50)
        
        if abs(pct_change) > threshold:
            return {
                'is_anomaly': True,
                'expected': last_value,
                'type': AnomalyType.SPIKE if pct_change > 0 else AnomalyType.DROP,
                'severity': min(abs(pct_change) / 100, 1.0),
                'confidence': min(abs(pct_change) / threshold * 0.8, 0.99),
                'details': {'pct_change': pct_change, 'threshold': threshold}
            }
        return {'is_anomaly': False}
    
    def _threshold_detect(self, values: List[float], 
                          current: float) -> Optional[Dict]:
        """阈值检测"""
        thresholds = self.config.get('thresholds', {})
        
        # 这里简化处理，实际应该根据指标名称获取阈值
        upper = thresholds.get('upper', 90)
        lower = thresholds.get('lower', 10)
        
        if current > upper:
            return {
                'is_anomaly': True,
                'expected': upper,
                'type': AnomalyType.SPIKE,
                'severity': min((current - upper) / upper, 1.0),
                'confidence': 0.9,
                'details': {'threshold': upper, 'direction': 'upper'}
            }
        elif current < lower:
            return {
                'is_anomaly': True,
                'expected': lower,
                'type': AnomalyType.DROP,
                'severity': min((lower - current) / lower, 1.0),
                'confidence': 0.9,
                'details': {'threshold': lower, 'direction': 'lower'}
            }
        return {'is_anomaly': False}
    
    def register_anomaly_handler(self, handler: Callable[[Anomaly], None]):
        """注册异常处理器"""
        self.anomaly_handlers.append(handler)
        logger.info("异常处理器注册完成")
    
    def _trigger_anomaly_handlers(self, anomaly: Anomaly):
        """触发异常处理器"""
        for handler in self.anomaly_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(anomaly))
                else:
                    handler(anomaly)
            except Exception as e:
                logger.warning(f"异常处理器失败: {e}")
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """获取指标统计"""
        series = self.metrics.get(metric_name)
        if not series or not series.values:
            return {}
        
        values = list(series.values)
        return {
            'count': len(values),
            'mean': mean(values),
            'std': stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }
    
    def get_all_metrics(self) -> List[str]:
        """获取所有指标名称"""
        return list(self.metrics.keys())
    
    def detect_trend(self, metric_name: str, 
                     window: int = 24) -> Optional[Dict]:
        """检测趋势"""
        series = self.metrics.get(metric_name)
        if not series or len(series.values) < window:
            return None
        
        timestamps, values = series.get_recent(window)
        
        # 简单线性回归
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # 计算R²
        y_pred = slope * x + intercept
        ss_res = np.sum((np.array(values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(values) - mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        trend_type = AnomalyType.TREND_UP if slope > 0 else AnomalyType.TREND_DOWN
        
        return {
            'metric': metric_name,
            'trend': trend_type.value,
            'slope': slope,
            'r_squared': r_squared,
            'confidence': r_squared,
            'window': window
        }


import asyncio
