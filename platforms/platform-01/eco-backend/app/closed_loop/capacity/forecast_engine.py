# =============================================================================
# Capacity Forecast Engine
# =============================================================================
# 容量预测引擎 - Phase 2 核心组件
# 基于历史数据预测未来负载
# =============================================================================

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque


@dataclass
class ForecastResult:
    """预测结果"""
    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper)
    model_name: str
    accuracy: float  # MAPE or similar metric


class ForecastModel:
    """预测模型基类"""
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """预测未来值"""
        raise NotImplementedError
    
    def get_accuracy(self, actual: List[float], predicted: List[float]) -> float:
        """计算预测准确度 (MAPE)"""
        if not actual or not predicted:
            return 0.0
        
        mape = np.mean([
            abs(a - p) / max(abs(a), 1e-10) * 100
            for a, p in zip(actual, predicted)
        ])
        
        return 100 - mape  # 返回准确度 (100 - MAPE)


class HoltWintersModel(ForecastModel):
    """
    Holt-Winters 季节性预测模型
    
    适用于具有趋势和季节性的时间序列
    """
    
    def __init__(
        self,
        seasonal_periods: int = 24,  # 默认 24 小时周期
        alpha: float = 0.3,  # 水平平滑因子
        beta: float = 0.1,   # 趋势平滑因子
        gamma: float = 0.1,  # 季节性平滑因子
    ):
        self.seasonal_periods = seasonal_periods
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.level = None
        self.trend = None
        self.seasonal = []
        self.last_timestamp = None
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """训练 Holt-Winters 模型"""
        if len(values) < self.seasonal_periods * 2:
            # 数据不足，使用简单初始化
            self.level = np.mean(values)
            self.trend = 0
            self.seasonal = [0] * self.seasonal_periods
            return
        
        # 初始化季节性成分
        season_averages = []
        for i in range(self.seasonal_periods):
            season_values = values[i::self.seasonal_periods]
            season_averages.append(np.mean(season_values))
        
        self.seasonal = [
            values[i] - season_averages[i % self.seasonal_periods]
            for i in range(self.seasonal_periods)
        ]
        
        # 初始化水平和趋势
        self.level = np.mean(values[:self.seasonal_periods])
        self.trend = np.mean([
            values[i + self.seasonal_periods] - values[i]
            for i in range(self.seasonal_periods)
        ]) / self.seasonal_periods
        
        # 训练
        for i in range(self.seasonal_periods, len(values)):
            value = values[i]
            season_idx = i % self.seasonal_periods
            
            # 更新水平
            last_level = self.level
            self.level = (
                self.alpha * (value - self.seasonal[season_idx]) +
                (1 - self.alpha) * (self.level + self.trend)
            )
            
            # 更新趋势
            self.trend = (
                self.beta * (self.level - last_level) +
                (1 - self.beta) * self.trend
            )
            
            # 更新季节性
            self.seasonal[season_idx] = (
                self.gamma * (value - self.level) +
                (1 - self.gamma) * self.seasonal[season_idx]
            )
        
        self.last_timestamp = timestamps[-1]
    
    def predict(self, steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """预测未来值"""
        predictions = []
        confidence_intervals = []
        
        for i in range(steps):
            season_idx = (len(self.seasonal) + i) % self.seasonal_periods
            forecast = self.level + (i + 1) * self.trend + self.seasonal[season_idx]
            predictions.append(max(0, forecast))  # 确保非负
            
            # 简单的置信区间
            std_dev = abs(self.trend) * (i + 1) * 0.5
            confidence_intervals.append((
                max(0, forecast - 2 * std_dev),
                forecast + 2 * std_dev
            ))
        
        return predictions, confidence_intervals


class SimpleLinearModel(ForecastModel):
    """简单线性回归模型"""
    
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.std_error = 0.0
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """训练线性回归模型"""
        if len(values) < 2:
            self.slope = 0.0
            self.intercept = values[0] if values else 0.0
            return
        
        # 使用索引作为 x
        x = np.arange(len(values))
        y = np.array(values)
        
        # 计算斜率和截距
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        self.slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        self.intercept = y_mean - self.slope * x_mean
        
        # 计算标准误差
        predictions = self.slope * x + self.intercept
        residuals = y - predictions
        self.std_error = np.std(residuals)
    
    def predict(self, steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """预测未来值"""
        predictions = []
        confidence_intervals = []
        
        for i in range(steps):
            x = len(self.slope) + i if isinstance(self.slope, np.ndarray) else i
            forecast = self.slope * x + self.intercept
            predictions.append(max(0, forecast))
            
            # 置信区间
            confidence_intervals.append((
                max(0, forecast - 2 * self.std_error),
                forecast + 2 * self.std_error
            ))
        
        return predictions, confidence_intervals


class MovingAverageModel(ForecastModel):
    """移动平均模型"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.std_dev = 0.0
    
    def fit(self, timestamps: List[datetime], values: List[float]) -> None:
        """训练移动平均模型"""
        self.history.clear()
        self.history.extend(values[-self.window_size:])
        self.std_dev = np.std(values) if values else 0.0
    
    def predict(self, steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """预测未来值"""
        if not self.history:
            return [0.0] * steps, [(0.0, 0.0)] * steps
        
        forecast = np.mean(self.history)
        predictions = [forecast] * steps
        
        confidence_intervals = [
            (max(0, forecast - 2 * self.std_dev), forecast + 2 * self.std_dev)
            for _ in range(steps)
        ]
        
        return predictions, confidence_intervals


class ForecastEngine:
    """
    预测引擎
    
    管理多个预测模型，自动选择最佳模型
    """
    
    def __init__(self):
        self.models: Dict[str, ForecastModel] = {}
        self._metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._max_history = 1000
    
    def register_model(self, name: str, model: ForecastModel) -> None:
        """注册预测模型"""
        self.models[name] = model
    
    def record_metric(
        self,
        metric_name: str,
        timestamp: datetime,
        value: float
    ) -> None:
        """记录指标值"""
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []
        
        self._metric_history[metric_name].append((timestamp, value))
        
        # 限制历史大小
        if len(self._metric_history[metric_name]) > self._max_history:
            self._metric_history[metric_name].pop(0)
    
    def forecast(
        self,
        metric_name: str,
        horizon_minutes: int = 60,
        model_name: Optional[str] = None,
    ) -> Optional[ForecastResult]:
        """
        预测未来值
        
        Args:
            metric_name: 指标名称
            horizon_minutes: 预测时间范围（分钟）
            model_name: 指定模型名称，如果为 None 则自动选择
        
        Returns:
            预测结果
        """
        history = self._metric_history.get(metric_name, [])
        
        if len(history) < 10:
            return None  # 数据不足
        
        timestamps = [t for t, _ in history]
        values = [v for _, v in history]
        
        # 选择模型
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self._select_best_model(timestamps, values)
        
        # 训练模型
        model.fit(timestamps, values)
        
        # 预测
        steps = horizon_minutes // 5  # 假设 5 分钟一个数据点
        predictions, confidence_intervals = model.predict(steps)
        
        # 生成未来时间戳
        last_timestamp = timestamps[-1]
        future_timestamps = [
            last_timestamp + timedelta(minutes=5 * (i + 1))
            for i in range(steps)
        ]
        
        # 计算准确度
        # 使用最后 10% 的数据验证
        validation_size = max(1, len(values) // 10)
        train_values = values[:-validation_size]
        test_values = values[-validation_size:]
        
        model.fit(timestamps[:-validation_size], train_values)
        pred_values, _ = model.predict(validation_size)
        accuracy = model.get_accuracy(test_values, pred_values)
        
        return ForecastResult(
            metric_name=metric_name,
            timestamps=future_timestamps,
            values=predictions,
            confidence_intervals=confidence_intervals,
            model_name=model.__class__.__name__,
            accuracy=accuracy,
        )
    
    def _select_best_model(
        self,
        timestamps: List[datetime],
        values: List[float]
    ) -> ForecastModel:
        """选择最佳模型"""
        if len(values) < 20:
            return MovingAverageModel(window_size=5)
        
        # 检测趋势
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # 检测季节性
        has_seasonality = self._detect_seasonality(values)
        
        if has_seasonality and len(values) >= 48:
            return HoltWintersModel(seasonal_periods=24)
        elif abs(slope) > 0.01:
            return SimpleLinearModel()
        else:
            return MovingAverageModel(window_size=10)
    
    def _detect_seasonality(self, values: List[float], period: int = 24) -> bool:
        """检测时间序列是否有季节性"""
        if len(values) < period * 2:
            return False
        
        # 简单的季节性检测：计算周期内的方差
        seasonal_variances = []
        for i in range(period):
            seasonal_values = values[i::period]
            if len(seasonal_values) > 1:
                seasonal_variances.append(np.var(seasonal_values))
        
        if not seasonal_variances:
            return False
        
        # 如果周期内方差较小，说明有季节性
        mean_variance = np.mean(seasonal_variances)
        total_variance = np.var(values)
        
        return mean_variance < total_variance * 0.5
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """获取指标统计信息"""
        history = self._metric_history.get(metric_name, [])
        
        if not history:
            return {"error": "No data available"}
        
        values = [v for _, v in history]
        
        return {
            "metric_name": metric_name,
            "data_points": len(history),
            "time_range": {
                "start": history[0][0].isoformat(),
                "end": history[-1][0].isoformat(),
            },
            "statistics": {
                "mean": round(np.mean(values), 2),
                "std": round(np.std(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            },
        }


# 全局预测引擎实例
forecast_engine = ForecastEngine()

# 注册默认模型
forecast_engine.register_model("holt_winters", HoltWintersModel())
forecast_engine.register_model("linear", SimpleLinearModel())
forecast_engine.register_model("moving_average", MovingAverageModel())


__all__ = [
    "ForecastResult",
    "ForecastModel",
    "HoltWintersModel",
    "SimpleLinearModel",
    "MovingAverageModel",
    "ForecastEngine",
    "forecast_engine",
]
