"""
预测引擎
实现容量需求预测
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ForecastModel(Enum):
    """预测模型"""
    NAIVE = "naive"                    # 简单平均
    MOVING_AVERAGE = "ma"              # 移动平均
    EXPONENTIAL_SMOOTHING = "es"       # 指数平滑
    HOLT_WINTERS = "holt_winters"      # Holt-Winters (季节性)
    LINEAR_REGRESSION = "linear"       # 线性回归
    PROPHET = "prophet"                # Facebook Prophet


@dataclass
class ForecastResult:
    """预测结果"""
    metric_name: str
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]  # (lower, upper)
    model_used: ForecastModel
    accuracy_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_peak_value(self) -> Tuple[float, datetime]:
        """获取预测峰值"""
        max_idx = np.argmax(self.forecast_values)
        return self.forecast_values[max_idx], self.forecast_timestamps[max_idx]
    
    def get_average_value(self) -> float:
        """获取预测平均值"""
        return np.mean(self.forecast_values)


class ForecastEngine:
    """预测引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 配置
        self.default_model = ForecastModel(
            self.config.get('default_model', 'holt_winters')
        )
        self.forecast_horizon = self.config.get('forecast_horizon_hours', 24)
        self.history_window = self.config.get('history_window_days', 7)
        
        # 历史数据存储
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.forecast_cache: Dict[str, ForecastResult] = {}
        
        # 模型参数
        self.model_params = self.config.get('model_params', {})
        
        logger.info(f"预测引擎初始化完成 (默认模型: {self.default_model.value})")
    
    def add_metric_point(self, metric_name: str, 
                         timestamp: datetime, 
                         value: float):
        """添加指标数据点"""
        self.metric_history[metric_name].append((timestamp, value))
        
        # 清理过期数据
        cutoff = datetime.now() - timedelta(days=self.history_window)
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts > cutoff
        ]
    
    def add_metric_batch(self, metric_name: str, 
                         data: List[Tuple[datetime, float]]):
        """批量添加指标数据"""
        self.metric_history[metric_name].extend(data)
        
        # 清理和排序
        cutoff = datetime.now() - timedelta(days=self.history_window)
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts > cutoff
        ]
        self.metric_history[metric_name].sort(key=lambda x: x[0])
    
    async def forecast(self, 
                       metric_name: str,
                       horizon_hours: int = None,
                       model: ForecastModel = None) -> Optional[ForecastResult]:
        """执行预测"""
        horizon_hours = horizon_hours or self.forecast_horizon
        model = model or self.default_model
        
        # 获取历史数据
        history = self.metric_history.get(metric_name, [])
        if len(history) < 10:
            logger.warning(f"历史数据不足: {metric_name} ({len(history)} 点)")
            return None
        
        # 提取数值
        timestamps = [ts for ts, _ in history]
        values = np.array([val for _, val in history])
        
        # 选择模型并预测
        if model == ForecastModel.NAIVE:
            forecast_values, intervals = self._naive_forecast(values, horizon_hours)
        elif model == ForecastModel.MOVING_AVERAGE:
            forecast_values, intervals = self._moving_average_forecast(values, horizon_hours)
        elif model == ForecastModel.EXPONENTIAL_SMOOTHING:
            forecast_values, intervals = self._exponential_smoothing_forecast(values, horizon_hours)
        elif model == ForecastModel.HOLT_WINTERS:
            forecast_values, intervals = self._holt_winters_forecast(values, horizon_hours)
        elif model == ForecastModel.LINEAR_REGRESSION:
            forecast_values, intervals = self._linear_regression_forecast(values, horizon_hours)
        else:
            logger.warning(f"不支持的模型: {model.value}")
            return None
        
        # 生成未来时间戳
        last_timestamp = timestamps[-1]
        forecast_timestamps = [
            last_timestamp + timedelta(hours=i+1)
            for i in range(horizon_hours)
        ]
        
        # 计算准确度指标
        accuracy = self._calculate_accuracy(values, forecast_values[:len(values)])
        
        result = ForecastResult(
            metric_name=metric_name,
            forecast_values=forecast_values.tolist(),
            forecast_timestamps=forecast_timestamps,
            confidence_intervals=intervals,
            model_used=model,
            accuracy_metrics=accuracy
        )
        
        # 缓存结果
        self.forecast_cache[metric_name] = result
        
        logger.info(f"预测完成: {metric_name} (模型: {model.value}, 预测点数: {horizon_hours})")
        return result
    
    def _naive_forecast(self, values: np.ndarray, 
                        horizon: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """简单预测（使用最后一个值）"""
        last_value = values[-1]
        forecast = np.full(horizon, last_value)
        
        # 简单置信区间
        std = np.std(values[-24:]) if len(values) >= 24 else np.std(values)
        intervals = [(last_value - 2*std, last_value + 2*std) for _ in range(horizon)]
        
        return forecast, intervals
    
    def _moving_average_forecast(self, values: np.ndarray, 
                                  horizon: int,
                                  window: int = 24) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """移动平均预测"""
        # 使用最后window个点的平均值
        ma = np.mean(values[-window:])
        forecast = np.full(horizon, ma)
        
        # 置信区间
        std = np.std(values[-window:])
        intervals = [(ma - 2*std, ma + 2*std) for _ in range(horizon)]
        
        return forecast, intervals
    
    def _exponential_smoothing_forecast(self, values: np.ndarray,
                                         horizon: int,
                                         alpha: float = 0.3) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """指数平滑预测"""
        # 初始化
        smoothed = values[0]
        
        # 平滑历史数据
        for val in values[1:]:
            smoothed = alpha * val + (1 - alpha) * smoothed
        
        forecast = np.full(horizon, smoothed)
        
        # 置信区间
        residuals = values[1:] - values[:-1]
        std = np.std(residuals)
        intervals = [(smoothed - 2*std, smoothed + 2*std) for _ in range(horizon)]
        
        return forecast, intervals
    
    def _holt_winters_forecast(self, values: np.ndarray,
                                horizon: int,
                                seasonal_period: int = 24) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Holt-Winters 季节性预测"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # 确保数据长度足够
            if len(values) < seasonal_period * 2:
                logger.warning("数据不足，降级为简单指数平滑")
                return self._exponential_smoothing_forecast(values, horizon)
            
            # 拟合模型
            model = ExponentialSmoothing(
                values,
                seasonal_periods=seasonal_period,
                trend='add',
                seasonal='add'
            )
            fitted = model.fit()
            
            # 预测
            forecast = fitted.forecast(horizon)
            
            # 置信区间
            resid_std = np.std(fitted.resid)
            intervals = [
                (forecast[i] - 2*resid_std, forecast[i] + 2*resid_std)
                for i in range(horizon)
            ]
            
            return forecast, intervals
        
        except ImportError:
            logger.warning("statsmodels 未安装，降级为指数平滑")
            return self._exponential_smoothing_forecast(values, horizon)
        except Exception as e:
            logger.warning(f"Holt-Winters 失败: {e}，降级为指数平滑")
            return self._exponential_smoothing_forecast(values, horizon)
    
    def _linear_regression_forecast(self, values: np.ndarray,
                                     horizon: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """线性回归预测"""
        # 创建时间序列特征
        x = np.arange(len(values))
        
        # 拟合线性回归
        coeffs = np.polyfit(x, values, 1)
        
        # 预测
        future_x = np.arange(len(values), len(values) + horizon)
        forecast = np.polyval(coeffs, future_x)
        
        # 确保预测值非负（对于资源指标）
        forecast = np.maximum(forecast, 0)
        
        # 置信区间
        predictions = np.polyval(coeffs, x)
        residuals = values - predictions
        std = np.std(residuals)
        intervals = [
            (max(0, forecast[i] - 2*std), forecast[i] + 2*std)
            for i in range(horizon)
        ]
        
        return forecast, intervals
    
    def _calculate_accuracy(self, actual: np.ndarray, 
                           predicted: np.ndarray) -> Dict[str, float]:
        """计算准确度指标"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return {}
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # MAPE
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        
        # 方向准确度
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2),
            'direction_accuracy': round(direction_accuracy, 2)
        }
    
    def detect_anomaly_in_forecast(self, forecast: ForecastResult,
                                    threshold_multiplier: float = 2.0) -> List[Dict]:
        """在预测中检测异常"""
        anomalies = []
        
        history = self.metric_history.get(forecast.metric_name, [])
        if not history:
            return anomalies
        
        # 计算历史统计
        values = [val for _, val in history]
        mean = np.mean(values)
        std = np.std(values)
        
        threshold = mean + threshold_multiplier * std
        
        for i, (value, timestamp) in enumerate(zip(forecast.forecast_values, 
                                                    forecast.forecast_timestamps)):
            if value > threshold:
                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'predicted_value': value,
                    'threshold': threshold,
                    'severity': 'high' if value > mean + 3*std else 'medium'
                })
        
        return anomalies
    
    def get_forecast(self, metric_name: str) -> Optional[ForecastResult]:
        """获取缓存的预测结果"""
        return self.forecast_cache.get(metric_name)
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """获取指标统计"""
        history = self.metric_history.get(metric_name, [])
        if not history:
            return {}
        
        values = [val for _, val in history]
        
        return {
            'data_points': len(values),
            'mean': round(np.mean(values), 2),
            'std': round(np.std(values), 2),
            'min': round(np.min(values), 2),
            'max': round(np.max(values), 2),
            'latest': round(values[-1], 2) if values else None
        }
    
    def get_all_metrics(self) -> List[str]:
        """获取所有指标名称"""
        return list(self.metric_history.keys())
    
    def clear_history(self, metric_name: str = None):
        """清理历史数据"""
        if metric_name:
            if metric_name in self.metric_history:
                del self.metric_history[metric_name]
            if metric_name in self.forecast_cache:
                del self.forecast_cache[metric_name]
        else:
            self.metric_history.clear()
            self.forecast_cache.clear()
    
    def forecast(self, data: List[float], model: str = 'linear', periods: int = 3) -> List[float]:
        """预测 (简化接口)"""
        # 注册临时指标 (直接使用字典)
        metric_name = 'temp_forecast_metric'
        self.metric_history[metric_name] = []
        
        # 添加历史数据
        now = datetime.now()
        for i, value in enumerate(data):
            self.add_metric_point(metric_name, now + timedelta(hours=i), value)
        
        # 执行预测
        import asyncio
        model_type = ForecastModel(model) if isinstance(model, str) else model
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._forecast_async(metric_name, model_type, periods)
            )
            
            if result:
                return result.forecast_values
        except:
            pass
        
        # 如果预测失败，返回简单的线性外推
        if len(data) >= 2:
            slope = (data[-1] - data[0]) / len(data)
            return [data[-1] + slope * (i + 1) for i in range(periods)]
        
        return [data[-1]] * periods if data else [0] * periods
    
    async def _forecast_async(self, metric_name: str, model: ForecastModel, periods: int):
        """异步预测"""
        return await self.forecast(metric_name, horizon_hours=periods, model=model)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'metrics_tracked': len(self.metric_history),
            'forecasts_cached': len(self.forecast_cache),
            'default_model': self.default_model.value,
            'forecast_horizon_hours': self.forecast_horizon,
            'history_window_days': self.history_window
        }
