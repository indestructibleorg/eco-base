"""
故障预测引擎
基于 LSTM/Transformer 进行时序预测，预测未来故障
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailurePrediction:
    """故障预测结果"""
    service_id: str
    horizon_hours: int
    failure_probability: float
    failure_type: str
    confidence: float
    features_importance: Dict[str, float]
    predicted_at: datetime
    valid_until: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_id': self.service_id,
            'horizon_hours': self.horizon_hours,
            'failure_probability': self.failure_probability,
            'failure_type': self.failure_type,
            'confidence': self.confidence,
            'features_importance': self.features_importance,
            'predicted_at': self.predicted_at.isoformat(),
            'valid_until': self.valid_until.isoformat()
        }


class LSTMPredictor(nn.Module):
    """LSTM 故障预测模型"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2,
                 output_horizons: List[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_horizons = output_horizons or [1, 6, 24, 168]  # 1h, 6h, 24h, 7d
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 为每个时间窗口创建预测头
        self.predictors = nn.ModuleDict({
            f'h{h}': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 2)  # 二分类: 故障/正常
            )
            for h in self.output_horizons
        })
        
        # 故障类型分类
        self.failure_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 5)  # 5种故障类型
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, input_size]
        
        Returns:
            预测结果字典
        """
        # LSTM 编码
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个隐藏状态
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # 多时间尺度预测
        predictions = {}
        for horizon in self.output_horizons:
            logits = self.predictors[f'h{horizon}'](last_hidden)
            probs = torch.softmax(logits, dim=-1)
            predictions[f'{horizon}h'] = probs
        
        # 故障类型预测
        type_logits = self.failure_type_classifier(last_hidden)
        type_probs = torch.softmax(type_logits, dim=-1)
        
        return {
            'failure_probabilities': predictions,
            'failure_type_probs': type_probs,
            'hidden_state': last_hidden
        }


class TransformerPredictor(nn.Module):
    """Transformer 故障预测模型"""
    
    def __init__(self, input_size: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4, 
                 dropout: float = 0.1,
                 output_horizons: List[int] = None):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_horizons = output_horizons or [1, 6, 24, 168]
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 预测头
        self.predictors = nn.ModuleDict({
            f'h{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 2)
            )
            for h in self.output_horizons
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer 编码
        encoded = self.transformer(x)
        
        # 使用最后一个时间步
        last_step = encoded[:, -1, :]
        
        # 多时间尺度预测
        predictions = {}
        for horizon in self.output_horizons:
            logits = self.predictors[f'h{horizon}'](last_step)
            probs = torch.softmax(logits, dim=-1)
            predictions[f'{horizon}h'] = probs
        
        return {'failure_probabilities': predictions}


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        
    def extract_features(self, metrics: Dict[str, List[float]]) -> np.ndarray:
        """从指标序列中提取特征"""
        features = []
        
        for metric_name, values in metrics.items():
            if len(values) < self.window_size:
                # 填充
                values = [0] * (self.window_size - len(values)) + values
            
            values = values[-self.window_size:]
            
            # 统计特征
            features.append(np.mean(values))
            features.append(np.std(values))
            features.append(np.max(values))
            features.append(np.min(values))
            features.append(np.percentile(values, 95))
            features.append(np.percentile(values, 99))
            
            # 趋势特征
            if len(values) >= 2:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                features.append(slope)
            else:
                features.append(0)
            
            # 变化率
            if len(values) >= 2:
                change_rate = (values[-1] - values[0]) / (values[0] + 1e-8)
                features.append(change_rate)
            else:
                features.append(0)
        
        return np.array(features)
    
    def create_sequence(self, metrics_history: List[Dict[str, float]], 
                       seq_length: int = 24) -> np.ndarray:
        """创建序列数据"""
        sequences = []
        
        for i in range(len(metrics_history) - seq_length + 1):
            seq = metrics_history[i:i+seq_length]
            
            # 将每个时间步的指标转换为向量
            seq_vectors = []
            for metrics in seq:
                vector = [
                    metrics.get('cpu_usage', 0) / 100,
                    metrics.get('memory_usage', 0) / 100,
                    metrics.get('error_rate', 0),
                    min(metrics.get('latency_p99', 0) / 5000, 1),
                    metrics.get('qps', 0) / 10000,
                    metrics.get('disk_usage', 0) / 100,
                ]
                seq_vectors.append(vector)
            
            sequences.append(seq_vectors)
        
        return np.array(sequences)


class FailurePredictor:
    """
    故障预测引擎主类
    整合 LSTM/Transformer 模型和预测逻辑
    """
    
    FAILURE_TYPES = ['cpu_exhaustion', 'memory_leak', 'disk_full', 
                     'network_partition', 'dependency_failure']
    
    def __init__(self, model_type: str = 'lstm', config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_type = model_type
        
        # 特征维度
        self.input_size = 6  # cpu, memory, error_rate, latency, qps, disk
        self.seq_length = self.config.get('seq_length', 24)
        
        # 初始化模型
        if model_type == 'lstm':
            self.model = LSTMPredictor(
                input_size=self.input_size,
                hidden_size=self.config.get('hidden_size', 128),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )
        elif model_type == 'transformer':
            self.model = TransformerPredictor(
                input_size=self.input_size,
                d_model=self.config.get('d_model', 128),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 4)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 预测历史
        self.prediction_history: Dict[str, List[FailurePrediction]] = {}
        
    def predict(self, service_id: str, metrics_history: List[Dict[str, float]],
               horizon_hours: List[int] = None) -> List[FailurePrediction]:
        """
        预测故障
        
        Args:
            service_id: 服务 ID
            metrics_history: 历史指标数据
            horizon_hours: 预测时间窗口
        
        Returns:
            预测结果列表
        """
        if len(metrics_history) < self.seq_length:
            logger.warning(f"Insufficient data for prediction: {len(metrics_history)} < {self.seq_length}")
            return []
        
        # 准备输入
        sequence = self.feature_extractor.create_sequence(
            metrics_history, self.seq_length
        )
        
        if len(sequence) == 0:
            return []
        
        # 使用最后一个序列进行预测
        input_tensor = torch.FloatTensor(sequence[-1:])  # [1, seq_len, input_size]
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            result = self.model(input_tensor)
        
        # 解析结果
        predictions = []
        predicted_at = datetime.now()
        
        failure_probs = result['failure_probabilities']
        
        # 故障类型预测
        if 'failure_type_probs' in result:
            type_probs = result['failure_type_probs'][0]
            predicted_type_idx = torch.argmax(type_probs).item()
            predicted_type = self.FAILURE_TYPES[predicted_type_idx]
            type_confidence = type_probs[predicted_type_idx].item()
        else:
            predicted_type = 'unknown'
            type_confidence = 0.5
        
        # 为每个时间窗口生成预测
        horizons = horizon_hours or [1, 6, 24, 168]
        
        for horizon in horizons:
            key = f'{horizon}h'
            if key in failure_probs:
                probs = failure_probs[key][0]
                failure_prob = probs[1].item()  # 故障概率
                
                # 置信度计算
                confidence = self._calculate_confidence(failure_prob, len(metrics_history))
                
                # 特征重要性 (简化版)
                feature_importance = self._calculate_feature_importance(metrics_history)
                
                prediction = FailurePrediction(
                    service_id=service_id,
                    horizon_hours=horizon,
                    failure_probability=failure_prob,
                    failure_type=predicted_type,
                    confidence=confidence,
                    features_importance=feature_importance,
                    predicted_at=predicted_at,
                    valid_until=predicted_at + timedelta(hours=horizon)
                )
                
                predictions.append(prediction)
                
                # 存储历史
                if service_id not in self.prediction_history:
                    self.prediction_history[service_id] = []
                self.prediction_history[service_id].append(prediction)
        
        return predictions
    
    def _calculate_confidence(self, probability: float, data_points: int) -> float:
        """计算预测置信度"""
        # 基于数据量和概率分布计算置信度
        data_confidence = min(data_points / 100, 1.0)  # 至少100个数据点
        
        # 概率接近 0.5 时置信度较低
        prob_confidence = 1 - abs(probability - 0.5) * 2
        
        return (data_confidence + prob_confidence) / 2
    
    def _calculate_feature_importance(self, metrics_history: List[Dict[str, float]]) -> Dict[str, float]:
        """计算特征重要性"""
        # 简化的特征重要性计算
        if not metrics_history:
            return {}
        
        latest = metrics_history[-1]
        
        importance = {}
        
        # CPU 重要性
        cpu = latest.get('cpu_usage', 0)
        importance['cpu_usage'] = min(cpu / 100, 1.0)
        
        # 内存重要性
        memory = latest.get('memory_usage', 0)
        importance['memory_usage'] = min(memory / 100, 1.0)
        
        # 错误率重要性
        error_rate = latest.get('error_rate', 0)
        importance['error_rate'] = min(error_rate * 10, 1.0)
        
        # 延迟重要性
        latency = latest.get('latency_p99', 0)
        importance['latency'] = min(latency / 5000, 1.0)
        
        # QPS 重要性
        qps = latest.get('qps', 0)
        importance['qps'] = min(qps / 10000, 1.0)
        
        return importance
    
    def get_prediction_history(self, service_id: str, 
                               since: datetime = None) -> List[FailurePrediction]:
        """获取预测历史"""
        history = self.prediction_history.get(service_id, [])
        
        if since:
            history = [h for h in history if h.predicted_at >= since]
        
        return history
    
    def evaluate_prediction_accuracy(self, service_id: str, 
                                     actual_failures: List[datetime]) -> Dict[str, float]:
        """评估预测准确性"""
        history = self.prediction_history.get(service_id, [])
        
        if not history or not actual_failures:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        # 计算精确率和召回率
        true_positives = 0
        false_positives = 0
        false_negatives = len(actual_failures)
        
        for prediction in history:
            if prediction.failure_probability > 0.5:  # 预测有故障
                # 检查是否在实际故障附近
                predicted_time = prediction.predicted_at + timedelta(hours=prediction.horizon_hours)
                
                matched = False
                for failure_time in actual_failures:
                    time_diff = abs((predicted_time - failure_time).total_seconds())
                    if time_diff < 3600:  # 1小时内
                        matched = True
                        false_negatives -= 1
                        break
                
                if matched:
                    true_positives += 1
                else:
                    false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
