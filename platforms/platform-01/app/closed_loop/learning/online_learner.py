"""
在线学习系统
支持增量学习、概念漂移检测、模型热更新
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import json
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """概念漂移检测器"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_detected = False
        self.drift_timestamp = None
        
    def add_sample(self, prediction: float, actual: float):
        """添加样本"""
        error = abs(prediction - actual)
        
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(error)
            return {'drift_detected': False, 'status': 'collecting_reference'}
        
        self.current_window.append(error)
        
        if len(self.current_window) >= self.window_size:
            return self._check_drift()
        
        return {'drift_detected': False, 'status': 'collecting_current'}
    
    def _check_drift(self) -> Dict[str, Any]:
        """检查漂移"""
        from scipy import stats
        
        reference = list(self.reference_window)
        current = list(self.current_window)
        
        # KS 检验
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        
        # 均值变化
        mean_change = abs(np.mean(current) - np.mean(reference)) / (np.mean(reference) + 1e-8)
        
        # 方差变化
        var_change = abs(np.var(current) - np.var(reference)) / (np.var(reference) + 1e-8)
        
        drift_detected = (p_value < self.threshold or 
                         mean_change > 0.2 or 
                         var_change > 0.3)
        
        if drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.drift_timestamp = datetime.now()
            logger.warning(f"Concept drift detected! p_value={p_value:.4f}, "
                          f"mean_change={mean_change:.4f}")
        
        return {
            'drift_detected': drift_detected,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'mean_change': mean_change,
            'var_change': var_change,
            'reference_mean': np.mean(reference),
            'current_mean': np.mean(current)
        }
    
    def reset(self):
        """重置检测器"""
        self.reference_window.clear()
        self.current_window.clear()
        self.drift_detected = False
        self.drift_timestamp = None


class IncrementalLearner:
    """增量学习器"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.sample_count = 0
        self.update_count = 0
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                    batch_size: int = 32) -> Dict[str, float]:
        """增量训练"""
        self.model.train()
        
        # 转换为 tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # 小批量训练
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(batch_X)
            loss = nn.MSELoss()(predictions.squeeze(), batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        self.sample_count += len(X)
        self.update_count += 1
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0,
            'samples': self.sample_count,
            'updates': self.update_count
        }


@dataclass
class ModelVersion:
    """模型版本信息"""
    version_id: str
    created_at: datetime
    description: str
    performance_metrics: Dict[str, float]
    training_samples: int
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version_id': self.version_id,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'performance_metrics': self.performance_metrics,
            'training_samples': self.training_samples,
            'parent_version': self.parent_version
        }


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version: Optional[str] = None
        self.version_history: List[str] = []
        
        os.makedirs(model_dir, exist_ok=True)
        self._load_version_history()
        
    def _load_version_history(self):
        """加载版本历史"""
        history_file = os.path.join(self.model_dir, "version_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                data = json.load(f)
                for v_data in data.get('versions', []):
                    version = ModelVersion(
                        version_id=v_data['version_id'],
                        created_at=datetime.fromisoformat(v_data['created_at']),
                        description=v_data['description'],
                        performance_metrics=v_data['performance_metrics'],
                        training_samples=v_data['training_samples'],
                        parent_version=v_data.get('parent_version')
                    )
                    self.versions[version.version_id] = version
                self.version_history = data.get('history', [])
                self.current_version = data.get('current_version')
    
    def _save_version_history(self):
        """保存版本历史"""
        history_file = os.path.join(self.model_dir, "version_history.json")
        data = {
            'versions': [v.to_dict() for v in self.versions.values()],
            'history': self.version_history,
            'current_version': self.current_version
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(self, model: nn.Module, description: str,
                      performance_metrics: Dict[str, float],
                      training_samples: int) -> str:
        """创建新版本"""
        # 生成版本 ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        version_id = f"v{timestamp}_{model_hash}"
        
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now(),
            description=description,
            performance_metrics=performance_metrics,
            training_samples=training_samples,
            parent_version=self.current_version
        )
        
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{version_id}.pt")
        torch.save(model.state_dict(), model_path)
        
        # 更新版本信息
        self.versions[version_id] = version
        self.version_history.append(version_id)
        self.current_version = version_id
        
        self._save_version_history()
        
        logger.info(f"Created model version: {version_id}")
        return version_id
    
    def load_version(self, version_id: str, model: nn.Module) -> bool:
        """加载指定版本"""
        model_path = os.path.join(self.model_dir, f"{version_id}.pt")
        
        if not os.path.exists(model_path):
            logger.error(f"Model version not found: {version_id}")
            return False
        
        model.load_state_dict(torch.load(model_path))
        self.current_version = version_id
        
        logger.info(f"Loaded model version: {version_id}")
        return True
    
    def rollback(self, model: nn.Module, steps: int = 1) -> Optional[str]:
        """回滚到之前的版本"""
        if len(self.version_history) <= steps:
            logger.error("Cannot rollback: insufficient version history")
            return None
        
        target_version = self.version_history[-(steps + 1)]
        
        if self.load_version(target_version, model):
            logger.info(f"Rolled back to version: {target_version}")
            return target_version
        
        return None
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return [v.to_dict() for v in self.versions.values()]
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本"""
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        comparison = {
            'version1': v1.to_dict(),
            'version2': v2.to_dict(),
            'performance_diff': {
                k: v2.performance_metrics.get(k, 0) - v1.performance_metrics.get(k, 0)
                for k in set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys())
            },
            'time_diff': (v2.created_at - v1.created_at).total_seconds()
        }
        
        return comparison


class OnlineLearner:
    """
    在线学习主类
    整合增量学习、概念漂移检测、版本管理
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = model
        
        # 增量学习器
        self.incremental_learner = IncrementalLearner(
            model, 
            learning_rate=self.config.get('learning_rate', 1e-4)
        )
        
        # 漂移检测器
        self.drift_detector = ConceptDriftDetector(
            window_size=self.config.get('drift_window_size', 100),
            threshold=self.config.get('drift_threshold', 0.05)
        )
        
        # 版本管理器
        self.version_manager = ModelVersionManager(
            model_dir=self.config.get('model_dir', './models')
        )
        
        # 学习状态
        self.learning_enabled = True
        self.min_samples_before_update = self.config.get('min_samples', 10)
        self.update_interval = self.config.get('update_interval', 100)
        
        # 性能监控
        self.performance_history = deque(maxlen=1000)
        
    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> Dict[str, Any]:
        """在线学习"""
        if not self.learning_enabled:
            return {'status': 'learning_disabled'}
        
        # 准备训练数据
        X = state.reshape(1, -1)
        
        # 计算目标值 (简化版 Q-learning)
        with torch.no_grad():
            next_q_values = self.model(torch.FloatTensor(next_state).unsqueeze(0))
            target = reward + (0.99 * next_q_values.max().item() if not done else 0)
        
        y = np.array([target])
        
        # 增量训练
        result = self.incremental_learner.partial_fit(X, y)
        
        # 检测漂移
        with torch.no_grad():
            prediction = self.model(torch.FloatTensor(state).unsqueeze(0)).max().item()
        drift_result = self.drift_detector.add_sample(prediction, target)
        
        # 如果检测到漂移，触发模型更新
        if drift_result.get('drift_detected'):
            self._handle_drift()
        
        return {
            'loss': result['loss'],
            'samples': result['samples'],
            'drift_detected': drift_result.get('drift_detected', False)
        }
    
    def _handle_drift(self):
        """处理概念漂移"""
        logger.warning("Handling concept drift...")
        
        # 保存当前版本
        self.version_manager.create_version(
            model=self.model,
            description="Auto-saved before drift handling",
            performance_metrics=self._get_current_performance(),
            training_samples=self.incremental_learner.sample_count
        )
        
        # 重置漂移检测器
        self.drift_detector.reset()
        
        # 可以增加学习率等策略
        for param_group in self.incremental_learner.optimizer.param_groups:
            param_group['lr'] *= 1.5  # 临时增加学习率
    
    def _get_current_performance(self) -> Dict[str, float]:
        """获取当前性能指标"""
        if not self.performance_history:
            return {}
        
        recent = list(self.performance_history)[-100:]
        return {
            'mean_reward': np.mean([p['reward'] for p in recent]),
            'success_rate': np.mean([p['success'] for p in recent]),
            'avg_loss': np.mean([p.get('loss', 0) for p in recent])
        }
    
    def record_performance(self, reward: float, success: bool, loss: float = 0):
        """记录性能"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'reward': reward,
            'success': success,
            'loss': loss
        })
    
    def save_checkpoint(self, description: str = "") -> str:
        """保存检查点"""
        return self.version_manager.create_version(
            model=self.model,
            description=description or f"Checkpoint at {datetime.now()}",
            performance_metrics=self._get_current_performance(),
            training_samples=self.incremental_learner.sample_count
        )
    
    def rollback(self, steps: int = 1) -> Optional[str]:
        """回滚模型"""
        return self.version_manager.rollback(self.model, steps)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            'learning_enabled': self.learning_enabled,
            'samples_trained': self.incremental_learner.sample_count,
            'updates': self.incremental_learner.update_count,
            'current_version': self.version_manager.current_version,
            'drift_detected': self.drift_detector.drift_detected,
            'performance': self._get_current_performance()
        }
