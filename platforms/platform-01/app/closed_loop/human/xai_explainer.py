"""
可解释 AI 系统 (XAI)
提供决策解释、特征重要性分析、决策路径可视化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """特征重要性"""
    feature_name: str
    importance_score: float
    direction: str  # positive, negative
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'importance_score': self.importance_score,
            'direction': self.direction,
            'description': self.description
        }


@dataclass
class DecisionExplanation:
    """决策解释"""
    decision_id: str
    decision_type: str
    summary: str
    feature_importance: List[FeatureImportance]
    decision_path: List[Dict[str, Any]]
    confidence: float
    counterfactual: Optional[Dict[str, Any]]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'decision_type': self.decision_type,
            'summary': self.summary,
            'feature_importance': [f.to_dict() for f in self.feature_importance],
            'decision_path': self.decision_path,
            'confidence': self.confidence,
            'counterfactual': self.counterfactual,
            'generated_at': self.generated_at.isoformat()
        }


class SHAPExplainer:
    """SHAP 解释器"""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or [
            'cpu_usage', 'memory_usage', 'error_rate', 'latency_p99', 
            'qps', 'disk_usage', 'load_trend', 'time_of_day'
        ]
        
    def explain(self, features: np.ndarray, 
                prediction: float,
                baseline: np.ndarray = None) -> List[FeatureImportance]:
        """
        使用 SHAP 值解释预测
        
        简化实现 - 实际应使用 shap 库
        """
        if baseline is None:
            baseline = np.zeros_like(features)
        
        # 简化的 SHAP 值计算
        shap_values = self._approximate_shap(features, baseline)
        
        importance_list = []
        for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            importance_list.append(FeatureImportance(
                feature_name=name,
                importance_score=abs(shap_val),
                direction='positive' if shap_val > 0 else 'negative',
                description=self._generate_feature_description(name, features[i], shap_val)
            ))
        
        # 排序
        importance_list.sort(key=lambda x: x.importance_score, reverse=True)
        
        return importance_list
    
    def _approximate_shap(self, features: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """近似计算 SHAP 值"""
        # 简化的线性近似
        diff = features - baseline
        shap_values = diff * np.random.uniform(0.5, 1.5, size=diff.shape)
        return shap_values
    
    def _generate_feature_description(self, name: str, value: float, shap_val: float) -> str:
        """生成特征描述"""
        descriptions = {
            'cpu_usage': f"CPU 使用率 {'高' if value > 80 else '正常'} ({value:.1f}%)",
            'memory_usage': f"内存使用率 {'高' if value > 80 else '正常'} ({value:.1f}%)",
            'error_rate': f"错误率 {'高' if value > 0.1 else '正常'} ({value:.3f})",
            'latency_p99': f"P99 延迟 {'高' if value > 1000 else '正常'} ({value:.0f}ms)",
            'qps': f"QPS {'高' if value > 5000 else '正常'} ({value:.0f})",
            'disk_usage': f"磁盘使用率 {'高' if value > 80 else '正常'} ({value:.1f}%)",
        }
        
        return descriptions.get(name, f"{name}: {value:.2f}")


class LIMEExplainer:
    """LIME 解释器"""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or [
            'cpu_usage', 'memory_usage', 'error_rate', 'latency_p99', 'qps'
        ]
        
    def explain(self, instance: np.ndarray,
                prediction_fn: callable,
                num_samples: int = 100) -> List[FeatureImportance]:
        """
        使用 LIME 解释单个预测
        
        简化实现 - 实际应使用 lime 库
        """
        # 生成扰动样本
        perturbations = self._generate_perturbations(instance, num_samples)
        
        # 获取预测
        predictions = [prediction_fn(p) for p in perturbations]
        
        # 拟合局部线性模型
        weights = self._fit_local_model(perturbations, predictions, instance)
        
        importance_list = []
        for i, (name, weight) in enumerate(zip(self.feature_names, weights)):
            importance_list.append(FeatureImportance(
                feature_name=name,
                importance_score=abs(weight),
                direction='positive' if weight > 0 else 'negative',
                description=f"局部权重: {weight:.3f}"
            ))
        
        importance_list.sort(key=lambda x: x.importance_score, reverse=True)
        
        return importance_list
    
    def _generate_perturbations(self, instance: np.ndarray, 
                                num_samples: int) -> List[np.ndarray]:
        """生成扰动样本"""
        perturbations = [instance]
        
        for _ in range(num_samples - 1):
            noise = np.random.normal(0, 0.1, size=instance.shape)
            perturbed = instance + noise
            perturbations.append(perturbed)
        
        return perturbations
    
    def _fit_local_model(self, X: List[np.ndarray], y: List[float], 
                        instance: np.ndarray) -> np.ndarray:
        """拟合局部线性模型"""
        # 简化的线性回归
        X_matrix = np.array(X)
        y_vector = np.array(y)
        
        # 添加截距
        X_with_intercept = np.column_stack([np.ones(len(X)), X_matrix])
        
        # 最小二乘
        try:
            coeffs = np.linalg.lstsq(X_with_intercept, y_vector, rcond=None)[0]
            return coeffs[1:]  # 返回特征系数 (不包括截距)
        except:
            return np.zeros(len(instance))


class DecisionPathExplainer:
    """决策路径解释器"""
    
    def __init__(self):
        self.decision_rules = []
        
    def extract_rules(self, model: Any, 
                     feature_names: List[str]) -> List[Dict[str, Any]]:
        """从模型提取决策规则"""
        # 简化实现 - 实际应根据模型类型提取
        rules = []
        
        # 模拟规则
        rules.append({
            'condition': 'cpu_usage > 80%',
            'action': 'scale_up',
            'confidence': 0.85
        })
        
        rules.append({
            'condition': 'error_rate > 0.1 AND latency_p99 > 1000ms',
            'action': 'restart',
            'confidence': 0.75
        })
        
        return rules
    
    def trace_decision_path(self, features: Dict[str, float],
                           rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """追踪决策路径"""
        path = []
        
        for rule in rules:
            # 检查条件是否满足
            condition_met = self._evaluate_condition(rule['condition'], features)
            
            path.append({
                'step': len(path) + 1,
                'condition': rule['condition'],
                'condition_met': condition_met,
                'action': rule['action'],
                'confidence': rule['confidence']
            })
            
            if condition_met:
                break
        
        return path
    
    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """评估条件"""
        # 简化的条件评估
        if 'cpu_usage > 80' in condition:
            return features.get('cpu_usage', 0) > 80
        elif 'error_rate > 0.1' in condition:
            return features.get('error_rate', 0) > 0.1
        
        return False


class CounterfactualExplainer:
    """反事实解释器"""
    
    def __init__(self):
        pass
        
    def generate_counterfactual(self, instance: Dict[str, float],
                               desired_outcome: str,
                               feature_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """生成反事实解释"""
        # 简化的反事实生成
        counterfactual = instance.copy()
        changes = []
        
        if desired_outcome == 'no_action':
            # 要达到不采取行动的结果
            if instance.get('cpu_usage', 0) > 80:
                counterfactual['cpu_usage'] = 70
                changes.append({
                    'feature': 'cpu_usage',
                    'original': instance['cpu_usage'],
                    'changed_to': 70,
                    'change': 'reduce CPU usage'
                })
            
            if instance.get('error_rate', 0) > 0.1:
                counterfactual['error_rate'] = 0.05
                changes.append({
                    'feature': 'error_rate',
                    'original': instance['error_rate'],
                    'changed_to': 0.05,
                    'change': 'reduce error rate'
                })
        
        return {
            'original_instance': instance,
            'counterfactual': counterfactual,
            'changes_required': changes,
            'desired_outcome': desired_outcome
        }


class XAIExplainer:
    """
    可解释 AI 主类
    整合多种解释方法
    """
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or [
            'cpu_usage', 'memory_usage', 'error_rate', 'latency_p99', 
            'qps', 'disk_usage', 'load_trend'
        ]
        
        self.shap_explainer = SHAPExplainer(self.feature_names)
        self.lime_explainer = LIMEExplainer(self.feature_names)
        self.path_explainer = DecisionPathExplainer()
        self.counterfactual_explainer = CounterfactualExplainer()
        
    def explain_decision(self, decision_id: str,
                        decision_type: str,
                        features: Dict[str, float],
                        prediction: float,
                        model: Any = None) -> DecisionExplanation:
        """解释决策"""
        # 转换特征为数组
        feature_array = np.array([features.get(f, 0) for f in self.feature_names])
        
        # SHAP 解释
        shap_importance = self.shap_explainer.explain(feature_array, prediction)
        
        # 决策路径
        rules = self.path_explainer.extract_rules(model, self.feature_names) if model else []
        decision_path = self.path_explainer.trace_decision_path(features, rules)
        
        # 反事实
        counterfactual = self.counterfactual_explainer.generate_counterfactual(
            features, 'no_action', {}
        )
        
        # 生成自然语言摘要
        summary = self._generate_summary(decision_type, shap_importance, decision_path)
        
        return DecisionExplanation(
            decision_id=decision_id,
            decision_type=decision_type,
            summary=summary,
            feature_importance=shap_importance,
            decision_path=decision_path,
            confidence=prediction,
            counterfactual=counterfactual,
            generated_at=datetime.now()
        )
    
    def _generate_summary(self, decision_type: str, 
                         importance: List[FeatureImportance],
                         path: List[Dict[str, Any]]) -> str:
        """生成自然语言摘要"""
        top_features = importance[:3]
        
        summary = f"决策类型: {decision_type}\n"
        summary += "主要影响因素:\n"
        
        for feat in top_features:
            summary += f"  - {feat.feature_name}: {feat.description} (重要性: {feat.importance_score:.2f})\n"
        
        if path:
            summary += f"\n决策路径: 经过 {len(path)} 个规则检查"
        
        return summary
    
    def explain_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """解释异常"""
        features = anomaly_data.get('features', {})
        anomaly_score = anomaly_data.get('score', 0)
        
        # 找出最异常的特征
        feature_scores = []
        for name, value in features.items():
            # 简化的异常分数
            score = abs(value - 0.5) * 2  # 假设 0.5 是正常值
            feature_scores.append({
                'feature': name,
                'value': value,
                'anomaly_contribution': score
            })
        
        feature_scores.sort(key=lambda x: x['anomaly_contribution'], reverse=True)
        
        return {
            'anomaly_score': anomaly_score,
            'top_contributing_features': feature_scores[:5],
            'explanation': f"异常主要由以下因素导致: {', '.join([f['feature'] for f in feature_scores[:3]])}"
        }
    
    def get_feature_importance_summary(self, 
                                       explanations: List[DecisionExplanation]) -> Dict[str, Any]:
        """获取特征重要性汇总"""
        aggregated = {}
        
        for exp in explanations:
            for feat in exp.feature_importance:
                if feat.feature_name not in aggregated:
                    aggregated[feat.feature_name] = {
                        'total_importance': 0,
                        'count': 0,
                        'positive_count': 0,
                        'negative_count': 0
                    }
                
                aggregated[feat.feature_name]['total_importance'] += feat.importance_score
                aggregated[feat.feature_name]['count'] += 1
                
                if feat.direction == 'positive':
                    aggregated[feat.feature_name]['positive_count'] += 1
                else:
                    aggregated[feat.feature_name]['negative_count'] += 1
        
        # 计算平均
        summary = {}
        for name, data in aggregated.items():
            summary[name] = {
                'average_importance': data['total_importance'] / data['count'],
                'occurrence_count': data['count'],
                'positive_ratio': data['positive_count'] / data['count']
            }
        
        # 排序
        sorted_summary = dict(sorted(summary.items(), 
                                    key=lambda x: x[1]['average_importance'], 
                                    reverse=True))
        
        return sorted_summary
