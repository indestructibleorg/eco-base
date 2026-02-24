"""
决策效果评估系统
用于评估修复决策的效果，支持 A/B 测试、因果推断分析
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentType(Enum):
    """实验类型"""
    AB_TEST = "ab_test"
    CANARY = "canary"
    SHADOW = "shadow"
    ROLLOUT = "rollout"


class EvaluationStatus(Enum):
    """评估状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DecisionRecord:
    """决策记录"""
    decision_id: str
    timestamp: datetime
    action: str
    action_params: Dict[str, Any]
    context: Dict[str, Any]  # 决策时的上下文
    target_service: str
    experiment_group: Optional[str] = None  # A/B 测试分组
    
    
@dataclass
class OutcomeMetrics:
    """结果指标"""
    mttr: float  # 平均修复时间 (秒)
    cost: float  # 成本
    error_rate_before: float
    error_rate_after: float
    latency_p99_before: float
    latency_p99_after: float
    availability_before: float
    availability_after: float
    user_impact_score: float  # 0-100
    side_effects: List[str] = field(default_factory=list)
    
    def calculate_improvement(self) -> Dict[str, float]:
        """计算改善指标"""
        return {
            'error_rate_reduction': self.error_rate_before - self.error_rate_after,
            'latency_reduction': self.latency_p99_before - self.latency_p99_after,
            'availability_improvement': self.availability_after - self.availability_before,
            'mttr': self.mttr,
            'cost': self.cost
        }


@dataclass
class Experiment:
    """实验定义"""
    experiment_id: str
    name: str
    type: ExperimentType
    decision_type: str  # 决策类型
    start_time: datetime
    end_time: Optional[datetime] = None
    control_group: Dict[str, Any] = field(default_factory=dict)
    treatment_group: Dict[str, Any] = field(default_factory=dict)
    traffic_split: float = 0.5  # 流量分配比例
    min_sample_size: int = 100
    status: EvaluationStatus = EvaluationStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)


class ABTestFramework:
    """A/B 测试框架"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.decisions: Dict[str, List[DecisionRecord]] = defaultdict(list)
        self.outcomes: Dict[str, List[OutcomeMetrics]] = defaultdict(list)
        
    def create_experiment(self, name: str, decision_type: str,
                         control_config: Dict[str, Any],
                         treatment_config: Dict[str, Any],
                         traffic_split: float = 0.5,
                         min_sample_size: int = 100) -> str:
        """创建 A/B 测试实验"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{name}"
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            type=ExperimentType.AB_TEST,
            decision_type=decision_type,
            start_time=datetime.now(),
            control_group=control_config,
            treatment_group=treatment_config,
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
            status=EvaluationStatus.RUNNING
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created A/B test experiment: {experiment_id}")
        return experiment_id
    
    def assign_group(self, experiment_id: str, decision_id: str) -> str:
        """为决策分配实验组"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return "control"
        
        # 基于决策 ID 的哈希进行分组，确保一致性
        hash_val = hash(decision_id) % 1000
        if hash_val < experiment.traffic_split * 1000:
            return "treatment"
        return "control"
    
    def record_decision(self, experiment_id: str, decision: DecisionRecord):
        """记录决策"""
        self.decisions[experiment_id].append(decision)
        
    def record_outcome(self, experiment_id: str, outcome: OutcomeMetrics):
        """记录结果"""
        self.outcomes[experiment_id].append(outcome)
        
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """分析实验结果"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        control_outcomes = []
        treatment_outcomes = []
        
        for decision, outcome in zip(self.decisions[experiment_id], self.outcomes[experiment_id]):
            if decision.experiment_group == "control":
                control_outcomes.append(outcome)
            else:
                treatment_outcomes.append(outcome)
        
        if len(control_outcomes) < experiment.min_sample_size or \
           len(treatment_outcomes) < experiment.min_sample_size:
            return {
                "status": "insufficient_data",
                "control_samples": len(control_outcomes),
                "treatment_samples": len(treatment_outcomes),
                "min_required": experiment.min_sample_size
            }
        
        # 计算指标
        results = {}
        
        # MTTR 比较
        control_mttr = [o.mttr for o in control_outcomes]
        treatment_mttr = [o.mttr for o in treatment_outcomes]
        results['mttr'] = self._compare_metrics(control_mttr, treatment_mttr, "lower_is_better")
        
        # 错误率改善比较
        control_error_improvement = [o.error_rate_before - o.error_rate_after for o in control_outcomes]
        treatment_error_improvement = [o.error_rate_before - o.error_rate_after for o in treatment_outcomes]
        results['error_rate_improvement'] = self._compare_metrics(
            control_error_improvement, treatment_error_improvement, "higher_is_better"
        )
        
        # 成本比较
        control_cost = [o.cost for o in control_outcomes]
        treatment_cost = [o.cost for o in treatment_outcomes]
        results['cost'] = self._compare_metrics(control_cost, treatment_cost, "lower_is_better")
        
        # 综合评估
        results['winner'] = self._determine_winner(results)
        results['control_samples'] = len(control_outcomes)
        results['treatment_samples'] = len(treatment_outcomes)
        
        experiment.results = results
        experiment.status = EvaluationStatus.COMPLETED
        experiment.end_time = datetime.now()
        
        return results
    
    def _compare_metrics(self, control: List[float], treatment: List[float],
                        direction: str) -> Dict[str, Any]:
        """比较两组指标"""
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        
        # t-test
        t_stat, p_value = stats.ttest_ind(control, treatment)
        
        # 效应量 (Cohen's d)
        pooled_std = np.sqrt((np.std(control)**2 + np.std(treatment)**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # 相对改善
        relative_improvement = ((treatment_mean - control_mean) / control_mean * 100) \
                               if control_mean != 0 else 0
        
        # 判断显著性
        is_significant = p_value < 0.05
        is_better = (treatment_mean < control_mean) if direction == "lower_is_better" else \
                    (treatment_mean > control_mean)
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'relative_improvement': relative_improvement,
            'p_value': p_value,
            'is_significant': is_significant,
            'is_better': is_better,
            'cohens_d': cohens_d,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    def _determine_winner(self, results: Dict[str, Any]) -> str:
        """确定获胜组"""
        significant_improvements = 0
        significant_regressions = 0
        
        for metric, result in results.items():
            if isinstance(result, dict) and 'is_significant' in result:
                if result['is_significant'] and result['is_better']:
                    significant_improvements += 1
                elif result['is_significant'] and not result['is_better']:
                    significant_regressions += 1
        
        if significant_improvements > significant_regressions:
            return "treatment"
        elif significant_regressions > significant_improvements:
            return "control"
        return "tie"


class CausalInferenceAnalyzer:
    """因果推断分析器"""
    
    def __init__(self):
        self.observations: List[Dict[str, Any]] = []
        
    def add_observation(self, timestamp: datetime, treatment: bool, 
                       outcome: float, covariates: Dict[str, float]):
        """添加观测数据"""
        self.observations.append({
            'timestamp': timestamp,
            'treatment': treatment,
            'outcome': outcome,
            'covariates': covariates
        })
        
    def propensity_score_matching(self) -> Dict[str, Any]:
        """倾向得分匹配"""
        # 简化的倾向得分估计
        treatment_group = [o for o in self.observations if o['treatment']]
        control_group = [o for o in self.observations if not o['treatment']]
        
        if not treatment_group or not control_group:
            return {"error": "Insufficient data for matching"}
        
        # 计算平均处理效应 (ATE)
        treatment_outcomes = [o['outcome'] for o in treatment_group]
        control_outcomes = [o['outcome'] for o in control_group]
        
        ate = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        
        # 置信区间
        se = np.sqrt(np.var(treatment_outcomes) / len(treatment_outcomes) + 
                     np.var(control_outcomes) / len(control_outcomes))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'ate': ate,
            'confidence_interval': (ci_lower, ci_upper),
            'treatment_samples': len(treatment_group),
            'control_samples': len(control_group)
        }
    
    def difference_in_differences(self, pre_period: Tuple[datetime, datetime],
                                  post_period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """双重差分分析"""
        # 筛选数据
        pre_treatment = [o for o in self.observations 
                        if o['treatment'] and pre_period[0] <= o['timestamp'] <= pre_period[1]]
        pre_control = [o for o in self.observations 
                      if not o['treatment'] and pre_period[0] <= o['timestamp'] <= pre_period[1]]
        post_treatment = [o for o in self.observations 
                         if o['treatment'] and post_period[0] <= o['timestamp'] <= post_period[1]]
        post_control = [o for o in self.observations 
                       if not o['treatment'] and post_period[0] <= o['timestamp'] <= post_period[1]]
        
        if not all([pre_treatment, pre_control, post_treatment, post_control]):
            return {"error": "Insufficient data for DiD analysis"}
        
        # 计算均值
        pre_treatment_mean = np.mean([o['outcome'] for o in pre_treatment])
        pre_control_mean = np.mean([o['outcome'] for o in pre_control])
        post_treatment_mean = np.mean([o['outcome'] for o in post_treatment])
        post_control_mean = np.mean([o['outcome'] for o in post_control])
        
        # DiD 估计
        did_estimate = (post_treatment_mean - pre_treatment_mean) - \
                       (post_control_mean - pre_control_mean)
        
        return {
            'did_estimate': did_estimate,
            'pre_treatment_mean': pre_treatment_mean,
            'post_treatment_mean': post_treatment_mean,
            'pre_control_mean': pre_control_mean,
            'post_control_mean': post_control_mean
        }


class LongTermEffectTracker:
    """长期效果追踪器"""
    
    def __init__(self, tracking_window_days: int = 30):
        self.tracking_window = timedelta(days=tracking_window_days)
        self.effect_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def record_effect(self, decision_id: str, timestamp: datetime, 
                     metrics: Dict[str, float]):
        """记录效果"""
        self.effect_history[decision_id].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
    def get_effect_trend(self, decision_id: str) -> Dict[str, Any]:
        """获取效果趋势"""
        history = self.effect_history.get(decision_id, [])
        if not history:
            return {"error": "No history found"}
        
        # 按时间排序
        history.sort(key=lambda x: x['timestamp'])
        
        # 计算趋势
        trends = {}
        all_metrics = set()
        for record in history:
            all_metrics.update(record['metrics'].keys())
        
        for metric in all_metrics:
            values = [r['metrics'].get(metric, 0) for r in history]
            if len(values) >= 2:
                # 线性回归计算趋势
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                trends[metric] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend': 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
                }
        
        return {
            'decision_id': decision_id,
            'data_points': len(history),
            'time_span_days': (history[-1]['timestamp'] - history[0]['timestamp']).days if len(history) > 1 else 0,
            'trends': trends
        }
    
    def calculate_long_term_roi(self, decision_id: str, 
                                cost_baseline: float = 100) -> Dict[str, Any]:
        """计算长期 ROI"""
        history = self.effect_history.get(decision_id, [])
        if not history:
            return {"error": "No history found"}
        
        # 计算累积效果
        total_improvement = 0
        for record in history:
            # 假设每个记录包含 error_rate 和 latency
            error_improvement = record['metrics'].get('error_rate_improvement', 0)
            latency_improvement = record['metrics'].get('latency_improvement', 0)
            total_improvement += error_improvement + latency_improvement
        
        # 简化 ROI 计算
        roi = (total_improvement / cost_baseline) * 100 if cost_baseline > 0 else 0
        
        return {
            'decision_id': decision_id,
            'total_improvement': total_improvement,
            'cost_baseline': cost_baseline,
            'roi_percentage': roi,
            'data_points': len(history)
        }


class EffectEvaluator:
    """
    效果评估主类
    整合 A/B 测试、因果推断、长期追踪
    """
    
    def __init__(self):
        self.ab_test_framework = ABTestFramework()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.long_term_tracker = LongTermEffectTracker()
        
        self.decision_outcomes: Dict[str, OutcomeMetrics] = {}
        
    def evaluate_decision(self, decision_id: str, action: str,
                         before_metrics: Dict[str, float],
                         after_metrics: Dict[str, float],
                         execution_time: float,
                         cost: float) -> Dict[str, Any]:
        """评估单个决策的效果"""
        outcome = OutcomeMetrics(
            mttr=execution_time,
            cost=cost,
            error_rate_before=before_metrics.get('error_rate', 0),
            error_rate_after=after_metrics.get('error_rate', 0),
            latency_p99_before=before_metrics.get('latency_p99', 0),
            latency_p99_after=after_metrics.get('latency_p99', 0),
            availability_before=before_metrics.get('availability', 1.0),
            availability_after=after_metrics.get('availability', 1.0),
            user_impact_score=after_metrics.get('user_impact', 0),
            side_effects=after_metrics.get('side_effects', [])
        )
        
        self.decision_outcomes[decision_id] = outcome
        
        # 计算改善指标
        improvements = outcome.calculate_improvement()
        
        # 综合评分
        score = self._calculate_score(improvements, outcome)
        
        return {
            'decision_id': decision_id,
            'action': action,
            'improvements': improvements,
            'score': score,
            'has_side_effects': len(outcome.side_effects) > 0,
            'side_effects': outcome.side_effects
        }
    
    def _calculate_score(self, improvements: Dict[str, float], 
                        outcome: OutcomeMetrics) -> float:
        """计算综合评分"""
        weights = {
            'error_rate_reduction': 0.3,
            'latency_reduction': 0.2,
            'availability_improvement': 0.3,
            'mttr': -0.1,  # 负权重，越小越好
            'cost': -0.1   # 负权重，越小越好
        }
        
        score = 0
        for metric, weight in weights.items():
            value = improvements.get(metric, 0)
            # 归一化
            if metric == 'mttr':
                value = -min(value / 300, 1)  # 假设 5 分钟为基准
            elif metric == 'cost':
                value = -min(value / 100, 1)  # 假设 $100 为基准
            score += weight * value
        
        # 惩罚副作用
        if outcome.side_effects:
            score -= len(outcome.side_effects) * 0.1
        
        return max(0, min(1, score + 0.5))  # 归一化到 0-1
    
    def create_ab_test(self, name: str, decision_type: str,
                      control_config: Dict[str, Any],
                      treatment_config: Dict[str, Any]) -> str:
        """创建 A/B 测试"""
        return self.ab_test_framework.create_experiment(
            name, decision_type, control_config, treatment_config
        )
    
    def analyze_ab_test(self, experiment_id: str) -> Dict[str, Any]:
        """分析 A/B 测试结果"""
        return self.ab_test_framework.analyze_experiment(experiment_id)
    
    def get_decision_feedback(self, decision_id: str) -> Dict[str, Any]:
        """获取决策反馈"""
        outcome = self.decision_outcomes.get(decision_id)
        if not outcome:
            return {"error": "Decision not found"}
        
        return {
            'decision_id': decision_id,
            'mttr': outcome.mttr,
            'cost': outcome.cost,
            'improvements': outcome.calculate_improvement(),
            'side_effects': outcome.side_effects,
            'long_term_trend': self.long_term_tracker.get_effect_trend(decision_id)
        }
