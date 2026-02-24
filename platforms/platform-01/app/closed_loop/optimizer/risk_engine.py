"""
风险评估引擎
评估故障概率、影响范围和风险等级
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class RiskCategory(Enum):
    """风险类别"""
    TECHNICAL = "technical"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"


@dataclass
class RiskFactor:
    """风险因子"""
    name: str
    category: RiskCategory
    probability: float  # 0-1
    impact: float  # 0-1
    weight: float = 1.0
    
    @property
    def risk_score(self) -> float:
        return self.probability * self.impact * self.weight


@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_risk: RiskLevel
    risk_score: float
    probability: float
    impact: float
    factors: List[RiskFactor]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_risk': self.overall_risk.value,
            'risk_score': self.risk_score,
            'probability': self.probability,
            'impact': self.impact,
            'factors': [
                {
                    'name': f.name,
                    'category': f.category.value,
                    'probability': f.probability,
                    'impact': f.impact,
                    'risk_score': f.risk_score
                }
                for f in self.factors
            ],
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class FailureProbabilityPredictor:
    """故障概率预测器"""
    
    def __init__(self):
        # 历史故障模式
        self.failure_patterns = defaultdict(lambda: defaultdict(int))
        self.total_observations = 0
        
    def train(self, historical_data: List[Dict[str, Any]]):
        """训练概率模型"""
        for record in historical_data:
            service = record.get('service')
            conditions = record.get('conditions', {})
            failed = record.get('failed', False)
            
            # 记录条件组合
            condition_key = self._encode_conditions(conditions)
            
            if failed:
                self.failure_patterns[service][condition_key] += 1
            self.total_observations += 1
    
    def _encode_conditions(self, conditions: Dict[str, Any]) -> str:
        """编码条件为字符串"""
        # 简化编码
        encoded = []
        if conditions.get('cpu_high'):
            encoded.append('cpu_high')
        if conditions.get('memory_high'):
            encoded.append('mem_high')
        if conditions.get('error_spike'):
            encoded.append('err_spike')
        if conditions.get('latency_high'):
            encoded.append('lat_high')
        return '|'.join(encoded) if encoded else 'normal'
    
    def predict(self, service: str, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """预测故障概率"""
        condition_key = self._encode_conditions(current_conditions)
        
        # 基于历史数据计算概率
        service_failures = sum(self.failure_patterns[service].values())
        condition_failures = self.failure_patterns[service].get(condition_key, 0)
        
        if service_failures == 0:
            base_probability = 0.1  # 默认概率
        else:
            base_probability = condition_failures / service_failures
        
        # 调整因子
        adjustment = 1.0
        
        # 时间因子 (夜间和周末风险略高)
        hour = datetime.now().hour
        if hour < 6 or hour > 22:
            adjustment *= 1.2
        
        # 负载因子
        if current_conditions.get('load_trend') == 'increasing':
            adjustment *= 1.3
        
        # 近期故障因子
        if current_conditions.get('recent_failures', 0) > 0:
            adjustment *= (1 + 0.2 * current_conditions['recent_failures'])
        
        final_probability = min(base_probability * adjustment, 1.0)
        
        return {
            'service': service,
            'probability': final_probability,
            'confidence': 0.7 if service_failures > 10 else 0.5,
            'base_probability': base_probability,
            'adjustment': adjustment,
            'conditions': current_conditions
        }


class ImpactAssessor:
    """影响评估器"""
    
    def __init__(self, service_graph: Dict[str, List[str]] = None):
        self.service_graph = service_graph or {}
        
    def assess_service_impact(self, service: str, 
                              failure_type: str) -> Dict[str, Any]:
        """评估服务故障影响"""
        # 直接影响
        direct_impact = self._calculate_direct_impact(service, failure_type)
        
        # 级联影响
        cascade_impact = self._calculate_cascade_impact(service)
        
        # 业务影响
        business_impact = self._calculate_business_impact(service, cascade_impact)
        
        # 用户影响
        user_impact = self._calculate_user_impact(service, cascade_impact)
        
        return {
            'direct_impact': direct_impact,
            'cascade_impact': cascade_impact,
            'business_impact': business_impact,
            'user_impact': user_impact,
            'overall_impact_score': self._calculate_overall_impact(
                direct_impact, cascade_impact, business_impact, user_impact
            )
        }
    
    def _calculate_direct_impact(self, service: str, failure_type: str) -> Dict[str, Any]:
        """计算直接影响"""
        # 基于服务关键性和故障类型
        criticality_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        failure_severity = {
            'complete_outage': 1.0,
            'partial_degradation': 0.6,
            'intermittent_errors': 0.3,
            'performance_degradation': 0.2
        }
        
        # 假设服务关键性 (实际应从配置获取)
        criticality = 'high'  # 默认值
        
        impact_score = (criticality_scores.get(criticality, 0.5) * 
                       failure_severity.get(failure_type, 0.5))
        
        return {
            'score': impact_score,
            'criticality': criticality,
            'failure_type': failure_type
        }
    
    def _calculate_cascade_impact(self, service: str) -> Dict[str, Any]:
        """计算级联影响"""
        # BFS 遍历依赖图
        affected_services = set()
        queue = [service]
        depth = {service: 0}
        max_depth = 0
        
        while queue:
            current = queue.pop(0)
            affected_services.add(current)
            
            # 找到依赖当前服务的服务
            for svc, dependencies in self.service_graph.items():
                if current in dependencies and svc not in affected_services:
                    queue.append(svc)
                    depth[svc] = depth[current] + 1
                    max_depth = max(max_depth, depth[svc])
        
        # 计算级联影响分数
        cascade_score = len(affected_services) * 0.1 + max_depth * 0.05
        
        return {
            'affected_services': list(affected_services),
            'affected_count': len(affected_services),
            'max_depth': max_depth,
            'score': min(cascade_score, 1.0)
        }
    
    def _calculate_business_impact(self, service: str, 
                                   cascade_impact: Dict[str, Any]) -> Dict[str, Any]:
        """计算业务影响"""
        # 假设的业务价值 (实际应从配置获取)
        business_value_per_minute = 1000  # $1000/minute
        
        affected_count = cascade_impact['affected_count']
        
        # 业务影响与受影响服务数量成正比
        revenue_impact = business_value_per_minute * (1 + affected_count * 0.2)
        
        return {
            'revenue_per_minute': revenue_impact,
            'affected_business_functions': affected_count,
            'score': min(affected_count * 0.15, 1.0)
        }
    
    def _calculate_user_impact(self, service: str, 
                               cascade_impact: Dict[str, Any]) -> Dict[str, Any]:
        """计算用户影响"""
        # 假设的用户基数
        base_users = 10000
        affected_services = cascade_impact['affected_count']
        
        # 受影响用户数
        affected_users = base_users * (0.3 + affected_services * 0.1)
        
        return {
            'estimated_affected_users': int(affected_users),
            'percentage': min(affected_users / base_users * 100, 100),
            'score': min(affected_users / base_users, 1.0)
        }
    
    def _calculate_overall_impact(self, direct: Dict, cascade: Dict,
                                  business: Dict, user: Dict) -> float:
        """计算整体影响分数"""
        weights = {
            'direct': 0.3,
            'cascade': 0.2,
            'business': 0.3,
            'user': 0.2
        }
        
        overall = (
            weights['direct'] * direct['score'] +
            weights['cascade'] * cascade['score'] +
            weights['business'] * business['score'] +
            weights['user'] * user['score']
        )
        
        return min(overall, 1.0)


class RiskMatrix:
    """风险矩阵"""
    
    def __init__(self):
        # 概率 x 影响 -> 风险等级
        self.matrix = {
            # 概率: minimal, low, medium, high, critical
            # 影响: minimal, low, medium, high, critical
            ('minimal', 'minimal'): RiskLevel.MINIMAL,
            ('minimal', 'low'): RiskLevel.MINIMAL,
            ('minimal', 'medium'): RiskLevel.LOW,
            ('minimal', 'high'): RiskLevel.LOW,
            ('minimal', 'critical'): RiskLevel.MEDIUM,
            
            ('low', 'minimal'): RiskLevel.MINIMAL,
            ('low', 'low'): RiskLevel.LOW,
            ('low', 'medium'): RiskLevel.LOW,
            ('low', 'high'): RiskLevel.MEDIUM,
            ('low', 'critical'): RiskLevel.HIGH,
            
            ('medium', 'minimal'): RiskLevel.LOW,
            ('medium', 'low'): RiskLevel.LOW,
            ('medium', 'medium'): RiskLevel.MEDIUM,
            ('medium', 'high'): RiskLevel.HIGH,
            ('medium', 'critical'): RiskLevel.HIGH,
            
            ('high', 'minimal'): RiskLevel.LOW,
            ('high', 'low'): RiskLevel.MEDIUM,
            ('high', 'medium'): RiskLevel.HIGH,
            ('high', 'high'): RiskLevel.HIGH,
            ('high', 'critical'): RiskLevel.CRITICAL,
            
            ('critical', 'minimal'): RiskLevel.MEDIUM,
            ('critical', 'low'): RiskLevel.HIGH,
            ('critical', 'medium'): RiskLevel.HIGH,
            ('critical', 'high'): RiskLevel.CRITICAL,
            ('critical', 'critical'): RiskLevel.CRITICAL,
        }
    
    def get_risk_level(self, probability: float, impact: float) -> RiskLevel:
        """根据概率和影响获取风险等级"""
        prob_level = self._score_to_level(probability)
        impact_level = self._score_to_level(impact)
        
        return self.matrix.get((prob_level, impact_level), RiskLevel.MEDIUM)
    
    def _score_to_level(self, score: float) -> str:
        """分数转换为等级"""
        if score < 0.1:
            return 'minimal'
        elif score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        elif score < 0.9:
            return 'high'
        else:
            return 'critical'


class RiskTrendAnalyzer:
    """风险趋势分析器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.risk_history = []
        
    def add_risk_record(self, risk_score: float, timestamp: datetime = None):
        """添加风险记录"""
        self.risk_history.append({
            'score': risk_score,
            'timestamp': timestamp or datetime.now()
        })
        
        # 保持窗口大小
        if len(self.risk_history) > self.window_size:
            self.risk_history.pop(0)
    
    def analyze_trend(self) -> Dict[str, Any]:
        """分析风险趋势"""
        if len(self.risk_history) < 3:
            return {'trend': 'insufficient_data'}
        
        scores = [r['score'] for r in self.risk_history]
        
        # 线性回归计算趋势
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        # 判断趋势
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'current_level': self._score_to_level(scores[-1]),
            'average_level': self._score_to_level(np.mean(scores)),
            'prediction': self._predict_future(scores, slope)
        }
    
    def _score_to_level(self, score: float) -> str:
        """分数转等级"""
        if score < 0.2:
            return 'low'
        elif score < 0.5:
            return 'medium'
        elif score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _predict_future(self, scores: List[float], slope: float) -> Dict[str, Any]:
        """预测未来风险"""
        current = scores[-1]
        
        # 预测 1h, 6h, 24h 后的风险
        predictions = {
            '1h': min(max(current + slope * 1, 0), 1),
            '6h': min(max(current + slope * 6, 0), 1),
            '24h': min(max(current + slope * 24, 0), 1)
        }
        
        return predictions


class RiskEngine:
    """
    风险评估引擎主类
    整合故障预测、影响评估、风险矩阵
    """
    
    def __init__(self, service_graph: Dict[str, List[str]] = None):
        self.probability_predictor = FailureProbabilityPredictor()
        self.impact_assessor = ImpactAssessor(service_graph)
        self.risk_matrix = RiskMatrix()
        self.trend_analyzer = RiskTrendAnalyzer()
        
    def assess_risk(self, service: str, current_conditions: Dict[str, Any],
                   failure_type: str = 'partial_degradation') -> RiskAssessment:
        """综合风险评估"""
        # 预测故障概率
        prob_result = self.probability_predictor.predict(service, current_conditions)
        probability = prob_result['probability']
        
        # 评估影响
        impact_result = self.impact_assessor.assess_service_impact(service, failure_type)
        impact = impact_result['overall_impact_score']
        
        # 确定风险等级
        risk_level = self.risk_matrix.get_risk_level(probability, impact)
        risk_score = probability * impact
        
        # 识别风险因子
        factors = self._identify_risk_factors(current_conditions, impact_result)
        
        # 生成建议
        recommendations = self._generate_recommendations(risk_level, factors)
        
        # 记录趋势
        self.trend_analyzer.add_risk_record(risk_score)
        
        return RiskAssessment(
            overall_risk=risk_level,
            risk_score=risk_score,
            probability=probability,
            impact=impact,
            factors=factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _identify_risk_factors(self, conditions: Dict[str, Any],
                               impact: Dict[str, Any]) -> List[RiskFactor]:
        """识别风险因子"""
        factors = []
        
        # 技术风险因子
        if conditions.get('cpu_high'):
            factors.append(RiskFactor(
                name='high_cpu_usage',
                category=RiskCategory.TECHNICAL,
                probability=0.7,
                impact=0.6
            ))
        
        if conditions.get('memory_high'):
            factors.append(RiskFactor(
                name='high_memory_usage',
                category=RiskCategory.TECHNICAL,
                probability=0.6,
                impact=0.5
            ))
        
        if conditions.get('error_spike'):
            factors.append(RiskFactor(
                name='error_rate_spike',
                category=RiskCategory.TECHNICAL,
                probability=0.8,
                impact=0.7
            ))
        
        # 业务风险因子
        if impact['business_impact']['score'] > 0.5:
            factors.append(RiskFactor(
                name='high_business_impact',
                category=RiskCategory.BUSINESS,
                probability=0.5,
                impact=impact['business_impact']['score']
            ))
        
        if impact['user_impact']['score'] > 0.5:
            factors.append(RiskFactor(
                name='high_user_impact',
                category=RiskCategory.BUSINESS,
                probability=0.5,
                impact=impact['user_impact']['score']
            ))
        
        return factors
    
    def _generate_recommendations(self, risk_level: RiskLevel, 
                                  factors: List[RiskFactor]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("立即通知相关团队，准备应急响应")
            recommendations.append("考虑启动降级策略")
        
        for factor in factors:
            if factor.name == 'high_cpu_usage':
                recommendations.append("考虑扩容或优化 CPU 密集型任务")
            elif factor.name == 'high_memory_usage':
                recommendations.append("检查内存泄漏，考虑重启服务")
            elif factor.name == 'error_rate_spike':
                recommendations.append("立即调查错误原因，考虑回滚最近的变更")
        
        if not recommendations:
            recommendations.append("持续监控，保持当前状态")
        
        return recommendations
    
    def get_risk_trend(self) -> Dict[str, Any]:
        """获取风险趋势"""
        return self.trend_analyzer.analyze_trend()
