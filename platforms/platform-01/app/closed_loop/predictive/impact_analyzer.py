"""
影响分析器
分析故障的影响范围，包括服务影响、业务影响、用户影响
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceImpact:
    """服务影响"""
    service_id: str
    impact_level: str  # critical, high, medium, low
    affected_dependencies: List[str]
    critical_path: bool
    estimated_downtime_minutes: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_id': self.service_id,
            'impact_level': self.impact_level,
            'affected_dependencies': self.affected_dependencies,
            'critical_path': self.critical_path,
            'estimated_downtime_minutes': self.estimated_downtime_minutes
        }


@dataclass
class BusinessImpact:
    """业务影响"""
    affected_functions: List[str]
    revenue_loss_per_minute: float
    transaction_impact: Dict[str, Any]
    sla_violation_risk: str  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'affected_functions': self.affected_functions,
            'revenue_loss_per_minute': self.revenue_loss_per_minute,
            'transaction_impact': self.transaction_impact,
            'sla_violation_risk': self.sla_violation_risk
        }


@dataclass
class UserImpact:
    """用户影响"""
    estimated_affected_users: int
    geographic_distribution: Dict[str, int]
    user_segments: List[str]
    severity_score: float  # 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'estimated_affected_users': self.estimated_affected_users,
            'geographic_distribution': self.geographic_distribution,
            'user_segments': self.user_segments,
            'severity_score': self.severity_score
        }


@dataclass
class DataImpact:
    """数据影响"""
    data_loss_risk: str  # high, medium, low, none
    consistency_impact: str
    affected_datasets: List[str]
    recovery_complexity: str  # high, medium, low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_loss_risk': self.data_loss_risk,
            'consistency_impact': self.consistency_impact,
            'affected_datasets': self.affected_datasets,
            'recovery_complexity': self.recovery_complexity
        }


class DependencyGraphAnalyzer:
    """依赖图分析器"""
    
    def __init__(self, service_graph: Dict[str, List[str]] = None):
        self.service_graph = service_graph or {}
        self.reverse_graph = self._build_reverse_graph()
        
    def _build_reverse_graph(self) -> Dict[str, List[str]]:
        """构建反向图"""
        reverse = defaultdict(list)
        for service, deps in self.service_graph.items():
            for dep in deps:
                reverse[dep].append(service)
        return dict(reverse)
    
    def find_affected_services(self, source_service: str, 
                               max_depth: int = 5) -> Dict[str, Any]:
        """查找受影响的服务"""
        affected = set()
        depths = {}
        paths = {}
        
        # BFS 遍历
        queue = [(source_service, 0, [source_service])]
        visited = {source_service}
        
        while queue:
            current, depth, path = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            affected.add(current)
            depths[current] = depth
            paths[current] = path
            
            # 找到依赖 current 的服务
            dependents = self.reverse_graph.get(current, [])
            for dependent in dependents:
                if dependent not in visited:
                    visited.add(dependent)
                    queue.append((dependent, depth + 1, path + [dependent]))
        
        return {
            'affected_services': list(affected),
            'affected_count': len(affected),
            'depths': depths,
            'paths': paths
        }
    
    def find_critical_path(self, source: str, target: str) -> List[str]:
        """查找关键路径"""
        # 使用 BFS 找最短路径
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == target:
                return path
            
            for neighbor in self.service_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def calculate_service_criticality(self, service: str) -> float:
        """计算服务关键性分数"""
        # 基于依赖该服务的服务数量
        dependents = self.reverse_graph.get(service, [])
        
        # 考虑间接依赖
        all_dependents = set()
        queue = list(dependents)
        
        while queue:
            current = queue.pop(0)
            if current not in all_dependents:
                all_dependents.add(current)
                queue.extend(self.reverse_graph.get(current, []))
        
        # 关键性分数
        criticality = len(all_dependents) / max(len(self.service_graph), 1)
        
        return min(criticality, 1.0)


class BusinessFunctionMapper:
    """业务功能映射器"""
    
    def __init__(self):
        # 服务到业务功能的映射
        self.service_to_functions: Dict[str, List[str]] = defaultdict(list)
        
        # 业务功能价值
        self.function_values: Dict[str, float] = {}
        
    def register_service_function(self, service: str, functions: List[str]):
        """注册服务的业务功能"""
        self.service_to_functions[service] = functions
        
    def set_function_value(self, function: str, value_per_minute: float):
        """设置业务功能价值"""
        self.function_values[function] = value_per_minute
        
    def get_affected_functions(self, services: List[str]) -> Dict[str, Any]:
        """获取受影响的业务功能"""
        affected_functions = set()
        function_to_services = defaultdict(list)
        
        for service in services:
            functions = self.service_to_functions.get(service, [])
            for func in functions:
                affected_functions.add(func)
                function_to_services[func].append(service)
        
        # 计算总业务价值损失
        total_value_loss = sum(
            self.function_values.get(func, 0) 
            for func in affected_functions
        )
        
        return {
            'affected_functions': list(affected_functions),
            'function_to_services': dict(function_to_services),
            'total_value_loss_per_minute': total_value_loss
        }


class UserBaseAnalyzer:
    """用户基数分析器"""
    
    def __init__(self):
        # 服务用户分布
        self.service_users: Dict[str, int] = {}
        
        # 地理分布
        self.geo_distribution: Dict[str, Dict[str, int]] = {}
        
        # 用户细分
        self.user_segments: Dict[str, List[str]] = {}
        
    def set_service_users(self, service: str, user_count: int):
        """设置服务用户数"""
        self.service_users[service] = user_count
        
    def set_geo_distribution(self, service: str, distribution: Dict[str, int]):
        """设置地理分布"""
        self.geo_distribution[service] = distribution
        
    def set_user_segments(self, service: str, segments: List[str]):
        """设置用户细分"""
        self.user_segments[service] = segments
        
    def estimate_affected_users(self, services: List[str]) -> Dict[str, Any]:
        """估计受影响用户数"""
        total_users = 0
        merged_geo = defaultdict(int)
        all_segments = set()
        
        for service in services:
            # 用户数
            users = self.service_users.get(service, 0)
            total_users += users
            
            # 地理分布
            geo = self.geo_distribution.get(service, {})
            for region, count in geo.items():
                merged_geo[region] += count
            
            # 用户细分
            segments = self.user_segments.get(service, [])
            all_segments.update(segments)
        
        # 去重估算 (假设用户可能使用多个服务)
        estimated_unique = int(total_users * 0.7)  # 30% 重叠
        
        return {
            'estimated_total_users': total_users,
            'estimated_unique_users': estimated_unique,
            'geographic_distribution': dict(merged_geo),
            'affected_segments': list(all_segments)
        }


class ImpactAnalyzer:
    """
    影响分析器主类
    整合依赖分析、业务影响、用户影响分析
    """
    
    def __init__(self, service_graph: Dict[str, List[str]] = None, topology: Dict[str, List[str]] = None):
        # 支持两种参数名
        graph = service_graph or topology or {}
        self.dependency_analyzer = DependencyGraphAnalyzer(graph)
        self.business_mapper = BusinessFunctionMapper()
        self.user_analyzer = UserBaseAnalyzer()
        
    def analyze_impact(self, source_service: str, 
                      failure_type: str = 'partial_degradation',
                      estimated_duration_minutes: float = 30) -> Dict[str, Any]:
        """
        综合分析影响
        
        Args:
            source_service: 故障源服务
            failure_type: 故障类型
            estimated_duration_minutes: 预计故障持续时间
        
        Returns:
            完整的影响分析结果
        """
        # 服务影响分析
        service_impact = self._analyze_service_impact(source_service)
        
        # 业务影响分析
        business_impact = self._analyze_business_impact(
            service_impact['affected_services'], 
            estimated_duration_minutes
        )
        
        # 用户影响分析
        user_impact = self._analyze_user_impact(service_impact['affected_services'])
        
        # 数据影响分析
        data_impact = self._analyze_data_impact(source_service, failure_type)
        
        # 综合评分
        overall_score = self._calculate_overall_score(
            service_impact, business_impact, user_impact
        )
        
        return {
            'source_service': source_service,
            'failure_type': failure_type,
            'estimated_duration_minutes': estimated_duration_minutes,
            'overall_impact_score': overall_score,
            'overall_severity': self._score_to_severity(overall_score),
            'service_impact': service_impact,
            'business_impact': business_impact,
            'user_impact': user_impact,
            'data_impact': data_impact,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_service_impact(self, source_service: str) -> Dict[str, Any]:
        """分析服务影响"""
        affected = self.dependency_analyzer.find_affected_services(source_service)
        
        services_impact = []
        for service in affected['affected_services']:
            depth = affected['depths'].get(service, 0)
            
            # 确定影响级别
            if depth == 0:
                level = 'critical'
            elif depth <= 2:
                level = 'high'
            elif depth <= 4:
                level = 'medium'
            else:
                level = 'low'
            
            # 计算关键性
            criticality = self.dependency_analyzer.calculate_service_criticality(service)
            
            services_impact.append({
                'service_id': service,
                'impact_level': level,
                'distance_from_source': depth,
                'criticality_score': criticality,
                'is_source': service == source_service
            })
        
        return {
            'affected_services': services_impact,
            'total_affected_count': len(services_impact),
            'critical_path_services': [s for s in services_impact if s['criticality_score'] > 0.5]
        }
    
    def _analyze_business_impact(self, affected_services: List[Dict], 
                                 duration_minutes: float) -> Dict[str, Any]:
        """分析业务影响"""
        service_ids = [s['service_id'] for s in affected_services]
        
        affected = self.business_mapper.get_affected_functions(service_ids)
        
        # 计算收入损失
        revenue_loss = affected['total_value_loss_per_minute'] * duration_minutes
        
        # 评估 SLA 风险
        critical_count = sum(1 for s in affected_services if s['impact_level'] == 'critical')
        if critical_count > 0:
            sla_risk = 'high'
        elif any(s['impact_level'] == 'high' for s in affected_services):
            sla_risk = 'medium'
        else:
            sla_risk = 'low'
        
        return {
            'affected_business_functions': affected['affected_functions'],
            'revenue_loss_estimate': revenue_loss,
            'revenue_loss_per_minute': affected['total_value_loss_per_minute'],
            'sla_violation_risk': sla_risk,
            'transaction_impact': {
                'estimated_failed_transactions': int(revenue_loss / 10),  # 假设平均交易额
                'affected_transaction_types': affected['affected_functions']
            }
        }
    
    def _analyze_user_impact(self, affected_services: List[Dict]) -> Dict[str, Any]:
        """分析用户影响"""
        service_ids = [s['service_id'] for s in affected_services]
        
        user_estimate = self.user_analyzer.estimate_affected_users(service_ids)
        
        # 计算严重程度分数
        affected_users = user_estimate['estimated_unique_users']
        severity = min(affected_users / 10000, 1.0)  # 假设 10000 为基准
        
        # 根据服务关键性调整
        critical_services = [s for s in affected_services if s['impact_level'] == 'critical']
        if critical_services:
            severity = min(severity * 1.5, 1.0)
        
        return {
            'estimated_affected_users': affected_users,
            'geographic_distribution': user_estimate['geographic_distribution'],
            'affected_user_segments': user_estimate['affected_segments'],
            'severity_score': severity
        }
    
    def _analyze_data_impact(self, source_service: str, 
                            failure_type: str) -> Dict[str, Any]:
        """分析数据影响"""
        # 基于故障类型评估数据风险
        data_risk_map = {
            'complete_outage': 'high',
            'partial_degradation': 'medium',
            'intermittent_errors': 'low',
            'performance_degradation': 'none',
            'data_corruption': 'high'
        }
        
        data_loss_risk = data_risk_map.get(failure_type, 'medium')
        
        # 一致性影响
        consistency_map = {
            'high': 'strong_consistency_violation',
            'medium': 'eventual_consistency_delay',
            'low': 'minor_inconsistency',
            'none': 'no_impact'
        }
        
        return {
            'data_loss_risk': data_loss_risk,
            'consistency_impact': consistency_map.get(data_loss_risk, 'unknown'),
            'affected_datasets': [f"{source_service}_data"],
            'recovery_complexity': 'high' if data_loss_risk == 'high' else 'medium'
        }
    
    def _calculate_overall_score(self, service: Dict, business: Dict, 
                                 user: Dict) -> float:
        """计算整体影响分数"""
        # 权重
        weights = {
            'service': 0.3,
            'business': 0.35,
            'user': 0.35
        }
        
        # 服务影响分数
        critical_count = sum(1 for s in service['affected_services'] 
                           if s['impact_level'] == 'critical')
        service_score = min(critical_count * 0.2 + 
                          len(service['affected_services']) * 0.05, 1.0)
        
        # 业务影响分数
        revenue_loss = business.get('revenue_loss_per_minute', 0)
        business_score = min(revenue_loss / 10000, 1.0)  # 假设 $10000/min 为基准
        
        # 用户影响分数
        user_score = user.get('severity_score', 0)
        
        overall = (weights['service'] * service_score +
                  weights['business'] * business_score +
                  weights['user'] * user_score)
        
        return min(overall, 1.0)
    
    def _score_to_severity(self, score: float) -> str:
        """分数转换为严重级别"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        elif score >= 0.2:
            return 'low'
        return 'minimal'
