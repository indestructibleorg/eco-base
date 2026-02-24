"""
成本模型构建器
计算各种决策的直接成本、间接成本和机会成本
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CostType(Enum):
    """成本类型"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    OPPORTUNITY = "opportunity"


@dataclass
class CostBreakdown:
    """成本明细"""
    cost_type: CostType
    category: str
    amount: float
    currency: str = "USD"
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cost_type': self.cost_type.value,
            'category': self.category,
            'amount': self.amount,
            'currency': self.currency,
            'details': self.details or {}
        }


class CostModelBuilder:
    """
    成本模型构建器
    
    成本维度:
    - 直接成本: 计算资源、存储、网络、人力
    - 间接成本: 收入损失、客户流失、声誉损害
    - 机会成本: 延迟修复代价、资源转移成本
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 成本基准配置
        self.cost_rates = {
            # 计算资源 ($/小时)
            'compute': {
                'cpu_core': self.config.get('cpu_cost', 0.05),
                'memory_gb': self.config.get('memory_cost', 0.01),
                'gpu': self.config.get('gpu_cost', 2.0)
            },
            # 存储 ($/GB/月)
            'storage': {
                'ssd': self.config.get('ssd_cost', 0.10),
                'hdd': self.config.get('hdd_cost', 0.02),
                'backup': self.config.get('backup_cost', 0.05)
            },
            # 网络 ($/GB)
            'network': {
                'ingress': self.config.get('ingress_cost', 0.01),
                'egress': self.config.get('egress_cost', 0.09),
                'cross_region': self.config.get('cross_region_cost', 0.02)
            },
            # 人力 ($/小时)
            'labor': {
                'engineer': self.config.get('engineer_cost', 100),
                'sre': self.config.get('sre_cost', 150),
                'manager': self.config.get('manager_cost', 200)
            }
        }
        
        # 业务成本系数
        self.business_cost_factors = {
            'revenue_per_minute': self.config.get('revenue_per_minute', 1000),
            'customer_acquisition_cost': self.config.get('cac', 50),
            'customer_lifetime_value': self.config.get('clv', 500),
            'reputation_recovery_cost': self.config.get('reputation_cost', 10000)
        }
        
    def calculate_direct_cost(self, action: str, action_params: Dict[str, Any],
                             resource_usage: Dict[str, Any]) -> List[CostBreakdown]:
        """计算直接成本"""
        costs = []
        
        # 计算资源成本
        if 'compute' in resource_usage:
            compute = resource_usage['compute']
            duration_hours = compute.get('duration_minutes', 0) / 60
            
            cpu_cost = compute.get('cpu_cores', 0) * self.cost_rates['compute']['cpu_core'] * duration_hours
            memory_cost = compute.get('memory_gb', 0) * self.cost_rates['compute']['memory_gb'] * duration_hours
            gpu_cost = compute.get('gpu_hours', 0) * self.cost_rates['compute']['gpu']
            
            total_compute = cpu_cost + memory_cost + gpu_cost
            if total_compute > 0:
                costs.append(CostBreakdown(
                    cost_type=CostType.DIRECT,
                    category='compute',
                    amount=total_compute,
                    details={
                        'cpu_cost': cpu_cost,
                        'memory_cost': memory_cost,
                        'gpu_cost': gpu_cost,
                        'duration_hours': duration_hours
                    }
                ))
        
        # 存储成本
        if 'storage' in resource_usage:
            storage = resource_usage['storage']
            ssd_cost = storage.get('ssd_gb', 0) * self.cost_rates['storage']['ssd'] / 30 / 24  # 转换为小时
            hdd_cost = storage.get('hdd_gb', 0) * self.cost_rates['storage']['hdd'] / 30 / 24
            
            total_storage = ssd_cost + hdd_cost
            if total_storage > 0:
                costs.append(CostBreakdown(
                    cost_type=CostType.DIRECT,
                    category='storage',
                    amount=total_storage,
                    details={'ssd_cost': ssd_cost, 'hdd_cost': hdd_cost}
                ))
        
        # 网络成本
        if 'network' in resource_usage:
            network = resource_usage['network']
            ingress_cost = network.get('ingress_gb', 0) * self.cost_rates['network']['ingress']
            egress_cost = network.get('egress_gb', 0) * self.cost_rates['network']['egress']
            cross_region_cost = network.get('cross_region_gb', 0) * self.cost_rates['network']['cross_region']
            
            total_network = ingress_cost + egress_cost + cross_region_cost
            if total_network > 0:
                costs.append(CostBreakdown(
                    cost_type=CostType.DIRECT,
                    category='network',
                    amount=total_network,
                    details={
                        'ingress_cost': ingress_cost,
                        'egress_cost': egress_cost,
                        'cross_region_cost': cross_region_cost
                    }
                ))
        
        # 人力成本
        if 'labor' in resource_usage:
            labor = resource_usage['labor']
            engineer_hours = labor.get('engineer_hours', 0)
            sre_hours = labor.get('sre_hours', 0)
            manager_hours = labor.get('manager_hours', 0)
            
            engineer_cost = engineer_hours * self.cost_rates['labor']['engineer']
            sre_cost = sre_hours * self.cost_rates['labor']['sre']
            manager_cost = manager_hours * self.cost_rates['labor']['manager']
            
            total_labor = engineer_cost + sre_cost + manager_cost
            if total_labor > 0:
                costs.append(CostBreakdown(
                    cost_type=CostType.DIRECT,
                    category='labor',
                    amount=total_labor,
                    details={
                        'engineer_cost': engineer_cost,
                        'sre_cost': sre_cost,
                        'manager_cost': manager_cost
                    }
                ))
        
        return costs
    
    def calculate_indirect_cost(self, incident_impact: Dict[str, Any]) -> List[CostBreakdown]:
        """计算间接成本"""
        costs = []
        
        # 收入损失
        downtime_minutes = incident_impact.get('downtime_minutes', 0)
        degraded_minutes = incident_impact.get('degraded_minutes', 0)
        
        revenue_loss = (downtime_minutes * self.business_cost_factors['revenue_per_minute'] +
                       degraded_minutes * self.business_cost_factors['revenue_per_minute'] * 0.5)
        
        if revenue_loss > 0:
            costs.append(CostBreakdown(
                cost_type=CostType.INDIRECT,
                category='revenue_loss',
                amount=revenue_loss,
                details={
                    'downtime_minutes': downtime_minutes,
                    'degraded_minutes': degraded_minutes
                }
            ))
        
        # 客户流失成本
        affected_customers = incident_impact.get('affected_customers', 0)
        churn_rate = incident_impact.get('churn_rate', 0.01)  # 默认 1%
        churned_customers = affected_customers * churn_rate
        
        customer_churn_cost = churned_customers * self.business_cost_factors['customer_lifetime_value']
        
        if customer_churn_cost > 0:
            costs.append(CostBreakdown(
                cost_type=CostType.INDIRECT,
                category='customer_churn',
                amount=customer_churn_cost,
                details={
                    'affected_customers': affected_customers,
                    'churned_customers': churned_customers,
                    'churn_rate': churn_rate
                }
            ))
        
        # 声誉损害成本
        severity = incident_impact.get('severity', 'low')
        reputation_multiplier = {'low': 0.1, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
        reputation_cost = (self.business_cost_factors['reputation_recovery_cost'] * 
                          reputation_multiplier.get(severity, 0.1))
        
        if reputation_cost > 0:
            costs.append(CostBreakdown(
                cost_type=CostType.INDIRECT,
                category='reputation_damage',
                amount=reputation_cost,
                details={'severity': severity}
            ))
        
        return costs
    
    def calculate_opportunity_cost(self, decision_context: Dict[str, Any]) -> List[CostBreakdown]:
        """计算机会成本"""
        costs = []
        
        # 延迟修复代价
        delayed_minutes = decision_context.get('delayed_minutes', 0)
        if delayed_minutes > 0:
            delayed_cost = delayed_minutes * self.business_cost_factors['revenue_per_minute']
            costs.append(CostBreakdown(
                cost_type=CostType.OPPORTUNITY,
                category='delayed_fix',
                amount=delayed_cost,
                details={'delayed_minutes': delayed_minutes}
            ))
        
        # 资源转移成本
        diverted_resources = decision_context.get('diverted_resources', {})
        if diverted_resources:
            diverted_cost = (diverted_resources.get('engineer_hours', 0) * self.cost_rates['labor']['engineer'] +
                           diverted_resources.get('sre_hours', 0) * self.cost_rates['labor']['sre'])
            
            # 机会成本 = 资源转移导致的其他工作延迟
            opportunity_multiplier = 1.5  # 资源转移的机会成本倍数
            total_diverted_cost = diverted_cost * opportunity_multiplier
            
            costs.append(CostBreakdown(
                cost_type=CostType.OPPORTUNITY,
                category='resource_diversion',
                amount=total_diverted_cost,
                details={
                    'direct_cost': diverted_cost,
                    'opportunity_multiplier': opportunity_multiplier
                }
            ))
        
        return costs
    
    def calculate_total_cost(self, action: str, action_params: Dict[str, Any],
                            resource_usage: Dict[str, Any],
                            incident_impact: Dict[str, Any],
                            decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """计算总成本"""
        direct_costs = self.calculate_direct_cost(action, action_params, resource_usage)
        indirect_costs = self.calculate_indirect_cost(incident_impact)
        opportunity_costs = self.calculate_opportunity_cost(decision_context)
        
        all_costs = direct_costs + indirect_costs + opportunity_costs
        
        # 汇总
        total_by_type = {
            'direct': sum(c.amount for c in direct_costs),
            'indirect': sum(c.amount for c in indirect_costs),
            'opportunity': sum(c.amount for c in opportunity_costs)
        }
        
        total_cost = sum(total_by_type.values())
        
        return {
            'total_cost': total_cost,
            'breakdown_by_type': total_by_type,
            'detailed_costs': [c.to_dict() for c in all_costs],
            'currency': 'USD'
        }


class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, cost_model: CostModelBuilder):
        self.cost_model = cost_model
        
    def find_cheapest_option(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """找出成本最低的选项"""
        costs = []
        
        for option in options:
            cost_result = self.cost_model.calculate_total_cost(
                action=option['action'],
                action_params=option.get('params', {}),
                resource_usage=option.get('resource_usage', {}),
                incident_impact=option.get('incident_impact', {}),
                decision_context=option.get('decision_context', {})
            )
            costs.append({
                'option': option,
                'cost': cost_result['total_cost'],
                'breakdown': cost_result['breakdown_by_type']
            })
        
        # 排序
        costs.sort(key=lambda x: x['cost'])
        
        return {
            'cheapest': costs[0] if costs else None,
            'all_options': costs,
            'cost_difference': costs[-1]['cost'] - costs[0]['cost'] if len(costs) > 1 else 0
        }
    
    def optimize_resource_allocation(self, budget: float, 
                                    requirements: Dict[str, Any]) -> Dict[str, Any]:
        """优化资源分配"""
        # 简化的资源分配优化
        # 实际应用中可以使用线性规划或整数规划
        
        allocation = {
            'compute': budget * 0.4,
            'storage': budget * 0.2,
            'network': budget * 0.1,
            'labor': budget * 0.3
        }
        
        return {
            'budget': budget,
            'allocation': allocation,
            'expected_efficiency': 0.85
        }
