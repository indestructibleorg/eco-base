"""
规则引擎
管理闭环规则和执行策略
"""

import logging
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class RuleCondition(Enum):
    """规则条件类型"""
    METRIC_THRESHOLD = "metric_threshold"
    ANOMALY_DETECTED = "anomaly_detected"
    TIME_BASED = "time_based"
    COMPOSITE = "composite"


class RuleAction(Enum):
    """规则动作类型"""
    ALERT = "alert"
    REMEDIATE = "remediate"
    ESCALATE = "escalate"
    SUPPRESS = "suppress"
    WORKFLOW = "workflow"


@dataclass
class Rule:
    """规则定义"""
    rule_id: str
    name: str
    description: str
    enabled: bool = True
    condition_type: RuleCondition = RuleCondition.METRIC_THRESHOLD
    condition_config: Dict[str, Any] = field(default_factory=dict)
    action_type: RuleAction = RuleAction.ALERT
    action_config: Dict[str, Any] = field(default_factory=dict)
    cooldown_minutes: int = 10
    priority: int = 5
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        
        # 规则存储
        self.rules: Dict[str, Rule] = {}
        self.rule_groups: Dict[str, List[str]] = {}
        
        # 动作处理器
        self.action_handlers: Dict[RuleAction, Callable] = {}
        
        # 冷却记录
        self.cooldown_log: Dict[str, datetime] = {}
        
        # 加载配置
        if config_path and Path(config_path).exists():
            self.load_rules(config_path)
        
        logger.info("规则引擎初始化完成")
    
    def register_action_handler(self, action_type: RuleAction, handler: Callable):
        """注册动作处理器"""
        self.action_handlers[action_type] = handler
        logger.info(f"动作处理器注册: {action_type.value}")
    
    def add_rule(self, rule: Rule) -> str:
        """添加规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"规则添加: {rule.name} (ID: {rule.rule_id})")
        return rule.rule_id
    
    def add_rule_from_dict(self, rule_dict: Dict[str, Any]) -> str:
        """从字典添加规则"""
        condition = rule_dict.get('condition', {})
        action = rule_dict.get('action', 'alert')
        
        rule = Rule(
            rule_id=rule_dict.get('id', f"rule_{datetime.now().timestamp()}"),
            name=rule_dict.get('name', 'Unnamed Rule'),
            description=rule_dict.get('description', ''),
            enabled=rule_dict.get('enabled', True),
            condition_type=RuleCondition.METRIC_THRESHOLD,
            condition_config={
                'metric': condition.get('metric', 'cpu'),
                'operator': condition.get('operator', '>'),
                'threshold': condition.get('threshold', 80)
            },
            action_type=RuleAction(action) if action in [a.value for a in RuleAction] else RuleAction.ALERT,
            action_config=rule_dict.get('action_config', {}),
            priority=rule_dict.get('priority', 5)
        )
        
        return self.add_rule(rule)
    
    def evaluate_rules(self, metrics: Dict[str, float]) -> bool:
        """评估规则是否触发"""
        triggered = False
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 检查冷却期
            if rule.rule_id in self.cooldown_log:
                elapsed = (datetime.now() - self.cooldown_log[rule.rule_id]).total_seconds() / 60
                if elapsed < rule.cooldown_minutes:
                    continue
            
            # 评估条件
            context = {'metrics': metrics}
            if self._evaluate_condition(rule.condition_type, rule.condition_config, context):
                triggered = True
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
                self.cooldown_log[rule.rule_id] = datetime.now()
                logger.info(f"规则触发: {rule.name}")
        
        return triggered
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"规则移除: {rule_id}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def list_rules(self, enabled_only: bool = False, 
                   tags: List[str] = None) -> List[Rule]:
        """列出规则"""
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if tags:
            rules = [r for r in rules if any(t in r.tags for t in tags)]
        
        return sorted(rules, key=lambda r: r.priority, reverse=True)
    
    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> bool:
        """评估单个规则"""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return False
        
        # 检查冷却
        if self._is_in_cooldown(rule_id, rule.cooldown_minutes):
            return False
        
        # 评估条件
        condition_met = self._evaluate_condition(
            rule.condition_type, 
            rule.condition_config, 
            context
        )
        
        if condition_met:
            # 更新触发记录
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            self.cooldown_log[rule_id] = datetime.now()
            
            logger.info(f"规则触发: {rule.name}")
        
        return condition_met
    
    def evaluate_all(self, context: Dict[str, Any]) -> List[Rule]:
        """评估所有规则"""
        triggered = []
        
        for rule in self.list_rules(enabled_only=True):
            if self.evaluate_rule(rule.rule_id, context):
                triggered.append(rule)
        
        return triggered
    
    async def execute_actions(self, rules: List[Rule], 
                              context: Dict[str, Any]) -> List[Dict]:
        """执行规则动作"""
        results = []
        
        for rule in rules:
            handler = self.action_handlers.get(rule.action_type)
            if not handler:
                logger.warning(f"未找到动作处理器: {rule.action_type.value}")
                continue
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(rule, context)
                else:
                    result = handler(rule, context)
                
                results.append({
                    'rule_id': rule.rule_id,
                    'action': rule.action_type.value,
                    'result': result
                })
                
                logger.info(f"动作执行: {rule.name} -> {rule.action_type.value}")
            
            except Exception as e:
                logger.exception(f"动作执行失败: {rule.name}")
                results.append({
                    'rule_id': rule.rule_id,
                    'action': rule.action_type.value,
                    'error': str(e)
                })
        
        return results
    
    def _evaluate_condition(self, condition_type: RuleCondition,
                            config: Dict[str, Any],
                            context: Dict[str, Any]) -> bool:
        """评估条件"""
        if condition_type == RuleCondition.METRIC_THRESHOLD:
            return self._evaluate_metric_threshold(config, context)
        elif condition_type == RuleCondition.ANOMALY_DETECTED:
            return self._evaluate_anomaly(config, context)
        elif condition_type == RuleCondition.TIME_BASED:
            return self._evaluate_time_based(config, context)
        elif condition_type == RuleCondition.COMPOSITE:
            return self._evaluate_composite(config, context)
        return False
    
    def _evaluate_metric_threshold(self, config: Dict, context: Dict) -> bool:
        """评估指标阈值"""
        metric_name = config.get('metric')
        threshold = config.get('threshold')
        operator = config.get('operator', '>')
        
        value = context.get('metrics', {}).get(metric_name)
        if value is None:
            return False
        
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        
        return False
    
    def _evaluate_anomaly(self, config: Dict, context: Dict) -> bool:
        """评估异常检测"""
        anomaly_type = config.get('anomaly_type')
        min_severity = config.get('min_severity', 0.5)
        
        anomalies = context.get('anomalies', [])
        
        for anomaly in anomalies:
            if anomaly_type and anomaly.get('type') != anomaly_type:
                continue
            if anomaly.get('severity', 0) >= min_severity:
                return True
        
        return False
    
    def _evaluate_time_based(self, config: Dict, context: Dict) -> bool:
        """评估时间条件"""
        start_time = config.get('start_time')
        end_time = config.get('end_time')
        days_of_week = config.get('days_of_week', [])
        
        now = datetime.now()
        
        if days_of_week and now.weekday() not in days_of_week:
            return False
        
        if start_time and end_time:
            current_time = now.strftime('%H:%M')
            return start_time <= current_time <= end_time
        
        return True
    
    def _evaluate_composite(self, config: Dict, context: Dict) -> bool:
        """评估复合条件"""
        conditions = config.get('conditions', [])
        operator = config.get('operator', 'AND')
        
        if not conditions:
            return False
        
        results = []
        for cond in conditions:
            cond_type = RuleCondition(cond.get('type'))
            cond_config = cond.get('config', {})
            results.append(self._evaluate_condition(cond_type, cond_config, context))
        
        if operator == 'AND':
            return all(results)
        elif operator == 'OR':
            return any(results)
        elif operator == 'NOT':
            return not any(results)
        
        return False
    
    def _is_in_cooldown(self, rule_id: str, cooldown_minutes: int) -> bool:
        """检查是否在冷却期"""
        last_triggered = self.cooldown_log.get(rule_id)
        if not last_triggered:
            return False
        
        elapsed = datetime.now() - last_triggered
        return elapsed < timedelta(minutes=cooldown_minutes)
    
    def load_rules(self, path: str):
        """从文件加载规则"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            for rule_data in data.get('rules', []):
                rule = Rule(
                    rule_id=rule_data['id'],
                    name=rule_data['name'],
                    description=rule_data.get('description', ''),
                    enabled=rule_data.get('enabled', True),
                    condition_type=RuleCondition(rule_data['condition']['type']),
                    condition_config=rule_data['condition'].get('config', {}),
                    action_type=RuleAction(rule_data['action']['type']),
                    action_config=rule_data['action'].get('config', {}),
                    cooldown_minutes=rule_data.get('cooldown_minutes', 10),
                    priority=rule_data.get('priority', 5),
                    tags=rule_data.get('tags', [])
                )
                self.add_rule(rule)
            
            logger.info(f"规则加载完成: {path} ({len(data.get('rules', []))} 条)")
        
        except Exception as e:
            logger.error(f"规则加载失败: {e}")
    
    def save_rules(self, path: str):
        """保存规则到文件"""
        data = {
            'rules': [
                {
                    'id': r.rule_id,
                    'name': r.name,
                    'description': r.description,
                    'enabled': r.enabled,
                    'condition': {
                        'type': r.condition_type.value,
                        'config': r.condition_config
                    },
                    'action': {
                        'type': r.action_type.value,
                        'config': r.action_config
                    },
                    'cooldown_minutes': r.cooldown_minutes,
                    'priority': r.priority,
                    'tags': r.tags
                }
                for r in self.rules.values()
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"规则保存完成: {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.rules)
        enabled = sum(1 for r in self.rules.values() if r.enabled)
        
        by_condition = {}
        by_action = {}
        
        for rule in self.rules.values():
            ct = rule.condition_type.value
            at = rule.action_type.value
            by_condition[ct] = by_condition.get(ct, 0) + 1
            by_action[at] = by_action.get(at, 0) + 1
        
        total_triggers = sum(r.trigger_count for r in self.rules.values())
        
        return {
            'total_rules': total,
            'enabled': enabled,
            'by_condition': by_condition,
            'by_action': by_action,
            'total_triggers': total_triggers
        }


import asyncio
