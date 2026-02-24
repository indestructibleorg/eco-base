# =============================================================================
# Rule Engine
# =============================================================================
# 規則引擎 - Phase 1 基礎閉環核心組件
# 支持 YAML/JSON 規則定義、條件評估、動作調度
# =============================================================================

import re
import operator
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import json
import yaml

import asyncio

from app.core.logging import get_logger
from app.closed_loop.remediator.remediator import (
    Remediator, RemediationAction, RemediationType, RemediationResult
)
from app.closed_loop.governance.approval_service import (
    ApprovalService, ApprovalStatus, ApprovalPriority, ApprovalResult
)

logger = get_logger("rule_engine")


class ConditionOperator(Enum):
    """條件操作符"""
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class LogicalOperator(Enum):
    """邏輯操作符"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Condition:
    """條件定義"""
    field: str
    operator: ConditionOperator
    value: Any = None
    
    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """評估條件"""
        actual_value = self._get_nested_value(facts, self.field)
        
        if self.operator == ConditionOperator.EXISTS:
            return actual_value is not None
        
        if self.operator == ConditionOperator.NOT_EXISTS:
            return actual_value is None
        
        if actual_value is None:
            return False
        
        ops = {
            ConditionOperator.EQ: operator.eq,
            ConditionOperator.NE: operator.ne,
            ConditionOperator.GT: operator.gt,
            ConditionOperator.GTE: operator.ge,
            ConditionOperator.LT: operator.lt,
            ConditionOperator.LTE: operator.le,
        }
        
        if self.operator in ops:
            return ops[self.operator](actual_value, self.value)
        
        if self.operator == ConditionOperator.CONTAINS:
            return self.value in actual_value if isinstance(actual_value, (str, list, dict)) else False
        
        if self.operator == ConditionOperator.STARTS_WITH:
            return str(actual_value).startswith(str(self.value))
        
        if self.operator == ConditionOperator.ENDS_WITH:
            return str(actual_value).endswith(str(self.value))
        
        if self.operator == ConditionOperator.MATCHES:
            return bool(re.match(self.value, str(actual_value)))
        
        if self.operator == ConditionOperator.IN:
            return actual_value in self.value if isinstance(self.value, (list, set, tuple)) else False
        
        if self.operator == ConditionOperator.NOT_IN:
            return actual_value not in self.value if isinstance(self.value, (list, set, tuple)) else True
        
        return False
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """獲取嵌套值"""
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value


@dataclass
class CompoundCondition:
    """複合條件"""
    operator: LogicalOperator
    conditions: List[Union[Condition, "CompoundCondition"]]
    
    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """評估複合條件"""
        if not self.conditions:
            return True
        
        if self.operator == LogicalOperator.NOT:
            return not self.conditions[0].evaluate(facts)
        
        results = [c.evaluate(facts) for c in self.conditions]
        
        if self.operator == LogicalOperator.AND:
            return all(results)
        
        if self.operator == LogicalOperator.OR:
            return any(results)
        
        return False


@dataclass
class Action:
    """動作定義"""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay_seconds: int = 0
    require_approval: bool = False


@dataclass
class Rule:
    """規則定義"""
    name: str
    description: str = ""
    enabled: bool = True
    priority: int = 100
    
    # 觸發條件
    condition: Optional[Union[Condition, CompoundCondition]] = None
    
    # 執行動作
    actions: List[Action] = field(default_factory=list)
    
    # 冷卻配置
    cooldown_minutes: int = 5
    
    # 狀態
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    success_count: int = 0
    
    # 元數據
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def can_trigger(self) -> bool:
        """檢查是否可以觸發"""
        if not self.enabled:
            return False
        
        if self.last_triggered:
            cooldown = timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() - self.last_triggered < cooldown:
                return False
        
        return True
    
    def record_trigger(self, success: bool = True) -> None:
        """記錄觸發"""
        self.last_triggered = datetime.utcnow()
        self.trigger_count += 1
        if success:
            self.success_count += 1


class RuleParser:
    """規則解析器"""
    
    @staticmethod
    def parse_condition(data: Dict[str, Any]) -> Union[Condition, CompoundCondition]:
        """解析條件"""
        # 檢查是否是複合條件
        if "and" in data:
            return CompoundCondition(
                operator=LogicalOperator.AND,
                conditions=[RuleParser.parse_condition(c) for c in data["and"]]
            )
        
        if "or" in data:
            return CompoundCondition(
                operator=LogicalOperator.OR,
                conditions=[RuleParser.parse_condition(c) for c in data["or"]]
            )
        
        if "not" in data:
            return CompoundCondition(
                operator=LogicalOperator.NOT,
                conditions=[RuleParser.parse_condition(data["not"])]
            )
        
        # 解析簡單條件
        field = data.get("field")
        op_str = data.get("operator", "==")
        value = data.get("value")
        
        # 映射操作符字符串
        op_map = {
            "==": ConditionOperator.EQ,
            "!=": ConditionOperator.NE,
            ">": ConditionOperator.GT,
            ">=": ConditionOperator.GTE,
            "<": ConditionOperator.LT,
            "<=": ConditionOperator.LTE,
            "contains": ConditionOperator.CONTAINS,
            "starts_with": ConditionOperator.STARTS_WITH,
            "ends_with": ConditionOperator.ENDS_WITH,
            "matches": ConditionOperator.MATCHES,
            "in": ConditionOperator.IN,
            "not_in": ConditionOperator.NOT_IN,
            "exists": ConditionOperator.EXISTS,
            "not_exists": ConditionOperator.NOT_EXISTS,
        }
        
        operator = op_map.get(op_str, ConditionOperator.EQ)
        
        return Condition(field=field, operator=operator, value=value)
    
    @staticmethod
    def parse_action(data: Dict[str, Any]) -> Action:
        """解析動作"""
        return Action(
            action_type=data.get("type", ""),
            parameters=data.get("parameters", {}),
            delay_seconds=data.get("delay_seconds", 0),
            require_approval=data.get("require_approval", False)
        )
    
    @staticmethod
    def parse_rule(data: Dict[str, Any]) -> Rule:
        """解析規則"""
        condition = None
        if "condition" in data:
            condition = RuleParser.parse_condition(data["condition"])
        
        actions = [RuleParser.parse_action(a) for a in data.get("actions", [])]
        
        return Rule(
            name=data.get("name", ""),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            condition=condition,
            actions=actions,
            cooldown_minutes=data.get("cooldown_minutes", 5)
        )
    
    @staticmethod
    def load_from_yaml(yaml_content: str) -> List[Rule]:
        """從 YAML 加載規則"""
        data = yaml.safe_load(yaml_content)
        rules_data = data.get("rules", []) if isinstance(data, dict) else data
        return [RuleParser.parse_rule(r) for r in rules_data]
    
    @staticmethod
    def load_from_json(json_content: str) -> List[Rule]:
        """從 JSON 加載規則"""
        data = json.loads(json_content)
        rules_data = data.get("rules", []) if isinstance(data, dict) else data
        return [RuleParser.parse_rule(r) for r in rules_data]


class RuleEngine:
    """
    規則引擎
    
    管理規則的加載、評估和執行
    """
    
    def __init__(
        self,
        remediator: Optional[Remediator] = None,
        approval_service: Optional[ApprovalService] = None,
    ):
        self.rules: Dict[str, Rule] = {}
        self.remediator = remediator or Remediator()
        self.approval_service = approval_service
        self.parser = RuleParser()
        self._fact_providers: List[Callable[[], Dict[str, Any]]] = []
        self._approval_results: Dict[str, ApprovalResult] = {}  # 缓存审批结果
    
    def register_fact_provider(self, provider: Callable[[], Dict[str, Any]]) -> None:
        """註冊事實提供者"""
        self._fact_providers.append(provider)
    
    def add_rule(self, rule: Rule) -> None:
        """添加規則"""
        self.rules[rule.name] = rule
        logger.info(
            "rule_added",
            rule_name=rule.name,
            priority=rule.priority,
            action_count=len(rule.actions)
        )
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除規則"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info("rule_removed", rule_name=rule_name)
            return True
        return False
    
    def load_rules_from_yaml(self, yaml_content: str) -> int:
        """從 YAML 加載規則"""
        rules = self.parser.load_from_yaml(yaml_content)
        for rule in rules:
            self.add_rule(rule)
        return len(rules)
    
    def load_rules_from_json(self, json_content: str) -> int:
        """從 JSON 加載規則"""
        rules = self.parser.load_from_json(json_content)
        for rule in rules:
            self.add_rule(rule)
        return len(rules)
    
    def evaluate_rule(self, rule_name: str, facts: Dict[str, Any]) -> bool:
        """
        評估單個規則
        
        Args:
            rule_name: 規則名稱
            facts: 事實數據
            
        Returns:
            條件是否滿足
        """
        rule = self.rules.get(rule_name)
        if not rule:
            logger.warning("rule_not_found", rule_name=rule_name)
            return False
        
        if not rule.can_trigger():
            return False
        
        if not rule.condition:
            return True
        
        return rule.condition.evaluate(facts)
    
    def evaluate_all(self, facts: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        評估所有規則
        
        Args:
            facts: 事實數據（為 None 時使用註冊的事實提供者）
            
        Returns:
            觸發的規則名稱列表
        """
        if facts is None:
            facts = self._collect_facts()
        
        triggered = []
        
        # 按優先級排序
        sorted_rules = sorted(self.rules.values(), key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if self.evaluate_rule(rule.name, facts):
                triggered.append(rule.name)
        
        return triggered
    
    def _collect_facts(self) -> Dict[str, Any]:
        """收集所有事實"""
        facts = {}
        
        for provider in self._fact_providers:
            try:
                provider_facts = provider()
                if provider_facts:
                    facts.update(provider_facts)
            except Exception as e:
                logger.error("fact_provider_failed", error=str(e))
        
        return facts
    
    async def execute_rule(self, rule_name: str, facts: Optional[Dict[str, Any]] = None) -> List[RemediationResult]:
        """
        執行規則
        
        Args:
            rule_name: 規則名稱
            facts: 事實數據
            
        Returns:
            執行結果列表
        """
        rule = self.rules.get(rule_name)
        if not rule:
            logger.warning("rule_not_found", rule_name=rule_name)
            return []
        
        if facts is None:
            facts = self._collect_facts()
        
        # 評估條件
        if not self.evaluate_rule(rule_name, facts):
            return []
        
        logger.info("rule_triggered", rule_name=rule_name, action_count=len(rule.actions))
        
        results = []
        all_success = True
        
        for action_def in rule.actions:
            # 延遲執行
            if action_def.delay_seconds > 0:
                await asyncio.sleep(action_def.delay_seconds)
            
            # 檢查是否需要人工審批
            if action_def.require_approval:
                approval_result = await self._handle_approval(rule_name, action_def, facts)
                if not approval_result.approved:
                    # 审批未通过，创建失败结果
                    result = RemediationResult(
                        action_id="",
                        action_type=RemediationType.RESTART_POD,
                        status=RemediationResult.status,
                        message=f"Action rejected: {approval_result.message}",
                        started_at=datetime.utcnow(),
                        completed_at=datetime.utcnow(),
                    )
                    results.append(result)
                    all_success = False
                    continue
            
            # 執行動作
            result = await self._execute_action(action_def, facts)
            results.append(result)
            
            if result.status.value != "success":
                all_success = False
        
        # 記錄觸發
        rule.record_trigger(success=all_success)
        
        return results
    
    async def _execute_action(self, action_def: Action, facts: Dict[str, Any]) -> RemediationResult:
        """執行單個動作"""
        # 映射動作類型到 RemediationType
        type_map = {
            "restart_pod": RemediationType.RESTART_POD,
            "clear_cache": RemediationType.CLEAR_CACHE,
            "switch_provider": RemediationType.SWITCH_PROVIDER,
            "scale_up": RemediationType.SCALE_UP,
            "scale_down": RemediationType.SCALE_DOWN,
            "enable_circuit_breaker": RemediationType.ENABLE_CIRCUIT_BREAKER,
            "disable_circuit_breaker": RemediationType.DISABLE_CIRCUIT_BREAKER,
            "rollback_deployment": RemediationType.ROLLBACK_DEPLOYMENT,
        }
        
        remediation_type = type_map.get(action_def.action_type)
        
        if not remediation_type:
            return RemediationResult(
                action_id="",
                action_type=RemediationType.RESTART_POD,  # 默認值
                status=RemediationResult.status,
                message=f"Unknown action type: {action_def.action_type}",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        
        # 構建修復動作
        remediation_action = RemediationAction(
            action_type=remediation_type,
            target=action_def.parameters.get("target", ""),
            parameters=action_def.parameters
        )
        
        # 執行
        return await self.remediator.execute(remediation_action)
    
    async def _handle_approval(
        self,
        rule_name: str,
        action_def: Action,
        facts: Dict[str, Any],
    ) -> ApprovalResult:
        """
        处理审批流程
        
        Args:
            rule_name: 规则名称
            action_def: 动作定义
            facts: 事实数据
            
        Returns:
            审批结果
        """
        if not self.approval_service:
            logger.warning(
                "approval_service_not_configured",
                rule_name=rule_name,
                action_type=action_def.action_type,
            )
            return ApprovalResult(
                request_id="",
                approved=False,
                status=ApprovalStatus.REJECTED,
                message="Approval service not configured",
            )
        
        # 提取风险信息
        risk_level = facts.get("risk_level", "MEDIUM")
        risk_score = facts.get("risk_score", 0.5)
        
        # 确定优先级
        priority_map = {
            "LOW": ApprovalPriority.LOW,
            "MEDIUM": ApprovalPriority.MEDIUM,
            "HIGH": ApprovalPriority.HIGH,
            "CRITICAL": ApprovalPriority.CRITICAL,
        }
        priority = priority_map.get(risk_level, ApprovalPriority.MEDIUM)
        
        # 创建审批请求
        request = await self.approval_service.create_request(
            rule_name=rule_name,
            action_type=action_def.action_type,
            action_parameters=action_def.parameters,
            risk_level=risk_level,
            risk_score=risk_score,
            context={
                "affected_services": facts.get("affected_services", []),
                "anomaly_type": facts.get("anomaly_type"),
                "metric_name": facts.get("metric_name"),
            },
            facts_snapshot=facts,
            priority=priority,
            timeout_minutes=30,  # 默认30分钟超时
        )
        
        logger.info(
            "approval_request_created",
            request_id=request.request_id,
            rule_name=rule_name,
            action_type=action_def.action_type,
        )
        
        # 等待审批结果
        result = await self.approval_service.wait_for_approval(
            request_id=request.request_id,
            timeout_seconds=30 * 60,  # 30分钟
            check_interval=5.0,  # 每5秒检查一次
        )
        
        # 缓存结果
        self._approval_results[request.request_id] = result
        
        logger.info(
            "approval_result_received",
            request_id=request.request_id,
            approved=result.approved,
            status=result.status.value,
        )
        
        return result
    
    async def execute_all(self, facts: Optional[Dict[str, Any]] = None) -> Dict[str, List[RemediationResult]]:
        """
        執行所有觸發的規則
        
        Args:
            facts: 事實數據
            
        Returns:
            規則名稱到執行結果的映射
        """
        triggered = self.evaluate_all(facts)
        results = {}
        
        for rule_name in triggered:
            results[rule_name] = await self.execute_rule(rule_name, facts)
        
        return results
    
    def get_rule_status(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """獲取規則狀態"""
        rule = self.rules.get(rule_name)
        if not rule:
            return None
        
        return {
            "name": rule.name,
            "description": rule.description,
            "enabled": rule.enabled,
            "priority": rule.priority,
            "trigger_count": rule.trigger_count,
            "success_count": rule.success_count,
            "success_rate": rule.success_count / rule.trigger_count if rule.trigger_count > 0 else 0,
            "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
            "can_trigger": rule.can_trigger()
        }
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """列出所有規則"""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "action_count": len(rule.actions)
            }
            for rule in sorted(self.rules.values(), key=lambda r: r.priority)
        ]
    
    def export_rules_to_yaml(self) -> str:
        """導出規則為 YAML"""
        rules_data = []
        
        for rule in self.rules.values():
            rule_dict = {
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "cooldown_minutes": rule.cooldown_minutes,
            }
            
            if rule.condition:
                rule_dict["condition"] = self._condition_to_dict(rule.condition)
            
            rule_dict["actions"] = [
                {
                    "type": action.action_type,
                    "parameters": action.parameters,
                    "delay_seconds": action.delay_seconds,
                    "require_approval": action.require_approval
                }
                for action in rule.actions
            ]
            
            rules_data.append(rule_dict)
        
        return yaml.dump({"rules": rules_data}, default_flow_style=False)
    
    def _condition_to_dict(self, condition: Union[Condition, CompoundCondition]) -> Dict[str, Any]:
        """將條件轉換為字典"""
        if isinstance(condition, Condition):
            return {
                "field": condition.field,
                "operator": condition.operator.value,
                "value": condition.value
            }
        
        if isinstance(condition, CompoundCondition):
            key = condition.operator.value
            value = [self._condition_to_dict(c) for c in condition.conditions]
            return {key: value}
        
        return {}


# 導入 asyncio
import asyncio

# 全局規則引擎實例
rule_engine = RuleEngine()
