# =============================================================================
# RCA Root Cause Identifier
# =============================================================================
# 根因分析 - 根因识别器
# 使用贝叶斯网络、决策树和图遍历算法识别根本原因
# =============================================================================

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import uuid

from app.closed_loop.rca.event_collector import Event, EventType, EventSeverity
from app.closed_loop.rca.correlation_analyzer import (
    CorrelationAnalyzer, EventCluster, CausalInferenceAnalyzer
)


@dataclass
class RootCause:
    """根因结果"""
    event: Event
    confidence: float
    evidence: List[str]
    affected_services: List[str]
    cascading_impact: List[str]


@dataclass
class RCAResult:
    """RCA 分析结果"""
    id: str
    timestamp: datetime
    root_causes: List[RootCause]
    contributing_factors: List[Event]
    affected_services: List[str]
    timeline: List[Event]
    confidence: float
    recommended_actions: List[str]
    analysis_method: str


class CausalGraph:
    """因果图"""
    
    def __init__(self):
        self.nodes: Dict[str, Event] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.node_scores: Dict[str, float] = {}
    
    def add_node(self, event: Event) -> None:
        """添加节点"""
        self.nodes[event.id] = event
        if event.id not in self.edges:
            self.edges[event.id] = []
    
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        confidence: float
    ) -> None:
        """添加因果边"""
        self.edges[from_id].append((to_id, confidence))
    
    def get_parents(self, node_id: str) -> List[Tuple[str, float]]:
        """获取节点的父节点（可能的原因）"""
        parents = []
        for from_id, edges in self.edges.items():
            for to_id, confidence in edges:
                if to_id == node_id:
                    parents.append((from_id, confidence))
        return parents
    
    def get_children(self, node_id: str) -> List[Tuple[str, float]]:
        """获取节点的子节点（可能的结果）"""
        return self.edges.get(node_id, [])
    
    def get_root_nodes(self) -> List[str]:
        """获取根节点（没有父节点的节点）"""
        all_children = set()
        for edges in self.edges.values():
            for to_id, _ in edges:
                all_children.add(to_id)
        
        roots = []
        for node_id in self.nodes.keys():
            if node_id not in all_children:
                roots.append(node_id)
        
        return roots
    
    def calculate_node_scores(self) -> Dict[str, float]:
        """
        计算节点作为根因的分数
        
        分数基于：
        1. 出度（影响的节点数）
        2. 时间（越早越高）
        3. 入度（被影响程度，越低越可能是根因）
        """
        scores = {}
        
        for node_id, event in self.nodes.items():
            score = 0.0
            
            # 出度分数
            out_degree = len(self.edges.get(node_id, []))
            score += out_degree * 0.3
            
            # 入度分数（越低越好）
            in_degree = len(self.get_parents(node_id))
            score += (1.0 / (in_degree + 1)) * 0.3
            
            # 时间分数
            timestamps = [e.timestamp for e in self.nodes.values()]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
                time_range = (latest - earliest).total_seconds()
                if time_range > 0:
                    time_score = 1.0 - (
                        (event.timestamp - earliest).total_seconds() / time_range
                    )
                    score += time_score * 0.4
            
            scores[node_id] = score
        
        self.node_scores = scores
        return scores


class BayesianNetworkAnalyzer:
    """贝叶斯网络分析器"""
    
    # 条件概率表：P(结果 | 原因)
    CPT = {
        # 如果 Pod 崩溃，高错误率的概率
        (EventType.POD_CRASH, EventType.HIGH_ERROR_RATE): 0.8,
        # 如果数据库错误，高延迟的概率
        (EventType.DATABASE_ERROR, EventType.HIGH_LATENCY): 0.7,
        # 如果网络错误，提供者错误的概率
        (EventType.NETWORK_ERROR, EventType.PROVIDER_ERROR): 0.6,
        # 如果高延迟，熔断器打开的概率
        (EventType.HIGH_LATENCY, EventType.CIRCUIT_BREAKER_OPEN): 0.5,
    }
    
    # 先验概率：P(原因)
    PRIOR_PROBABILITIES = {
        EventType.POD_CRASH: 0.1,
        EventType.DATABASE_ERROR: 0.05,
        EventType.NETWORK_ERROR: 0.08,
        EventType.HIGH_LATENCY: 0.15,
        EventType.DEPLOYMENT_FAILURE: 0.03,
    }
    
    def calculate_posterior(
        self,
        cause_type: EventType,
        observed_effects: List[EventType]
    ) -> float:
        """
        使用贝叶斯定理计算后验概率
        
        P(原因 | 结果) = P(结果 | 原因) * P(原因) / P(结果)
        """
        # 获取先验概率
        prior = self.PRIOR_PROBABILITIES.get(cause_type, 0.1)
        
        # 计算似然 P(结果 | 原因)
        likelihood = 1.0
        for effect in observed_effects:
            prob = self.CPT.get((cause_type, effect), 0.3)
            likelihood *= prob
        
        # 简化计算：假设 P(结果) = 1
        posterior = likelihood * prior
        
        return min(posterior, 1.0)
    
    def identify_most_likely_causes(
        self,
        observed_events: List[Event]
    ) -> List[Tuple[EventType, float]]:
        """识别最可能的原因"""
        observed_types = [e.event_type for e in observed_events]
        
        candidates = []
        for cause_type in EventType:
            if cause_type not in observed_types:
                prob = self.calculate_posterior(cause_type, observed_types)
                if prob > 0.1:  # 只返回概率大于 10% 的
                    candidates.append((cause_type, prob))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)


class DecisionTreeAnalyzer:
    """决策树分析器"""
    
    # 决策规则
    DECISION_RULES = [
        {
            "name": "Pod Crash Root Cause",
            "conditions": [
                lambda e: e.event_type == EventType.POD_CRASH,
                lambda e: e.attributes.get("restart_count", 0) > 3,
            ],
            "root_cause": "OOM_KILLED",
            "confidence": 0.85,
        },
        {
            "name": "Database Connection Issue",
            "conditions": [
                lambda e: e.event_type == EventType.DATABASE_ERROR,
                lambda e: "connection" in e.message.lower(),
            ],
            "root_cause": "DB_CONNECTION_POOL_EXHAUSTED",
            "confidence": 0.8,
        },
        {
            "name": "Network Partition",
            "conditions": [
                lambda e: e.event_type == EventType.NETWORK_ERROR,
                lambda e: e.attributes.get("timeout_count", 0) > 5,
            ],
            "root_cause": "NETWORK_PARTITION",
            "confidence": 0.75,
        },
        {
            "name": "Deployment Issue",
            "conditions": [
                lambda e: e.event_type == EventType.DEPLOYMENT_FAILURE,
                lambda e: e.severity == EventSeverity.CRITICAL,
            ],
            "root_cause": "DEPLOYMENT_ROLLBACK_NEEDED",
            "confidence": 0.9,
        },
    ]
    
    def analyze_event(self, event: Event) -> Optional[Tuple[str, float]]:
        """使用决策规则分析单个事件"""
        for rule in self.DECISION_RULES:
            if all(condition(event) for condition in rule["conditions"]):
                return (rule["root_cause"], rule["confidence"])
        return None
    
    def analyze_events(
        self,
        events: List[Event]
    ) -> Dict[str, List[Tuple[Event, float]]]:
        """分析事件列表，返回根因分类"""
        results = defaultdict(list)
        
        for event in events:
            result = self.analyze_event(event)
            if result:
                root_cause, confidence = result
                results[root_cause].append((event, confidence))
        
        return dict(results)


class RootCauseIdentifier:
    """
    根因识别器
    
    综合使用多种方法识别根本原因
    """
    
    def __init__(self):
        self.correlation_analyzer = CorrelationAnalyzer()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.bayesian_analyzer = BayesianNetworkAnalyzer()
        self.decision_tree_analyzer = DecisionTreeAnalyzer()
    
    def identify_root_cause(
        self,
        events: List[Event]
    ) -> RCAResult:
        """
        识别根本原因
        
        使用多种方法综合分析：
        1. 因果图分析
        2. 贝叶斯网络
        3. 决策树规则
        4. 时间序列分析
        """
        if not events:
            return RCAResult(
                id=f"rca-{uuid.uuid4().hex[:8]}",
                timestamp=datetime.utcnow(),
                root_causes=[],
                contributing_factors=[],
                affected_services=[],
                timeline=[],
                confidence=0.0,
                recommended_actions=[],
                analysis_method="none",
            )
        
        # 构建因果图
        causal_graph = self._build_causal_graph(events)
        
        # 方法 1: 因果图分析
        graph_roots = self._analyze_causal_graph(causal_graph)
        
        # 方法 2: 贝叶斯网络
        bayesian_results = self.bayesian_analyzer.identify_most_likely_causes(events)
        
        # 方法 3: 决策树
        dt_results = self.decision_tree_analyzer.analyze_events(events)
        
        # 方法 4: 时间序列分析
        temporal_roots = self._analyze_temporal_sequence(events)
        
        # 综合结果
        root_causes = self._combine_results(
            events,
            graph_roots,
            bayesian_results,
            dt_results,
            temporal_roots
        )
        
        # 识别受影响的服务
        affected_services = self._identify_affected_services(events)
        
        # 生成推荐动作
        recommended_actions = self._generate_recommendations(root_causes)
        
        # 计算总体置信度
        confidence = (
            sum(rc.confidence for rc in root_causes) / len(root_causes)
            if root_causes else 0.0
        )
        
        # 按时间排序
        timeline = sorted(events, key=lambda e: e.timestamp)
        
        return RCAResult(
            id=f"rca-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow(),
            root_causes=root_causes,
            contributing_factors=[e for e in events if e not in [rc.event for rc in root_causes]],
            affected_services=affected_services,
            timeline=timeline,
            confidence=confidence,
            recommended_actions=recommended_actions,
            analysis_method="combined",
        )
    
    def _build_causal_graph(self, events: List[Event]) -> CausalGraph:
        """构建因果图"""
        graph = CausalGraph()
        
        # 添加所有节点
        for event in events:
            graph.add_node(event)
        
        # 添加因果边
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                is_causal, confidence, _ = self.causal_analyzer.infer_causal_relationship(
                    e1, e2
                )
                if is_causal:
                    graph.add_edge(e1.id, e2.id, confidence)
        
        return graph
    
    def _analyze_causal_graph(
        self,
        graph: CausalGraph
    ) -> List[Tuple[Event, float]]:
        """分析因果图找出根因候选"""
        # 计算节点分数
        scores = graph.calculate_node_scores()
        
        # 获取根节点
        root_nodes = graph.get_root_nodes()
        
        # 结合分数排序
        candidates = []
        for node_id in root_nodes:
            event = graph.nodes[node_id]
            score = scores.get(node_id, 0.0)
            candidates.append((event, score))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)
    
    def _analyze_temporal_sequence(
        self,
        events: List[Event]
    ) -> List[Tuple[Event, float]]:
        """分析时间序列找出最早的异常事件"""
        if not events:
            return []
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # 找出最早的几个事件作为候选
        candidates = []
        for i, event in enumerate(sorted_events[:3]):  # 取前 3 个
            # 越早分数越高
            score = 1.0 - (i * 0.2)
            candidates.append((event, score))
        
        return candidates
    
    def _combine_results(
        self,
        events: List[Event],
        graph_roots: List[Tuple[Event, float]],
        bayesian_results: List[Tuple[EventType, float]],
        dt_results: Dict[str, List[Tuple[Event, float]]],
        temporal_roots: List[Tuple[Event, float]],
    ) -> List[RootCause]:
        """综合多种方法的结果"""
        # 收集所有候选事件及其分数
        event_scores = defaultdict(list)
        
        # 因果图结果
        for event, score in graph_roots:
            event_scores[event.id].append(score)
        
        # 贝叶斯结果
        for event_type, score in bayesian_results:
            for event in events:
                if event.event_type == event_type:
                    event_scores[event.id].append(score)
        
        # 决策树结果
        for root_cause, event_list in dt_results.items():
            for event, score in event_list:
                event_scores[event.id].append(score)
        
        # 时间序列结果
        for event, score in temporal_roots:
            event_scores[event.id].append(score)
        
        # 计算平均分数并排序
        final_scores = []
        for event_id, scores in event_scores.items():
            avg_score = sum(scores) / len(scores)
            event = next(e for e in events if e.id == event_id)
            final_scores.append((event, avg_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建 RootCause 对象
        root_causes = []
        for event, score in final_scores[:3]:  # 取前 3 个
            evidence = self._generate_evidence(event, events)
            affected = self._get_affected_services(event, events)
            cascading = self._get_cascading_impact(event, events)
            
            root_causes.append(RootCause(
                event=event,
                confidence=score,
                evidence=evidence,
                affected_services=affected,
                cascading_impact=cascading,
            ))
        
        return root_causes
    
    def _generate_evidence(
        self,
        root_event: Event,
        all_events: List[Event]
    ) -> List[str]:
        """生成证据列表"""
        evidence = []
        
        # 时间证据
        time_order = sorted(all_events, key=lambda e: e.timestamp)
        if time_order[0].id == root_event.id:
            evidence.append("First event in the incident timeline")
        
        # 因果证据
        for event in all_events:
            if event.id != root_event.id:
                is_causal, confidence, explanation = self.causal_analyzer.infer_causal_relationship(
                    root_event, event
                )
                if is_causal:
                    evidence.append(f"Causally linked to {event.event_type.value} (confidence: {confidence:.2f})")
        
        # 属性证据
        if root_event.attributes:
            for key, value in root_event.attributes.items():
                if key in ["error_type", "service", "metric_name"]:
                    evidence.append(f"{key}: {value}")
        
        return evidence
    
    def _identify_affected_services(
        self,
        events: List[Event]
    ) -> List[str]:
        """识别受影响的服务"""
        services = set()
        
        for event in events:
            service = event.attributes.get("service")
            if service:
                services.add(service)
            
            # 从来源提取服务名
            if "." in event.source:
                services.add(event.source.split(".")[0])
        
        return list(services)
    
    def _get_affected_services(
        self,
        root_event: Event,
        all_events: List[Event]
    ) -> List[str]:
        """获取根因影响的服务"""
        services = set()
        
        # 直接属性
        service = root_event.attributes.get("service")
        if service:
            services.add(service)
        
        # 从因果相关事件中提取
        for event in all_events:
            is_causal, _, _ = self.causal_analyzer.infer_causal_relationship(
                root_event, event
            )
            if is_causal:
                service = event.attributes.get("service")
                if service:
                    services.add(service)
        
        return list(services)
    
    def _get_cascading_impact(
        self,
        root_event: Event,
        all_events: List[Event]
    ) -> List[str]:
        """获取级联影响"""
        impacts = []
        
        for event in all_events:
            if event.id == root_event.id:
                continue
            
            is_causal, confidence, _ = self.causal_analyzer.infer_causal_relationship(
                root_event, event
            )
            if is_causal:
                impacts.append(f"{event.event_type.value} (confidence: {confidence:.2f})")
        
        return impacts
    
    def _generate_recommendations(
        self,
        root_causes: List[RootCause]
    ) -> List[str]:
        """生成推荐动作"""
        recommendations = []
        
        for root_cause in root_causes:
            event = root_cause.event
            
            # 根据事件类型生成推荐
            if event.event_type == EventType.POD_CRASH:
                recommendations.append(
                    f"Restart pod and investigate OOM cause for {event.source}"
                )
            
            elif event.event_type == EventType.DATABASE_ERROR:
                recommendations.append(
                    "Check database connection pool settings and increase if needed"
                )
            
            elif event.event_type == EventType.NETWORK_ERROR:
                recommendations.append(
                    "Investigate network connectivity and DNS resolution"
                )
            
            elif event.event_type == EventType.HIGH_LATENCY:
                recommendations.append(
                    "Scale up service instances or optimize database queries"
                )
            
            elif event.event_type == EventType.DEPLOYMENT_FAILURE:
                recommendations.append(
                    "Consider rolling back to previous stable version"
                )
            
            elif event.event_type == EventType.CIRCUIT_BREAKER_OPEN:
                recommendations.append(
                    "Check downstream service health and reduce timeout thresholds"
                )
        
        # 添加通用推荐
        if not recommendations:
            recommendations.append("Investigate logs and metrics for more details")
        
        recommendations.append("Set up alerts to detect similar issues early")
        
        return recommendations
