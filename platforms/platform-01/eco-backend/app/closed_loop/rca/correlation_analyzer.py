# =============================================================================
# RCA Correlation Analyzer
# =============================================================================
# 根因分析 - 关联分析器
# 分析事件之间的关联关系
# =============================================================================

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from app.closed_loop.rca.event_collector import Event, EventType, EventSeverity


@dataclass
class CorrelationResult:
    """关联分析结果"""
    event_id: str
    correlated_events: List[Tuple[str, float]]  # (event_id, correlation_score)
    correlation_type: str  # temporal, causal, attribute
    confidence: float
    explanation: str


@dataclass
class EventCluster:
    """事件簇 (高度相关的事件集合)"""
    id: str
    events: List[Event]
    centroid: datetime  # 时间中心点
    common_attributes: Dict[str, Any]
    correlation_strength: float
    
    @property
    def time_span_seconds(self) -> float:
        """时间跨度"""
        if len(self.events) < 2:
            return 0.0
        timestamps = [e.timestamp for e in self.events]
        return (max(timestamps) - min(timestamps)).total_seconds()
    
    @property
    def primary_event(self) -> Optional[Event]:
        """主要事件 (最早发生的)"""
        if not self.events:
            return None
        return min(self.events, key=lambda e: e.timestamp)


class TemporalCorrelationAnalyzer:
    """时间关联分析器"""
    
    def __init__(self, time_window_seconds: int = 300):
        self.time_window_seconds = time_window_seconds
    
    def find_temporally_related(
        self,
        target_event: Event,
        events: List[Event]
    ) -> List[Tuple[Event, float]]:
        """
        查找时间相关的事件
        
        Returns:
            相关事件列表，每个事件带有时间关联分数
        """
        related = []
        
        for event in events:
            if event.id == target_event.id:
                continue
            
            time_diff = abs(
                (event.timestamp - target_event.timestamp).total_seconds()
            )
            
            if time_diff <= self.time_window_seconds:
                # 时间越近，关联度越高
                score = 1.0 - (time_diff / self.time_window_seconds)
                related.append((event, score))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def detect_cascade_pattern(
        self,
        events: List[Event]
    ) -> Optional[List[Event]]:
        """
        检测级联故障模式
        
        级联模式：事件按时间顺序发生，且每个事件可能触发下一个
        """
        if len(events) < 3:
            return None
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # 检查时间间隔是否递减（级联特征）
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = (
                sorted_events[i].timestamp - sorted_events[i-1].timestamp
            ).total_seconds()
            intervals.append(interval)
        
        # 如果间隔相对稳定，可能是级联
        if len(intervals) >= 2:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # 变异系数小于 0.5 认为是稳定的
            if mean_interval > 0 and std_interval / mean_interval < 0.5:
                return sorted_events
        
        return None


class AttributeCorrelationAnalyzer:
    """属性关联分析器"""
    
    def find_attribute_matches(
        self,
        target_event: Event,
        events: List[Event],
        key_attributes: Optional[List[str]] = None
    ) -> List[Tuple[Event, float]]:
        """
        查找属性匹配的事件
        
        Args:
            key_attributes: 关键属性列表，如果为 None 则比较所有属性
        """
        related = []
        
        attrs_to_compare = key_attributes or list(target_event.attributes.keys())
        
        for event in events:
            if event.id == target_event.id:
                continue
            
            matches = 0
            total = 0
            
            for attr in attrs_to_compare:
                total += 1
                if attr in target_event.attributes and attr in event.attributes:
                    if target_event.attributes[attr] == event.attributes[attr]:
                        matches += 1
            
            if total > 0:
                score = matches / total
                if score > 0.5:  # 至少 50% 匹配
                    related.append((event, score))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def extract_common_attributes(
        self,
        events: List[Event]
    ) -> Dict[str, Any]:
        """提取事件的共同属性"""
        if not events:
            return {}
        
        common = {}
        
        # 获取第一个事件的所有属性
        first_attrs = events[0].attributes
        
        for key, value in first_attrs.items():
            # 检查所有事件是否都有这个属性且值相同
            if all(
                key in e.attributes and e.attributes[key] == value
                for e in events[1:]
            ):
                common[key] = value
        
        return common


class CausalInferenceAnalyzer:
    """因果推断分析器"""
    
    # 因果规则：如果 A 发生，可能导致 B
    CAUSAL_RULES = {
        EventType.POD_CRASH: [
            EventType.HIGH_ERROR_RATE,
            EventType.CIRCUIT_BREAKER_OPEN,
        ],
        EventType.DATABASE_ERROR: [
            EventType.HIGH_LATENCY,
            EventType.PROVIDER_ERROR,
        ],
        EventType.NETWORK_ERROR: [
            EventType.PROVIDER_ERROR,
            EventType.TRACE_SPAN_ERROR,
        ],
        EventType.HIGH_LATENCY: [
            EventType.CIRCUIT_BREAKER_OPEN,
        ],
    }
    
    def infer_causal_relationship(
        self,
        cause_event: Event,
        effect_event: Event
    ) -> Tuple[bool, float, str]:
        """
        推断两个事件之间的因果关系
        
        Returns:
            (是否存在因果关系, 置信度, 解释)
        """
        # 检查时间顺序：原因必须在结果之前
        if cause_event.timestamp >= effect_event.timestamp:
            return False, 0.0, "Cause must occur before effect"
        
        # 检查类型关系
        possible_effects = self.CAUSAL_RULES.get(cause_event.event_type, [])
        
        if effect_event.event_type not in possible_effects:
            return False, 0.0, "No known causal relationship"
        
        # 计算时间间隔置信度
        time_diff = (
            effect_event.timestamp - cause_event.timestamp
        ).total_seconds()
        
        # 时间间隔在 0-60 秒内置信度最高
        if time_diff <= 60:
            time_confidence = 1.0
        elif time_diff <= 300:
            time_confidence = 0.7
        elif time_diff <= 600:
            time_confidence = 0.4
        else:
            time_confidence = 0.1
        
        # 检查属性关联
        common_attrs = set(cause_event.attributes.keys()) & set(effect_event.attributes.keys())
        attr_confidence = len(common_attrs) / max(len(cause_event.attributes), 1)
        
        # 综合置信度
        confidence = (time_confidence + attr_confidence) / 2
        
        explanation = (
            f"{cause_event.event_type.value} commonly causes "
            f"{effect_event.event_type.value} within {time_diff:.0f}s"
        )
        
        return True, confidence, explanation
    
    def build_causal_chain(
        self,
        events: List[Event]
    ) -> List[Tuple[Event, Event, float]]:
        """
        构建因果链
        
        Returns:
            因果链列表，每个元素为 (原因事件, 结果事件, 置信度)
        """
        chain = []
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        for i, cause in enumerate(sorted_events):
            for effect in sorted_events[i+1:]:
                is_causal, confidence, _ = self.infer_causal_relationship(
                    cause, effect
                )
                if is_causal and confidence > 0.5:
                    chain.append((cause, effect, confidence))
        
        return sorted(chain, key=lambda x: x[2], reverse=True)


class CorrelationAnalyzer:
    """
    关联分析器
    
    综合分析事件之间的时间、属性和因果关系
    """
    
    def __init__(
        self,
        time_window_seconds: int = 300,
        correlation_threshold: float = 0.5
    ):
        self.temporal_analyzer = TemporalCorrelationAnalyzer(time_window_seconds)
        self.attribute_analyzer = AttributeCorrelationAnalyzer()
        self.causal_analyzer = CausalInferenceAnalyzer()
        self.correlation_threshold = correlation_threshold
    
    def analyze_correlation(
        self,
        target_event: Event,
        events: List[Event]
    ) -> CorrelationResult:
        """
        分析目标事件与其他事件的关联关系
        """
        correlated = []
        
        # 时间关联
        temporal_related = self.temporal_analyzer.find_temporally_related(
            target_event, events
        )
        
        # 属性关联
        attribute_related = self.attribute_analyzer.find_attribute_matches(
            target_event, events
        )
        
        # 因果推断
        for event in events:
            if event.id == target_event.id:
                continue
            
            is_causal, confidence, explanation = (
                self.causal_analyzer.infer_causal_relationship(
                    target_event, event
                )
                if target_event.timestamp < event.timestamp
                else self.causal_analyzer.infer_causal_relationship(
                    event, target_event
                )
            )
            
            if is_causal and confidence > self.correlation_threshold:
                correlated.append((event.id, confidence, "causal", explanation))
        
        # 合并结果
        for event, score in temporal_related:
            if score > self.correlation_threshold:
                correlated.append((event.id, score, "temporal", "Time proximity"))
        
        for event, score in attribute_related:
            if score > self.correlation_threshold:
                correlated.append((event.id, score, "attribute", "Attribute match"))
        
        # 去重并取最高分数
        best_scores = {}
        for event_id, score, corr_type, explanation in correlated:
            if event_id not in best_scores or best_scores[event_id][0] < score:
                best_scores[event_id] = (score, corr_type, explanation)
        
        # 构建结果
        correlated_list = [
            (event_id, score) for event_id, (score, _, _) in best_scores.items()
        ]
        correlated_list.sort(key=lambda x: x[1], reverse=True)
        
        # 确定主要关联类型
        type_counts = defaultdict(int)
        for _, (_, corr_type, _) in best_scores.items():
            type_counts[corr_type] += 1
        
        primary_type = max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else "unknown"
        
        # 计算总体置信度
        confidence = (
            sum(score for _, score in correlated_list) / len(correlated_list)
            if correlated_list else 0.0
        )
        
        return CorrelationResult(
            event_id=target_event.id,
            correlated_events=correlated_list,
            correlation_type=primary_type,
            confidence=confidence,
            explanation=f"Found {len(correlated_list)} correlated events via {primary_type} analysis"
        )
    
    def cluster_events(
        self,
        events: List[Event],
        time_window_seconds: int = 300
    ) -> List[EventCluster]:
        """
        将事件聚类成簇
        """
        if not events:
            return []
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            time_diff = (
                sorted_events[i].timestamp - sorted_events[i-1].timestamp
            ).total_seconds()
            
            if time_diff <= time_window_seconds:
                current_cluster.append(sorted_events[i])
            else:
                # 保存当前簇
                if current_cluster:
                    clusters.append(self._create_cluster(current_cluster))
                current_cluster = [sorted_events[i]]
        
        # 保存最后一个簇
        if current_cluster:
            clusters.append(self._create_cluster(current_cluster))
        
        return clusters
    
    def _create_cluster(self, events: List[Event]) -> EventCluster:
        """从事件列表创建簇"""
        import uuid
        
        timestamps = [e.timestamp for e in events]
        centroid = timestamps[len(timestamps) // 2]
        
        common_attrs = self.attribute_analyzer.extract_common_attributes(events)
        
        # 计算关联强度
        correlation_scores = []
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                result = self.analyze_correlation(e1, [e2])
                correlation_scores.append(result.confidence)
        
        strength = np.mean(correlation_scores) if correlation_scores else 0.0
        
        return EventCluster(
            id=f"cluster-{uuid.uuid4().hex[:8]}",
            events=events,
            centroid=centroid,
            common_attributes=common_attrs,
            correlation_strength=strength,
        )
    
    def find_root_cause_candidates(
        self,
        events: List[Event]
    ) -> List[Tuple[Event, float]]:
        """
        查找可能的根因事件候选
        
        根因特征：
        1. 时间最早
        2. 被其他事件因果关联
        3. 高严重级别
        """
        if not events:
            return []
        
        candidates = []
        
        for event in events:
            score = 0.0
            
            # 时间分数 (越早越高)
            time_rank = sorted(events, key=lambda e: e.timestamp).index(event)
            time_score = 1.0 - (time_rank / len(events))
            score += time_score * 0.3
            
            # 被因果关联分数
            causal_count = 0
            for other in events:
                if other.id != event.id:
                    is_causal, confidence, _ = self.causal_analyzer.infer_causal_relationship(
                        event, other
                    )
                    if is_causal:
                        causal_count += 1
            
            causal_score = causal_count / max(len(events) - 1, 1)
            score += causal_score * 0.4
            
            # 严重级别分数
            severity_scores = {
                EventSeverity.INFO: 0.1,
                EventSeverity.WARNING: 0.3,
                EventSeverity.ERROR: 0.6,
                EventSeverity.CRITICAL: 1.0,
            }
            severity_score = severity_scores.get(event.severity, 0.0)
            score += severity_score * 0.3
            
            candidates.append((event, score))
        
        return sorted(candidates, key=lambda x: x[1], reverse=True)
