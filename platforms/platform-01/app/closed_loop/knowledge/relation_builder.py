"""
关系构建器
构建实体之间的关系，包括依赖关系、调用关系、因果关系等
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """关系"""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'properties': self.properties,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class DependencyDiscoverer:
    """依赖关系发现器"""
    
    def __init__(self):
        self.call_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.timing_correlations: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
    def add_call_record(self, caller: str, callee: str, timestamp: datetime = None):
        """添加调用记录"""
        self.call_graph[caller][callee] += 1
        
    def discover_from_logs(self, log_entries: List[Dict[str, Any]]) -> List[Relationship]:
        """从日志中发现依赖关系"""
        relationships = []
        
        # 分析日志序列
        for i, log in enumerate(log_entries):
            service = log.get('service', '')
            message = log.get('message', '')
            timestamp = log.get('timestamp', datetime.now())
            
            # 查找调用模式
            if 'calling' in message.lower() or 'request to' in message.lower():
                # 提取被调用服务
                for other_log in log_entries[i+1:i+5]:  # 查看后续日志
                    other_service = other_log.get('service', '')
                    if other_service and other_service != service:
                        time_diff = (other_log.get('timestamp', timestamp) - timestamp).total_seconds()
                        if 0 < time_diff < 5:  # 5秒内响应
                            self.call_graph[service][other_service] += 1
        
        # 生成关系
        for caller, callees in self.call_graph.items():
            total_calls = sum(callees.values())
            for callee, count in callees.items():
                confidence = min(count / 10, 1.0)  # 至少10次调用达到最高置信度
                relationships.append(Relationship(
                    source_id=caller,
                    target_id=callee,
                    relation_type='CALLS',
                    properties={
                        'call_count': count,
                        'frequency': count / total_calls if total_calls > 0 else 0
                    },
                    confidence=confidence,
                    timestamp=datetime.now()
                ))
        
        return relationships
    
    def discover_from_traces(self, traces: List[Dict[str, Any]]) -> List[Relationship]:
        """从分布式追踪中发现依赖关系"""
        relationships = []
        
        for trace in traces:
            spans = trace.get('spans', [])
            
            # 构建 span 父子关系
            span_map = {span['id']: span for span in spans}
            
            for span in spans:
                parent_id = span.get('parent_id')
                if parent_id and parent_id in span_map:
                    parent = span_map[parent_id]
                    
                    relationships.append(Relationship(
                        source_id=parent.get('service', ''),
                        target_id=span.get('service', ''),
                        relation_type='CALLS',
                        properties={
                            'latency': span.get('duration', 0),
                            'trace_id': trace.get('id')
                        },
                        confidence=0.9,
                        timestamp=datetime.now()
                    ))
        
        return relationships
    
    def discover_from_config(self, config: Dict[str, Any]) -> List[Relationship]:
        """从配置中发现依赖关系"""
        relationships = []
        
        # 解析服务配置中的依赖声明
        for service_name, service_config in config.get('services', {}).items():
            dependencies = service_config.get('depends_on', [])
            
            for dep in dependencies:
                relationships.append(Relationship(
                    source_id=service_name,
                    target_id=dep,
                    relation_type='DEPENDS_ON',
                    properties={'source': 'config'},
                    confidence=1.0,
                    timestamp=datetime.now()
                ))
        
        return relationships


class CausalityAnalyzer:
    """因果关系分析器"""
    
    def __init__(self, time_window_seconds: float = 60.0):
        self.time_window = time_window_seconds
        self.event_sequences: List[List[Dict[str, Any]]] = []
        
    def add_event_sequence(self, events: List[Dict[str, Any]]):
        """添加事件序列"""
        self.event_sequences.append(events)
        
    def analyze_temporal_causality(self, events: List[Dict[str, Any]]) -> List[Relationship]:
        """分析时间因果关系"""
        relationships = []
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', datetime.now()))
        
        # 分析事件对
        for i, event1 in enumerate(sorted_events):
            for event2 in sorted_events[i+1:]:
                time_diff = (event2.get('timestamp', datetime.now()) - 
                           event1.get('timestamp', datetime.now())).total_seconds()
                
                # 在时间窗口内
                if 0 < time_diff < self.time_window:
                    # 计算因果分数
                    causality_score = self._calculate_causality_score(event1, event2, time_diff)
                    
                    if causality_score > 0.5:  # 阈值
                        relationships.append(Relationship(
                            source_id=event1.get('entity_id', event1.get('id', '')),
                            target_id=event2.get('entity_id', event2.get('id', '')),
                            relation_type='CAUSES',
                            properties={
                                'time_difference': time_diff,
                                'causality_score': causality_score
                            },
                            confidence=causality_score,
                            timestamp=datetime.now()
                        ))
        
        return relationships
    
    def _calculate_causality_score(self, event1: Dict, event2: Dict, time_diff: float) -> float:
        """计算因果分数"""
        score = 0.0
        
        # 时间接近度 (指数衰减)
        time_score = np.exp(-time_diff / self.time_window)
        score += 0.4 * time_score
        
        # 事件类型相关性
        type1 = event1.get('type', '')
        type2 = event2.get('type', '')
        
        if type1 == 'error' and type2 == 'error':
            score += 0.3  # 错误传播
        elif type1 == 'deployment' and type2 == 'error':
            score += 0.4  # 部署导致错误
        elif type1 == 'scaling' and type2 == 'performance_improvement':
            score += 0.3  # 扩容改善性能
        
        # 服务相关性
        service1 = event1.get('service', '')
        service2 = event2.get('service', '')
        
        if service1 and service2:
            if service1 == service2:
                score += 0.3  # 同一服务
            else:
                score += 0.1  # 可能相关服务
        
        return min(score, 1.0)
    
    def discover_granger_causality(self, time_series: Dict[str, List[float]], 
                                   max_lag: int = 5) -> List[Relationship]:
        """发现 Granger 因果关系"""
        relationships = []
        
        services = list(time_series.keys())
        
        for i, service1 in enumerate(services):
            for service2 in services[i+1:]:
                # 计算 Granger 因果
                causality_1_to_2 = self._granger_test(
                    time_series[service1], 
                    time_series[service2], 
                    max_lag
                )
                causality_2_to_1 = self._granger_test(
                    time_series[service2], 
                    time_series[service1], 
                    max_lag
                )
                
                if causality_1_to_2 > 0.5:
                    relationships.append(Relationship(
                        source_id=service1,
                        target_id=service2,
                        relation_type='GRANGER_CAUSES',
                        properties={'granger_score': causality_1_to_2},
                        confidence=causality_1_to_2,
                        timestamp=datetime.now()
                    ))
                
                if causality_2_to_1 > 0.5:
                    relationships.append(Relationship(
                        source_id=service2,
                        target_id=service1,
                        relation_type='GRANGER_CAUSES',
                        properties={'granger_score': causality_2_to_1},
                        confidence=causality_2_to_1,
                        timestamp=datetime.now()
                    ))
        
        return relationships
    
    def _granger_test(self, x: List[float], y: List[float], max_lag: int) -> float:
        """简化的 Granger 因果检验"""
        if len(x) < max_lag + 2 or len(y) < max_lag + 2:
            return 0.0
        
        # 使用相关性作为简化的 Granger 因果度量
        correlations = []
        for lag in range(1, max_lag + 1):
            if len(x) > lag and len(y) > lag:
                corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return max(correlations) if correlations else 0.0


class SimilarityCalculator:
    """相似度计算器"""
    
    def __init__(self):
        self.entity_features: Dict[str, np.ndarray] = {}
        
    def add_entity_features(self, entity_id: str, features: np.ndarray):
        """添加实体特征"""
        self.entity_features[entity_id] = features
        
    def calculate_cosine_similarity(self, entity1_id: str, entity2_id: str) -> float:
        """计算余弦相似度"""
        f1 = self.entity_features.get(entity1_id)
        f2 = self.entity_features.get(entity2_id)
        
        if f1 is None or f2 is None:
            return 0.0
        
        dot_product = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_entities(self, entity_id: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """查找相似实体"""
        similarities = []
        
        for other_id, features in self.entity_features.items():
            if other_id != entity_id:
                sim = self.calculate_cosine_similarity(entity_id, other_id)
                if sim >= threshold:
                    similarities.append({
                        'entity_id': other_id,
                        'similarity': sim
                    })
        
        # 排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities


class RelationBuilder:
    """
    关系构建器主类
    整合依赖发现、因果分析、相似度计算
    """
    
    def __init__(self):
        self.dependency_discoverer = DependencyDiscoverer()
        self.causality_analyzer = CausalityAnalyzer()
        self.similarity_calculator = SimilarityCalculator()
        
        self.relationships: Dict[str, Relationship] = {}
        
    def build_relations(self, entities: List[Any], 
                       logs: List[Dict] = None,
                       traces: List[Dict] = None,
                       config: Dict = None) -> List[Relationship]:
        """构建所有关系"""
        all_relationships = []
        
        # 从日志发现依赖
        if logs:
            dep_relations = self.dependency_discoverer.discover_from_logs(logs)
            all_relationships.extend(dep_relations)
        
        # 从追踪发现依赖
        if traces:
            trace_relations = self.dependency_discoverer.discover_from_traces(traces)
            all_relationships.extend(trace_relations)
        
        # 从配置发现依赖
        if config:
            config_relations = self.dependency_discoverer.discover_from_config(config)
            all_relationships.extend(config_relations)
        
        # 去重和存储
        for rel in all_relationships:
            rel_id = f"{rel.source_id}:{rel.relation_type}:{rel.target_id}"
            if rel_id in self.relationships:
                # 更新置信度
                existing = self.relationships[rel_id]
                existing.confidence = max(existing.confidence, rel.confidence)
                existing.properties.update(rel.properties)
            else:
                self.relationships[rel_id] = rel
        
        return list(self.relationships.values())
    
    def add_causality_relations(self, events: List[Dict[str, Any]]) -> List[Relationship]:
        """添加因果关系"""
        causality_rels = self.causality_analyzer.analyze_temporal_causality(events)
        
        for rel in causality_rels:
            rel_id = f"{rel.source_id}:{rel.relation_type}:{rel.target_id}"
            self.relationships[rel_id] = rel
        
        return causality_rels
    
    def add_similarity_relations(self, entity_id: str, 
                                 threshold: float = 0.7) -> List[Relationship]:
        """添加相似关系"""
        similar = self.similarity_calculator.find_similar_entities(entity_id, threshold)
        
        relationships = []
        for sim in similar:
            rel = Relationship(
                source_id=entity_id,
                target_id=sim['entity_id'],
                relation_type='SIMILAR_TO',
                properties={'similarity_score': sim['similarity']},
                confidence=sim['similarity'],
                timestamp=datetime.now()
            )
            rel_id = f"{rel.source_id}:{rel.relation_type}:{rel.target_id}"
            self.relationships[rel_id] = rel
            relationships.append(rel)
        
        return relationships
    
    def get_relationships(self, entity_id: str = None, 
                         relation_type: str = None) -> List[Relationship]:
        """获取关系"""
        result = list(self.relationships.values())
        
        if entity_id:
            result = [r for r in result if r.source_id == entity_id or r.target_id == entity_id]
        
        if relation_type:
            result = [r for r in result if r.relation_type == relation_type]
        
        return result
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取子图"""
        nodes = {entity_id}
        edges = []
        
        current_level = {entity_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                rels = self.get_relationships(entity_id=node)
                for rel in rels:
                    other = rel.target_id if rel.source_id == node else rel.source_id
                    nodes.add(other)
                    next_level.add(other)
                    edges.append(rel.to_dict())
            current_level = next_level
        
        return {
            'nodes': list(nodes),
            'edges': edges,
            'center': entity_id,
            'depth': depth
        }
