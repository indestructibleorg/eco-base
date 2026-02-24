"""
关联分析器
分析事件之间的关联关系
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .event_collector import Event, EventType, EventCollector

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """关联分析结果"""
    primary_event_id: str
    related_events: List[Tuple[str, float]]  # (event_id, correlation_score)
    correlation_type: str
    confidence: float
    time_window: Tuple[datetime, datetime]
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def get_high_correlation_events(self, threshold: float = 0.7) -> List[str]:
        """获取高关联度事件"""
        return [eid for eid, score in self.related_events if score >= threshold]


@dataclass
class EventCluster:
    """事件簇"""
    cluster_id: str
    events: List[str]
    centroid_time: datetime
    dominant_source: str
    dominant_type: EventType
    severity_score: float
    created_at: datetime = field(default_factory=datetime.now)


class CorrelationAnalyzer:
    """关联分析器"""
    
    def __init__(self, event_collector: EventCollector = None, config: Optional[Dict] = None):
        self.event_collector = event_collector
        self.config = config or {}
        
        # 配置参数
        self.time_window_minutes = self.config.get('time_window_minutes', 30)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.5)
        self.max_distance_seconds = self.config.get('max_distance_seconds', 300)
        
        # 历史关联数据
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.event_clusters: Dict[str, EventCluster] = {}
        
        logger.info("关联分析器初始化完成")
    
    def analyze_temporal_correlation(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析时间关联 (简化接口)"""
        if len(events) < 2:
            return []
        
        results = []
        
        # 按时间排序
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', datetime.now()))
        
        # 简单的时间窗口关联
        for i, event1 in enumerate(sorted_events):
            for event2 in sorted_events[i+1:]:
                time_diff = abs((event2.get('timestamp', datetime.now()) - 
                                event1.get('timestamp', datetime.now())).total_seconds())
                
                if time_diff < self.max_distance_seconds:
                    correlation_score = 1.0 - (time_diff / self.max_distance_seconds)
                    results.append({
                        'event1': event1.get('id'),
                        'event2': event2.get('id'),
                        'correlation_score': correlation_score,
                        'time_diff_seconds': time_diff
                    })
        
        return results
    
    def analyze(self, event_id: str) -> CorrelationResult:
        """分析单个事件的关联"""
        event = self.event_collector.get_event(event_id)
        if not event:
            logger.warning(f"事件不存在: {event_id}")
            return CorrelationResult(
                primary_event_id=event_id,
                related_events=[],
                correlation_type="none",
                confidence=0.0,
                time_window=(datetime.now(), datetime.now())
            )
        
        # 获取时间窗口内的事件
        start_time = event.timestamp - timedelta(minutes=self.time_window_minutes)
        end_time = event.timestamp + timedelta(minutes=self.time_window_minutes)
        
        candidate_events = self.event_collector.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # 排除自己
        candidate_events = [e for e in candidate_events if e.event_id != event_id]
        
        if not candidate_events:
            return CorrelationResult(
                primary_event_id=event_id,
                related_events=[],
                correlation_type="none",
                confidence=1.0,
                time_window=(start_time, end_time)
            )
        
        # 计算关联分数
        correlations = []
        for candidate in candidate_events:
            score = self._calculate_correlation(event, candidate)
            if score >= self.correlation_threshold:
                correlations.append((candidate.event_id, score))
        
        # 排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 确定关联类型
        correlation_type = self._determine_correlation_type(event, correlations)
        
        # 计算置信度
        confidence = self._calculate_confidence(correlations)
        
        result = CorrelationResult(
            primary_event_id=event_id,
            related_events=correlations[:20],  # 最多返回20个
            correlation_type=correlation_type,
            confidence=confidence,
            time_window=(start_time, end_time)
        )
        
        # 存储关联结果
        for related_id, score in correlations:
            self.correlation_matrix[(event_id, related_id)] = score
        
        logger.info(f"关联分析完成: {event_id}, 找到 {len(correlations)} 个关联事件")
        return result
    
    def _calculate_correlation(self, event1: Event, event2: Event) -> float:
        """计算两个事件的关联分数"""
        scores = []
        
        # 1. 时间邻近性 (权重: 0.3)
        time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
        if time_diff > self.max_distance_seconds:
            time_score = 0.0
        else:
            time_score = 1.0 - (time_diff / self.max_distance_seconds)
        scores.append(time_score * 0.3)
        
        # 2. 来源相似性 (权重: 0.25)
        if event1.source == event2.source:
            source_score = 1.0
        elif event1.source.split('.')[0] == event2.source.split('.')[0]:
            source_score = 0.5
        else:
            source_score = 0.0
        scores.append(source_score * 0.25)
        
        # 3. 类型关联性 (权重: 0.2)
        type_score = self._get_type_correlation(event1.event_type, event2.event_type)
        scores.append(type_score * 0.2)
        
        # 4. 标签重叠 (权重: 0.15)
        if event1.tags and event2.tags:
            common_tags = set(event1.tags) & set(event2.tags)
            all_tags = set(event1.tags) | set(event2.tags)
            tag_score = len(common_tags) / len(all_tags) if all_tags else 0.0
        else:
            tag_score = 0.0
        scores.append(tag_score * 0.15)
        
        # 5. 严重级别相似性 (权重: 0.1)
        severity_diff = abs(event1.severity.value - event2.severity.value)
        severity_score = 1.0 - (severity_diff / 4.0)
        scores.append(severity_score * 0.1)
        
        return sum(scores)
    
    def _get_type_correlation(self, type1: EventType, type2: EventType) -> float:
        """获取事件类型关联度"""
        # 定义类型关联矩阵
        correlations = {
            (EventType.ALERT, EventType.METRIC_ANOMALY): 0.9,
            (EventType.ALERT, EventType.LOG_ERROR): 0.8,
            (EventType.METRIC_ANOMALY, EventType.LOG_ERROR): 0.7,
            (EventType.DEPLOYMENT, EventType.ALERT): 0.6,
            (EventType.CONFIG_CHANGE, EventType.ALERT): 0.7,
            (EventType.DEPLOYMENT, EventType.METRIC_ANOMALY): 0.5,
            (EventType.CONFIG_CHANGE, EventType.METRIC_ANOMALY): 0.6,
        }
        
        if type1 == type2:
            return 1.0
        
        key = (type1, type2)
        reverse_key = (type2, type1)
        
        return correlations.get(key, correlations.get(reverse_key, 0.3))
    
    def _determine_correlation_type(self, event: Event, 
                                    correlations: List[Tuple[str, float]]) -> str:
        """确定关联类型"""
        if not correlations:
            return "isolated"
        
        high_corr = [c for c in correlations if c[1] >= 0.7]
        medium_corr = [c for c in correlations if 0.4 <= c[1] < 0.7]
        
        if len(high_corr) >= 5:
            return "cascade"  # 级联故障
        elif len(high_corr) >= 2:
            return "correlated"  # 强关联
        elif len(medium_corr) >= 3:
            return "possibly_related"  # 可能关联
        else:
            return "weakly_related"  # 弱关联
    
    def _calculate_confidence(self, correlations: List[Tuple[str, float]]) -> float:
        """计算置信度"""
        if not correlations:
            return 1.0
        
        # 基于关联分数的分布计算置信度
        scores = [s for _, s in correlations]
        if len(scores) == 1:
            return scores[0]
        
        # 使用标准差和平均值
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 置信度 = 平均分 * (1 - 变异系数)
        cv = std_score / mean_score if mean_score > 0 else 1.0
        confidence = mean_score * (1 - min(cv, 1.0))
        
        return round(confidence, 2)
    
    def cluster_events(self, 
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[EventCluster]:
        """对事件进行聚类分析"""
        # 获取事件
        if not start_time:
            start_time = datetime.now() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.now()
        
        events = self.event_collector.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        if len(events) < 3:
            logger.info("事件数量不足，无法进行聚类")
            return []
        
        # 构建特征向量
        features = self._extract_features(events)
        
        # 计算距离矩阵
        distances = pdist(features, metric='euclidean')
        
        # 层次聚类
        linkage_matrix = linkage(distances, method='ward')
        
        # 确定聚类数量
        n_clusters = min(max(2, len(events) // 10), 10)
        
        # 分配簇标签
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # 构建簇
        clusters = defaultdict(list)
        for i, event in enumerate(events):
            clusters[cluster_labels[i]].append(event)
        
        # 创建EventCluster对象
        result = []
        for cluster_id, cluster_events in clusters.items():
            if len(cluster_events) < 2:
                continue
            
            event_ids = [e.event_id for e in cluster_events]
            
            # 计算中心时间
            timestamps = [e.timestamp for e in cluster_events]
            centroid_time = timestamps[len(timestamps) // 2]
            
            # 确定主导来源和类型
            source_counts = defaultdict(int)
            type_counts = defaultdict(int)
            severity_sum = 0
            
            for e in cluster_events:
                source_counts[e.source] += 1
                type_counts[e.event_type] += 1
                severity_sum += e.severity.value
            
            dominant_source = max(source_counts, key=source_counts.get)
            dominant_type = max(type_counts, key=type_counts.get)
            avg_severity = severity_sum / len(cluster_events)
            
            cluster = EventCluster(
                cluster_id=f"cluster_{cluster_id}",
                events=event_ids,
                centroid_time=centroid_time,
                dominant_source=dominant_source,
                dominant_type=dominant_type,
                severity_score=avg_severity
            )
            
            self.event_clusters[cluster.cluster_id] = cluster
            result.append(cluster)
        
        logger.info(f"事件聚类完成: {len(result)} 个簇")
        return result
    
    def _extract_features(self, events: List[Event]) -> np.ndarray:
        """提取事件特征向量"""
        features = []
        
        # 获取所有来源和类型的集合
        all_sources = list(set(e.source for e in events))
        all_types = list(set(e.event_type for e in events))
        
        base_time = min(e.timestamp for e in events)
        
        for event in events:
            # 时间特征 (归一化)
            time_feature = (event.timestamp - base_time).total_seconds() / 3600
            
            # 来源特征 (one-hot)
            source_features = [1.0 if s == event.source else 0.0 for s in all_sources]
            
            # 类型特征 (one-hot)
            type_features = [1.0 if t == event.event_type else 0.0 for t in all_types]
            
            # 严重级别
            severity_feature = event.severity.value / 5.0
            
            # 组合特征
            feature_vector = [time_feature, severity_feature] + source_features + type_features
            features.append(feature_vector)
        
        return np.array(features)
    
    def find_causal_chain(self, event_id: str, 
                          max_depth: int = 5) -> List[List[str]]:
        """查找因果链"""
        event = self.event_collector.get_event(event_id)
        if not event:
            return []
        
        chains = []
        
        def dfs(current_id: str, chain: List[str], depth: int):
            if depth > max_depth:
                return
            
            # 获取关联事件
            result = self.analyze(current_id)
            
            # 按时间排序，找之前发生的事件（可能是原因）
            current_event = self.event_collector.get_event(current_id)
            
            for related_id, score in result.related_events:
                if score < 0.6:  # 只考虑强关联
                    continue
                
                related_event = self.event_collector.get_event(related_id)
                if not related_event:
                    continue
                
                # 找在current_event之前发生的事件
                if related_event.timestamp < current_event.timestamp:
                    new_chain = chain + [related_id]
                    chains.append(new_chain)
                    dfs(related_id, new_chain, depth + 1)
        
        dfs(event_id, [event_id], 1)
        
        # 去重并排序
        unique_chains = []
        seen = set()
        for chain in chains:
            chain_tuple = tuple(chain)
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique_chains.append(chain)
        
        # 按链长度排序
        unique_chains.sort(key=len, reverse=True)
        
        return unique_chains[:10]  # 返回前10条
    
    def get_temporal_patterns(self, 
                              source: str,
                              minutes: int = 60) -> Dict[str, Any]:
        """获取时间模式"""
        events = self.event_collector.get_recent_events(minutes=minutes, source=source)
        
        if len(events) < 5:
            return {"pattern": "insufficient_data"}
        
        # 按时间排序
        events.sort(key=lambda x: x.timestamp)
        
        # 计算时间间隔
        intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        # 统计分析
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # 检测周期性
        if std_interval / mean_interval < 0.3 if mean_interval > 0 else False:
            pattern = "periodic"
        elif any(i < 60 for i in intervals):  # 有小于1分钟的间隔
            pattern = "burst"
        else:
            pattern = "random"
        
        # 检测趋势
        severity_values = [e.severity.value for e in events]
        if len(severity_values) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(severity_values)), severity_values
            )
            trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        else:
            trend = "unknown"
        
        return {
            "pattern": pattern,
            "trend": trend,
            "mean_interval_seconds": round(mean_interval, 2),
            "interval_std": round(std_interval, 2),
            "event_count": len(events),
            "severity_trend": trend
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "correlation_pairs": len(self.correlation_matrix),
            "clusters": len(self.event_clusters),
            "time_window_minutes": self.time_window_minutes,
            "correlation_threshold": self.correlation_threshold
        }
