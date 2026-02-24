"""
根因识别器
使用多种算法识别问题的根本原因
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from enum import Enum

import numpy as np

from .event_collector import Event, EventType, EventSeverity, EventCollector
from .correlation_analyzer import CorrelationAnalyzer, CorrelationResult

logger = logging.getLogger(__name__)


class RootCauseCategory(Enum):
    """根因类别"""
    INFRASTRUCTURE = "infrastructure"    # 基础设施问题
    DEPLOYMENT = "deployment"            # 部署问题
    CONFIGURATION = "configuration"      # 配置问题
    CODE_BUG = "code_bug"                # 代码缺陷
    DEPENDENCY = "dependency"            # 依赖问题
    RESOURCE = "resource"                # 资源问题
    NETWORK = "network"                  # 网络问题
    SECURITY = "security"                # 安全问题
    EXTERNAL = "external"                # 外部因素
    UNKNOWN = "unknown"                  # 未知


@dataclass
class RootCause:
    """根因"""
    cause_id: str
    category: RootCauseCategory
    description: str
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    identified_at: datetime = field(default_factory=datetime.now)


@dataclass
class RootCauseAnalysis:
    """根因分析结果"""
    analysis_id: str
    primary_event_id: str
    root_causes: List[RootCause]
    correlation_result: CorrelationResult
    analysis_time_ms: int
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_top_cause(self, min_confidence: float = 0.5) -> Optional[RootCause]:
        """获取最可能的根因"""
        valid_causes = [c for c in self.root_causes if c.confidence >= min_confidence]
        if not valid_causes:
            return None
        return max(valid_causes, key=lambda x: x.confidence)
    
    def get_all_actions(self) -> List[str]:
        """获取所有建议动作"""
        actions = []
        for cause in self.root_causes:
            actions.extend(cause.suggested_actions)
        return list(set(actions))


class RootCauseIdentifier:
    """根因识别器"""
    
    def __init__(self, 
                 event_collector: EventCollector = None,
                 correlation_analyzer: CorrelationAnalyzer = None,
                 config: Optional[Dict] = None):
        self.event_collector = event_collector
        self.correlation_analyzer = correlation_analyzer
        self.config = config or {}
        
        # 配置
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.max_causes = self.config.get('max_causes', 5)
        self.analysis_history: List[RootCauseAnalysis] = []
        
        # 因果规则库
        self.causal_rules = self._init_causal_rules()
        
        logger.info("根因识别器初始化完成")
    
    def identify_root_causes_bayesian(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """贝叶斯根因识别 (简化接口)"""
        causes = []
        
        # 简单的启发式根因识别
        for event in events:
            event_type = event.get('type', '')
            metric = event.get('metric', '')
            value = event.get('value', 0)
            
            # CPU 高 -> 资源问题
            if metric == 'cpu' and value > 80:
                causes.append({
                    'cause': 'cpu_exhaustion',
                    'category': 'resource',
                    'confidence': 0.8,
                    'evidence': [f"CPU usage: {value}%"]
                })
            
            # 错误 -> 代码问题
            if event_type == 'error':
                causes.append({
                    'cause': 'code_bug',
                    'category': 'code',
                    'confidence': 0.7,
                    'evidence': [f"Error event: {event}"]
                })
        
        return causes
    
    def _init_causal_rules(self) -> Dict[str, Any]:
        """初始化因果规则"""
        return {
            # 部署后出现问题
            "post_deployment_issue": {
                "pattern": lambda events: any(
                    e.event_type == EventType.DEPLOYMENT for e in events
                ) and any(
                    e.event_type in [EventType.ALERT, EventType.METRIC_ANOMALY, EventType.LOG_ERROR]
                    for e in events
                ),
                "category": RootCauseCategory.DEPLOYMENT,
                "confidence_boost": 0.3
            },
            # 配置变更后出现问题
            "post_config_change": {
                "pattern": lambda events: any(
                    e.event_type == EventType.CONFIG_CHANGE for e in events
                ) and any(
                    e.event_type in [EventType.ALERT, EventType.METRIC_ANOMALY]
                    for e in events
                ),
                "category": RootCauseCategory.CONFIGURATION,
                "confidence_boost": 0.25
            },
            # 资源问题模式
            "resource_exhaustion": {
                "pattern": lambda events: any(
                    'cpu' in e.title.lower() or 'memory' in e.title.lower() or
                    'disk' in e.title.lower() or 'resource' in e.title.lower()
                    for e in events if e.event_type == EventType.METRIC_ANOMALY
                ),
                "category": RootCauseCategory.RESOURCE,
                "confidence_boost": 0.2
            },
            # 依赖问题模式
            "dependency_failure": {
                "pattern": lambda events: any(
                    'timeout' in e.description.lower() or 
                    'connection' in e.description.lower() or
                    'unavailable' in e.description.lower()
                    for e in events
                ),
                "category": RootCauseCategory.DEPENDENCY,
                "confidence_boost": 0.2
            },
            # 网络问题模式
            "network_issue": {
                "pattern": lambda events: any(
                    'network' in e.title.lower() or 
                    'dns' in e.title.lower() or
                    'latency' in e.title.lower()
                    for e in events
                ),
                "category": RootCauseCategory.NETWORK,
                "confidence_boost": 0.2
            }
        }
    
    async def analyze(self, event_id: str) -> RootCauseAnalysis:
        """执行根因分析"""
        import time
        start_time = time.time()
        
        # 获取关联分析结果
        correlation_result = self.correlation_analyzer.analyze(event_id)
        
        # 获取所有相关事件
        all_event_ids = [event_id] + [eid for eid, _ in correlation_result.related_events]
        events = [
            self.event_collector.get_event(eid) 
            for eid in all_event_ids 
            if self.event_collector.get_event(eid)
        ]
        
        # 识别根因
        root_causes = []
        
        # 1. 基于时间序列分析
        temporal_causes = self._analyze_temporal_patterns(events, event_id)
        root_causes.extend(temporal_causes)
        
        # 2. 基于因果规则
        rule_causes = self._apply_causal_rules(events)
        root_causes.extend(rule_causes)
        
        # 3. 基于事件链分析
        chain_causes = self._analyze_causal_chains(event_id)
        root_causes.extend(chain_causes)
        
        # 4. 基于贝叶斯推理
        bayesian_causes = self._bayesian_inference(events, event_id)
        root_causes.extend(bayesian_causes)
        
        # 合并相似根因
        root_causes = self._merge_similar_causes(root_causes)
        
        # 排序并限制数量
        root_causes.sort(key=lambda x: x.confidence, reverse=True)
        root_causes = root_causes[:self.max_causes]
        
        analysis_time = int((time.time() - start_time) * 1000)
        
        analysis = RootCauseAnalysis(
            analysis_id=f"rca_{datetime.now().strftime('%Y%m%d%H%M%S')}_{event_id[:8]}",
            primary_event_id=event_id,
            root_causes=root_causes,
            correlation_result=correlation_result,
            analysis_time_ms=analysis_time
        )
        
        self.analysis_history.append(analysis)
        
        logger.info(f"根因分析完成: {analysis.analysis_id}, 找到 {len(root_causes)} 个可能原因")
        return analysis
    
    def _analyze_temporal_patterns(self, events: List[Event], 
                                   primary_event_id: str) -> List[RootCause]:
        """分析时间模式识别根因"""
        causes = []
        
        if not events:
            return causes
        
        # 按时间排序
        events.sort(key=lambda x: x.timestamp)
        
        # 找最早的事件
        earliest_event = events[0]
        
        # 如果最早事件比主事件早超过5分钟，可能是根因
        primary_event = next((e for e in events if e.event_id == primary_event_id), None)
        if primary_event:
            time_diff = (primary_event.timestamp - earliest_event.timestamp).total_seconds()
            
            if time_diff > 300:  # 5分钟
                # 根据最早事件类型确定根因类别
                category = self._categorize_event(earliest_event)
                
                cause = RootCause(
                    cause_id=f"temporal_{earliest_event.event_id}",
                    category=category,
                    description=f"最早发生的事件: {earliest_event.title}",
                    confidence=min(0.7, 0.4 + time_diff / 600),  # 时间差越大，置信度越高
                    evidence=[
                        f"事件时间: {earliest_event.timestamp.isoformat()}",
                        f"比主事件早 {time_diff:.0f} 秒"
                    ],
                    related_events=[earliest_event.event_id],
                    suggested_actions=self._get_suggested_actions(category)
                )
                causes.append(cause)
        
        return causes
    
    def _apply_causal_rules(self, events: List[Event]) -> List[RootCause]:
        """应用因果规则"""
        causes = []
        
        for rule_name, rule in self.causal_rules.items():
            try:
                if rule['pattern'](events):
                    # 找到触发规则的事件
                    triggering_events = [
                        e for e in events 
                        if self._is_triggering_event(e, rule['category'])
                    ]
                    
                    if triggering_events:
                        trigger = triggering_events[0]
                        
                        cause = RootCause(
                            cause_id=f"rule_{rule_name}_{trigger.event_id}",
                            category=rule['category'],
                            description=f"匹配规则 '{rule_name}': {trigger.title}",
                            confidence=min(0.9, 0.5 + rule['confidence_boost']),
                            evidence=[
                                f"触发事件: {trigger.title}",
                                f"规则: {rule_name}"
                            ],
                            related_events=[e.event_id for e in triggering_events],
                            suggested_actions=self._get_suggested_actions(rule['category'])
                        )
                        causes.append(cause)
            
            except Exception as e:
                logger.warning(f"规则应用失败 {rule_name}: {e}")
        
        return causes
    
    def _analyze_causal_chains(self, event_id: str) -> List[RootCause]:
        """分析因果链"""
        causes = []
        
        # 获取因果链
        chains = self.correlation_analyzer.find_causal_chain(event_id, max_depth=5)
        
        if chains:
            # 取最长的链
            longest_chain = max(chains, key=len)
            
            if len(longest_chain) >= 3:
                # 链的最后一个元素可能是最早的原因
                root_event_id = longest_chain[-1]
                root_event = self.event_collector.get_event(root_event_id)
                
                if root_event:
                    category = self._categorize_event(root_event)
                    
                    cause = RootCause(
                        cause_id=f"chain_{root_event_id}",
                        category=category,
                        description=f"因果链分析: {root_event.title}",
                        confidence=min(0.85, 0.5 + len(longest_chain) * 0.1),
                        evidence=[
                            f"因果链长度: {len(longest_chain)}",
                            f"链: {' -> '.join(longest_chain[:5])}"
                        ],
                        related_events=longest_chain,
                        suggested_actions=self._get_suggested_actions(category)
                    )
                    causes.append(cause)
        
        return causes
    
    def _bayesian_inference(self, events: List[Event], 
                           primary_event_id: str) -> List[RootCause]:
        """基于贝叶斯推理识别根因"""
        causes = []
        
        if len(events) < 3:
            return causes
        
        # 统计各类别事件的先验概率
        category_counts = defaultdict(int)
        for event in events:
            category = self._categorize_event(event)
            category_counts[category] += 1
        
        total = len(events)
        
        # 计算后验概率
        for category, count in category_counts.items():
            prior = count / total
            
            # 似然：假设是该类别导致问题的概率
            likelihood = self._get_likelihood(category, events)
            
            # 后验概率 (简化计算)
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
            if posterior >= self.min_confidence:
                # 找该类别的事件
                category_events = [
                    e for e in events 
                    if self._categorize_event(e) == category
                ]
                
                if category_events:
                    representative = category_events[0]
                    
                    cause = RootCause(
                        cause_id=f"bayesian_{category.value}_{representative.event_id}",
                        category=category,
                        description=f"贝叶斯推理: {category.value}",
                        confidence=round(posterior, 2),
                        evidence=[
                            f"先验概率: {prior:.2f}",
                            f"似然: {likelihood:.2f}",
                            f"相关事件数: {len(category_events)}"
                        ],
                        related_events=[e.event_id for e in category_events[:5]],
                        suggested_actions=self._get_suggested_actions(category)
                    )
                    causes.append(cause)
        
        return causes
    
    def _categorize_event(self, event: Event) -> RootCauseCategory:
        """对事件进行分类"""
        title_lower = event.title.lower()
        desc_lower = event.description.lower()
        
        if event.event_type == EventType.DEPLOYMENT:
            return RootCauseCategory.DEPLOYMENT
        elif event.event_type == EventType.CONFIG_CHANGE:
            return RootCauseCategory.CONFIGURATION
        elif any(kw in title_lower for kw in ['cpu', 'memory', 'disk', 'resource']):
            return RootCauseCategory.RESOURCE
        elif any(kw in title_lower for kw in ['network', 'dns', 'latency']):
            return RootCauseCategory.NETWORK
        elif any(kw in desc_lower for kw in ['timeout', 'connection', 'unavailable']):
            return RootCauseCategory.DEPENDENCY
        elif event.event_type == EventType.SECURITY:
            return RootCauseCategory.SECURITY
        elif event.event_type == EventType.INFRASTRUCTURE:
            return RootCauseCategory.INFRASTRUCTURE
        else:
            return RootCauseCategory.CODE_BUG
    
    def _is_triggering_event(self, event: Event, category: RootCauseCategory) -> bool:
        """判断事件是否是触发事件"""
        event_category = self._categorize_event(event)
        return event_category == category
    
    def _get_likelihood(self, category: RootCauseCategory, 
                       events: List[Event]) -> float:
        """获取似然概率"""
        # 基于历史数据或启发式规则
        likelihoods = {
            RootCauseCategory.DEPLOYMENT: 0.8,
            RootCauseCategory.CONFIGURATION: 0.75,
            RootCauseCategory.RESOURCE: 0.7,
            RootCauseCategory.DEPENDENCY: 0.65,
            RootCauseCategory.NETWORK: 0.6,
            RootCauseCategory.CODE_BUG: 0.5,
            RootCauseCategory.INFRASTRUCTURE: 0.7,
            RootCauseCategory.SECURITY: 0.6,
            RootCauseCategory.EXTERNAL: 0.4,
            RootCauseCategory.UNKNOWN: 0.3
        }
        return likelihoods.get(category, 0.5)
    
    def _get_suggested_actions(self, category: RootCauseCategory) -> List[str]:
        """获取建议动作"""
        actions = {
            RootCauseCategory.DEPLOYMENT: [
                "检查最近的部署记录",
                "考虑回滚到上一个稳定版本",
                "查看部署日志"
            ],
            RootCauseCategory.CONFIGURATION: [
                "检查最近的配置变更",
                "验证配置参数",
                "考虑恢复之前的配置"
            ],
            RootCauseCategory.RESOURCE: [
                "检查资源使用情况",
                "考虑扩容",
                "优化资源使用"
            ],
            RootCauseCategory.DEPENDENCY: [
                "检查依赖服务状态",
                "查看依赖调用日志",
                "联系依赖服务负责人"
            ],
            RootCauseCategory.NETWORK: [
                "检查网络连接",
                "查看网络监控",
                "联系网络团队"
            ],
            RootCauseCategory.CODE_BUG: [
                "查看错误日志",
                "分析代码变更",
                "准备热修复"
            ],
            RootCauseCategory.INFRASTRUCTURE: [
                "检查基础设施状态",
                "查看云平台状态页",
                "联系基础设施团队"
            ],
            RootCauseCategory.SECURITY: [
                "检查安全日志",
                "评估安全影响",
                "联系安全团队"
            ],
            RootCauseCategory.EXTERNAL: [
                "检查第三方服务状态",
                "查看外部依赖",
                "准备降级方案"
            ],
            RootCauseCategory.UNKNOWN: [
                "收集更多信息",
                "扩大监控范围",
                "联系相关团队"
            ]
        }
        return actions.get(category, ["进一步调查"])
    
    def _merge_similar_causes(self, causes: List[RootCause]) -> List[RootCause]:
        """合并相似的根因"""
        if not causes:
            return causes
        
        merged = []
        seen_categories = set()
        
        for cause in causes:
            if cause.category not in seen_categories:
                # 找同类别的其他原因
                similar = [c for c in causes if c.category == cause.category]
                
                if len(similar) > 1:
                    # 合并
                    max_confidence = max(c.confidence for c in similar)
                    all_evidence = []
                    all_events = []
                    all_actions = []
                    
                    for c in similar:
                        all_evidence.extend(c.evidence)
                        all_events.extend(c.related_events)
                        all_actions.extend(c.suggested_actions)
                    
                    merged_cause = RootCause(
                        cause_id=f"merged_{cause.category.value}",
                        category=cause.category,
                        description=f"合并分析: {cause.category.value}",
                        confidence=max_confidence,
                        evidence=list(set(all_evidence)),
                        related_events=list(set(all_events)),
                        suggested_actions=list(set(all_actions))
                    )
                    merged.append(merged_cause)
                else:
                    merged.append(cause)
                
                seen_categories.add(cause.category)
        
        return merged
    
    def get_analysis_history(self, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> List[RootCauseAnalysis]:
        """获取分析历史"""
        history = self.analysis_history
        
        if start_time:
            history = [h for h in history if h.created_at >= start_time]
        if end_time:
            history = [h for h in history if h.created_at <= end_time]
        
        return sorted(history, key=lambda x: x.created_at, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.analysis_history:
            return {"total_analyses": 0}
        
        total = len(self.analysis_history)
        
        # 按类别统计
        category_counts = defaultdict(int)
        for analysis in self.analysis_history:
            for cause in analysis.root_causes:
                category_counts[cause.category.value] += 1
        
        # 平均分析时间
        avg_time = np.mean([a.analysis_time_ms for a in self.analysis_history])
        
        # 平均根因数量
        avg_causes = np.mean([len(a.root_causes) for a in self.analysis_history])
        
        return {
            "total_analyses": total,
            "by_category": dict(category_counts),
            "avg_analysis_time_ms": round(avg_time, 2),
            "avg_causes_per_analysis": round(avg_causes, 2)
        }
