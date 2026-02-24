"""
报告生成器
生成根因分析报告
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from .event_collector import Event, EventCollector
    from .correlation_analyzer import CorrelationAnalyzer, CorrelationResult, EventCluster
    from .root_cause_identifier import RootCauseIdentifier, RootCauseAnalysis, RootCause, RootCauseCategory
except ImportError:
    # 处理循环导入
    Event = None
    EventCollector = None
    CorrelationAnalyzer = None
    CorrelationResult = None
    EventCluster = None
    RootCauseIdentifier = None
    RootCauseAnalysis = None
    RootCause = None
    RootCauseCategory = None

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """报告格式"""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


@dataclass
class RCAReport:
    """RCA 报告"""
    report_id: str
    title: str
    summary: str
    primary_event: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    root_causes: List[Dict[str, Any]]
    event_timeline: List[Dict[str, Any]]
    recommendations: List[str]
    related_clusters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'title': self.title,
            'summary': self.summary,
            'primary_event': self.primary_event,
            'correlation_analysis': self.correlation_analysis,
            'root_causes': self.root_causes,
            'event_timeline': self.event_timeline,
            'recommendations': self.recommendations,
            'related_clusters': self.related_clusters,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, 
                 event_collector: EventCollector,
                 correlation_analyzer: CorrelationAnalyzer,
                 root_cause_identifier: RootCauseIdentifier,
                 config: Optional[Dict] = None):
        self.event_collector = event_collector
        self.correlation_analyzer = correlation_analyzer
        self.root_cause_identifier = root_cause_identifier
        self.config = config or {}
        
        # 配置
        self.default_format = self.config.get('default_format', ReportFormat.MARKDOWN)
        self.include_raw_data = self.config.get('include_raw_data', False)
        
        # 报告历史
        self.report_history: List[RCAReport] = []
        
        logger.info("报告生成器初始化完成")
    
    async def generate(self, 
                       event_id: str,
                       format: ReportFormat = None) -> RCAReport:
        """生成报告"""
        format = format or self.default_format
        
        # 获取主事件
        primary_event = self.event_collector.get_event(event_id)
        if not primary_event:
            raise ValueError(f"事件不存在: {event_id}")
        
        # 执行根因分析
        analysis = await self.root_cause_identifier.analyze(event_id)
        
        # 生成报告
        report = self._build_report(primary_event, analysis)
        
        self.report_history.append(report)
        
        logger.info(f"报告生成完成: {report.report_id}")
        return report
    
    def _build_report(self, primary_event: Event, 
                      analysis: RootCauseAnalysis) -> RCAReport:
        """构建报告"""
        # 1. 主事件信息
        primary_event_dict = primary_event.to_dict()
        
        # 2. 关联分析
        correlation_dict = {
            'correlation_type': analysis.correlation_result.correlation_type,
            'confidence': analysis.correlation_result.confidence,
            'related_event_count': len(analysis.correlation_result.related_events),
            'related_events': [
                {
                    'event_id': eid,
                    'correlation_score': score,
                    'event_summary': self._get_event_summary(eid)
                }
                for eid, score in analysis.correlation_result.related_events[:10]
            ]
        }
        
        # 3. 根因列表
        root_causes_dict = [
            {
                'category': cause.category.value,
                'description': cause.description,
                'confidence': cause.confidence,
                'evidence': cause.evidence,
                'related_events': cause.related_events,
                'suggested_actions': cause.suggested_actions
            }
            for cause in analysis.root_causes
        ]
        
        # 4. 事件时间线
        timeline = self._build_timeline(primary_event, analysis)
        
        # 5. 建议
        recommendations = self._generate_recommendations(analysis)
        
        # 6. 相关簇
        clusters = self.correlation_analyzer.cluster_events(
            start_time=analysis.correlation_result.time_window[0],
            end_time=analysis.correlation_result.time_window[1]
        )
        clusters_dict = [
            {
                'cluster_id': c.cluster_id,
                'event_count': len(c.events),
                'dominant_source': c.dominant_source,
                'dominant_type': c.dominant_type.value,
                'severity_score': c.severity_score
            }
            for c in clusters[:5]
        ]
        
        # 7. 摘要
        summary = self._generate_summary(primary_event, analysis)
        
        # 8. 元数据
        metadata = {
            'analysis_id': analysis.analysis_id,
            'analysis_time_ms': analysis.analysis_time_ms,
            'correlation_pairs': len(analysis.correlation_result.related_events),
            'total_events_analyzed': len(analysis.correlation_result.related_events) + 1
        }
        
        report = RCAReport(
            report_id=f"rca_report_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=f"根因分析报告: {primary_event.title}",
            summary=summary,
            primary_event=primary_event_dict,
            correlation_analysis=correlation_dict,
            root_causes=root_causes_dict,
            event_timeline=timeline,
            recommendations=recommendations,
            related_clusters=clusters_dict,
            metadata=metadata
        )
        
        return report
    
    def _get_event_summary(self, event_id: str) -> str:
        """获取事件摘要"""
        event = self.event_collector.get_event(event_id)
        if not event:
            return "未知事件"
        return f"{event.source}: {event.title}"
    
    def _build_timeline(self, primary_event: Event, 
                        analysis: RootCauseAnalysis) -> List[Dict[str, Any]]:
        """构建事件时间线"""
        timeline = []
        
        # 获取所有相关事件
        all_event_ids = set([primary_event.event_id])
        for cause in analysis.root_causes:
            all_event_ids.update(cause.related_events)
        
        events = [
            self.event_collector.get_event(eid)
            for eid in all_event_ids
            if self.event_collector.get_event(eid)
        ]
        
        # 按时间排序
        events.sort(key=lambda x: x.timestamp)
        
        for event in events:
            timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'source': event.source,
                'title': event.title,
                'severity': event.severity.name,
                'is_primary': event.event_id == primary_event.event_id
            })
        
        return timeline
    
    def _generate_recommendations(self, analysis: RootCauseAnalysis) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于根因的建议
        for cause in analysis.root_causes:
            recommendations.extend(cause.suggested_actions)
        
        # 去重
        recommendations = list(set(recommendations))
        
        # 添加通用建议
        if analysis.correlation_result.correlation_type == "cascade":
            recommendations.append("这是一个级联故障，建议检查服务依赖关系")
        
        if len(analysis.root_causes) > 3:
            recommendations.append("问题可能涉及多个因素，建议分优先级处理")
        
        return recommendations
    
    def _generate_summary(self, primary_event: Event, 
                          analysis: RootCauseAnalysis) -> str:
        """生成摘要"""
        parts = []
        
        # 主事件描述
        parts.append(f"主事件: {primary_event.title}")
        parts.append(f"来源: {primary_event.source}")
        parts.append(f"时间: {primary_event.timestamp.isoformat()}")
        
        # 关联分析结果
        parts.append(f"\n关联类型: {analysis.correlation_result.correlation_type}")
        parts.append(f"关联置信度: {analysis.correlation_result.confidence:.2f}")
        parts.append(f"相关事件数: {len(analysis.correlation_result.related_events)}")
        
        # 根因
        if analysis.root_causes:
            parts.append(f"\n识别的根因 ({len(analysis.root_causes)} 个):")
            for i, cause in enumerate(analysis.root_causes[:3], 1):
                parts.append(f"  {i}. {cause.category.value} (置信度: {cause.confidence:.2f})")
                parts.append(f"     {cause.description}")
        else:
            parts.append("\n未能识别明确的根因")
        
        return "\n".join(parts)
    
    def export(self, report: RCAReport, 
               format: ReportFormat = None) -> str:
        """导出报告"""
        format = format or self.default_format
        
        if format == ReportFormat.JSON:
            return self._export_json(report)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report)
        elif format == ReportFormat.HTML:
            return self._export_html(report)
        elif format == ReportFormat.TEXT:
            return self._export_text(report)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _export_json(self, report: RCAReport) -> str:
        """导出为 JSON"""
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    
    def _export_markdown(self, report: RCAReport) -> str:
        """导出为 Markdown"""
        lines = []
        
        # 标题
        lines.append(f"# {report.title}")
        lines.append(f"\n**报告ID:** {report.report_id}")
        lines.append(f"**生成时间:** {report.created_at.isoformat()}")
        
        # 摘要
        lines.append("\n## 摘要\n")
        lines.append(report.summary)
        
        # 主事件
        lines.append("\n## 主事件\n")
        event = report.primary_event
        lines.append(f"- **事件ID:** {event['event_id']}")
        lines.append(f"- **类型:** {event['event_type']}")
        lines.append(f"- **来源:** {event['source']}")
        lines.append(f"- **时间:** {event['timestamp']}")
        lines.append(f"- **严重级别:** {event['severity']}")
        lines.append(f"- **标题:** {event['title']}")
        lines.append(f"- **描述:** {event['description']}")
        
        # 关联分析
        lines.append("\n## 关联分析\n")
        corr = report.correlation_analysis
        lines.append(f"- **关联类型:** {corr['correlation_type']}")
        lines.append(f"- **置信度:** {corr['confidence']}")
        lines.append(f"- **相关事件数:** {corr['related_event_count']}")
        
        if corr['related_events']:
            lines.append("\n### 相关事件\n")
            for rel in corr['related_events'][:5]:
                lines.append(f"- {rel['event_summary']} (关联度: {rel['correlation_score']:.2f})")
        
        # 根因
        lines.append("\n## 根因分析\n")
        if report.root_causes:
            for i, cause in enumerate(report.root_causes, 1):
                lines.append(f"### {i}. {cause['category'].upper()}\n")
                lines.append(f"**描述:** {cause['description']}")
                lines.append(f"**置信度:** {cause['confidence']:.2f}")
                
                if cause['evidence']:
                    lines.append("\n**证据:**")
                    for ev in cause['evidence']:
                        lines.append(f"- {ev}")
                
                if cause['suggested_actions']:
                    lines.append("\n**建议动作:**")
                    for action in cause['suggested_actions']:
                        lines.append(f"- {action}")
                
                lines.append("")
        else:
            lines.append("未能识别明确的根因。")
        
        # 时间线
        lines.append("\n## 事件时间线\n")
        lines.append("| 时间 | 来源 | 类型 | 标题 | 严重级别 |")
        lines.append("|------|------|------|------|----------|")
        for item in report.event_timeline:
            marker = " **(主事件)**" if item['is_primary'] else ""
            lines.append(f"| {item['timestamp']} | {item['source']} | {item['event_type']} | {item['title']}{marker} | {item['severity']} |")
        
        # 建议
        lines.append("\n## 建议\n")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        
        # 元数据
        lines.append("\n## 元数据\n")
        for key, value in report.metadata.items():
            lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def _export_html(self, report: RCAReport) -> str:
        """导出为 HTML"""
        # 简化的 HTML 导出
        md_content = self._export_markdown(report)
        
        # 这里可以使用 markdown 库转换为 HTML
        # 简化处理，直接返回基本 HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        h3 {{ color: #888; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .primary {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p><strong>报告ID:</strong> {report.report_id}</p>
    <p><strong>生成时间:</strong> {report.created_at.isoformat()}</p>
    
    <h2>摘要</h2>
    <div class="summary">
        <pre>{report.summary}</pre>
    </div>
    
    <h2>根因分析</h2>
    <div>
        {''.join(f'<h3>{i+1}. {cause["category"].upper()}</h3><p>{cause["description"]}</p><p>置信度: {cause["confidence"]:.2f}</p>' for i, cause in enumerate(report.root_causes))}
    </div>
    
    <h2>建议</h2>
    <ul>
        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
    </ul>
</body>
</html>"""
        return html
    
    def _export_text(self, report: RCAReport) -> str:
        """导出为纯文本"""
        lines = []
        
        lines.append("=" * 60)
        lines.append(report.title)
        lines.append("=" * 60)
        lines.append(f"报告ID: {report.report_id}")
        lines.append(f"生成时间: {report.created_at.isoformat()}")
        lines.append("")
        lines.append(report.summary)
        lines.append("")
        lines.append("-" * 60)
        lines.append("根因分析")
        lines.append("-" * 60)
        
        for i, cause in enumerate(report.root_causes, 1):
            lines.append(f"\n{i}. {cause['category'].upper()}")
            lines.append(f"   描述: {cause['description']}")
            lines.append(f"   置信度: {cause['confidence']:.2f}")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("建议")
        lines.append("-" * 60)
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def get_report_history(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> List[RCAReport]:
        """获取报告历史"""
        history = self.report_history
        
        if start_time:
            history = [h for h in history if h.created_at >= start_time]
        if end_time:
            history = [h for h in history if h.created_at <= end_time]
        
        return sorted(history, key=lambda x: x.created_at, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.report_history:
            return {"total_reports": 0}
        
        total = len(self.report_history)
        
        # 按根因类别统计
        from collections import defaultdict
        category_counts = defaultdict(int)
        for report in self.report_history:
            for cause in report.root_causes:
                category_counts[cause['category']] += 1
        
        return {
            "total_reports": total,
            "by_category": dict(category_counts),
            "avg_root_causes": sum(len(r.root_causes) for r in self.report_history) / total
        }
    
    def generate_report(self, incident_id: str, root_causes: List[Dict[str, Any]], 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """生成报告 (简化接口)"""
        metadata = metadata or {}
        
        # 转换根因为字典格式
        root_causes_dict = []
        recommendations = []
        for cause in root_causes:
            root_causes_dict.append({
                'category': cause.get('cause', 'unknown'),
                'description': cause.get('description', 'No description'),
                'confidence': cause.get('confidence', 0.5),
                'evidence': cause.get('evidence', []),
                'related_events': cause.get('related_events', []),
                'suggested_actions': cause.get('suggested_actions', [])
            })
            # 收集建议
            if 'suggested_actions' in cause:
                recommendations.extend(cause['suggested_actions'])
        
        # 去重建议
        recommendations = list(set(recommendations))
        if not recommendations:
            recommendations = ["建议进一步调查问题根因"]
        
        # 创建报告 - 使用 RCAReport 定义的字段
        report = RCAReport(
            report_id=f"rca_report_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=f"根因分析报告 - {incident_id}",
            summary=f"检测到 {len(root_causes_dict)} 个可能的根因",
            primary_event={
                'event_id': incident_id,
                'event_type': 'incident',
                'source': metadata.get('source', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'severity': metadata.get('severity', 'high'),
                'title': metadata.get('title', 'Unknown Incident'),
                'description': metadata.get('description', '')
            },
            correlation_analysis={
                'correlation_type': 'direct',
                'confidence': 0.8,
                'related_event_count': len(root_causes_dict),
                'related_events': []
            },
            root_causes=root_causes_dict,
            event_timeline=metadata.get('timeline', []),
            recommendations=recommendations,
            related_clusters=[],
            metadata={
                'incident_id': incident_id,
                'affected_services': metadata.get('affected_services', []),
                'generated_by': 'simplified_interface'
            }
        )
        
        # 存储报告
        self.report_history.append(report)
        
        return {
            'report_id': report.report_id,
            'incident_id': incident_id,
            'root_causes_count': len(report.root_causes),
            'recommendations_count': len(report.recommendations)
        }
