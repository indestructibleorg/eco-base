# =============================================================================
# RCA Report Generator
# =============================================================================
# 根因分析 - 报告生成器
# 生成 RCA 分析报告，支持多种格式导出
# =============================================================================

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.closed_loop.rca.root_cause_identifier import RCAResult, RootCause
from app.closed_loop.rca.event_collector import Event, EventSeverity


class ReportGenerator:
    """
    RCA 报告生成器
    
    支持生成多种格式的分析报告
    """
    
    def generate_report(self, result: RCAResult) -> Dict[str, Any]:
        """
        生成结构化报告
        
        Returns:
            结构化报告字典
        """
        report = {
            "metadata": {
                "id": result.id,
                "timestamp": result.timestamp.isoformat(),
                "analysis_method": result.analysis_method,
                "overall_confidence": round(result.confidence, 2),
            },
            "summary": self._generate_summary(result),
            "root_causes": self._format_root_causes(result.root_causes),
            "timeline": self._format_timeline(result.timeline),
            "affected_services": result.affected_services,
            "contributing_factors": self._format_contributing_factors(
                result.contributing_factors
            ),
            "recommended_actions": result.recommended_actions,
            "impact_analysis": self._generate_impact_analysis(result),
        }
        
        return report
    
    def _generate_summary(self, result: RCAResult) -> Dict[str, Any]:
        """生成报告摘要"""
        primary_cause = result.root_causes[0] if result.root_causes else None
        
        return {
            "primary_root_cause": primary_cause.event.message if primary_cause else "Unknown",
            "primary_confidence": round(primary_cause.confidence, 2) if primary_cause else 0.0,
            "total_events_analyzed": len(result.timeline),
            "root_causes_identified": len(result.root_causes),
            "affected_services_count": len(result.affected_services),
            "incident_duration_minutes": self._calculate_duration(result.timeline),
        }
    
    def _calculate_duration(self, timeline: List[Event]) -> float:
        """计算事件持续时间"""
        if len(timeline) < 2:
            return 0.0
        
        start = min(e.timestamp for e in timeline)
        end = max(e.timestamp for e in timeline)
        duration = (end - start).total_seconds() / 60
        
        return round(duration, 2)
    
    def _format_root_causes(self, root_causes: List[RootCause]) -> List[Dict[str, Any]]:
        """格式化根因列表"""
        formatted = []
        
        for i, rc in enumerate(root_causes, 1):
            formatted.append({
                "rank": i,
                "event_id": rc.event.id,
                "timestamp": rc.event.timestamp.isoformat(),
                "event_type": rc.event.event_type.value,
                "severity": rc.event.severity.value,
                "message": rc.event.message,
                "source": rc.event.source,
                "confidence": round(rc.confidence, 2),
                "evidence": rc.evidence,
                "affected_services": rc.affected_services,
                "cascading_impact": rc.cascading_impact,
            })
        
        return formatted
    
    def _format_timeline(self, timeline: List[Event]) -> List[Dict[str, Any]]:
        """格式化时间线"""
        sorted_events = sorted(timeline, key=lambda e: e.timestamp)
        
        formatted = []
        for i, event in enumerate(sorted_events):
            # 计算与上一个事件的时间差
            time_diff = ""
            if i > 0:
                diff_seconds = (
                    event.timestamp - sorted_events[i-1].timestamp
                ).total_seconds()
                time_diff = f"+{diff_seconds:.1f}s"
            
            formatted.append({
                "sequence": i + 1,
                "timestamp": event.timestamp.isoformat(),
                "time_diff": time_diff,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "source": event.source,
                "message": event.message,
                "event_id": event.id,
            })
        
        return formatted
    
    def _format_contributing_factors(
        self,
        factors: List[Event]
    ) -> List[Dict[str, Any]]:
        """格式化 contributing factors"""
        formatted = []
        
        for event in factors:
            formatted.append({
                "event_id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "source": event.source,
                "message": event.message,
            })
        
        return formatted
    
    def _generate_impact_analysis(self, result: RCAResult) -> Dict[str, Any]:
        """生成影响分析"""
        # 按严重级别统计事件
        severity_counts = {}
        for event in result.timeline:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 按类型统计事件
        type_counts = {}
        for event in result.timeline:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        return {
            "severity_distribution": severity_counts,
            "event_type_distribution": type_counts,
            "services_affected": result.affected_services,
            "estimated_blast_radius": self._estimate_blast_radius(result),
        }
    
    def _estimate_blast_radius(self, result: RCAResult) -> str:
        """估算影响范围"""
        event_count = len(result.timeline)
        service_count = len(result.affected_services)
        
        # 检查是否有严重事件
        has_critical = any(
            e.severity == EventSeverity.CRITICAL for e in result.timeline
        )
        
        if has_critical or event_count > 20:
            return "HIGH - Multiple services affected, immediate attention required"
        elif event_count > 10 or service_count > 2:
            return "MEDIUM - Several components affected, investigation needed"
        else:
            return "LOW - Limited impact, can be addressed during business hours"
    
    def export_to_json(self, result: RCAResult, indent: int = 2) -> str:
        """导出为 JSON 格式"""
        report = self.generate_report(result)
        return json.dumps(report, indent=indent, ensure_ascii=False)
    
    def export_to_markdown(self, result: RCAResult) -> str:
        """导出为 Markdown 格式"""
        report = self.generate_report(result)
        
        lines = []
        
        # 标题
        lines.append(f"# RCA Report: {result.id}")
        lines.append("")
        lines.append(f"**Generated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")
        
        # 摘要
        lines.append("## Summary")
        lines.append("")
        summary = report["summary"]
        lines.append(f"- **Primary Root Cause:** {summary['primary_root_cause']}")
        lines.append(f"- **Confidence:** {summary['primary_confidence'] * 100:.0f}%")
        lines.append(f"- **Total Events Analyzed:** {summary['total_events_analyzed']}")
        lines.append(f"- **Root Causes Identified:** {summary['root_causes_identified']}")
        lines.append(f"- **Affected Services:** {summary['affected_services_count']}")
        lines.append(f"- **Incident Duration:** {summary['incident_duration_minutes']} minutes")
        lines.append("")
        
        # 根因详情
        lines.append("## Root Causes")
        lines.append("")
        
        for rc in report["root_causes"]:
            lines.append(f"### #{rc['rank']} - {rc['event_type']}")
            lines.append("")
            lines.append(f"**Confidence:** {rc['confidence'] * 100:.0f}%")
            lines.append("")
            lines.append(f"**Message:** {rc['message']}")
            lines.append("")
            lines.append(f"**Source:** {rc['source']}")
            lines.append("")
            lines.append(f"**Time:** {rc['timestamp']}")
            lines.append("")
            
            if rc['evidence']:
                lines.append("**Evidence:**")
                for evidence in rc['evidence']:
                    lines.append(f"- {evidence}")
                lines.append("")
            
            if rc['affected_services']:
                lines.append(f"**Affected Services:** {', '.join(rc['affected_services'])}")
                lines.append("")
            
            if rc['cascading_impact']:
                lines.append("**Cascading Impact:**")
                for impact in rc['cascading_impact']:
                    lines.append(f"- {impact}")
                lines.append("")
        
        # 时间线
        lines.append("## Timeline")
        lines.append("")
        lines.append("| Sequence | Time | Diff | Type | Severity | Source | Message |")
        lines.append("|----------|------|------|------|----------|--------|---------|")
        
        for event in report["timeline"]:
            lines.append(
                f"| {event['sequence']} | {event['timestamp']} | {event['time_diff']} | "
                f"{event['event_type']} | {event['severity']} | {event['source']} | "
                f"{event['message'][:50]}... |"
            )
        
        lines.append("")
        
        # 受影响的服务
        if report["affected_services"]:
            lines.append("## Affected Services")
            lines.append("")
            for service in report["affected_services"]:
                lines.append(f"- {service}")
            lines.append("")
        
        # 推荐动作
        lines.append("## Recommended Actions")
        lines.append("")
        for i, action in enumerate(report["recommended_actions"], 1):
            lines.append(f"{i}. {action}")
        lines.append("")
        
        # 影响分析
        lines.append("## Impact Analysis")
        lines.append("")
        impact = report["impact_analysis"]
        lines.append(f"**Estimated Blast Radius:** {impact['estimated_blast_radius']}")
        lines.append("")
        
        if impact["severity_distribution"]:
            lines.append("### Severity Distribution")
            lines.append("")
            for severity, count in impact["severity_distribution"].items():
                lines.append(f"- {severity}: {count}")
            lines.append("")
        
        if impact["event_type_distribution"]:
            lines.append("### Event Type Distribution")
            lines.append("")
            for event_type, count in impact["event_type_distribution"].items():
                lines.append(f"- {event_type}: {count}")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_to_html(self, result: RCAResult) -> str:
        """导出为 HTML 格式"""
        report = self.generate_report(result)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RCA Report: {result.id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        h3 {{ color: #888; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .severity-critical {{ color: #d32f2f; font-weight: bold; }}
        .severity-error {{ color: #f57c00; }}
        .severity-warning {{ color: #fbc02d; }}
        .severity-info {{ color: #388e3c; }}
        .confidence-high {{ color: #388e3c; }}
        .confidence-medium {{ color: #fbc02d; }}
        .confidence-low {{ color: #f57c00; }}
        ul {{ line-height: 1.8; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>RCA Report: {result.id}</h1>
    
    <div class="metadata">
        <p><strong>Generated:</strong> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>Analysis Method:</strong> {result.analysis_method}</p>
        <p><strong>Overall Confidence:</strong> {result.confidence * 100:.0f}%</p>
    </div>
"""
        
        # 摘要
        html += "<h2>Summary</h2><ul>"
        summary = report["summary"]
        html += f"<li><strong>Primary Root Cause:</strong> {summary['primary_root_cause']}</li>"
        html += f"<li><strong>Confidence:</strong> {summary['primary_confidence'] * 100:.0f}%</li>"
        html += f"<li><strong>Total Events:</strong> {summary['total_events_analyzed']}</li>"
        html += f"<li><strong>Root Causes:</strong> {summary['root_causes_identified']}</li>"
        html += f"<li><strong>Duration:</strong> {summary['incident_duration_minutes']} minutes</li>"
        html += "</ul>"
        
        # 根因
        html += "<h2>Root Causes</h2>"
        for rc in report["root_causes"]:
            confidence_class = "confidence-high" if rc["confidence"] > 0.7 else \
                              "confidence-medium" if rc["confidence"] > 0.4 else "confidence-low"
            
            html += f"<h3>#{rc['rank']} - {rc['event_type']}</h3>"
            html += f"<p><strong>Confidence:</strong> <span class='{confidence_class}'>{rc['confidence'] * 100:.0f}%</span></p>"
            html += f"<p><strong>Message:</strong> {rc['message']}</p>"
            html += f"<p><strong>Source:</strong> {rc['source']}</p>"
            
            if rc["evidence"]:
                html += "<p><strong>Evidence:</strong></p><ul>"
                for evidence in rc["evidence"]:
                    html += f"<li>{evidence}</li>"
                html += "</ul>"
        
        # 时间线
        html += """
        <h2>Timeline</h2>
        <table>
            <tr>
                <th>Sequence</th>
                <th>Time</th>
                <th>Diff</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Source</th>
            </tr>
"""
        for event in report["timeline"]:
            severity_class = f"severity-{event['severity']}"
            html += f"""
            <tr>
                <td>{event['sequence']}</td>
                <td>{event['timestamp']}</td>
                <td>{event['time_diff']}</td>
                <td>{event['event_type']}</td>
                <td class="{severity_class}">{event['severity']}</td>
                <td>{event['source']}</td>
            </tr>
"""
        html += "</table>"
        
        # 推荐动作
        html += "<h2>Recommended Actions</h2><ol>"
        for action in report["recommended_actions"]:
            html += f"<li>{action}</li>"
        html += "</ol>"
        
        html += "</body></html>"
        
        return html


# 全局报告生成器实例
report_generator = ReportGenerator()
