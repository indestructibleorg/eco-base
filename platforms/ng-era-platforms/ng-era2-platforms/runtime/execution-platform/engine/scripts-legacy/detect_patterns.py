#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: detect_patterns
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Pattern Detection for Intermittent Failures
Analyzes logs to identify recurring error patterns
"""
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
class ErrorPatternAnalyzer:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.patterns = defaultdict(lambda: {"count": 0, "timestamps": [], "contexts": []})
        self.error_types = Counter()
        self.time_distribution = defaultdict(int)
    def analyze_logs(self):
        """Analyze all logs in directory"""
        log_files = list(self.log_dir.glob("*.log")) + list(self.log_dir.glob("*.txt"))
        for log_file in log_files:
            print(f"Analyzing: {log_file.name}")
            self._analyze_file(log_file)
        self._generate_report()
    def _analyze_file(self, log_file):
        """Analyze individual log file"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                for line in lines:
                    self._extract_patterns(line)
                    self._extract_time_patterns(line)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    def _extract_patterns(self, line):
        """Extract error patterns from log line"""
        # Common error patterns
        patterns = [
            r'Error:\s*(.+?)(?:\n|$)',
            r'Exception:\s*(.+?)(?:\n|$)',
            r'Failed\s+to\s+(.+?)(?:\n|$)',
            r'Timeout\s+(.+?)(?:\n|$)',
            r'Connection\s+(.+?)(?:\n|$)',
            r'race\s+condition',
            r'deadlock',
            r'concurrent',
            r'timeout',
            r'retry',
            r'attempt\s+\d+',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            for match in matches:
                pattern_key = match[:100]  # Limit length
                self.patterns[pattern_key]["count"] += 1
                self.patterns[pattern_key]["timestamps"].append(datetime.now().isoformat())
                self.patterns[pattern_key]["contexts"].append(line[:200])
                self.error_types[match.split()[0].lower()] += 1
    def _extract_time_patterns(self, line):
        """Extract time-based patterns"""
        time_patterns = [
            r'\d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
        ]
        for pattern in time_patterns:
            if re.search(pattern, line):
                # Extract hour for time distribution
                hour_match = re.search(r'(\d{2}):', line)
                if hour_match:
                    hour = int(hour_match.group(1))
                    self.time_distribution[hour] += 1
                break
    def _generate_report(self):
        """Generate analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"pattern_analysis_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Error Pattern Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            # Top error patterns
            f.write("## Top Error Patterns\n\n")
            sorted_patterns = sorted(self.patterns.items(), 
                                    key=lambda x: x[1]["count"], 
                                    reverse=True)[:20]
            for pattern, data in sorted_patterns:
                f.write(f"### Pattern: `{pattern}`\n")
                f.write(f"- **Occurrences:** {data['count']}\n")
                f.write(f"- **Sample Context:**\n```\n{data['contexts'][0] if data['contexts'] else 'N/A'}\n```\n\n")
            # Error type distribution
            f.write("## Error Type Distribution\n\n")
            for error_type, count in self.error_types.most_common(10):
                f.write(f"- **{error_type}:** {count}\n")
            # Time distribution
            f.write("\n## Time Distribution of Errors\n\n")
            for hour, count in sorted(self.time_distribution.items()):
                f.write(f"- **Hour {hour:02d}:** {count} errors\n")
            # Recommendations
            f.write("\n## Recommendations\n\n")
            f.write("Based on the analysis:\n\n")
            if any('timeout' in p.lower() for p in self.patterns.keys()):
                f.write("- ⚠️ **Timeout issues detected** - Consider increasing timeout values or implementing retry logic\n")
            if any('race' in p.lower() for p in self.patterns.keys()):
                f.write("- ⚠️ **Race condition indicators** - Review concurrent operations and add proper locking\n")
            if any('concurrent' in p.lower() for p in self.patterns.keys()):
                f.write("- ⚠️ **Concurrency issues** - Review thread safety and synchronization\n")
            if self.time_distribution:
                peak_hour = max(self.time_distribution.items(), key=lambda x: x[1])
                f.write(f"- ⚠️ **Peak error hour: {peak_hour[0]:02d}:00** - Check for scheduled tasks or load patterns\n")
        print(f"\n✅ Report generated: {report_file}")
        return report_file
if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"
    analyzer = ErrorPatternAnalyzer(log_dir)
    analyzer.analyze_logs()