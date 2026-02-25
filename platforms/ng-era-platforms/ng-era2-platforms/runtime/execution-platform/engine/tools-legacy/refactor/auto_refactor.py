# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: auto_refactor
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    Auto Refactor - Simple CLI Entry Point
                    自動重構 - 簡易命令行入口
═══════════════════════════════════════════════════════════════════════════════
Purpose: Simple interface for triggering automated refactoring and evolution
Version: 1.0.0
Usage:
    # Start automated refactoring
    python auto_refactor.py start
    # Quick analysis only
    python auto_refactor.py quick-scan
    # Full evolution cycle
    python auto_refactor.py evolve
    # Check status
    python auto_refactor.py status
═══════════════════════════════════════════════════════════════════════════════
"""
import argparse
import asyncio
import sys
from pathlib import Path
from refactor_evolution_workflow import REPORTS_DIR, RefactorEvolutionWorkflow
# Add parent directory to path
# Note: Consider setting PYTHONPATH environment variable as an alternative
BASE_PATH = Path(__file__).parent.parent.parent
refactor_path = str(BASE_PATH / "tools" / "refactor")
if refactor_path not in sys.path:
    sys.path.insert(0, refactor_path)
# ============================================================================
# Simple Commands
# ============================================================================
async def cmd_start(args):
    """Start automated refactoring workflow"""
    print(
        """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    🤖 Auto Refactor & Evolution System                        ║
║                    自動重構與演化系統                                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    )
    workflow = RefactorEvolutionWorkflow()
    result = await workflow.run_full_workflow(mode=args.mode)
    return 0 if result["success"] else 1
async def cmd_quick_scan(args):
    """Quick analysis of codebase"""
    print("\n🔍 Quick Scan - Analyzing codebase structure...\n")
    workflow = RefactorEvolutionWorkflow()
    if not workflow._initialize_engines():
        print("❌ Failed to initialize engines")
        return 1
    result = await workflow._run_analysis_phase()
    print("\n" + "=" * 70)
    print("📊 Quick Scan Results:")
    print("=" * 70)
    print(f"✅ Targets analyzed: {result.get('targets_analyzed', 0)}")
    print(f"📄 Output file: {result.get('output_file', 'N/A')}")
    metrics = result.get("metrics", {})
    print(f"📁 Total files: {metrics.get('total_files', 0)}")
    print(f"⚠️  Total issues: {metrics.get('total_issues', 0)}")
    print()
    return 0 if result["success"] else 1
async def cmd_evolve(args):
    """Run evolution cycle"""
    print("\n🚀 Evolution Mode - Analyzing and improving system...\n")
    workflow = RefactorEvolutionWorkflow()
    result = await workflow.run_full_workflow(mode="autonomous")
    return 0 if result["success"] else 1
async def cmd_status(args):
    """Show status of latest workflow"""
    import yaml
    reports = sorted(REPORTS_DIR.glob("workflow_report_*.yaml"), reverse=True)
    if not reports:
        print("\n❌ No workflow reports found")
        print("💡 Run 'python auto_refactor.py start' to begin\n")
        return 1
    latest_report = reports[0]
    with open(latest_report, "r", encoding="utf-8") as f:
        report = yaml.safe_load(f)
    print("\n" + "=" * 70)
    print("📊 Latest Workflow Status")
    print("=" * 70)
    print(f"Workflow ID: {report.get('workflow_id', 'N/A')}")
    print(f"Status: {report.get('status', 'N/A')}")
    print(f"Duration: {report.get('duration_seconds', 0):.2f}s")
    print()
    print(
        f"Phases completed: {report.get('phases', {}).get('completed', 0)}/{report.get('phases', {}).get('total', 0)}"
    )
    if report.get("phases", {}).get("failed", 0) > 0:
        print(f"⚠️  Phases failed: {report.get('phases', {}).get('failed', 0)}")
    print()
    print(f"Success rate: {report.get('summary', {}).get('success_rate', 0)*100:.1f}%")
    print()
    print(f"Report file: {latest_report}")
    print()
    return 0
# ============================================================================
# Main
# ============================================================================
async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Auto Refactor - Automated Refactoring & Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start        Start automated refactoring workflow
  quick-scan   Quick analysis of codebase structure
  evolve       Run full evolution cycle
  status       Show status of latest workflow
Examples:
  python auto_refactor.py start
  python auto_refactor.py quick-scan
  python auto_refactor.py evolve
  python auto_refactor.py status
Chinese (中文):
  請開始：使用引擎自動化重構專案
  python auto_refactor.py start
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # start
    start_parser = subparsers.add_parser("start", help="Start automated refactoring")
    start_parser.add_argument(
        "--mode",
        "-m",
        choices=["autonomous", "supervised", "interactive"],
        default="autonomous",
        help="Execution mode",
    )
    # quick-scan
    subparsers.add_parser("quick-scan", help="Quick codebase analysis")
    # evolve
    subparsers.add_parser("evolve", help="Run evolution cycle")
    # status
    subparsers.add_parser("status", help="Show latest workflow status")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    # Route to command handlers
    handlers = {
        "start": cmd_start,
        "quick-scan": cmd_quick_scan,
        "evolve": cmd_evolve,
        "status": cmd_status,
    }
    handler = handlers.get(args.command)
    if handler:
        return await handler(args)
    else:
        parser.print_help()
        return 1
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
