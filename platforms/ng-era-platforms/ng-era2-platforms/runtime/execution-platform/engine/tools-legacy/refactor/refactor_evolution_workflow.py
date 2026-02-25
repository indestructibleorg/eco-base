# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: refactor_evolution_workflow
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
            Refactor & Evolution Workflow Orchestrator
            重構與演化工作流編排器
═══════════════════════════════════════════════════════════════════════════════
Purpose: Orchestrates automated refactoring and evolution using existing engines
Version: 1.0.0
Integration Points:
- tools/refactor/refactor_engine.py (RefactorEngine)
- automation/intelligent/synergymesh_core/self_evolution_engine.py (SelfEvolutionEngine)
- island-ai agents (architectural analysis)
- automation_launcher.py (pipeline execution)
Usage:
    # Full workflow
    python refactor_evolution_workflow.py run --mode autonomous
    # Individual phases
    python refactor_evolution_workflow.py analyze --target core/
    python refactor_evolution_workflow.py plan --analysis-result reports/analysis.yaml
    python refactor_evolution_workflow.py execute --plan reports/plan.yaml
    python refactor_evolution_workflow.py evolve --learning-data reports/metrics.yaml
    # Status and monitoring
    python refactor_evolution_workflow.py status
    python refactor_evolution_workflow.py report
═══════════════════════════════════════════════════════════════════════════════
"""
# MNGA-002: Import organization needs review
import argparse
import asyncio
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
# ============================================================================
# Path Configuration
# ============================================================================
BASE_PATH = Path(__file__).parent.parent.parent
# ============================================================================
# Configuration Constants
# ============================================================================
# These defaults can be overridden via configuration file
DEFAULT_VALIDATION_TIMEOUT = 300  # seconds for validation checks
DEFAULT_MIN_PROBLEMS_THRESHOLD = 20  # minimum problems to warrant refactoring
DEFAULT_MAX_PROBLEMS_THRESHOLD = 200  # maximum problems to handle in one cycle
CONFIG_PATH = BASE_PATH / "config" / "refactor-evolution.yaml"
REPORTS_DIR = BASE_PATH / "reports" / "refactor-evolution"
BACKUP_DIR = BASE_PATH / ".refactor-backups"
# Add paths to Python path
sys.path.insert(0, str(BASE_PATH / "tools" / "refactor"))
sys.path.insert(0, str(BASE_PATH / "automation" / "intelligent"))
# ============================================================================
# Data Structures
# ============================================================================
class WorkflowPhase(Enum):
    """Workflow execution phases"""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    LEARNING = "learning"
    EVOLUTION = "evolution"
    VALIDATION = "validation"
class WorkflowStatus(Enum):
    """Workflow execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class WorkflowState:
    """Current workflow state"""
    workflow_id: str
    status: WorkflowStatus
    current_phase: Optional[WorkflowPhase]
    completed_phases: List[WorkflowPhase] = field(default_factory=list)
    failed_phases: List[WorkflowPhase] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
@dataclass
class PhaseResult:
    """Result of a workflow phase execution"""
    phase: WorkflowPhase
    success: bool
    duration_seconds: float
    output: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
# ============================================================================
# Workflow Orchestrator
# ============================================================================
class RefactorEvolutionWorkflow:
    """
    Orchestrates automated refactoring and evolution workflow
    Responsibilities:
    1. Load configuration and initialize engines
    2. Execute workflow phases in sequence
    3. Coordinate between refactor and evolution engines
    4. Manage state and error handling
    5. Generate reports and metrics
    """
    def __init__(self, config_path: Path = CONFIG_PATH):
        """Initialize workflow orchestrator"""
        self.config_path = config_path
        self.config = self._load_config()
        self.state = None
        self.refactor_engine = None
        self.evolution_engine = None
        self.results: List[PhaseResult] = []
        # Ensure output directories exist
        self._ensure_directories()
    def _load_config(self) -> Dict[str, Any]:
        """Load workflow configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    def _ensure_directories(self):
        """Ensure output directories exist"""
        for dir_key in ["reports_dir", "plans_dir", "logs_dir", "backup_dir"]:
            dir_path = Path(self.config["output"].get(dir_key, ""))
            if dir_path:
                dir_path = (
                    BASE_PATH / dir_path if not dir_path.is_absolute() else dir_path
                )
                dir_path.mkdir(parents=True, exist_ok=True)
    def _initialize_engines(self):
        """Initialize refactor and evolution engines"""
        try:
            # Check refactor engine exists
            refactor_engine_path = (
                BASE_PATH / self.config["engines"]["refactor_engine"]["path"]
            )
            if not refactor_engine_path.exists():
                print(f"❌ Refactor engine not found: {refactor_engine_path}")
                return False
            self.refactor_engine = str(refactor_engine_path)
            print("✅ Refactor engine found")
            # Check evolution engine exists
            evolution_engine_path = (
                BASE_PATH / self.config["engines"]["evolution_engine"]["path"]
            )
            if not evolution_engine_path.exists():
                print(f"⚠️  Evolution engine not found: {evolution_engine_path}")
                self.evolution_engine = None
            else:
                self.evolution_engine = str(evolution_engine_path)
                print("✅ Evolution engine found")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize engines: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _create_backup(self) -> bool:
        """Create backup of target directories"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DIR / timestamp
            backup_path.mkdir(parents=True, exist_ok=True)
            targets = self.config.get("targets", {}).get("primary", [])
            for target in targets:
                target_path = BASE_PATH / target["path"]
                if target_path.exists():
                    dest_path = backup_path / target["path"]
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if target_path.is_dir():
                        shutil.copytree(target_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(target_path, dest_path)
            print(f"✅ Backup created: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    def _run_safety_checks(self, check_type: str = "pre") -> bool:
        """Run safety checks before/after execution"""
        checks_key = f"{check_type}_checks"
        checks = self.config.get("safety", {}).get(checks_key, [])
        print(f"\n🔍 Running {check_type}-execution safety checks...")
        all_passed = True
        for check in checks:
            result = self._execute_safety_check(check)
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
            all_passed = all_passed and result
        return all_passed
    def _execute_safety_check(self, check: str) -> bool:
        """Execute a specific safety check"""
        try:
            if check == "git_status_clean":
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=BASE_PATH,
                    capture_output=True,
                    text=True,
                )
                return len(result.stdout.strip()) == 0
            elif check == "no_uncommitted_changes":
                result = subprocess.run(["git", "diff", "--quiet"], cwd=BASE_PATH)
                return result.returncode == 0
            elif check == "backup_created":
                return len(list(BACKUP_DIR.glob("*"))) > 0
            elif check in ["tests_passing", "tests_still_passing"]:
                # Check if tests exist and can be run
                if (BASE_PATH / "package.json").exists():
                    result = subprocess.run(
                        ["npm", "test", "--", "--passWithNoTests"],
                        cwd=BASE_PATH,
                        capture_output=True,
                        timeout=300,
                    )
                    return result.returncode == 0
                return True  # Pass if no tests defined
            elif check == "no_syntax_errors":
                # Basic syntax check for Python files
                py_files = [str(p) for p in BASE_PATH.glob("**/*.py")]
                result = subprocess.run(
                    ["python", "-m", "py_compile"] + py_files,
                    cwd=BASE_PATH,
                    capture_output=True,
                )
                return result.returncode == 0
            else:
                print(f"⚠️  Unknown check: {check}")
                return True  # Don't fail on unknown checks
        except Exception as e:
            print(f"⚠️  Check {check} failed with error: {e}")
            return False
    async def run_phase(self, phase: WorkflowPhase) -> PhaseResult:
        """Execute a single workflow phase"""
        print(f"\n{'='*70}")
        print(f"📍 Phase: {phase.value.upper()}")
        print(f"{'='*70}")
        start_time = datetime.now()
        try:
            if phase == WorkflowPhase.ANALYSIS:
                result = await self._run_analysis_phase()
            elif phase == WorkflowPhase.PLANNING:
                result = await self._run_planning_phase()
            elif phase == WorkflowPhase.EXECUTION:
                result = await self._run_execution_phase()
            elif phase == WorkflowPhase.LEARNING:
                result = await self._run_learning_phase()
            elif phase == WorkflowPhase.EVOLUTION:
                result = await self._run_evolution_phase()
            elif phase == WorkflowPhase.VALIDATION:
                result = await self._run_validation_phase()
            else:
                result = {"success": False, "error": f"Unknown phase: {phase}"}
            duration = (datetime.now() - start_time).total_seconds()
            phase_result = PhaseResult(
                phase=phase,
                success=result.get("success", False),
                duration_seconds=duration,
                output=result,
                metrics=result.get("metrics", {}),
                error=result.get("error"),
            )
            status = "✅" if phase_result.success else "❌"
            print(f"{status} Phase completed in {duration:.2f}s")
            return phase_result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"❌ Phase failed: {e}")
            import traceback
            traceback.print_exc()
            return PhaseResult(
                phase=phase,
                success=False,
                duration_seconds=duration,
                output={},
                error=str(e),
            )
    async def _run_analysis_phase(self) -> Dict[str, Any]:
        """Run analysis phase using refactor engine"""
        print("🔍 Analyzing codebase structure...")
        targets = self.config.get("targets", {}).get("primary", [])
        results = []
        for target in targets:
            target_path = BASE_PATH / target["path"]
            if not target_path.exists():
                print(f"⚠️  Target not found: {target_path}")
                continue
            print(f"  📂 Analyzing: {target['path']}")
            # Run refactor engine analysis
            analysis = self._analyze_target(target_path, target.get("focus", []))
            results.append(
                {
                    "path": target["path"],
                    "priority": target.get("priority", "medium"),
                    "analysis": analysis,
                }
            )
        # Save analysis results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = REPORTS_DIR / f"analysis_{timestamp}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "timestamp": timestamp,
                    "targets": results,
                    "summary": self._summarize_analysis(results),
                },
                f,
                allow_unicode=True,
                default_flow_style=False,
            )
        print(f"📄 Analysis saved: {output_file}")
        return {
            "success": True,
            "output_file": str(output_file),
            "targets_analyzed": len(results),
            "metrics": self._extract_analysis_metrics(results),
        }
    def _analyze_target(
        self, target_path: Path, focus_areas: List[str]
    ) -> Dict[str, Any]:
        """Analyze a specific target directory"""
        analysis = {
            "path": str(target_path),
            "exists": target_path.exists(),
            "is_directory": target_path.is_dir() if target_path.exists() else False,
            "focus_areas": focus_areas,
            "issues": [],
            "recommendations": [],
        }
        if not target_path.exists():
            return analysis
        # Count files and structure
        if target_path.is_dir():
            py_files = list(target_path.rglob("*.py"))
            js_files = list(target_path.rglob("*.js"))
            ts_files = list(target_path.rglob("*.ts"))
            analysis["file_counts"] = {
                "python": len(py_files),
                "javascript": len(js_files),
                "typescript": len(ts_files),
                "total": len(py_files) + len(js_files) + len(ts_files),
            }
            # Simple structure analysis
            subdirs = [
                d
                for d in target_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            analysis["subdirectories"] = len(subdirs)
            # Check for common issues
            if "structure" in focus_areas:
                min_threshold = self.config.get("analysis", {}).get(
                    "min_problems_threshold", DEFAULT_MIN_PROBLEMS_THRESHOLD
                )
                max_threshold = self.config.get("analysis", {}).get(
                    "max_problems_threshold", DEFAULT_MAX_PROBLEMS_THRESHOLD
                )
                if len(subdirs) > min_threshold:
                    analysis["issues"].append(
                        "High number of subdirectories - consider consolidation"
                    )
                if analysis["file_counts"]["total"] > max_threshold:
                    analysis["issues"].append(
                        "Large number of files - consider modularization"
                    )
            if "organization" in focus_areas:
                has_init = (target_path / "__init__.py").exists()
                has_readme = (target_path / "README.md").exists()
                if not has_init and analysis["file_counts"]["python"] > 0:
                    analysis["issues"].append("Missing __init__.py for Python package")
                if not has_readme:
                    analysis["recommendations"].append(
                        "Add README.md for documentation"
                    )
        return analysis
    def _summarize_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary of analysis results"""
        total_issues = sum(
            len(r.get("analysis", {}).get("issues", [])) for r in results
        )
        total_recommendations = sum(
            len(r.get("analysis", {}).get("recommendations", [])) for r in results
        )
        return {
            "targets_analyzed": len(results),
            "total_issues": total_issues,
            "total_recommendations": total_recommendations,
            "needs_refactoring": total_issues > 0,
        }
    def _extract_analysis_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Extract metrics from analysis results"""
        return {
            "targets_analyzed": len(results),
            "total_files": sum(
                r.get("analysis", {}).get("file_counts", {}).get("total", 0)
                for r in results
            ),
            "total_issues": sum(
                len(r.get("analysis", {}).get("issues", [])) for r in results
            ),
        }
    async def _run_planning_phase(self) -> Dict[str, Any]:
        """Run planning phase to create execution plan"""
        print("📋 Creating execution plan...")
        # Find latest analysis
        analysis_files = sorted(REPORTS_DIR.glob("analysis_*.yaml"), reverse=True)
        if not analysis_files:
            return {"success": False, "error": "No analysis results found"}
        latest_analysis = analysis_files[0]
        with open(latest_analysis, "r", encoding="utf-8") as f:
            analysis = yaml.safe_load(f)
        # Create execution plan
        plan = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "based_on_analysis": str(latest_analysis),
                "mode": self.config["workflow"]["mode"],
            },
            "phases": [],
            "validation": {},
        }
        # Generate plan phases based on analysis
        summary = analysis.get("summary", {})
        if summary.get("needs_refactoring", False):
            plan["phases"].append(
                {
                    "id": 1,
                    "name": "Structure Optimization",
                    "priority": "P1",
                    "description": "Optimize directory structure and organization",
                    "steps": [
                        "Consolidate scattered files",
                        "Improve naming consistency",
                        "Add missing documentation",
                    ],
                }
            )
        # Save plan
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = REPORTS_DIR / "plans" / f"plan_{timestamp}.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(plan, f, allow_unicode=True, default_flow_style=False)
        print(f"📄 Plan saved: {output_file}")
        return {
            "success": True,
            "output_file": str(output_file),
            "phases_planned": len(plan["phases"]),
            "metrics": {"phases": len(plan["phases"])},
        }
    async def _run_execution_phase(self) -> Dict[str, Any]:
        """Run execution phase to apply changes"""
        print("⚙️  Executing refactoring plan...")
        # This is a safe no-op execution for now
        # In real implementation, this would apply actual changes
        return {
            "success": True,
            "changes_applied": 0,
            "dry_run": True,
            "message": "Execution phase completed (dry-run mode)",
            "metrics": {"changes_applied": 0},
        }
    async def _run_learning_phase(self) -> Dict[str, Any]:
        """Run learning phase using evolution engine"""
        print("📚 Learning from execution results...")
        if not self.evolution_engine:
            return {
                "success": True,
                "message": "Learning phase skipped (evolution engine not available)",
                "metrics": {},
            }
        # Collect learning data
        learning_data = {
            "execution_results": [
                r for r in self.results if r.phase == WorkflowPhase.EXECUTION
            ],
            "metrics": {
                "total_phases": len(self.results),
                "successful_phases": len([r for r in self.results if r.success]),
            },
        }
        return {
            "success": True,
            "insights_collected": 0,
            "metrics": learning_data["metrics"],
        }
    async def _run_evolution_phase(self) -> Dict[str, Any]:
        """Run evolution phase to identify improvements"""
        print("🚀 Identifying evolution opportunities...")
        if not self.evolution_engine:
            return {
                "success": True,
                "message": "Evolution phase skipped (evolution engine not available)",
                "metrics": {},
            }
        return {
            "success": True,
            "opportunities_identified": 0,
            "optimizations_applied": 0,
            "metrics": {"opportunities": 0, "optimizations": 0},
        }
    async def _run_validation_phase(self) -> Dict[str, Any]:
        """Run validation phase to verify changes"""
        print("✅ Validating changes...")
        # Run post-execution safety checks
        checks_passed = self._run_safety_checks("post")
        return {
            "success": checks_passed,
            "checks_passed": checks_passed,
            "metrics": {"validation_passed": checks_passed},
        }
    async def run_full_workflow(self, mode: str = "autonomous") -> Dict[str, Any]:
        """Run complete refactor and evolution workflow"""
        print(f"\n{'='*70}")
        print("🚀 Starting Refactor & Evolution Workflow")
        print(f"{'='*70}")
        print(f"Mode: {mode}")
        print(f"Config: {self.config_path}")
        print()
        # Initialize state
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = WorkflowState(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_phase=None,
            start_time=datetime.now(),
        )
        # Initialize engines
        if not self._initialize_engines():
            self.state.status = WorkflowStatus.FAILED
            self.state.error = "Failed to initialize engines"
            return {"success": False, "error": self.state.error}
        # Pre-execution safety checks
        if not self._run_safety_checks("pre"):
            print("\n⚠️  Pre-execution safety checks failed")
            if mode == "autonomous":
                print("❌ Aborting in autonomous mode")
                self.state.status = WorkflowStatus.FAILED
                return {"success": False, "error": "Safety checks failed"}
        # Create backup
        if not self._create_backup():
            print("\n⚠️  Backup creation failed")
            if self.config["safety"].get("backup_required", True):
                self.state.status = WorkflowStatus.FAILED
                return {"success": False, "error": "Backup required but failed"}
        # Execute workflow phases
        phases = [
            WorkflowPhase.ANALYSIS,
            WorkflowPhase.PLANNING,
            WorkflowPhase.EXECUTION,
            WorkflowPhase.LEARNING,
            WorkflowPhase.EVOLUTION,
            WorkflowPhase.VALIDATION,
        ]
        for phase in phases:
            self.state.current_phase = phase
            result = await self.run_phase(phase)
            self.results.append(result)
            if result.success:
                self.state.completed_phases.append(phase)
            else:
                self.state.failed_phases.append(phase)
                # Check if we should continue or abort
                if mode == "autonomous" and phase in [
                    WorkflowPhase.ANALYSIS,
                    WorkflowPhase.VALIDATION,
                ]:
                    print(
                        f"\n❌ Critical phase {phase.value} failed, aborting workflow"
                    )
                    self.state.status = WorkflowStatus.FAILED
                    self.state.error = f"Phase {phase.value} failed"
                    break
        # Finalize
        self.state.end_time = datetime.now()
        self.state.status = (
            WorkflowStatus.COMPLETED
            if not self.state.failed_phases
            else WorkflowStatus.FAILED
        )
        self.state.current_phase = None
        # Generate final report
        report = self._generate_report()
        print(f"\n{'='*70}")
        print(
            f"{'✅' if self.state.status == WorkflowStatus.COMPLETED else '❌'} Workflow {self.state.status.value}"
        )
        print(f"{'='*70}")
        print(
            f"Duration: {(self.state.end_time - self.state.start_time).total_seconds():.2f}s"
        )
        print(f"Phases completed: {len(self.state.completed_phases)}/{len(phases)}")
        if self.state.failed_phases:
            print(
                f"Phases failed: {', '.join(p.value for p in self.state.failed_phases)}"
            )
        print()
        return {
            "success": self.state.status == WorkflowStatus.COMPLETED,
            "workflow_id": workflow_id,
            "report": report,
            "state": asdict(self.state),
        }
    def _generate_report(self) -> Dict[str, Any]:
        """Generate workflow execution report"""
        report = {
            "workflow_id": self.state.workflow_id,
            "status": self.state.status.value,
            "duration_seconds": (
                self.state.end_time - self.state.start_time
            ).total_seconds(),
            "phases": {
                "total": len(self.results),
                "completed": len(self.state.completed_phases),
                "failed": len(self.state.failed_phases),
            },
            "results": [
                {
                    "phase": r.phase.value,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in self.results
            ],
            "summary": {
                "total_duration": (
                    self.state.end_time - self.state.start_time
                ).total_seconds(),
                "success_rate": (
                    len(self.state.completed_phases) / len(self.results)
                    if self.results
                    else 0
                ),
            },
        }
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"workflow_report_{timestamp}.yaml"
        with open(report_file, "w", encoding="utf-8") as f:
            yaml.dump(report, f, allow_unicode=True, default_flow_style=False)
        print(f"📄 Report saved: {report_file}")
        return report
# ============================================================================
# CLI Interface
# ============================================================================
async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Refactor & Evolution Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow
  python refactor_evolution_workflow.py run --mode autonomous
  # Run specific phase
  python refactor_evolution_workflow.py analyze
  python refactor_evolution_workflow.py plan
  # Check status
  python refactor_evolution_workflow.py status
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # run command
    run_parser = subparsers.add_parser("run", help="Run full workflow")
    run_parser.add_argument(
        "--mode",
        "-m",
        choices=["autonomous", "supervised", "interactive"],
        default="autonomous",
        help="Execution mode",
    )
    run_parser.add_argument("--config", "-c", help="Configuration file path")
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis phase only")
    analyze_parser.add_argument("--target", "-t", help="Specific target to analyze")
    # plan command
    subparsers.add_parser("plan", help="Run planning phase only")
    # execute command
    execute_parser = subparsers.add_parser("execute", help="Run execution phase only")
    execute_parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    # status command
    subparsers.add_parser("status", help="Show workflow status")
    # report command
    subparsers.add_parser("report", help="Show latest report")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    # Initialize workflow
    config_path = (
        Path(args.config) if hasattr(args, "config") and args.config else CONFIG_PATH
    )
    workflow = RefactorEvolutionWorkflow(config_path)
    # Execute command
    if args.command == "run":
        result = await workflow.run_full_workflow(mode=args.mode)
        sys.exit(0 if result["success"] else 1)
    elif args.command == "analyze":
        if not workflow._initialize_engines():
            print("❌ Failed to initialize engines")
            sys.exit(1)
        result = await workflow._run_analysis_phase()
        print(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        sys.exit(0 if result["success"] else 1)
    elif args.command == "plan":
        result = await workflow._run_planning_phase()
        print(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        sys.exit(0 if result["success"] else 1)
    elif args.command == "execute":
        if not workflow._initialize_engines():
            print("❌ Failed to initialize engines")
            sys.exit(1)
        result = await workflow._run_execution_phase()
        print(yaml.dump(result, allow_unicode=True, default_flow_style=False))
        sys.exit(0 if result["success"] else 1)
    elif args.command == "status":
        # Show latest workflow status
        reports = sorted(REPORTS_DIR.glob("workflow_report_*.yaml"), reverse=True)
        if reports:
            with open(reports[0], "r", encoding="utf-8") as f:
                report = yaml.safe_load(f)
            print(yaml.dump(report, allow_unicode=True, default_flow_style=False))
        else:
            print("No workflow reports found")
    elif args.command == "report":
        # Show latest detailed report
        reports = sorted(REPORTS_DIR.glob("workflow_report_*.yaml"), reverse=True)
        if reports:
            with open(reports[0], "r", encoding="utf-8") as f:
                report = yaml.safe_load(f)
            print(yaml.dump(report, allow_unicode=True, default_flow_style=False))
        else:
            print("No reports found")
if __name__ == "__main__":
    asyncio.run(main())
