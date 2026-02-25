# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: emergency_mode_manager
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
緊急模式管理器 - Emergency Mode Manager
MachineNativeOps 驗證系統 v1.0.0
此模組管理驗證系統的緊急降級和恢復流程。
"""
# MNGA-002: Import organization needs review
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
# 配置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
class EmergencyLevel(Enum):
    """緊急等級枚舉"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
class FallbackStrategy(Enum):
    """降級策略枚舉"""
    STANDBY = "standby"
    LIGHTWEIGHT = "lightweight"
    CLASSIC_ONLY = "classic_only"
    CLASSIC_AGGRESSIVE = "classic_aggressive"
    FULL_FALLBACK = "full_fallback"
@dataclass
class SystemHealth:
    """系統健康狀態"""
    quantum_backend_status: str = "active"
    coherence: float = 0.792
    noise_level: float = 0.134
    error_rate: float = 0.02
    latency_ms: int = 172
    throughput: float = 47.0
@dataclass
class EmergencyState:
    """緊急狀態"""
    level: EmergencyLevel = EmergencyLevel.NORMAL
    strategy: FallbackStrategy = FallbackStrategy.STANDBY
    activated_at: Optional[str] = None
    reason: str = ""
    auto_recovery_enabled: bool = True
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
class HealthMonitor:
    """健康監控器"""
    def __init__(self):
        self.thresholds = {
            "coherence_warning": 0.78,
            "coherence_critical": 0.70,
            "noise_warning": 0.15,
            "noise_critical": 0.20,
            "error_rate_warning": 0.03,
            "error_rate_critical": 0.05,
            "latency_warning": 400,
            "latency_critical": 800,
        }
    def assess_health(self, health: SystemHealth) -> EmergencyLevel:
        """
        評估系統健康狀態
        Args:
            health: 當前系統健康狀態
        Returns:
            EmergencyLevel: 評估的緊急等級
        """
        issues = []
        # 檢查相干性
        if health.coherence < self.thresholds["coherence_critical"]:
            issues.append(("critical", "coherence"))
        elif health.coherence < self.thresholds["coherence_warning"]:
            issues.append(("warning", "coherence"))
        # 檢查噪聲水平
        if health.noise_level > self.thresholds["noise_critical"]:
            issues.append(("critical", "noise"))
        elif health.noise_level > self.thresholds["noise_warning"]:
            issues.append(("warning", "noise"))
        # 檢查錯誤率
        if health.error_rate > self.thresholds["error_rate_critical"]:
            issues.append(("critical", "error_rate"))
        elif health.error_rate > self.thresholds["error_rate_warning"]:
            issues.append(("warning", "error_rate"))
        # 檢查延遲
        if health.latency_ms > self.thresholds["latency_critical"]:
            issues.append(("critical", "latency"))
        elif health.latency_ms > self.thresholds["latency_warning"]:
            issues.append(("warning", "latency"))
        # 確定緊急等級
        critical_count = sum(1 for level, _ in issues if level == "critical")
        warning_count = sum(1 for level, _ in issues if level == "warning")
        if critical_count >= 2:
            return EmergencyLevel.EMERGENCY
        elif critical_count >= 1:
            return EmergencyLevel.CRITICAL
        elif warning_count >= 2:
            return EmergencyLevel.WARNING
        else:
            return EmergencyLevel.NORMAL
class EmergencyModeManager:
    """
    緊急模式管理器
    負責監控系統健康狀態，在異常情況下啟用降級策略，
    並在條件恢復時自動恢復正常運行。
    """
    def __init__(self):
        self.state = EmergencyState()
        self.monitor = HealthMonitor()
        self.history = []
    def check_and_respond(self, health: SystemHealth) -> dict:
        """
        檢查系統健康並響應
        Args:
            health: 當前系統健康狀態
        Returns:
            dict: 響應結果
        """
        level = self.monitor.assess_health(health)
        previous_level = self.state.level
        response = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "health_assessment": {
                "coherence": health.coherence,
                "noise_level": health.noise_level,
                "error_rate": health.error_rate,
                "latency_ms": health.latency_ms,
            },
            "previous_level": previous_level.value,
            "current_level": level.value,
            "action_taken": "none",
        }
        # 狀態轉換邏輯
        if (
            level == EmergencyLevel.EMERGENCY
            and previous_level != EmergencyLevel.EMERGENCY
        ):
            self._activate_emergency(health, "Multiple critical conditions detected")
            response["action_taken"] = "emergency_activated"
        elif (
            level == EmergencyLevel.CRITICAL and previous_level == EmergencyLevel.NORMAL
        ):
            self._activate_fallback(
                FallbackStrategy.LIGHTWEIGHT, "Critical condition detected"
            )
            response["action_taken"] = "fallback_activated"
        elif (
            level == EmergencyLevel.WARNING and previous_level == EmergencyLevel.NORMAL
        ):
            self._activate_fallback(
                FallbackStrategy.STANDBY, "Warning conditions detected"
            )
            response["action_taken"] = "standby_activated"
        elif level == EmergencyLevel.NORMAL and previous_level != EmergencyLevel.NORMAL:
            self._attempt_recovery()
            response["action_taken"] = "recovery_attempted"
        response["current_state"] = {
            "level": self.state.level.value,
            "strategy": self.state.strategy.value,
            "recovery_attempts": self.state.recovery_attempts,
        }
        self.history.append(response)
        return response
    def _activate_emergency(self, health: SystemHealth, reason: str):
        """啟用緊急模式"""
        logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")
        self.state.level = EmergencyLevel.EMERGENCY
        self.state.strategy = FallbackStrategy.CLASSIC_AGGRESSIVE
        self.state.activated_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.state.reason = reason
        self.state.recovery_attempts = 0
    def _activate_fallback(self, strategy: FallbackStrategy, reason: str):
        """啟用降級策略"""
        logger.warning(f"Fallback strategy activated: {strategy.value} - {reason}")
        self.state.level = EmergencyLevel.WARNING
        self.state.strategy = strategy
        self.state.activated_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.state.reason = reason
    def _attempt_recovery(self):
        """嘗試恢復"""
        if not self.state.auto_recovery_enabled:
            logger.info("Auto-recovery disabled, manual intervention required")
            return
        self.state.recovery_attempts += 1
        if self.state.recovery_attempts >= self.state.max_recovery_attempts:
            logger.warning(
                f"Max recovery attempts ({self.state.max_recovery_attempts}) reached"
            )
            return
        logger.info(
            f"Recovery attempt {self.state.recovery_attempts}/{self.state.max_recovery_attempts}"
        )
        # 恢復到正常狀態
        self.state.level = EmergencyLevel.NORMAL
        self.state.strategy = FallbackStrategy.STANDBY
        self.state.reason = "System recovered"
    def force_emergency(self, reason: str = "Manual activation"):
        """強制啟用緊急模式"""
        logger.warning(f"Emergency mode forced: {reason}")
        self._activate_emergency(SystemHealth(), reason)
    def force_recovery(self):
        """強制恢復"""
        logger.info("Forcing recovery to normal state")
        self.state.level = EmergencyLevel.NORMAL
        self.state.strategy = FallbackStrategy.STANDBY
        self.state.reason = "Forced recovery"
        self.state.recovery_attempts = 0
    def get_status(self) -> dict:
        """獲取當前狀態"""
        return {
            "emergency_mode": self.state.level == EmergencyLevel.EMERGENCY,
            "level": self.state.level.value,
            "strategy": self.state.strategy.value,
            "activated_at": self.state.activated_at,
            "reason": self.state.reason,
            "recovery_attempts": self.state.recovery_attempts,
            "auto_recovery_enabled": self.state.auto_recovery_enabled,
        }
    def get_cli_status(self) -> str:
        """獲取 CLI 格式的狀態"""
        status = self.get_status()
        lines = [
            f"EMERGENCY_MODE: {'Active' if status['emergency_mode'] else 'Inactive'}",
            f"LEVEL: {status['level'].upper()}",
            f"FALLBACK_STRATEGY: {status['strategy'].replace('_', ' ').title()}",
            f"AUTO_RECOVERY: {'Enabled' if status['auto_recovery_enabled'] else 'Disabled'}",
            f"RECOVERY_ATTEMPTS: {status['recovery_attempts']}/{self.state.max_recovery_attempts}",
        ]
        if status["activated_at"]:
            lines.append(f"ACTIVATED_AT: {status['activated_at']}")
        if status["reason"]:
            lines.append(f"REASON: {status['reason']}")
        return "\n".join(lines)
def run_demo():
    """運行演示"""
    manager = EmergencyModeManager()
    print("\n" + "=" * 70)
    print("演示場景：正常 -> 警告 -> 緊急 -> 恢復")
    print("=" * 70)
    # 場景 1：正常狀態
    print("\n📗 場景 1：正常運行")
    print("-" * 50)
    health_normal = SystemHealth(
        coherence=0.85, noise_level=0.10, error_rate=0.01, latency_ms=150
    )
    result = manager.check_and_respond(health_normal)
    print(
        f"健康評估: coherence={health_normal.coherence}, noise={health_normal.noise_level}"
    )
    print(f"行動: {result['action_taken']}")
    print(f"等級: {result['current_level']}")
    # 場景 2：警告狀態
    print("\n📙 場景 2：警告條件")
    print("-" * 50)
    health_warning = SystemHealth(
        coherence=0.76, noise_level=0.16, error_rate=0.02, latency_ms=200
    )
    result = manager.check_and_respond(health_warning)
    print(
        f"健康評估: coherence={health_warning.coherence}, noise={health_warning.noise_level}"
    )
    print(f"行動: {result['action_taken']}")
    print(f"等級: {result['current_level']}")
    # 場景 3：緊急狀態
    print("\n📕 場景 3：緊急條件")
    print("-" * 50)
    health_emergency = SystemHealth(
        coherence=0.65, noise_level=0.25, error_rate=0.08, latency_ms=900
    )
    result = manager.check_and_respond(health_emergency)
    print(
        f"健康評估: coherence={health_emergency.coherence}, noise={health_emergency.noise_level}"
    )
    print(f"行動: {result['action_taken']}")
    print(f"等級: {result['current_level']}")
    # 場景 4：恢復
    print("\n📗 場景 4：條件恢復")
    print("-" * 50)
    health_recovered = SystemHealth(
        coherence=0.82, noise_level=0.12, error_rate=0.02, latency_ms=180
    )
    result = manager.check_and_respond(health_recovered)
    print(
        f"健康評估: coherence={health_recovered.coherence}, noise={health_recovered.noise_level}"
    )
    print(f"行動: {result['action_taken']}")
    print(f"等級: {result['current_level']}")
    # 最終狀態
    print("\n" + "=" * 70)
    print("最終狀態")
    print("=" * 70)
    print(manager.get_cli_status())
def main():
    """主函數"""
    import argparse
    parser = argparse.ArgumentParser(
        description="緊急模式管理器 - MachineNativeOps 驗證系統"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["status", "activate", "recover", "demo"],
        default="status",
        help="執行的命令",
    )
    parser.add_argument(
        "--reason", default="Manual activation", help="啟用緊急模式的原因"
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式輸出")
    args = parser.parse_args()
    print("=" * 70)
    print("MachineNativeOps 緊急模式管理器 v1.0.0")
    print("=" * 70)
    manager = EmergencyModeManager()
    if args.command == "status":
        if args.json:
            print(json.dumps(manager.get_status(), indent=2))
        else:
            print("\n當前狀態:")
            print("-" * 50)
            print(manager.get_cli_status())
    elif args.command == "activate":
        manager.force_emergency(args.reason)
        print("\n✅ 緊急模式已啟用")
        print(f"原因: {args.reason}")
        print("\n" + manager.get_cli_status())
    elif args.command == "recover":
        manager.force_recovery()
        print("\n✅ 已強制恢復到正常狀態")
        print("\n" + manager.get_cli_status())
    elif args.command == "demo":
        run_demo()
    print("\n" + "=" * 70)
    return 0
if __name__ == "__main__":
    exit(main())
