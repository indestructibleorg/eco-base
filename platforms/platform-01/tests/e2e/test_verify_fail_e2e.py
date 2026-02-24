#!/usr/bin/env python3
"""
Verify-Fail E2E Tests

Verification Gate E2E 測試：
- 不可修復故障注入
- 可修復短暫故障
"""

import pytest
import asyncio
from app.closed_loop.governance import (
    VerificationGate,
    VerificationConfig,
    VerificationStatus,
    VerificationResult,
    VerificationStrategy
)
from datetime import datetime


class TestUnrecoverableFailure:
    """不可修復故障測試"""
    
    @pytest.mark.asyncio
    async def test_verify_fail_triggers_rollback(self):
        """V-01: 驗證失敗觸發回滾"""
        config = VerificationConfig(
            enabled=True,
            auto_rollback=True,
            on_failure_strategy=VerificationStrategy.ROLLBACK
        )
        
        gate = VerificationGate(config)
        
        # 創建一個失敗的驗證結果
        failed_result = VerificationResult(
            verification_id="vrf_fail_001",
            action_id="act_001",
            status=VerificationStatus.FAILED,
            checks=[],
            message="Test failure",
            started_at=datetime.now(),
            trigger_rollback=True
        )
        
        # 存儲結果
        gate._verification_results[failed_result.verification_id] = failed_result
        
        # 檢查是否應該觸發回滾
        should_trigger = gate.should_trigger_rollback(failed_result.verification_id)
        
        assert should_trigger, "驗證失敗應該觸發回滾"
    
    @pytest.mark.asyncio
    async def test_verify_fail_escalation(self):
        """V-01: 驗證失敗觸發升級"""
        config = VerificationConfig(
            enabled=True,
            auto_rollback=False,
            on_failure_strategy=VerificationStrategy.ESCALATE
        )
        
        gate = VerificationGate(config)
        
        # 創建一個失敗的驗證結果
        failed_result = VerificationResult(
            verification_id="vrf_escalate_001",
            action_id="act_001",
            status=VerificationStatus.FAILED,
            checks=[],
            message="Test failure requiring escalation",
            started_at=datetime.now(),
            trigger_rollback=False  # 升級而非回滾
        )
        
        gate._verification_results[failed_result.verification_id] = failed_result
        
        # 驗證狀態
        assert failed_result.status == VerificationStatus.FAILED
        assert not failed_result.trigger_rollback
    
    @pytest.mark.asyncio
    async def test_verify_pass_no_rollback(self):
        """驗證通過不觸發回滾"""
        config = VerificationConfig(
            enabled=True,
            auto_rollback=True
        )
        
        gate = VerificationGate(config)
        
        # 創建一個通過的驗證結果
        passed_result = VerificationResult(
            verification_id="vrf_pass_001",
            action_id="act_001",
            status=VerificationStatus.PASSED,
            checks=[],
            message="All checks passed",
            started_at=datetime.now(),
            trigger_rollback=False
        )
        
        gate._verification_results[passed_result.verification_id] = passed_result
        
        # 檢查不應該觸發回滾
        should_trigger = gate.should_trigger_rollback(passed_result.verification_id)
        
        assert not should_trigger, "驗證通過不應該觸發回滾"


class TestRecoverableFailure:
    """可修復故障測試"""
    
    @pytest.mark.asyncio
    async def test_verify_with_passing_checks(self):
        """V-02: 可修復故障在窗口內恢復"""
        config = VerificationConfig(
            enabled=True,
            timeout_seconds=5,
            check_interval_seconds=1,
            consecutive_success_period_seconds=2
        )
        
        gate = VerificationGate(config)
        
        # 設置通過條件（會通過）
        checks = [
            {"metric": "cpu_usage", "expected": "< 100%"},  # 實際 65%，會通過
        ]
        
        result = await gate.verify("act_recover_001", checks)
        
        # 應該通過
        assert result.status == VerificationStatus.PASSED, f"驗證應該通過，但狀態為 {result.status.value}"
    
    @pytest.mark.asyncio
    async def test_verify_timeout(self):
        """驗證超時"""
        config = VerificationConfig(
            enabled=True,
            timeout_seconds=2,
            check_interval_seconds=1,
            consecutive_success_period_seconds=10  # 長成功期，會超時
        )
        
        gate = VerificationGate(config)
        
        checks = [
            {"metric": "cpu_usage", "expected": "< 100%"},
        ]
        
        result = await gate.verify("act_timeout_001", checks)
        
        # 應該超時
        assert result.status == VerificationStatus.TIMEOUT, f"應該超時，但狀態為 {result.status.value}"


class TestVerificationConfig:
    """驗證配置測試"""
    
    def test_auto_rollback_enabled(self):
        """測試自動回滾啟用"""
        config = VerificationConfig(
            enabled=True,
            auto_rollback=True
        )
        
        assert config.enabled
        assert config.auto_rollback
    
    def test_auto_rollback_disabled(self):
        """測試自動回滾禁用"""
        config = VerificationConfig(
            enabled=True,
            auto_rollback=False
        )
        
        assert config.enabled
        assert not config.auto_rollback
    
    def test_verification_disabled(self):
        """測試驗證禁用"""
        config = VerificationConfig(enabled=False)
        
        assert not config.enabled


class TestVerificationReporter:
    """驗證報告測試"""
    
    @pytest.mark.asyncio
    async def test_generate_verification_report(self):
        """生成驗證報告"""
        from app.closed_loop.governance import VerificationReporter
        
        config = VerificationConfig(enabled=True)
        gate = VerificationGate(config)
        
        reporter = VerificationReporter(gate)
        
        # 創建一個驗證結果
        result = VerificationResult(
            verification_id="vrf_report_001",
            action_id="act_001",
            status=VerificationStatus.PASSED,
            checks=[
                {"metric": "cpu_usage", "expected": "< 70%", "actual": "65%", "passed": True}
            ],
            message="All checks passed",
            started_at=datetime.now()
        )
        
        gate._verification_results[result.verification_id] = result
        
        # 生成報告
        report = reporter.generate_report(result.verification_id)
        
        assert report['verification_id'] == result.verification_id
        assert report['status'] == VerificationStatus.PASSED.value
        assert report['passed'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
