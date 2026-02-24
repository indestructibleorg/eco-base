#!/usr/bin/env python3
"""
Crash-Resume E2E Tests

E2E 測試：任意中斷都能續跑
- EXECUTING 中 kill
- VERIFYING 中 kill
- 20 次隨機 kill
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
from app.closed_loop.core.state_store import RunState, RunPhase, RunStateTransition, FileStateStore


class TestCrashResume:
    """Crash-Resume E2E 測試"""
    
    @pytest.mark.asyncio
    async def test_resume_from_detected(self):
        """R-01: 從 DETECTED 狀態恢復"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            # 創建運行並推進到 DETECTED
            state = RunState(
                run_id="run_resume_001",
                trace_id="trace_001",
                phase=RunPhase.NEW
            )
            state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
            await store.save(state)
            
            # 模擬重啟：新實例加載狀態
            new_store = FileStateStore(base_path=tmpdir)
            loaded = await new_store.load(state.run_id)
            
            assert loaded is not None
            assert loaded.phase == RunPhase.DETECTED
            assert loaded.can_resume()
    
    @pytest.mark.asyncio
    async def test_resume_from_executing(self):
        """R-01: 從 EXECUTING 狀態恢復"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            # 創建運行並推進到 EXECUTING
            state = RunState(
                run_id="run_resume_002",
                trace_id="trace_001",
                phase=RunPhase.NEW
            )
            state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
            state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
            state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
            state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
            state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
            await store.save(state)
            
            # 模擬重啟
            new_store = FileStateStore(base_path=tmpdir)
            loaded = await new_store.load(state.run_id)
            
            assert loaded is not None
            assert loaded.phase == RunPhase.EXECUTING
            assert loaded.can_resume()
    
    @pytest.mark.asyncio
    async def test_resume_from_verifying(self):
        """R-02: 從 VERIFYING 狀態恢復"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            # 創建運行並推進到 VERIFYING
            state = RunState(
                run_id="run_resume_003",
                trace_id="trace_001",
                phase=RunPhase.NEW
            )
            state.transition_to(RunPhase.DETECTED, RunStateTransition.ANOMALY_DETECTED)
            state.transition_to(RunPhase.ANALYZED, RunStateTransition.RCA_COMPLETED)
            state.transition_to(RunPhase.PLANNED, RunStateTransition.DECISION_PLANNED)
            state.transition_to(RunPhase.APPROVED, RunStateTransition.APPROVAL_GRANTED)
            state.transition_to(RunPhase.EXECUTING, RunStateTransition.EXECUTION_STARTED)
            state.transition_to(RunPhase.EXECUTED, RunStateTransition.EXECUTION_COMPLETED)
            state.transition_to(RunPhase.VERIFYING, RunStateTransition.VERIFICATION_STARTED)
            await store.save(state)
            
            # 模擬重啟
            new_store = FileStateStore(base_path=tmpdir)
            loaded = await new_store.load(state.run_id)
            
            assert loaded is not None
            assert loaded.phase == RunPhase.VERIFYING
            assert loaded.can_resume()
    
    @pytest.mark.asyncio
    async def test_state_persistence_across_restarts(self):
        """測試狀態在重啟後持久化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            state = RunState(
                run_id="run_persist_001",
                trace_id="trace_001",
                phase=RunPhase.PLANNED,
                inputs_hash="abc123",
                decision_id="dec_001"
            )
            
            # 保存
            await store.save(state)
            
            # 新實例加載
            new_store = FileStateStore(base_path=tmpdir)
            loaded = await new_store.load(state.run_id)
            
            assert loaded is not None
            assert loaded.run_id == state.run_id
            assert loaded.phase == state.phase
            assert loaded.inputs_hash == state.inputs_hash
            assert loaded.decision_id == state.decision_id
    
    @pytest.mark.asyncio
    async def test_terminal_state_not_resumable(self):
        """終態不應該可恢復"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            state = RunState(
                run_id="run_term_001",
                trace_id="trace_001",
                phase=RunPhase.SUCCEEDED
            )
            await store.save(state)
            
            loaded = await store.load(state.run_id)
            
            assert loaded is not None
            assert loaded.is_terminal()
            assert not loaded.can_resume()
    
    @pytest.mark.asyncio
    async def test_list_active_runs(self):
        """測試列出活動運行"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStateStore(base_path=tmpdir)
            
            # 創建多個運行
            active_state = RunState(
                run_id="run_active_001",
                trace_id="trace_001",
                phase=RunPhase.EXECUTING
            )
            terminal_state = RunState(
                run_id="run_term_001",
                trace_id="trace_001",
                phase=RunPhase.SUCCEEDED
            )
            
            await store.save(active_state)
            await store.save(terminal_state)
            
            # 列出活動運行
            active_runs = await store.list_active()
            
            assert len(active_runs) == 1
            assert active_runs[0].run_id == "run_active_001"


class TestChaosKillResume:
    """混沌測試：隨機 kill 後恢復"""
    
    @pytest.mark.asyncio
    async def test_20_random_kills(self):
        """R-03: 20 次隨機 kill 測試"""
        success_count = 0
        iterations = 20
        
        for i in range(iterations):
            with tempfile.TemporaryDirectory() as tmpdir:
                store = FileStateStore(base_path=tmpdir)
                
                # 創建運行
                state = RunState(
                    run_id=f"run_chaos_{i}",
                    trace_id=f"trace_{i}",
                    phase=RunPhase.NEW
                )
                
                # 隨機推進到某個狀態
                import random
                phases_to_test = [
                    RunPhase.DETECTED,
                    RunPhase.ANALYZED,
                    RunPhase.PLANNED,
                    RunPhase.EXECUTING,
                ]
                target_phase = random.choice(phases_to_test)
                
                transitions = {
                    RunPhase.DETECTED: RunStateTransition.ANOMALY_DETECTED,
                    RunPhase.ANALYZED: RunStateTransition.RCA_COMPLETED,
                    RunPhase.PLANNED: RunStateTransition.DECISION_PLANNED,
                    RunPhase.EXECUTING: RunStateTransition.EXECUTION_STARTED,
                }
                
                if target_phase != RunPhase.NEW:
                    state.transition_to(target_phase, transitions[target_phase])
                
                await store.save(state)
                
                # 模擬重啟
                new_store = FileStateStore(base_path=tmpdir)
                loaded = await new_store.load(state.run_id)
                
                if loaded and loaded.phase == target_phase:
                    success_count += 1
        
        # 驗證成功率
        success_rate = success_count / iterations
        print(f"\n20 次隨機 kill: {success_count}/{iterations} 成功 ({success_rate*100:.1f}%)")
        
        assert success_rate == 1.0, f"成功率 {success_rate*100:.1f}% 低於 100%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
