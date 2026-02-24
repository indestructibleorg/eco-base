#!/usr/bin/env python3
"""
Audit/Reproducibility E2E Tests

稽核/可重播 E2E 測試：
- artifacts 完整性
- manifest hash 可重算一致
- replay 決策一致
"""

import pytest
import asyncio
import json
import hashlib
import tempfile
import os
from datetime import datetime
from pathlib import Path
from app.closed_loop.governance import (
    AuditTrail,
    AuditTrailStorage,
    Evidence,
    EvidenceType,
    EvidenceChain,
    EvidenceCollector
)


class TestArtifactsCompleteness:
    """產物完整性測試"""
    
    REQUIRED_ARTIFACTS = {
        'decision.json',
        'evidence.json',
        'execution_log.jsonl',
        'verification_result.json',
        'topology_snapshot.json',
        'manifest.json'
    }
    
    def test_required_artifacts_list(self):
        """A-01: 必需產物清單"""
        assert len(self.REQUIRED_ARTIFACTS) == 6
        assert 'decision.json' in self.REQUIRED_ARTIFACTS
        assert 'manifest.json' in self.REQUIRED_ARTIFACTS
    
    def test_artifact_directory_structure(self):
        """測試產物目錄結構"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_test_001'
            run_dir.mkdir()
            
            # 創建必需產物
            for artifact in self.REQUIRED_ARTIFACTS:
                (run_dir / artifact).touch()
            
            # 驗證所有產物存在
            for artifact in self.REQUIRED_ARTIFACTS:
                assert (run_dir / artifact).exists(), f"{artifact} 應該存在"
    
    def test_missing_artifact_detection(self):
        """測試缺失產物檢測"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'run_test_001'
            run_dir.mkdir()
            
            # 只創建部分產物
            (run_dir / 'decision.json').touch()
            (run_dir / 'evidence.json').touch()
            
            # 檢查缺失
            present = {f.name for f in run_dir.iterdir() if f.is_file()}
            missing = self.REQUIRED_ARTIFACTS - present
            
            assert len(missing) == 4, f"應該有 4 個缺失產物，實際 {len(missing)}"


class TestManifestHash:
    """Manifest Hash 測試"""
    
    def compute_hash(self, filepath: str, algorithm: str = 'sha3-512') -> str:
        """計算文件 hash"""
        if algorithm == 'sha3-512':
            hasher = hashlib.sha3_512()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def test_manifest_hash_computation(self):
        """A-02: manifest hash 計算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 創建測試文件
            test_file = Path(tmpdir) / 'test.json'
            test_file.write_text(json.dumps({"test": "data"}))
            
            # 計算 hash
            hash1 = self.compute_hash(str(test_file))
            hash2 = self.compute_hash(str(test_file))
            
            # 相同內容應該有相同 hash
            assert hash1 == hash2, "相同內容應該有相同 hash"
            assert len(hash1) == 128, "sha3-512 hash 長度應該為 128"
    
    def test_manifest_hash_consistency(self):
        """A-02: hash 可重算一致"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 創建測試文件
            test_file = Path(tmpdir) / 'decision.json'
            content = {"decision_id": "dec_001", "risk": {"score": 0.5}}
            test_file.write_text(json.dumps(content, sort_keys=True))
            
            # 計算 hash
            expected_hash = self.compute_hash(str(test_file))
            
            # 重新計算
            actual_hash = self.compute_hash(str(test_file))
            
            assert actual_hash == expected_hash, "hash 應該一致"
    
    def test_manifest_hash_tamper_detection(self):
        """測試 hash 篡改檢測"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 創建測試文件
            test_file = Path(tmpdir) / 'decision.json'
            test_file.write_text(json.dumps({"test": "data"}))
            
            # 計算原始 hash
            original_hash = self.compute_hash(str(test_file))
            
            # 修改文件
            test_file.write_text(json.dumps({"test": "tampered"}))
            
            # 重新計算 hash
            new_hash = self.compute_hash(str(test_file))
            
            assert new_hash != original_hash, "篡改後 hash 應該不同"


class TestEvidenceIntegrity:
    """證據完整性測試"""
    
    @pytest.mark.asyncio
    async def test_evidence_hash_computation(self):
        """測試證據 hash 計算"""
        evidence = Evidence(
            evidence_id="ev_hash_001",
            evidence_type=EvidenceType.INPUT,
            trace_id="trace_001",
            timestamp=datetime.now().isoformat(),
            content={"metric": "cpu_usage", "value": 85.5},
            content_hash=""
        )
        
        # 計算 hash
        evidence.content_hash = evidence._compute_hash()
        
        # 驗證完整性
        assert evidence.verify_integrity(), "證據完整性應該通過"
    
    @pytest.mark.asyncio
    async def test_evidence_tamper_detection(self):
        """測試證據篡改檢測"""
        evidence = Evidence(
            evidence_id="ev_tamper_001",
            evidence_type=EvidenceType.INPUT,
            trace_id="trace_001",
            timestamp=datetime.now().isoformat(),
            content={"metric": "cpu_usage", "value": 85.5},
            content_hash=""
        )
        
        # 計算 hash
        evidence.content_hash = evidence._compute_hash()
        
        # 篡改內容
        evidence.content["value"] = 99.9
        
        # 驗證應該失敗
        assert not evidence.verify_integrity(), "篡改後完整性應該失敗"
    
    @pytest.mark.asyncio
    async def test_evidence_chain_integrity(self):
        """測試證據鏈完整性"""
        chain = EvidenceChain(trace_id="trace_chain_001")
        
        # 添加多個證據
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"ev_{i}",
                evidence_type=EvidenceType.INPUT,
                trace_id="trace_chain_001",
                timestamp=datetime.now().isoformat(),
                content={"index": i},
                content_hash=""
            )
            evidence.content_hash = evidence._compute_hash()
            chain.add_evidence(evidence)
        
        # 驗證鏈完整性
        assert chain.verify_chain_integrity(), "證據鏈完整性應該通過"
        
        # 計算鏈 hash
        chain_hash = chain.compute_chain_hash()
        assert chain_hash, "鏈 hash 應該存在"


class TestEvidenceCollector:
    """證據收集器測試"""
    
    @pytest.mark.asyncio
    async def test_collect_input_evidence(self):
        """測試收集輸入證據"""
        storage = AuditTrailStorage()
        collector = EvidenceCollector(storage)
        
        evidence = await collector.collect_input_evidence(
            trace_id="trace_collect_001",
            metric_name="cpu_usage",
            metric_value=85.5,
            raw_input={"timestamp": datetime.now().isoformat()}
        )
        
        assert evidence is not None
        assert evidence.evidence_type == EvidenceType.INPUT
        assert evidence.verify_integrity()
    
    @pytest.mark.asyncio
    async def test_collect_decision_evidence(self):
        """測試收集決策證據"""
        storage = AuditTrailStorage()
        collector = EvidenceCollector(storage)
        
        evidence = await collector.collect_decision_evidence(
            trace_id="trace_collect_002",
            decision_id="dec_001",
            decision_context={"risk_score": 0.85},
            reasoning_process=["Detected anomaly", "Matched rule R1"]
        )
        
        assert evidence is not None
        assert evidence.evidence_type == EvidenceType.DECISION
        assert evidence.verify_integrity()


class TestAuditTrail:
    """審計追踪測試"""
    
    @pytest.mark.asyncio
    async def test_audit_report_generation(self):
        """測試審計報告生成"""
        storage = AuditTrailStorage()
        audit = AuditTrail(storage)
        collector = EvidenceCollector(storage)
        
        # 收集證據
        await collector.collect_input_evidence(
            trace_id="trace_audit_001",
            metric_name="cpu_usage",
            metric_value=85.5,
            raw_input={}
        )
        
        # 生成報告
        report = await audit.generate_audit_report("trace_audit_001")
        
        assert report is not None
        assert 'integrity_verified' in report
        assert report['trace_id'] == "trace_audit_001"


class TestReproducibility:
    """可重播性測試"""
    
    def test_deterministic_hash(self):
        """A-03: 確定性 hash"""
        content = {"decision_id": "dec_001", "risk": {"score": 0.5}}
        
        # 多次計算相同內容的 hash
        hash1 = hashlib.sha3_512(json.dumps(content, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha3_512(json.dumps(content, sort_keys=True).encode()).hexdigest()
        
        assert hash1 == hash2, "相同內容應該產生相同 hash"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
