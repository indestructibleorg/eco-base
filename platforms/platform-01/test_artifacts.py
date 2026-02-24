"""
Artifact 验证测试

验证：
- 产物完整性（hash 校验）
- manifest.json 完整性
- 缺失产物检测
"""

import asyncio
import json
import hashlib
import os
from datetime import datetime
from app.closed_loop.governance import (
    AuditTrail,
    AuditTrailStorage,
    EvidenceType,
    EvidenceCollector,
    InputEvidence,
    DecisionEvidence
)


class ArtifactManager:
    """产物管理器"""
    
    def __init__(self, base_path: str = "artifacts"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def create_run_directory(self, run_id: str) -> str:
        """创建运行产物目录"""
        run_path = os.path.join(self.base_path, run_id)
        os.makedirs(run_path, exist_ok=True)
        return run_path
    
    def save_artifact(self, run_id: str, name: str, content: dict) -> str:
        """保存产物"""
        run_path = self.create_run_directory(run_id)
        filepath = os.path.join(run_path, name)
        
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2, default=str)
        
        return filepath
    
    def compute_hash(self, filepath: str) -> str:
        """计算文件哈希"""
        with open(filepath, 'rb') as f:
            return hashlib.sha3_512(f.read()).hexdigest()
    
    def create_manifest(self, run_id: str) -> dict:
        """创建 manifest"""
        run_path = os.path.join(self.base_path, run_id)
        
        if not os.path.exists(run_path):
            return {"error": "Run directory not found"}
        
        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "files": {}
        }
        
        for filename in os.listdir(run_path):
            if filename == "manifest.json":
                continue
            
            filepath = os.path.join(run_path, filename)
            if os.path.isfile(filepath):
                manifest["files"][filename] = {
                    "hash": self.compute_hash(filepath),
                    "size": os.path.getsize(filepath),
                    "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                }
        
        # 计算 manifest 本身的哈希
        manifest_content = json.dumps(manifest, sort_keys=True)
        manifest["manifest_hash"] = hashlib.sha3_512(manifest_content.encode()).hexdigest()
        
        return manifest
    
    def save_manifest(self, run_id: str, manifest: dict) -> str:
        """保存 manifest"""
        run_path = os.path.join(self.base_path, run_id)
        filepath = os.path.join(run_path, "manifest.json")
        
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return filepath
    
    def verify_manifest(self, run_id: str) -> dict:
        """验证 manifest"""
        run_path = os.path.join(self.base_path, run_id)
        manifest_path = os.path.join(run_path, "manifest.json")
        
        if not os.path.exists(manifest_path):
            return {"valid": False, "error": "manifest.json not found"}
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # 验证每个文件
        results = {"valid": True, "files": {}}
        
        for filename, info in manifest.get("files", {}).items():
            filepath = os.path.join(run_path, filename)
            
            if not os.path.exists(filepath):
                results["valid"] = False
                results["files"][filename] = {"valid": False, "error": "File not found"}
                continue
            
            current_hash = self.compute_hash(filepath)
            expected_hash = info.get("hash")
            
            if current_hash != expected_hash:
                results["valid"] = False
                results["files"][filename] = {
                    "valid": False,
                    "error": "Hash mismatch",
                    "expected": expected_hash,
                    "actual": current_hash
                }
            else:
                results["files"][filename] = {"valid": True}
        
        return results


async def test_artifact_creation():
    """测试产物创建"""
    print("\n[Test] Artifact creation")
    
    manager = ArtifactManager(base_path="/tmp/test_artifacts")
    run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 创建产物
    decision = {"decision_id": "dec_001", "risk": {"score": 0.5}}
    evidence = {"anomaly": {"type": "cpu_high"}}
    execution = {"action_id": "act_001", "result": "success"}
    
    manager.save_artifact(run_id, "decision.json", decision)
    manager.save_artifact(run_id, "evidence.json", evidence)
    manager.save_artifact(run_id, "execution_log.json", execution)
    
    # 创建 manifest
    manifest = manager.create_manifest(run_id)
    manager.save_manifest(run_id, manifest)
    
    print(f"  Created {len(manifest['files'])} artifacts")
    print(f"  Manifest hash: {manifest['manifest_hash'][:16]}...")
    
    if len(manifest['files']) == 3:
        print(f"  ✅ Artifacts created successfully")
        return True
    else:
        print(f"  ❌ Artifact creation failed")
        return False


async def test_manifest_verification():
    """测试 manifest 验证"""
    print("\n[Test] Manifest verification")
    
    manager = ArtifactManager(base_path="/tmp/test_artifacts_verify")
    run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 创建产物和 manifest
    manager.save_artifact(run_id, "decision.json", {"id": "dec_001"})
    manifest = manager.create_manifest(run_id)
    manager.save_manifest(run_id, manifest)
    
    # 验证
    result = manager.verify_manifest(run_id)
    
    print(f"  Manifest valid: {result['valid']}")
    
    if result['valid']:
        print(f"  ✅ Manifest verification passed")
        return True
    else:
        print(f"  ❌ Manifest verification failed")
        return False


async def test_missing_artifact_detection():
    """测试缺失产物检测"""
    print("\n[Test] Missing artifact detection")
    
    manager = ArtifactManager(base_path="/tmp/test_artifacts_missing")
    run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 创建产物
    manager.save_artifact(run_id, "decision.json", {"id": "dec_001"})
    manager.save_artifact(run_id, "evidence.json", {"data": "test"})
    
    # 创建 manifest
    manifest = manager.create_manifest(run_id)
    manager.save_manifest(run_id, manifest)
    
    # 删除一个产物
    run_path = os.path.join(manager.base_path, run_id)
    os.remove(os.path.join(run_path, "evidence.json"))
    
    # 验证
    result = manager.verify_manifest(run_id)
    
    print(f"  Manifest valid: {result['valid']}")
    print(f"  Missing file detected: {'evidence.json' in result['files'] and not result['files']['evidence.json'].get('valid', True)}")
    
    if not result['valid'] and 'evidence.json' in result['files']:
        print(f"  ✅ Missing artifact correctly detected")
        return True
    else:
        print(f"  ❌ Missing artifact not detected")
        return False


async def test_hash_mismatch_detection():
    """测试哈希不匹配检测"""
    print("\n[Test] Hash mismatch detection")
    
    manager = ArtifactManager(base_path="/tmp/test_artifacts_hash")
    run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 创建产物
    filepath = manager.save_artifact(run_id, "decision.json", {"id": "dec_001"})
    
    # 创建 manifest
    manifest = manager.create_manifest(run_id)
    manager.save_manifest(run_id, manifest)
    
    # 修改产物
    with open(filepath, 'w') as f:
        json.dump({"id": "dec_001", "tampered": True}, f)
    
    # 验证
    result = manager.verify_manifest(run_id)
    
    print(f"  Manifest valid: {result['valid']}")
    
    if not result['valid']:
        print(f"  ✅ Hash mismatch correctly detected")
        return True
    else:
        print(f"  ❌ Hash mismatch not detected")
        return False


async def test_evidence_collector():
    """测试证据收集器"""
    print("\n[Test] Evidence collector")
    
    storage = AuditTrailStorage()
    collector = EvidenceCollector(storage)
    
    # 收集输入证据
    input_evidence = await collector.collect_input_evidence(
        trace_id="trace_001",
        metric_name="cpu_usage",
        metric_value=85.5,
        raw_input={"timestamp": datetime.now().isoformat()}
    )
    
    # 收集决策证据
    decision_evidence = await collector.collect_decision_evidence(
        trace_id="trace_001",
        decision_id="dec_001",
        decision_context={"risk_score": 0.85},
        reasoning_process=["Detected anomaly"]
    )
    
    print(f"  Input evidence: {input_evidence.evidence_id}")
    print(f"  Decision evidence: {decision_evidence.evidence_id}")
    print(f"  Input integrity: {input_evidence.verify_integrity()}")
    print(f"  Decision integrity: {decision_evidence.verify_integrity()}")
    
    if input_evidence.verify_integrity() and decision_evidence.verify_integrity():
        print(f"  ✅ Evidence collector working correctly")
        return True
    else:
        print(f"  ❌ Evidence integrity check failed")
        return False


async def test_required_artifacts():
    """测试必需产物检查"""
    print("\n[Test] Required artifacts check")
    
    manager = ArtifactManager(base_path="/tmp/test_artifacts_required")
    run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 定义必需产物
    required_artifacts = [
        "decision.json",
        "evidence.json",
        "execution_log.json",
        "verification_result.json",
        "manifest.json"
    ]
    
    # 只创建部分产物
    manager.save_artifact(run_id, "decision.json", {"id": "dec_001"})
    manager.save_artifact(run_id, "evidence.json", {"data": "test"})
    # 缺少 execution_log.json 和 verification_result.json
    
    # 创建 manifest
    manifest = manager.create_manifest(run_id)
    manager.save_manifest(run_id, manifest)
    
    # 检查必需产物
    run_path = os.path.join(manager.base_path, run_id)
    missing = []
    for artifact in required_artifacts:
        if not os.path.exists(os.path.join(run_path, artifact)):
            missing.append(artifact)
    
    print(f"  Required artifacts: {len(required_artifacts)}")
    print(f"  Missing artifacts: {len(missing)}")
    print(f"  Missing: {missing}")
    
    if len(missing) == 2 and "execution_log.json" in missing and "verification_result.json" in missing:
        print(f"  ✅ Missing required artifacts correctly detected")
        return True
    else:
        print(f"  ❌ Required artifact check failed")
        return False


async def run_artifact_tests():
    """运行所有 artifact 测试"""
    print("\n" + "="*60)
    print("Artifact Tests")
    print("="*60)
    
    results = []
    results.append(("Artifact creation", await test_artifact_creation()))
    results.append(("Manifest verification", await test_manifest_verification()))
    results.append(("Missing artifact detection", await test_missing_artifact_detection()))
    results.append(("Hash mismatch detection", await test_hash_mismatch_detection()))
    results.append(("Evidence collector", await test_evidence_collector()))
    results.append(("Required artifacts check", await test_required_artifacts()))
    
    # 汇总
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # 清理
    import shutil
    for path in ["/tmp/test_artifacts", "/tmp/test_artifacts_verify", "/tmp/test_artifacts_missing",
                 "/tmp/test_artifacts_hash", "/tmp/test_artifacts_required"]:
        if os.path.exists(path):
            shutil.rmtree(path)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_artifact_tests())
    exit(0 if success else 1)
