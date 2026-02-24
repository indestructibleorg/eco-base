"""
证据链系统 (Audit Trail)

强制治理规范核心组件
每次执行产出 artifacts，可机器验证（hash/版本/输入快照）
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, BinaryIO
import logging

logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """证据类型"""
    INPUT = "input"           # 输入证据
    DECISION = "decision"     # 决策证据
    EXECUTION = "execution"   # 执行证据
    VERIFICATION = "verification"  # 验证证据
    ROLLBACK = "rollback"     # 回滚证据


class StorageType(Enum):
    """存储类型"""
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    S3 = "s3"


@dataclass
class Evidence:
    """
    证据基类
    
    所有证据必须包含:
    - 唯一ID
    - 时间戳
    - 内容哈希（用于完整性验证）
    - 版本信息
    """
    evidence_id: str
    evidence_type: EvidenceType
    trace_id: str
    timestamp: str
    content: Dict[str, Any]
    content_hash: str
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def verify_integrity(self) -> bool:
        """验证证据完整性"""
        computed_hash = self._compute_hash()
        return computed_hash == self.content_hash
    
    def _compute_hash(self) -> str:
        """计算内容哈希"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class InputEvidence(Evidence):
    """输入证据"""
    metric_name: str = ""
    metric_value: float = 0.0
    raw_input: Dict[str, Any] = field(default_factory=dict)
    snapshot_hash: str = ""


@dataclass
class DecisionEvidence(Evidence):
    """决策证据"""
    decision_id: str = ""
    decision_context: Dict[str, Any] = field(default_factory=dict)
    reasoning_process: List[str] = field(default_factory=list)
    rule_matches: List[Dict] = field(default_factory=list)


@dataclass
class ExecutionEvidence(Evidence):
    """执行证据"""
    action_id: str = ""
    action_type: str = ""
    action_result: Dict[str, Any] = field(default_factory=dict)
    state_changes: List[Dict] = field(default_factory=list)


@dataclass
class VerificationEvidence(Evidence):
    """验证证据"""
    verification_id: str = ""
    verification_result: Dict[str, Any] = field(default_factory=dict)
    metric_snapshots: List[Dict] = field(default_factory=list)


class EvidenceChain:
    """
    证据链
    
    管理一个完整闭环的所有证据
    """
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.evidences: Dict[EvidenceType, List[Evidence]] = {
            et: [] for et in EvidenceType
        }
        self._chain_hash: Optional[str] = None
    
    def add_evidence(self, evidence: Evidence):
        """添加证据"""
        self.evidences[evidence.evidence_type].append(evidence)
        self._chain_hash = None  # 重置链哈希
    
    def get_evidences(self, evidence_type: Optional[EvidenceType] = None) -> List[Evidence]:
        """获取证据列表"""
        if evidence_type:
            return self.evidences.get(evidence_type, [])
        
        all_evidences = []
        for evidences in self.evidences.values():
            all_evidences.extend(evidences)
        
        # 按时间排序
        all_evidences.sort(key=lambda e: e.timestamp)
        return all_evidences
    
    def compute_chain_hash(self) -> str:
        """计算证据链哈希"""
        if self._chain_hash:
            return self._chain_hash
        
        all_evidences = self.get_evidences()
        content = json.dumps(
            [e.content_hash for e in all_evidences],
            sort_keys=True
        )
        self._chain_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._chain_hash
    
    def verify_chain_integrity(self) -> bool:
        """验证证据链完整性"""
        for evidences in self.evidences.values():
            for evidence in evidences:
                if not evidence.verify_integrity():
                    logger.error(
                        f"Evidence integrity check failed: {evidence.evidence_id}"
                    )
                    return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trace_id": self.trace_id,
            "chain_hash": self.compute_chain_hash(),
            "evidence_count": sum(len(e) for e in self.evidences.values()),
            "evidences": {
                et.value: [asdict(e) for e in evidences]
                for et, evidences in self.evidences.items()
            }
        }


class AuditTrailStorage:
    """
    审计存储
    
    支持多种存储后端
    """
    
    # 保留期限配置（天）
    RETENTION_DAYS = {
        StorageType.POSTGRESQL: 7 * 365,    # 7年
        StorageType.ELASTICSEARCH: 365,      # 1年
        StorageType.INFLUXDB: 90,            # 90天
        StorageType.NEO4J: 7 * 365,          # 7年
        StorageType.S3: 7 * 365,             # 7年
    }
    
    def __init__(self):
        self._storages: Dict[StorageType, Any] = {}
        self._evidence_chains: Dict[str, EvidenceChain] = {}
    
    def register_storage(self, storage_type: StorageType, storage: Any):
        """注册存储后端"""
        self._storages[storage_type] = storage
    
    async def store_evidence(
        self,
        evidence: Evidence,
        storage_types: Optional[List[StorageType]] = None
    ) -> bool:
        """
        存储证据
        
        Args:
            evidence: 证据对象
            storage_types: 存储类型列表，默认存储到所有可用存储
        
        Returns:
            是否成功
        """
        if storage_types is None:
            storage_types = list(self._storages.keys())
        
        success = True
        for storage_type in storage_types:
            storage = self._storages.get(storage_type)
            if not storage:
                logger.warning(f"Storage {storage_type.value} not available")
                continue
            
            try:
                # 实际实现中应调用存储后端的写入接口
                logger.debug(
                    f"Storing evidence {evidence.evidence_id} to {storage_type.value}"
                )
            except Exception as e:
                logger.exception(f"Failed to store evidence to {storage_type.value}: {e}")
                success = False
        
        # 更新证据链
        chain = self._get_or_create_chain(evidence.trace_id)
        chain.add_evidence(evidence)
        
        return success
    
    async def get_evidence(
        self,
        evidence_id: str,
        evidence_type: EvidenceType
    ) -> Optional[Evidence]:
        """获取证据"""
        # 从内存缓存中查找
        for chain in self._evidence_chains.values():
            for evidence in chain.get_evidences(evidence_type):
                if evidence.evidence_id == evidence_id:
                    return evidence
        
        # 实际实现中应从存储后端查询
        return None
    
    async def get_evidence_chain(self, trace_id: str) -> Optional[EvidenceChain]:
        """获取证据链"""
        return self._evidence_chains.get(trace_id)
    
    def _get_or_create_chain(self, trace_id: str) -> EvidenceChain:
        """获取或创建证据链"""
        if trace_id not in self._evidence_chains:
            self._evidence_chains[trace_id] = EvidenceChain(trace_id)
        return self._evidence_chains[trace_id]
    
    async def query_by_timerange(
        self,
        start: datetime,
        end: datetime,
        evidence_type: Optional[EvidenceType] = None
    ) -> List[Evidence]:
        """按时间范围查询证据"""
        results = []
        
        for chain in self._evidence_chains.values():
            for evidence in chain.get_evidences(evidence_type):
                evidence_time = datetime.fromisoformat(evidence.timestamp)
                if start <= evidence_time <= end:
                    results.append(evidence)
        
        # 按时间排序
        results.sort(key=lambda e: e.timestamp)
        return results
    
    async def cleanup_expired(self) -> int:
        """清理过期证据"""
        # 实际实现中应根据保留策略清理过期数据
        logger.info("Cleaning up expired evidence")
        return 0


class AuditTrail:
    """
    证据链查询接口
    
    提供证据的查询、验证和报告功能
    """
    
    def __init__(self, storage: AuditTrailStorage):
        self.storage = storage
    
    async def get_decision_evidence(
        self,
        decision_id: str
    ) -> Optional[DecisionEvidence]:
        """
        获取决策证据
        
        Args:
            decision_id: 决策ID
        
        Returns:
            决策证据
        """
        evidence = await self.storage.get_evidence(
            decision_id,
            EvidenceType.DECISION
        )
        
        if evidence and isinstance(evidence, DecisionEvidence):
            return evidence
        
        return None
    
    async def get_execution_evidence(
        self,
        action_id: str
    ) -> Optional[ExecutionEvidence]:
        """
        获取执行证据
        
        Args:
            action_id: 动作ID
        
        Returns:
            执行证据
        """
        evidence = await self.storage.get_evidence(
            action_id,
            EvidenceType.EXECUTION
        )
        
        if evidence and isinstance(evidence, ExecutionEvidence):
            return evidence
        
        return None
    
    async def verify_evidence_integrity(
        self,
        evidence_id: str,
        evidence_type: EvidenceType
    ) -> bool:
        """
        验证证据完整性
        
        Args:
            evidence_id: 证据ID
            evidence_type: 证据类型
        
        Returns:
            是否通过验证
        """
        evidence = await self.storage.get_evidence(evidence_id, evidence_type)
        
        if not evidence:
            logger.error(f"Evidence not found: {evidence_id}")
            return False
        
        return evidence.verify_integrity()
    
    async def verify_chain_integrity(self, trace_id: str) -> bool:
        """
        验证证据链完整性
        
        Args:
            trace_id: 追踪ID
        
        Returns:
            是否通过验证
        """
        chain = await self.storage.get_evidence_chain(trace_id)
        
        if not chain:
            logger.error(f"Evidence chain not found: {trace_id}")
            return False
        
        return chain.verify_chain_integrity()
    
    async def generate_audit_report(
        self,
        trace_id: str
    ) -> Dict[str, Any]:
        """
        生成审计报告
        
        Args:
            trace_id: 追踪ID
        
        Returns:
            审计报告
        """
        chain = await self.storage.get_evidence_chain(trace_id)
        
        if not chain:
            return {"error": "Evidence chain not found"}
        
        # 验证完整性
        integrity_passed = chain.verify_chain_integrity()
        
        # 统计证据
        evidences = chain.get_evidences()
        type_counts = {}
        for e in evidences:
            type_counts[e.evidence_type.value] = type_counts.get(e.evidence_type.value, 0) + 1
        
        return {
            "trace_id": trace_id,
            "chain_hash": chain.compute_chain_hash(),
            "integrity_verified": integrity_passed,
            "evidence_count": len(evidences),
            "type_distribution": type_counts,
            "evidences": [
                {
                    "evidence_id": e.evidence_id,
                    "type": e.evidence_type.value,
                    "timestamp": e.timestamp,
                    "content_hash": e.content_hash,
                    "integrity": e.verify_integrity()
                }
                for e in evidences
            ]
        }
    
    async def generate_compliance_report(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """
        生成合规报告
        
        Args:
            start: 开始时间
            end: 结束时间
        
        Returns:
            合规报告
        """
        evidences = await self.storage.query_by_timerange(start, end)
        
        total = len(evidences)
        integrity_passed = sum(1 for e in evidences if e.verify_integrity())
        
        return {
            "period": {
                "start": start.isoformat(),
                "end": end.isoformat()
            },
            "total_evidence": total,
            "integrity_passed": integrity_passed,
            "integrity_failed": total - integrity_passed,
            "integrity_rate": integrity_passed / total if total > 0 else 0,
            "evidence_types": list(set(e.evidence_type.value for e in evidences))
        }


class EvidenceCollector:
    """
    证据收集器
    
    简化证据收集过程
    """
    
    def __init__(self, storage: AuditTrailStorage):
        self.storage = storage
    
    async def collect_input_evidence(
        self,
        trace_id: str,
        metric_name: str,
        metric_value: float,
        raw_input: Dict[str, Any]
    ) -> InputEvidence:
        """收集输入证据"""
        evidence = InputEvidence(
            evidence_id=f"inp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            evidence_type=EvidenceType.INPUT,
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            content={"metric": metric_name, "value": metric_value},
            content_hash="",  # 将在创建时计算
            metric_name=metric_name,
            metric_value=metric_value,
            raw_input=raw_input,
            snapshot_hash=hashlib.sha256(
                json.dumps(raw_input, sort_keys=True).encode()
            ).hexdigest()[:16]
        )
        
        # 计算内容哈希
        evidence.content_hash = evidence._compute_hash()
        
        await self.storage.store_evidence(evidence)
        return evidence
    
    async def collect_decision_evidence(
        self,
        trace_id: str,
        decision_id: str,
        decision_context: Dict[str, Any],
        reasoning_process: List[str]
    ) -> DecisionEvidence:
        """收集决策证据"""
        evidence = DecisionEvidence(
            evidence_id=f"dec_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            evidence_type=EvidenceType.DECISION,
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            content={"decision_id": decision_id, "context": decision_context},
            content_hash="",
            decision_id=decision_id,
            decision_context=decision_context,
            reasoning_process=reasoning_process
        )
        
        evidence.content_hash = evidence._compute_hash()
        
        await self.storage.store_evidence(evidence)
        return evidence
    
    async def collect_execution_evidence(
        self,
        trace_id: str,
        action_id: str,
        action_type: str,
        action_result: Dict[str, Any]
    ) -> ExecutionEvidence:
        """收集执行证据"""
        evidence = ExecutionEvidence(
            evidence_id=f"exe_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            evidence_type=EvidenceType.EXECUTION,
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            content={"action_id": action_id, "result": action_result},
            content_hash="",
            action_id=action_id,
            action_type=action_type,
            action_result=action_result
        )
        
        evidence.content_hash = evidence._compute_hash()
        
        await self.storage.store_evidence(evidence)
        return evidence
