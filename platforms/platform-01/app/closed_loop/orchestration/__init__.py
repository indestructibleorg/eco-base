"""
跨系统协同模块

提供以下功能:
- 服务发现与拓扑构建
- 协同决策与共识
- 级联控制 (故障隔离、优雅降级、恢复协调)
"""

from .topology_builder import (
    TopologyBuilder,
    ServiceNode,
    TopologyEdge,
    KubernetesDiscovery,
    ServiceMeshDiscovery,
    LogBasedDiscovery
)

from .consensus_engine import (
    ConsensusEngine,
    DecisionProposal,
    ConsensusDecision,
    Vote,
    VotingStrategy,
    ConflictResolver,
    DecisionStatus,
    VoteType
)

from .cascade_controller import (
    CascadeController,
    CascadeAction,
    ServiceState,
    CircuitBreaker,
    FaultIsolator,
    GracefulDegradation,
    RecoveryCoordinator,
    CascadeActionType,
    ServiceStatus
)

__all__ = [
    # Topology Builder
    'TopologyBuilder',
    'ServiceNode',
    'TopologyEdge',
    'KubernetesDiscovery',
    'ServiceMeshDiscovery',
    'LogBasedDiscovery',
    
    # Consensus Engine
    'ConsensusEngine',
    'DecisionProposal',
    'ConsensusDecision',
    'Vote',
    'VotingStrategy',
    'ConflictResolver',
    'DecisionStatus',
    'VoteType',
    
    # Cascade Controller
    'CascadeController',
    'CascadeAction',
    'ServiceState',
    'CircuitBreaker',
    'FaultIsolator',
    'GracefulDegradation',
    'RecoveryCoordinator',
    'CascadeActionType',
    'ServiceStatus'
]
