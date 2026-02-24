"""
协同决策引擎
分布式共识和冲突解决
"""

import time
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DecisionStatus(Enum):
    """决策状态"""
    PROPOSED = "proposed"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class VoteType(Enum):
    """投票类型"""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class DecisionProposal:
    """决策提案"""
    proposal_id: str
    initiator: str
    action: str
    target: str
    parameters: Dict[str, Any]
    priority: int
    timestamp: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proposal_id': self.proposal_id,
            'initiator': self.initiator,
            'action': self.action,
            'target': self.target,
            'parameters': self.parameters,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }


@dataclass
class Vote:
    """投票"""
    proposal_id: str
    voter: str
    vote_type: VoteType
    reason: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proposal_id': self.proposal_id,
            'voter': self.voter,
            'vote_type': self.vote_type.value,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConsensusDecision:
    """共识决策"""
    decision_id: str
    proposal: DecisionProposal
    status: DecisionStatus
    votes: List[Vote]
    result: Dict[str, Any]
    decided_at: Optional[datetime]
    executed_at: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decision_id': self.decision_id,
            'proposal': self.proposal.to_dict(),
            'status': self.status.value,
            'votes': [v.to_dict() for v in self.votes],
            'result': self.result,
            'decided_at': self.decided_at.isoformat() if self.decided_at else None,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None
        }


class VotingStrategy:
    """投票策略"""
    
    def __init__(self, strategy_type: str = 'majority'):
        self.strategy_type = strategy_type
        
    def evaluate(self, votes: List[Vote], quorum: int = 3) -> Tuple[bool, Dict[str, Any]]:
        """评估投票结果"""
        if len(votes) < quorum:
            return False, {'reason': 'quorum_not_met', 'votes_count': len(votes), 'required': quorum}
        
        approve_count = sum(1 for v in votes if v.vote_type == VoteType.APPROVE)
        reject_count = sum(1 for v in votes if v.vote_type == VoteType.REJECT)
        abstain_count = sum(1 for v in votes if v.vote_type == VoteType.ABSTAIN)
        
        if self.strategy_type == 'majority':
            # 简单多数
            approved = approve_count > reject_count
        elif self.strategy_type == 'two_thirds':
            # 三分之二多数
            total = len(votes) - abstain_count
            approved = approve_count >= (total * 2 / 3) if total > 0 else False
        elif self.strategy_type == 'unanimous':
            # 全体一致
            approved = reject_count == 0 and approve_count > 0
        else:
            approved = approve_count > reject_count
        
        return approved, {
            'approve': approve_count,
            'reject': reject_count,
            'abstain': abstain_count,
            'total': len(votes)
        }


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.resolution_strategies = {
            'priority': self._resolve_by_priority,
            'timestamp': self._resolve_by_timestamp,
            'impact': self._resolve_by_impact,
        }
        
    def resolve_conflict(self, proposals: List[DecisionProposal], 
                        strategy: str = 'priority') -> DecisionProposal:
        """解决冲突"""
        resolver = self.resolution_strategies.get(strategy, self._resolve_by_priority)
        return resolver(proposals)
    
    def _resolve_by_priority(self, proposals: List[DecisionProposal]) -> DecisionProposal:
        """按优先级解决"""
        # 优先级数字越小越优先
        return min(proposals, key=lambda p: p.priority)
    
    def _resolve_by_timestamp(self, proposals: List[DecisionProposal]) -> DecisionProposal:
        """按时间戳解决"""
        return min(proposals, key=lambda p: p.timestamp)
    
    def _resolve_by_impact(self, proposals: List[DecisionProposal]) -> DecisionProposal:
        """按影响范围解决"""
        # 假设参数中有 impact_score
        return max(proposals, key=lambda p: p.parameters.get('impact_score', 0))
    
    def detect_conflicts(self, proposals: List[DecisionProposal]) -> List[List[DecisionProposal]]:
        """检测冲突的提案"""
        conflict_groups = []
        
        # 按目标分组
        by_target = defaultdict(list)
        for proposal in proposals:
            by_target[proposal.target].append(proposal)
        
        # 同一目标的多个提案视为冲突
        for target, group in by_target.items():
            if len(group) > 1:
                conflict_groups.append(group)
        
        return conflict_groups


class ConsensusEngine:
    """
    协同决策引擎主类
    实现分布式共识机制
    """
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id or f"node_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        self.voting_strategy = VotingStrategy('majority')
        self.conflict_resolver = ConflictResolver()
        
        # 决策存储
        self.proposals: Dict[str, DecisionProposal] = {}
        self.decisions: Dict[str, ConsensusDecision] = {}
        self.votes: Dict[str, List[Vote]] = defaultdict(list)
        
        # 配置
        self.quorum = 3
        self.voting_timeout_seconds = 60
        
    def propose(self, action: str, target: str, parameters: Dict[str, Any],
               priority: int = 5, initiator: str = None) -> DecisionProposal:
        """提出决策"""
        proposal_id = f"prop_{int(time.time() * 1000)}_{self.node_id}"
        
        proposal = DecisionProposal(
            proposal_id=proposal_id,
            initiator=initiator or self.node_id,
            action=action,
            target=target,
            parameters=parameters,
            priority=priority,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.voting_timeout_seconds)
        )
        
        self.proposals[proposal_id] = proposal
        
        # 创建决策记录
        decision = ConsensusDecision(
            decision_id=f"dec_{proposal_id}",
            proposal=proposal,
            status=DecisionStatus.PROPOSED,
            votes=[],
            result={},
            decided_at=None,
            executed_at=None
        )
        
        self.decisions[decision.decision_id] = decision
        
        logger.info(f"Proposed decision: {proposal_id} - {action} on {target}")
        
        return proposal
    
    def vote(self, proposal_id: str, vote_type: VoteType, 
            reason: str = "", voter: str = None) -> bool:
        """投票"""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"Proposal not found: {proposal_id}")
            return False
        
        # 检查是否过期
        if datetime.now() > proposal.expires_at:
            logger.warning(f"Proposal expired: {proposal_id}")
            return False
        
        vote = Vote(
            proposal_id=proposal_id,
            voter=voter or self.node_id,
            vote_type=vote_type,
            reason=reason,
            timestamp=datetime.now()
        )
        
        self.votes[proposal_id].append(vote)
        
        # 更新决策状态
        decision_id = f"dec_{proposal_id}"
        decision = self.decisions.get(decision_id)
        if decision:
            decision.votes.append(vote)
            decision.status = DecisionStatus.VOTING
        
        logger.info(f"Vote recorded: {voter} voted {vote_type.value} for {proposal_id}")
        
        # 检查是否达到共识
        self._check_consensus(proposal_id)
        
        return True
    
    def _check_consensus(self, proposal_id: str):
        """检查是否达到共识"""
        votes = self.votes.get(proposal_id, [])
        
        approved, stats = self.voting_strategy.evaluate(votes, self.quorum)
        
        decision_id = f"dec_{proposal_id}"
        decision = self.decisions.get(decision_id)
        
        if decision and len(votes) >= self.quorum:
            if approved:
                decision.status = DecisionStatus.APPROVED
                decision.decided_at = datetime.now()
                decision.result = {'approved': True, 'vote_stats': stats}
                logger.info(f"Consensus reached: {proposal_id} approved")
            else:
                decision.status = DecisionStatus.REJECTED
                decision.decided_at = datetime.now()
                decision.result = {'approved': False, 'vote_stats': stats}
                logger.info(f"Consensus reached: {proposal_id} rejected")
    
    def get_decision(self, decision_id: str) -> Optional[ConsensusDecision]:
        """获取决策"""
        return self.decisions.get(decision_id)
    
    def get_pending_proposals(self) -> List[DecisionProposal]:
        """获取待处理的提案"""
        pending = []
        for proposal in self.proposals.values():
            decision_id = f"dec_{proposal.proposal_id}"
            decision = self.decisions.get(decision_id)
            if decision and decision.status in [DecisionStatus.PROPOSED, DecisionStatus.VOTING]:
                if datetime.now() <= proposal.expires_at:
                    pending.append(proposal)
        return pending
    
    def resolve_conflicts(self) -> List[ConsensusDecision]:
        """解决冲突"""
        pending = self.get_pending_proposals()
        
        # 检测冲突
        conflict_groups = self.conflict_resolver.detect_conflicts(pending)
        
        resolved = []
        for group in conflict_groups:
            # 选择获胜者
            winner = self.conflict_resolver.resolve_conflict(group, 'priority')
            
            # 批准获胜者
            decision_id = f"dec_{winner.proposal_id}"
            decision = self.decisions.get(decision_id)
            if decision:
                decision.status = DecisionStatus.APPROVED
                decision.decided_at = datetime.now()
                decision.result = {'approved': True, 'conflict_resolved': True}
                resolved.append(decision)
            
            # 拒绝其他冲突提案
            for proposal in group:
                if proposal.proposal_id != winner.proposal_id:
                    other_decision_id = f"dec_{proposal.proposal_id}"
                    other_decision = self.decisions.get(other_decision_id)
                    if other_decision:
                        other_decision.status = DecisionStatus.REJECTED
                        other_decision.decided_at = datetime.now()
                        other_decision.result = {'approved': False, 'conflict_resolved': True}
        
        return resolved
    
    def execute_decision(self, decision_id: str, 
                        executor: Callable = None) -> Dict[str, Any]:
        """执行决策"""
        decision = self.decisions.get(decision_id)
        if not decision:
            return {'error': 'Decision not found'}
        
        if decision.status != DecisionStatus.APPROVED:
            return {'error': f'Decision not approved, current status: {decision.status}'}
        
        decision.status = DecisionStatus.EXECUTING
        
        try:
            if executor:
                result = executor(decision.proposal)
            else:
                # 默认执行逻辑
                result = self._default_execute(decision.proposal)
            
            decision.status = DecisionStatus.COMPLETED
            decision.executed_at = datetime.now()
            decision.result['execution_result'] = result
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            decision.status = DecisionStatus.FAILED
            decision.result['error'] = str(e)
            return {'success': False, 'error': str(e)}
    
    def _default_execute(self, proposal: DecisionProposal) -> Dict[str, Any]:
        """默认执行逻辑"""
        logger.info(f"Executing proposal: {proposal.proposal_id}")
        return {
            'action': proposal.action,
            'target': proposal.target,
            'status': 'simulated_execution'
        }
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """获取共识统计"""
        total = len(self.decisions)
        approved = sum(1 for d in self.decisions.values() if d.status == DecisionStatus.APPROVED)
        rejected = sum(1 for d in self.decisions.values() if d.status == DecisionStatus.REJECTED)
        pending = sum(1 for d in self.decisions.values() if d.status in [DecisionStatus.PROPOSED, DecisionStatus.VOTING])
        
        return {
            'total_decisions': total,
            'approved': approved,
            'rejected': rejected,
            'pending': pending,
            'approval_rate': approved / total if total > 0 else 0
        }
