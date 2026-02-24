"""
Phase 3 自治闭环控制器
整合所有自治组件，实现完全自治的闭环系统
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto

from .controller import ClosedLoopController, Phase2Controller, LoopState

logger = logging.getLogger(__name__)


class AutonomousMode(Enum):
    """自治模式"""
    MANUAL = auto()          # 完全人工
    ASSISTED = auto()        # 辅助模式
    SEMI_AUTONOMOUS = auto() # 半自治
    FULLY_AUTONOMOUS = auto()# 完全自治


class SystemCapability(Enum):
    """系统能力"""
    SELF_LEARNING = "self_learning"
    MULTI_OBJECTIVE = "multi_objective"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    PREDICTIVE = "predictive"
    ORCHESTRATION = "orchestration"
    HUMAN_AI = "human_ai"


@dataclass
class AutonomousContext:
    """自治上下文"""
    timestamp: datetime
    mode: AutonomousMode
    confidence: float
    decisions: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    learnings: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Phase3Controller:
    """Phase 3 自治闭环控制器"""
    
    def __init__(self,
                 phase2_controller: Phase2Controller,
                 learning_engine=None,
                 optimizer=None,
                 knowledge_system=None,
                 predictive_system=None,
                 orchestration=None,
                 human_interface=None):
        self.phase2 = phase2_controller
        
        # Phase 3 组件
        self.learning_engine = learning_engine
        self.optimizer = optimizer
        self.knowledge_system = knowledge_system
        self.predictive_system = predictive_system
        self.orchestration = orchestration
        self.human_interface = human_interface
        
        # 自治状态
        self.mode = AutonomousMode.ASSISTED
        self.capabilities: Dict[SystemCapability, bool] = {
            SystemCapability.SELF_LEARNING: learning_engine is not None,
            SystemCapability.MULTI_OBJECTIVE: optimizer is not None,
            SystemCapability.KNOWLEDGE_GRAPH: knowledge_system is not None,
            SystemCapability.PREDICTIVE: predictive_system is not None,
            SystemCapability.ORCHESTRATION: orchestration is not None,
            SystemCapability.HUMAN_AI: human_interface is not None,
        }
        
        # 运行时数据
        self.decision_history: List[Dict] = []
        self.learning_history: List[Dict] = []
        
        logger.info("Phase 3 控制器初始化完成")
    
    def set_mode(self, mode: AutonomousMode):
        """设置自治模式"""
        self.mode = mode
        logger.info(f"自治模式切换为: {mode.name}")
    
    def get_mode(self) -> AutonomousMode:
        """获取当前自治模式"""
        return self.mode
    
    def is_capability_available(self, capability: SystemCapability) -> bool:
        """检查能力是否可用"""
        return self.capabilities.get(capability, False)
    
    async def autonomous_optimize(self, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """自治优化"""
        if not self.is_capability_available(SystemCapability.SELF_LEARNING):
            return {'error': '自学习引擎不可用'}
        
        result = await self.learning_engine.optimize(context)
        
        self.learning_history.append({
            'timestamp': datetime.now(),
            'type': 'optimization',
            'result': result
        })
        
        return result
    
    async def multi_objective_decision(self,
                                        objectives: List[Dict],
                                        constraints: Dict) -> Dict[str, Any]:
        """多目标决策"""
        if not self.is_capability_available(SystemCapability.MULTI_OBJECTIVE):
            return {'error': '多目标优化器不可用'}
        
        result = await self.optimizer.optimize(objectives, constraints)
        
        self.decision_history.append({
            'timestamp': datetime.now(),
            'type': 'multi_objective',
            'result': result
        })
        
        return result
    
    async def query_knowledge(self, query: str) -> Dict[str, Any]:
        """知识查询"""
        if not self.is_capability_available(SystemCapability.KNOWLEDGE_GRAPH):
            return {'error': '知识图谱系统不可用'}
        
        return await self.knowledge_system.query(query)
    
    async def predict_failure(self, 
                              service: str,
                              horizon: int = 24) -> Dict[str, Any]:
        """故障预测"""
        if not self.is_capability_available(SystemCapability.PREDICTIVE):
            return {'error': '预测系统不可用'}
        
        return await self.predictive_system.predict(service, horizon)
    
    async def orchestrate_systems(self,
                                   systems: List[str],
                                   action: str) -> Dict[str, Any]:
        """跨系统协同"""
        if not self.is_capability_available(SystemCapability.ORCHESTRATION):
            return {'error': '协同系统不可用'}
        
        return await self.orchestration.orchestrate(systems, action)
    
    async def explain_decision(self, 
                               decision_id: str) -> Dict[str, Any]:
        """解释决策"""
        if not self.is_capability_available(SystemCapability.HUMAN_AI):
            return {'error': '人机界面不可用'}
        
        return await self.human_interface.explain(decision_id)
    
    async def request_approval(self,
                               action: Dict,
                               context: Dict) -> Dict[str, Any]:
        """请求人工审批"""
        if not self.is_capability_available(SystemCapability.HUMAN_AI):
            return {'error': '人机界面不可用'}
        
        return await self.human_interface.request_approval(action, context)
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """获取自治状态"""
        return {
            'mode': self.mode.name,
            'capabilities': {
                cap.value: available
                for cap, available in self.capabilities.items()
            },
            'decision_count': len(self.decision_history),
            'learning_count': len(self.learning_history),
            'phase2': self.phase2.get_integrated_status()
        }
    
    def get_decision_history(self, 
                             limit: int = 100) -> List[Dict]:
        """获取决策历史"""
        return self.decision_history[-limit:]
    
    def get_learning_history(self,
                             limit: int = 100) -> List[Dict]:
        """获取学习历史"""
        return self.learning_history[-limit:]
