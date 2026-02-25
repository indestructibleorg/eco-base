# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: __init__
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
SynergyMesh Automation Framework - 全自動化引擎框架
100% 機器自主操作，零人類介入的自動化系統
架構層次：
├── MasterOrchestrator (根目錄主控)
│   ├── EngineRegistry (引擎註冊中心)
│   ├── EngineScheduler (引擎調度器)
│   ├── EventBus (事件總線)
│   └── HealthMonitor (健康監控)
├── BaseEngine (引擎基礎類)
│   ├── CognitiveEngine (認知引擎)
│   ├── ExecutionEngine (執行引擎)
│   ├── ValidationEngine (驗證引擎)
│   └── TransformEngine (轉換引擎)
└── DomainEngines (領域引擎)
    ├── RefactorEngine
    ├── IntegrationEngine
    ├── DeconstructionEngine
    └── ... (可擴展)
Version: 1.0.0
"""
__version__ = "1.0.0"
__author__ = "SynergyMesh"
"""
Module docstring
================

This module is part of the GL governance framework.
Please add specific module documentation here.
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engine_base import BaseEngine, EngineConfig, EngineState
    from .engine_registry import EngineRegistry
    from .engine_scheduler import EngineScheduler
    from .event_bus import EventBus
    from .master_orchestrator import MasterOrchestrator
__all__ = [
    "MasterOrchestrator",
    "BaseEngine",
    "EngineConfig",
    "EngineState",
    "EngineRegistry",
    "EngineScheduler",
    "EventBus",
]
