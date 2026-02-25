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
SynergyMesh Automation Engines - 自動化引擎集合
子目錄專用引擎：
├── refactor_automation_engine.py    - 重構自動化引擎
├── integration_automation_engine.py - 整合自動化引擎
├── validation_automation_engine.py  - 驗證自動化引擎
├── generation_engine.py             - 生成自動化引擎
└── baseline_validation_engine.py    - 基線驗證引擎
Future engines (planned):
├── deconstruction_engine.py         - 解構自動化引擎
├── monitoring_engine.py             - 監控自動化引擎
└── sync_engine.py                   - 同步自動化引擎
"""
__version__ = "1.0.0"
__author__ = "SynergyMesh"
# Lazy loading to avoid circular dependencies
"""
Module docstring
================

This module is part of the GL governance framework.
Please add specific module documentation here.
"""
def __getattr__(name):
    if name == "RefactorAutomationEngine":
        from .refactor_automation_engine import RefactorAutomationEngine
        return RefactorAutomationEngine
    elif name == "IntegrationAutomationEngine":
        from .integration_automation_engine import IntegrationAutomationEngine
        return IntegrationAutomationEngine
    elif name == "ValidationAutomationEngine":
        from .validation_automation_engine import ValidationAutomationEngine
        return ValidationAutomationEngine
    elif name == "GenerationEngine":
        from .generation_engine import GenerationEngine
        return GenerationEngine
    elif name == "BaselineValidationEngine":
        from .baseline_validation_engine import BaselineValidationEngine
        return BaselineValidationEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "RefactorAutomationEngine",
    "IntegrationAutomationEngine",
    "ValidationAutomationEngine",
    "GenerationEngine",
    "BaselineValidationEngine",
]
