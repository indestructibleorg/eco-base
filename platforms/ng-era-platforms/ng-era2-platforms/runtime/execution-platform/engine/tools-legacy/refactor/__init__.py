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
Refactor Tools Package - 重構工具包
提供完整的重構引擎工具集，包括：
模組：
- refactor_engine: 主重構引擎，執行分析、規劃、執行
- cognitive_engine: 認知推理引擎，提供高階理解/推理/搜尋/整合能力
- process_legacy_scratch: 暫存區處理器，處理遺留資產
- execute_integration: 集成執行器，執行整合操作
- update_indexes: 索引更新器，同步各類索引
- validate_structure: 結構驗證器，驗證目錄結構
使用方式：
    # CLI 使用
    python -m tools.refactor.refactor_engine analyze --target docs/refactor_playbooks
    python -m tools.refactor.cognitive_engine --input "重構 playbooks"
    python -m tools.refactor.process_legacy_scratch scan
    python -m tools.refactor.execute_integration reorganize --target docs/refactor_playbooks
    python -m tools.refactor.update_indexes all
    python -m tools.refactor.validate_structure full
    # 程式化使用
"""
Module docstring
================

This module is part of the GL governance framework.
Please add specific module documentation here.
"""
    from tools.refactor import RefactorEngine, CognitiveEngine
    engine = RefactorEngine()
    result = engine.analyze("docs/refactor_playbooks")
Version: 1.0.0
"""
from .cognitive_engine import CognitiveEngine
from .execute_integration import IntegrationExecutor
from .process_legacy_scratch import LegacyScratchProcessor
from .refactor_engine import DirectoryAnalyzer as RefactorEngine
from .update_indexes import IndexUpdater
from .validate_structure import StructureValidatorMain as StructureValidator
__version__ = "1.0.0"
__author__ = "SynergyMesh"
# 延遲導入以避免循環依賴
def __getattr__(name):
    if name == "RefactorEngine":
        from .refactor_engine import DirectoryAnalyzer
        return DirectoryAnalyzer
    elif name == "CognitiveEngine":
        from .cognitive_engine import CognitiveEngine
        return CognitiveEngine
    elif name == "LegacyScratchProcessor":
        from .process_legacy_scratch import LegacyScratchProcessor
        return LegacyScratchProcessor
    elif name == "IntegrationExecutor":
        from .execute_integration import IntegrationExecutor
        return IntegrationExecutor
    elif name == "IndexUpdater":
        from .update_indexes import IndexUpdater
        return IndexUpdater
    elif name == "StructureValidator":
        from .validate_structure import StructureValidatorMain
        return StructureValidatorMain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__all__ = [
    "RefactorEngine",
    "CognitiveEngine",
    "LegacyScratchProcessor",
    "IntegrationExecutor",
    "IndexUpdater",
    "StructureValidator",
]
# Eagerly bind all exported names to ensure they exist for `from ... import *`
