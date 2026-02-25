# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: adaptive_decision_engine
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
自適應決策引擎 - Adaptive Decision Engine
MachineNativeOps 驗證系統 v1.0.0
此模組實現量子-經典混合驗證的決策邏輯，支持動態參數調整。
"""
# MNGA-002: Import organization needs review
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
# 配置日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
class ValidationStatus(Enum):
    """驗證狀態枚舉"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    PENDING = "PENDING"
class FusionAlgorithm(Enum):
    """融合算法枚舉"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_FUSION = "bayesian_fusion"
    ENSEMBLE_VOTING = "ensemble_voting"
@dataclass
class DynamicParameters:
    """動態參數配置"""
    classic_weight: float = 0.43
    quantum_weight: float = 0.57
    circuit_depth: int = 12
    fusion_algorithm: str = "weighted_average"
    preset: str = "standard_v3"
@dataclass
class ValidationDimension:
    """驗證維度結果"""
    name: str
    status: ValidationStatus
    classic_score: float
    quantum_score: float
    hybrid_score: float
    quantum_entropy: float
    weight: float
@dataclass
class ClassicValidationResult:
    """經典驗證結果"""
    structural_compliance: float = 0.0
    content_accuracy: float = 0.0
    file_paths: float = 0.0
    naming_conventions: float = 0.0
    consistency: float = 0.0
    logical_coherence: float = 0.0
    contextual_continuity: float = 0.0
    final_correctness: float = 0.0
@dataclass
class QuantumValidationResult:
    """量子驗證結果"""
    structural_compliance_entropy: float = 0.0
    content_accuracy_entropy: float = 0.0
    consistency_entropy: float = 0.0
    logical_coherence_entropy: float = 0.0
    contextual_continuity_entropy: float = 0.0
    final_correctness_entropy: float = 0.0
@dataclass
class HybridDecision:
    """混合決策結果"""
    overall_status: ValidationStatus = ValidationStatus.PENDING
    confidence: float = 0.0
    dimensions: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
class DynamicPolicyController:
    """動態策略控制器"""
    def __init__(self):
        self.emergency_mode = False
        self.current_preset = "standard_v3"
        # 預設配置
        self.presets = {
            "standard_v3": DynamicParameters(
                classic_weight=0.43,
                quantum_weight=0.57,
                circuit_depth=12,
                fusion_algorithm="weighted_average",
                preset="standard_v3",
            ),
            "lightweight_v2": DynamicParameters(
                classic_weight=0.70,
                quantum_weight=0.30,
                circuit_depth=6,
                fusion_algorithm="weighted_average",
                preset="lightweight_v2",
            ),
            "high_precision": DynamicParameters(
                classic_weight=0.35,
                quantum_weight=0.65,
                circuit_depth=16,
                fusion_algorithm="bayesian_fusion",
                preset="high_precision",
            ),
            "emergency_fallback": DynamicParameters(
                classic_weight=0.90,
                quantum_weight=0.10,
                circuit_depth=4,
                fusion_algorithm="weighted_average",
                preset="emergency_fallback",
            ),
        }
    def adjust_parameters(
        self, noise_level: float, coherence_time: float
    ) -> DynamicParameters:
        """
        根據量子狀態調整參數
        Args:
            noise_level: 當前噪聲水平
            coherence_time: 當前相干時間
        Returns:
            DynamicParameters: 調整後的參數
        """
        logger.info(
            f"Adjusting parameters - Noise: {noise_level:.4f}, Coherence: {coherence_time:.4f}"
        )
        # 高噪聲情況
        if noise_level > 0.15:
            logger.warning("High quantum noise detected, shifting to lightweight mode")
            return self.presets["lightweight_v2"]
        # 低相干性情況
        if coherence_time < 0.78:
            logger.warning("Low coherence time detected, triggering recalibration mode")
            # Create a copy to avoid mutating shared preset
            params = copy.deepcopy(self.presets["standard_v3"])
            params.classic_weight += 0.12
            params.quantum_weight -= 0.12
            return params
        # 最優條件
        if noise_level < 0.08 and coherence_time > 0.85:
            logger.info("Optimal conditions detected, enabling high precision mode")
            return self.presets["high_precision"]
        # 標準模式
        return self.presets["standard_v3"]
    def activate_emergency_mode(
        self,
        strategy: str = "classic_aggressive",
        quantum_preset: str = "lightweight_v2",
    ):
        """啟用緊急模式"""
        self.emergency_mode = True
        self.current_preset = "emergency_fallback"
        logger.warning(f"Emergency mode activated - Strategy: {strategy}")
class FusionEngine:
    """融合引擎"""
    def __init__(self, algorithm: FusionAlgorithm = FusionAlgorithm.WEIGHTED_AVERAGE):
        self.algorithm = algorithm
    def fuse_scores(
        self,
        classic_score: float,
        quantum_score: float,
        classic_weight: float,
        quantum_weight: float,
    ) -> float:
        """融合經典和量子分數"""
        if self.algorithm == FusionAlgorithm.WEIGHTED_AVERAGE:
            return classic_score * classic_weight + quantum_score * quantum_weight
        elif self.algorithm == FusionAlgorithm.BAYESIAN_FUSION:
            # 簡化的貝葉斯融合
            prior = 0.5
            likelihood = classic_score * quantum_score
            return (likelihood * prior) / (
                (likelihood * prior) + ((1 - likelihood) * (1 - prior))
            )
        else:
            # 集成投票
            threshold = 0.6
            votes = sum(
                [
                    1 if classic_score > threshold else 0,
                    1 if quantum_score > threshold else 0,
                ]
            )
            return 1.0 if votes >= 1 else 0.0
    def decide(
        self,
        classic_results: ClassicValidationResult,
        quantum_results: QuantumValidationResult,
        params: DynamicParameters,
    ) -> HybridDecision:
        """
        執行混合決策
        Args:
            classic_results: 經典驗證結果
            quantum_results: 量子驗證結果
            params: 動態參數
        Returns:
            HybridDecision: 混合決策結果
        """
        dimensions = []
        dimension_weights = {
            "structural_compliance": 0.15,
            "content_accuracy": 0.15,
            "file_paths": 0.10,
            "naming_conventions": 0.10,
            "consistency": 0.15,
            "logical_coherence": 0.15,
            "contextual_continuity": 0.10,
            "final_correctness": 0.10,
        }
        # 處理每個驗證維度
        classic_dict = {
            "structural_compliance": classic_results.structural_compliance,
            "content_accuracy": classic_results.content_accuracy,
            "file_paths": classic_results.file_paths,
            "naming_conventions": classic_results.naming_conventions,
            "consistency": classic_results.consistency,
            "logical_coherence": classic_results.logical_coherence,
            "contextual_continuity": classic_results.contextual_continuity,
            "final_correctness": classic_results.final_correctness,
        }
        quantum_dict = {
            "structural_compliance": quantum_results.structural_compliance_entropy,
            "content_accuracy": quantum_results.content_accuracy_entropy,
            "file_paths": 0.9999,  # 非量子增強
            "naming_conventions": 0.9978,  # 非量子增強
            "consistency": quantum_results.consistency_entropy,
            "logical_coherence": quantum_results.logical_coherence_entropy,
            "contextual_continuity": quantum_results.contextual_continuity_entropy,
            "final_correctness": quantum_results.final_correctness_entropy,
        }
        total_score = 0.0
        for dim_name, weight in dimension_weights.items():
            classic_score = classic_dict.get(dim_name, 0.95)
            quantum_score = quantum_dict.get(dim_name, 0.99)
            hybrid_score = self.fuse_scores(
                classic_score,
                quantum_score,
                params.classic_weight,
                params.quantum_weight,
            )
            status = (
                ValidationStatus.PASS
                if hybrid_score > 0.9
                else (
                    ValidationStatus.WARNING
                    if hybrid_score > 0.7
                    else ValidationStatus.FAIL
                )
            )
            dimensions.append(
                ValidationDimension(
                    name=dim_name,
                    status=status,
                    classic_score=classic_score,
                    quantum_score=quantum_score,
                    hybrid_score=hybrid_score,
                    quantum_entropy=quantum_score,
                    weight=weight,
                )
            )
            total_score += hybrid_score * weight
        # 確定整體狀態
        overall_status = (
            ValidationStatus.PASS
            if total_score > 0.9
            else (
                ValidationStatus.WARNING if total_score > 0.7 else ValidationStatus.FAIL
            )
        )
        return HybridDecision(
            overall_status=overall_status,
            confidence=total_score,
            dimensions=dimensions,
            recommendations=self._generate_recommendations(dimensions),
        )
    def _generate_recommendations(self, dimensions: list) -> list:
        """生成建議"""
        recommendations = []
        for dim in dimensions:
            if dim.status == ValidationStatus.FAIL:
                recommendations.append(
                    f"Critical: {dim.name} validation failed (score: {dim.hybrid_score:.4f})"
                )
            elif dim.status == ValidationStatus.WARNING:
                recommendations.append(
                    f"Warning: {dim.name} requires attention (score: {dim.hybrid_score:.4f})"
                )
        if not recommendations:
            recommendations.append("All validations passed successfully")
        return recommendations
class AdaptiveDecisionEngine:
    """
    自適應決策引擎
    實現量子-經典混合驗證的完整決策流程。
    """
    def __init__(self):
        self.dynamic_controller = DynamicPolicyController()
        self.fusion_engine = FusionEngine()
    def execute_decision(
        self,
        classic_results: ClassicValidationResult,
        quantum_features: dict,
        document_path: Optional[str] = None,
    ) -> dict:
        """
        執行自適應決策
        Args:
            classic_results: 經典驗證結果
            quantum_features: 量子特徵
            document_path: 文檔路徑
        Returns:
            dict: 完整決策報告
        """
        # 調整動態參數
        params = self.dynamic_controller.adjust_parameters(
            noise_level=quantum_features.get("noise_level", 0.1),
            coherence_time=quantum_features.get("coherence_time", 0.79),
        )
        # 構建量子驗證結果
        quantum_results = QuantumValidationResult(
            structural_compliance_entropy=quantum_features.get("coherence", 0.99),
            content_accuracy_entropy=quantum_features.get("fidelity", 0.99),
            consistency_entropy=quantum_features.get("entanglement", 0.99),
            logical_coherence_entropy=quantum_features.get("coherence", 0.99),
            contextual_continuity_entropy=quantum_features.get("fidelity", 0.99),
            final_correctness_entropy=quantum_features.get("coherence", 0.99),
        )
        # 執行混合決策
        decision = self.fusion_engine.decide(classic_results, quantum_results, params)
        # 構建報告
        return {
            "validation_report": {
                "document": document_path,
                "timestamp": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "overall_status": decision.overall_status.value,
                "confidence": round(decision.confidence, 4),
                "verification_matrix": [
                    {
                        "dimension": dim.name,
                        "status": dim.status.value,
                        "quantum_entropy": round(dim.quantum_entropy, 4),
                    }
                    for dim in decision.dimensions
                ],
                "dynamic_parameters": {
                    "classic_weight": params.classic_weight,
                    "quantum_weight": params.quantum_weight,
                    "circuit_depth": params.circuit_depth,
                    "fusion_algorithm": params.fusion_algorithm,
                    "preset": params.preset,
                },
                "recommendations": decision.recommendations,
            }
        }
def run_demo_decision() -> dict:
    """運行演示決策"""
    engine = AdaptiveDecisionEngine()
    # 模擬經典驗證結果
    classic_results = ClassicValidationResult(
        structural_compliance=0.98,
        content_accuracy=0.97,
        file_paths=0.99,
        naming_conventions=0.96,
        consistency=0.95,
        logical_coherence=0.98,
        contextual_continuity=0.97,
        final_correctness=0.99,
    )
    # 模擬量子特徵
    quantum_features = {
        "coherence": 0.792,
        "entanglement": 0.965,
        "fidelity": 0.9997,
        "noise_level": 0.134,
        "coherence_time": 0.792,
    }
    return engine.execute_decision(
        classic_results, quantum_features, document_path="/demo/example-spec.json"
    )
def main():
    """主函數"""
    import argparse
    parser = argparse.ArgumentParser(
        description="自適應決策引擎 - MachineNativeOps 驗證系統"
    )
    parser.add_argument("--demo", action="store_true", help="運行演示模式")
    parser.add_argument("--output", default=None, help="輸出文件路徑")
    args = parser.parse_args()
    print("=" * 70)
    print("MachineNativeOps 自適應決策引擎 v1.0.0")
    print("=" * 70)
    result = run_demo_decision()
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output)
        print(f"\n結果已保存到: {args.output}")
    else:
        print("\n決策結果:")
        print(output)
    print("\n" + "=" * 70)
    print(f"決策完成 - 狀態: {result['validation_report']['overall_status']}")
    print(f"置信度: {result['validation_report']['confidence']:.4f}")
    print("=" * 70)
    return 0
if __name__ == "__main__":
    exit(main())
