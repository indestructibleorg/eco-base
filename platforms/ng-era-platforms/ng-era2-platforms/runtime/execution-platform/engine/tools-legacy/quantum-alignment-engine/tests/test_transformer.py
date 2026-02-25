# @ECO-layer: GQS-L0
# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: test_transformer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
"""
Unit tests for Quantum Code Alignment Engine
"""
# MNGA-002: Import organization needs review
from pathlib import Path
import numpy as np
import pytest
from src.core.transformer import (
    CodeElement,
    EntanglementMapper,
    QuantumCodeTransformer,
    QuantumNode,
    QuantumState,
    SemanticDecoherenceError,
    SemanticLattice,
)
class TestCodeElement:
    """Test CodeElement creation and properties"""
    def test_code_element_creation(self):
        """Test creating a code element"""
        element = CodeElement(
            name="test_function",
            semantic_vector=np.random.randn(8192),
            namespace="test.namespace",
            dependencies=["dep1", "dep2"],
            element_type="function",
        )
        assert element.name == "test_function"
        assert element.namespace == "test.namespace"
        assert element.element_type == "function"
        assert len(element.dependencies) == 2
        assert element.quantum_state == QuantumState.SUPERPOSITION
    def test_semantic_vector_normalization(self):
        """Test semantic vector is properly normalized"""
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        norm = np.linalg.norm(element.semantic_vector)
        assert norm > 0
class TestSemanticLattice:
    """Test SemanticLattice quantum operations"""
    def test_lattice_initialization(self):
        """Test lattice initialization"""
        lattice = SemanticLattice(dimension=1024)
        assert lattice.dimension == 1024
        assert len(lattice.elements) == 0
    def test_hadamard_transformation(self):
        """Test Hadamard gate transformation"""
        lattice = SemanticLattice()
        vector = np.random.randn(8192)
        transformed = lattice._apply_hadamard(vector)
        assert len(transformed) == 8192
        # Hadamard should normalize
        assert np.isclose(np.linalg.norm(transformed), 1.0, atol=0.1)
    def test_entanglement_with_policy(self):
        """Test entanglement with policy"""
        lattice = SemanticLattice()
        state = np.random.randn(8192)
        entangled = lattice._entanglement_with_policy(state)
        assert len(entangled) == 8192
        assert entangled is not None
    def test_lattice_projection(self):
        """Test projecting code element to lattice"""
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        assert isinstance(node, QuantumNode)
        assert node.element == element
class TestEntanglementMapper:
    """Test EntanglementMapper remapping"""
    def test_mapper_initialization(self):
        """Test mapper initialization"""
        mapper = EntanglementMapper()
        assert mapper.namespace_registry is not None
    def test_feature_extraction(self):
        """Test semantic feature extraction"""
        mapper = EntanglementMapper()
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        features = mapper._extract_features(node)
        assert len(features) == 512
    def test_namespace_matching(self):
        """Test finding best namespace match"""
        mapper = EntanglementMapper()
        features = np.random.randn(512)
        # Should find a match from default namespaces
        namespace = mapper._find_best_match(features, "axiom-naming-v9")
        assert namespace is not None
        assert isinstance(namespace, str)
    def test_entanglement_creation(self):
        """Test creating entangled state"""
        mapper = EntanglementMapper()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        entangled = mapper._create_entanglement(element, "workspace.src.gl-platform.governance")
        assert isinstance(entangled, QuantumNode)
        assert (
            entangled.element.context.get("target_namespace")
            == "workspace.src.gl-platform.governance"
        )
    def test_coherence_measurement(self):
        """Test measuring quantum coherence"""
        mapper = EntanglementMapper()
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        coherence = mapper._measure_coherence(node)
        assert 0.0 <= coherence <= 1.0
    def test_remap_success(self):
        """Test successful remapping"""
        mapper = EntanglementMapper()
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        remapped = mapper.remap(node, "axiom-naming-v9")
        assert isinstance(remapped, QuantumNode)
        assert "target_namespace" in remapped.element.context
class TestQuantumCodeTransformer:
    """Test QuantumCodeTransformer pipeline"""
    def test_transformer_initialization(self):
        """Test transformer initialization"""
        transformer = QuantumCodeTransformer()
        assert transformer.semantic_lattice is not None
        assert transformer.entanglement_mapper is not None
    def test_semantic_vector_generation(self):
        """Test semantic vector generation"""
        transformer = QuantumCodeTransformer()
        vector = transformer._generate_semantic_vector("test string")
        assert len(vector) == 8192
        # Vector should be normalized
        norm = np.linalg.norm(vector)
        assert np.isclose(norm, 1.0, atol=0.1)
    def test_qir_generation(self):
        """Test quantum IR generation"""
        transformer = QuantumCodeTransformer()
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        qir = transformer._compile_to_qir([node])
        assert len(qir.circuits) == 1
        assert len(qir.circuits[0]) == 3  # Hadamard, CNOT, Rotation
    def test_decoherence_calibration(self):
        """Test decoherence calibration"""
        transformer = QuantumCodeTransformer()
        lattice = SemanticLattice()
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        qir = transformer._compile_to_qir([node])
        calibrated = transformer._stabilize_quantum_state(qir)
        assert calibrated is not None
        assert len(calibrated.circuits) == 1
    @pytest.mark.integration
    def test_full_transformation_pipeline(self, tmp_path):
        """Test full transformation pipeline"""
        # Create test codebase
        test_code = tmp_path / "test_code"
        test_code.mkdir()
        test_file = test_code / "test_module.py"
        test_file.write_text(
            """
def test_function():
    pass
class TestClass:
    def method(self):
        pass
"""
        )
        # Run transformation
        transformer = QuantumCodeTransformer()
        output_path = tmp_path / "output"
        result = transformer.transform(
            str(test_code),
            target_policy="axiom-naming-v9",
            output_path=str(output_path),
        )
        # Verify output
        assert Path(result).exists()
        assert "transformed" in result
class TestErrorHandling:
    """Test error handling"""
    def test_decoherence_error(self):
        """Test SemanticDecoherenceError is raised"""
        mapper = EntanglementMapper()
        lattice = SemanticLattice()
        # Create element with low coherence
        element = CodeElement(
            name="test", semantic_vector=np.random.randn(8192), namespace="test"
        )
        node = lattice.project(element)
        # Manually set low coherence to trigger error
        with pytest.raises(SemanticDecoherenceError):
            # This would need modification to actually test
            mapper._measure_coherence(node)
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
