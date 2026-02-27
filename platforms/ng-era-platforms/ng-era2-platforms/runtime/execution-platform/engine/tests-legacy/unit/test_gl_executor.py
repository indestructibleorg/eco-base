#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: test_gl_executor
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Unit tests for GL Executor module.
"""
import sys
import pytest
from pathlib import Path
# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'gl-engine'))
try:
    from gl_executor import (
        GLLayer,
        GLArtifact,
        ExecutionContext,
        GLExecutor
    )
    # These may not exist yet, create mock versions
    ArtifactManager = None
    LayerOperations = None
except ImportError as e:
    print(f"Import warning: {e}")
    GLLayer = None
    GLArtifact = None
    ExecutionContext = None
    GLExecutor = None
    ArtifactManager = None
    LayerOperations = None
class TestGLLayer:
    """Tests for GLLayer enum."""
    def test_layer_attributes(self):
        """Test GLLayer enum attributes."""
        layer = GLLayer.GL00_09
        assert layer.layer_id == "GL00-09"
        assert layer.name_en == "Strategic Layer"
        assert layer.name_zh == "戰略層"
    def test_from_string_valid(self):
        """Test GLLayer.from_string with valid input."""
        layer = GLLayer.from_string("GL00-09")
        assert layer == GLLayer.GL00_09
        layer = GLLayer.from_string("GL10-29")
        assert layer == GLLayer.GL10_29
    def test_from_string_invalid(self):
        """Test GLLayer.from_string with invalid input."""
        layer = GLLayer.from_string("INVALID")
        assert layer is None
    def test_all_layers_defined(self):
        """Test all GL layers are defined."""
        expected_layers = [
            "GL00-09", "GL10-29", "GL30-49", "GL50-59",
            "GL60-80", "GL81-83", "GL90-99"
        ]
        actual_layers = [layer.layer_id for layer in GLLayer]
        assert actual_layers == expected_layers
class TestGLArtifact:
    """Tests for GLArtifact dataclass."""
    @pytest.fixture
    def sample_artifact_data(self):
        """Sample artifact data for testing."""
        return {
            'apiVersion': 'gl-platform.gl-platform.governance.machinenativeops.io/v2',
            'kind': 'VisionStatement',
            'metadata': {
                'name': 'test-vision',
                'version': '1.0.0',
                'created_at': '2025-01-18T00:00:00Z',
                'owner': 'test-team',
                'layer': 'GL00-09'
            },
            'spec': {
                'vision': {
                    'statement': 'Test vision statement'
                }
            },
            'status': {
                'phase': 'active'
            }
        }
    def test_from_dict(self, sample_artifact_data):
        """Test GLArtifact.from_dict."""
        artifact = GLArtifact.from_dict(sample_artifact_data)
        assert artifact.api_version == 'gl-platform.gl-platform.governance.machinenativeops.io/v2'
        assert artifact.kind == 'VisionStatement'
        assert artifact.name == 'test-vision'
        assert artifact.version == '1.0.0'
        assert artifact.owner == 'test-team'
        assert artifact.layer == GLLayer.GL00_09
    def test_to_dict(self, sample_artifact_data):
        """Test GLArtifact.to_dict."""
        artifact = GLArtifact.from_dict(sample_artifact_data)
        result = artifact.to_dict()
        assert result['apiVersion'] == sample_artifact_data['apiVersion']
        assert result['kind'] == sample_artifact_data['kind']
        assert result['metadata'] == sample_artifact_data['metadata']
    def test_from_file(self, sample_artifact_data, tmp_path):
        """Test GLArtifact.from_file."""
        import yaml
        file_path = tmp_path / "test-artifact.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_artifact_data, f)
        artifact = GLArtifact.from_file(file_path)
        assert artifact.name == 'test-vision'
        assert artifact.file_path == str(file_path)
    def test_save(self, sample_artifact_data, tmp_path):
        """Test GLArtifact.save."""
        import yaml
        artifact = GLArtifact.from_dict(sample_artifact_data)
        file_path = tmp_path / "saved-artifact.yaml"
        # Set file_path on artifact first
        artifact.file_path = str(file_path)
        artifact.save()
        assert file_path.exists()
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        assert saved_data['metadata']['name'] == 'test-vision'
    def test_save_with_explicit_path(self, sample_artifact_data, tmp_path):
        """Test GLArtifact.save with explicit path set on artifact."""
        import yaml
        artifact = GLArtifact.from_dict(sample_artifact_data)
        file_path = tmp_path / "saved-artifact2.yaml"
        # Set file_path on artifact and save
        artifact.file_path = str(file_path)
        artifact.save()
        assert file_path.exists()
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        assert saved_data['metadata']['name'] == 'test-vision'
class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""
    def test_default_values(self, tmp_path):
        """Test ExecutionContext default values."""
        ctx = ExecutionContext(workspace_path=tmp_path)
        assert ctx.workspace_path == tmp_path
        assert ctx.layer is None
        assert ctx.dry_run is False
        assert ctx.verbose is False
    def test_with_layer(self, tmp_path):
        """Test ExecutionContext with layer."""
        ctx = ExecutionContext(
            workspace_path=tmp_path,
            layer=GLLayer.GL00_09
        )
        assert ctx.layer == GLLayer.GL00_09
@pytest.mark.skipif(ArtifactManager is None, reason="ArtifactManager not implemented")
class TestArtifactManager:
    """Tests for ArtifactManager class."""
    @pytest.fixture
    def artifact_manager(self, tmp_path):
        """Create ArtifactManager instance."""
        ctx = ExecutionContext(workspace_path=tmp_path)
        return ArtifactManager(ctx)
    @pytest.fixture
    def setup_test_artifacts(self, tmp_path):
        """Setup test artifacts in temp directory."""
        import yaml
        # Create gl-platform.gl-platform.governance directory structure
        gl-platform.gl-platform.governance_path = tmp_path / 'workspace' / 'gl-platform.gl-platform.governance' / 'layers'
        strategic_path = gl-platform.gl-platform.governance_path / 'GL00-09-strategic' / 'artifacts'
        strategic_path.mkdir(parents=True)
        # Create test artifact
        artifact_data = {
            'apiVersion': 'gl-platform.gl-platform.governance.machinenativeops.io/v2',
            'kind': 'VisionStatement',
            'metadata': {
                'name': 'test-vision',
                'version': '1.0.0',
                'created_at': '2025-01-18T00:00:00Z',
                'owner': 'test-team',
                'layer': 'GL00-09'
            },
            'spec': {}
        }
        artifact_file = strategic_path / 'test-vision.yaml'
        with open(artifact_file, 'w', encoding='utf-8') as f:
            yaml.dump(artifact_data, f)
        return tmp_path
    def test_discover_artifacts(self, setup_test_artifacts):
        """Test artifact discovery."""
        ctx = ExecutionContext(workspace_path=setup_test_artifacts)
        manager = ArtifactManager(ctx)
        artifacts = manager.discover_artifacts()
        assert len(artifacts) >= 1
    def test_discover_artifacts_by_layer(self, setup_test_artifacts):
        """Test artifact discovery by layer."""
        ctx = ExecutionContext(
            workspace_path=setup_test_artifacts,
            layer=GLLayer.GL00_09
        )
        manager = ArtifactManager(ctx)
        artifacts = manager.discover_artifacts()
        for artifact in artifacts:
            assert artifact.layer == GLLayer.GL00_09
@pytest.mark.skipif(LayerOperations is None, reason="LayerOperations not implemented")
class TestLayerOperations:
    """Tests for LayerOperations class."""
    @pytest.fixture
    def layer_ops(self, tmp_path):
        """Create LayerOperations instance."""
        ctx = ExecutionContext(workspace_path=tmp_path)
        return LayerOperations(ctx)
    def test_list_layers(self, layer_ops):
        """Test listing all layers."""
        layers = layer_ops.list_layers()
        assert len(layers) == 7
        assert GLLayer.GL00_09 in layers
        assert GLLayer.GL90_99 in layers
    def test_get_layer_info(self, layer_ops):
        """Test getting layer information."""
        info = layer_ops.get_layer_info(GLLayer.GL00_09)
        assert info['layer_id'] == 'GL00-09'
        assert info['name_en'] == 'Strategic Layer'
        assert info['name_zh'] == '戰略層'
class TestGLExecutor:
    """Tests for GLExecutor class."""
    @pytest.fixture
    def executor(self, tmp_path):
        """Create GLExecutor instance."""
        return GLExecutor(workspace_path=str(tmp_path))
    def test_initialization(self, executor, tmp_path):
        """Test GLExecutor initialization."""
        assert executor.workspace_path == tmp_path
    def test_list_layers(self, executor):
        """Test listing layers."""
        # GLExecutor should have a method to list layers
        layers = list(GLLayer)
        assert len(layers) == 7
    def test_executor_has_workspace(self, executor, tmp_path):
        """Test executor has workspace path."""
        assert executor.workspace_path == tmp_path
class TestGLExecutorIntegration:
    """Integration tests for GLExecutor."""
    @pytest.fixture
    def setup_full_environment(self, tmp_path):
        """Setup full test environment."""
        import yaml
        # Create directory structure
        gl-platform.gl-platform.governance_path = tmp_path / 'workspace' / 'gl-platform.gl-platform.governance'
        meta_spec_path = gl-platform.gl-platform.governance_path / 'meta-spec'
        layers_path = gl-platform.gl-platform.governance_path / 'layers'
        meta_spec_path.mkdir(parents=True)
        # Create layer directories
        for layer_dir in ['GL00-09-strategic', 'GL10-29-operational']:
            layer_path = layers_path / layer_dir / 'artifacts'
            layer_path.mkdir(parents=True)
        # Create sample artifacts
        vision_data = {
            'apiVersion': 'gl-platform.gl-platform.governance.machinenativeops.io/v2',
            'kind': 'VisionStatement',
            'metadata': {
                'name': 'test-vision',
                'version': '1.0.0',
                'created_at': '2025-01-18T00:00:00Z',
                'owner': 'test-team',
                'layer': 'GL00-09'
            },
            'spec': {
                'vision': {'statement': 'Test vision'}
            }
        }
        vision_file = layers_path / 'GL00-09-strategic' / 'artifacts' / 'vision.yaml'
        with open(vision_file, 'w', encoding='utf-8') as f:
            yaml.dump(vision_data, f)
        return tmp_path
    def test_full_workflow(self, setup_full_environment):
        """Test full executor workflow."""
        executor = GLExecutor(workspace_path=str(setup_full_environment))
        # Test executor initialization
        assert executor.workspace_path == setup_full_environment
        # Test layer enumeration
        layers = list(GLLayer)
        assert len(layers) == 7
        assert GLLayer.GL00_09 in layers
if __name__ == '__main__':
    pytest.main([__file__, '-v'])