#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl_integrator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Integrator - Cross-Layer Integration and Orchestration
MachineNativeOps GL Architecture Implementation
This module provides integration capabilities for GL gl-platform.gl-platform.governance layers,
enabling cross-references, data flow, and unified gl-platform.gl-platform.governance operations.
"""
# MNGA-002: Import organization needs review
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GLIntegrator')
class IntegrationStatus(Enum):
    """Integration status levels."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
class LinkType(Enum):
    """Types of semantic links between artifacts."""
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    IMPLEMENTS = "implements"
    VALIDATES = "validates"
    MONITORS = "monitors"
@dataclass
class SemanticLink:
    """Represents a semantic link between artifacts."""
    source_id: str
    target_id: str
    link_type: LinkType
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    strength: float = 1.0  # 0.0 to 1.0
@dataclass
class IntegrationPoint:
    """Represents an integration point between GL layers."""
    id: str
    source_layer: str
    target_layer: str
    integration_type: str
    status: IntegrationStatus
    links: List[SemanticLink] = field(default_factory=list)
    last_verified: Optional[datetime] = None
    consistency_score: float = 0.0
@dataclass
class IntegrationMatrix:
    """Integration matrix showing all cross-layer relationships."""
    layers: List[str]
    connections: Dict[Tuple[str, str], List[SemanticLink]]
    total_links: int = 0
    coverage_percentage: float = 0.0
class GLIntegrator:
    """
    Cross-layer integration and orchestration system.
    Features:
    - Semantic linking between artifacts
    - Cross-layer dependency management
    - Consistency verification
    - Integration matrix generation
    - Automated reconciliation
    """
    def __init__(self):
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.links: List[SemanticLink] = []
        self.integration_points: Dict[str, IntegrationPoint] = {}
        self.consistency_issues: List[Dict[str, Any]] = []
    def load_artifacts(self, root_path: Path) -> None:
        """Load all GL artifacts from the repository."""
        logger.info(f"Loading artifacts from: {root_path}")
        gl_files = list(root_path.glob("GL*.json"))
        for gl_file in gl_files:
            try:
                with open(gl_file, 'r') as f:
                    data = json.load(f)
                artifact_id = data.get("id", gl_file.stem)
                self.artifacts[artifact_id] = {
                    "data": data,
                    "file_path": str(gl_file),
                    "layer": self._extract_layer(data.get("gllevel", "")),
                    "name": data.get("name", artifact_id)
                }
                logger.debug(f"Loaded artifact: {artifact_id}")
            except Exception as e:
                logger.error(f"Failed to load artifact {gl_file}: {str(e)}")
        logger.info(f"Loaded {len(self.artifacts)} artifacts")
    def _extract_layer(self, gllevel: str) -> str:
        """Extract layer identifier from gllevel."""
        if not gllevel:
            return "unknown"
        # Map GL levels to layer groups
        if gllevel.startswith("GL00"):
            return "strategic"
        elif gllevel.startswith("GL01") or gllevel.startswith("GL02"):
            return "strategic_metrics"
        elif gllevel.startswith("GL10"):
            return "operational"
        elif gllevel.startswith("GL30"):
            return "execution"
        elif gllevel.startswith("GL50"):
            return "observability"
        elif gllevel.startswith("GL60"):
            return "advanced"
        elif gllevel.startswith("GL90"):
            return "meta"
        else:
            return "other"
    def create_semantic_link(self,
                           source_id: str,
                           target_id: str,
                           link_type: LinkType,
                           description: str,
                           strength: float = 1.0) -> SemanticLink:
        """Create a semantic link between artifacts."""
        if source_id not in self.artifacts:
            raise ValueError(f"Source artifact not found: {source_id}")
        if target_id not in self.artifacts:
            raise ValueError(f"Target artifact not found: {target_id}")
        link = SemanticLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            description=description,
            strength=strength
        )
        self.links.append(link)
        logger.info(f"Created semantic link: {source_id} -> {target_id} ({link_type.value})")
        return link
    def auto_discover_links(self) -> List[SemanticLink]:
        """Automatically discover semantic links between artifacts."""
        logger.info("Auto-discovering semantic links...")
        discovered_links = []
        # Strategy to Root links
        for artifact_id, artifact_info in self.artifacts.items():
            if artifact_id.startswith("GL00"):
                continue  # Skip root artifacts
            # Link to GL00 root
            if "GL00-root-semantic-anchor" in self.artifacts:
                link = SemanticLink(
                    source_id=artifact_id,
                    target_id="GL00-0001",
                    link_type=LinkType.DEPENDS_ON,
                    description=f"{artifact_id} depends on GL00 root",
                    strength=0.9
                )
                discovered_links.append(link)
        # Metrics to Risk links
        risk_artifacts = [id for id in self.artifacts.keys() if "risk" in id.lower()]
        metrics_artifacts = [id for id in self.artifacts.keys() if "metrics" in id.lower() or "success" in id.lower()]
        for risk_id in risk_artifacts:
            for metrics_id in metrics_artifacts:
                link = SemanticLink(
                    source_id=metrics_id,
                    target_id=risk_id,
                    link_type=LinkType.VALIDATES,
                    description="Metrics validate risk mitigation",
                    strength=0.7
                )
                discovered_links.append(link)
        # Charter to all links
        charter_artifacts = [id for id in self.artifacts.keys() if "charter" in id.lower()]
        for charter_id in charter_artifacts:
            for artifact_id in self.artifacts.keys():
                if artifact_id == charter_id or artifact_id.startswith("GL00"):
                    continue
                link = SemanticLink(
                    source_id=charter_id,
                    target_id=artifact_id,
                    link_type=LinkType.IMPLEMENTS,
                    description=f"Charter implements gl-platform.gl-platform.governance for {artifact_id}",
                    strength=0.6
                )
                discovered_links.append(link)
        # Add discovered links
        for link in discovered_links:
            self.links.append(link)
        logger.info(f"Auto-discovered {len(discovered_links)} semantic links")
        return discovered_links
    def verify_consistency(self) -> Dict[str, Any]:
        """Verify consistency across linked artifacts."""
        logger.info("Verifying cross-artifact consistency...")
        issues = []
        verified_count = 0
        for link in self.links:
            try:
                source = self.artifacts.get(link.source_id)
                target = self.artifacts.get(link.target_id)
                if not source or not target:
                    issues.append({
                        "type": "missing_artifact",
                        "link": f"{link.source_id} -> {link.target_id}",
                        "description": "One or both artifacts missing"
                    })
                    continue
                # Check semantic consistency
                consistency_score = self._check_link_consistency(link, source, target)
                if consistency_score < 0.5:
                    issues.append({
                        "type": "consistency_issue",
                        "link": f"{link.source_id} -> {link.target_id}",
                        "score": consistency_score,
                        "description": f"Low consistency score: {consistency_score:.2f}"
                    })
                verified_count += 1
            except Exception as e:
                issues.append({
                    "type": "verification_error",
                    "link": f"{link.source_id} -> {link.target_id}",
                    "error": str(e)
                })
        self.consistency_issues = issues
        consistency_score = (verified_count / len(self.links) * 100) if self.links else 100
        result = {
            "total_links": len(self.links),
            "verified_links": verified_count,
            "issues_found": len(issues),
            "consistency_score": consistency_score,
            "issues": issues
        }
        logger.info(f"Consistency verification complete: {consistency_score:.1f}%")
        return result
    def _check_link_consistency(self,
                                link: SemanticLink,
                                source: Dict[str, Any],
                                target: Dict[str, Any]) -> float:
        """Check consistency score for a link."""
        score = 1.0
        source_data = source.get("data", {})
        target_data = target.get("data", {})
        # Check semantic anchor consistency
        source_anchor = source_data.get("semanticanchor")
        target_data.get("semanticanchor")
        if link.link_type == LinkType.DEPENDS_ON:
            # Source should depend on target
            if source_anchor == target_data.get("id"):
                score += 0.2
            else:
                score -= 0.2
        # Check temporal consistency (creation dates)
        source_created = source_data.get("createdat")
        target_created = target_data.get("createdat")
        if source_created and target_created:
            if source_created >= target_created:
                score += 0.1
            else:
                score -= 0.1
        # Check layer consistency
        source_layer = source.get("layer")
        target_layer = target.get("layer")
        expected_relationships = {
            ("strategic", "operational"),
            ("operational", "execution"),
            ("execution", "observability"),
            ("observability", "advanced")
        }
        if (source_layer, target_layer) in expected_relationships:
            score += 0.3
        return max(0.0, min(1.0, score))
    def generate_integration_matrix(self) -> IntegrationMatrix:
        """Generate integration matrix showing all relationships."""
        logger.info("Generating integration matrix...")
        # Extract unique layers
        layers = set()
        for artifact_id, artifact_info in self.artifacts.items():
            layers.add(artifact_info.get("layer", "unknown"))
        layers = sorted(list(layers))
        # Build connections matrix
        connections = {}
        for link in self.links:
            source = self.artifacts.get(link.source_id)
            target = self.artifacts.get(link.target_id)
            if not source or not target:
                continue
            source_layer = source.get("layer", "unknown")
            target_layer = target.get("layer", "unknown")
            key = (source_layer, target_layer)
            if key not in connections:
                connections[key] = []
            connections[key].append(link)
        # Calculate coverage
        possible_connections = len(layers) * len(layers)
        actual_connections = len(connections)
        coverage = (actual_connections / possible_connections * 100) if possible_connections > 0 else 0
        matrix = IntegrationMatrix(
            layers=layers,
            connections=connections,
            total_links=len(self.links),
            coverage_percentage=coverage
        )
        logger.info(f"Integration matrix generated: {len(self.links)} links, {coverage:.1f}% coverage")
        return matrix
    def generate_cross_references(self) -> Dict[str, List[str]]:
        """Generate cross-reference mapping for all artifacts."""
        logger.info("Generating cross-references...")
        cross_refs = {}
        for artifact_id in self.artifacts:
            # Find all incoming links
            incoming = [link.source_id for link in self.links if link.target_id == artifact_id]
            # Find all outgoing links
            outgoing = [link.target_id for link in self.links if link.source_id == artifact_id]
            cross_refs[artifact_id] = {
                "incoming": incoming,
                "outgoing": outgoing,
                "total": len(incoming) + len(outgoing)
            }
        logger.info(f"Generated cross-references for {len(cross_refs)} artifacts")
        return cross_refs
    def export_integration_data(self) -> Dict[str, Any]:
        """Export integration data for external systems."""
        return {
            "generated_at": datetime.now().isoformat(),
            "artifacts_count": len(self.artifacts),
            "links_count": len(self.links),
            "artifacts": {
                artifact_id: {
                    "name": info["name"],
                    "layer": info["layer"],
                    "file_path": info["file_path"]
                }
                for artifact_id, info in self.artifacts.items()
            },
            "links": [
                {
                    "source": link.source_id,
                    "target": link.target_id,
                    "type": link.link_type.value,
                    "description": link.description,
                    "strength": link.strength
                }
                for link in self.links
            ],
            "integration_matrix": self.generate_integration_matrix().__dict__,
            "consistency": self.verify_consistency(),
            "cross_references": self.generate_cross_references()
        }
if __name__ == "__main__":
    # Demo: Create and run integrator
    integrator = GLIntegrator()
    print("GL Integrator Initialized")
    print("=" * 50)
    # Load artifacts
    integrator.load_artifacts(Path.cwd())
    print(f"\nLoaded {len(integrator.artifacts)} artifacts")
    # Auto-discover links
    links = integrator.auto_discover_links()
    print(f"Discovered {len(links)} semantic links")
    # Verify consistency
    consistency = integrator.verify_consistency()
    print(f"\nConsistency Score: {consistency['consistency_score']:.1f}%")
    print(f"Issues Found: {consistency['issues_found']}")
    # Generate integration matrix
    matrix = integrator.generate_integration_matrix()
    print(f"\nIntegration Coverage: {matrix.coverage_percentage:.1f}%")
    print(f"Layers: {', '.join(matrix.layers)}")
    # Export data
    export_data = integrator.export_integration_data()
    print(f"\nExport data ready: {len(export_data['links'])} links")