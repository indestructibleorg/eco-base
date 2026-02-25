# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: project_to_superroot
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Project Knowledge Graph entities to SuperRoot format.
This script transforms the repository knowledge graph into SuperRoot-compatible
entity definitions that can be consumed by SuperRoot gl-platform.governance systems.
Usage:
  python tools/docs/project_to_superroot.py \
    --kg docs/knowledge-graph.yaml \
    --output docs/superroot-entities.yaml
  python tools/docs/project_to_superroot.py --help
"""
import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
# Try to import yaml, provide helpful error if not available
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)
# Node type mapping from KG to SuperRoot entity types
# KG uses lowercase types per knowledge-graph.schema.json:
#   system, subsystem, component, module, file, directory, config, document, workflow, schema, capability, feature, tag
# SuperRoot uses PascalCase types
KG_TO_SUPERROOT_TYPE = {
    "system": "System",
    "subsystem": "Subsystem",
    "component": "Component",
    "module": "Component",  # Modules become components in SuperRoot
    "directory": "Directory",
    "config": "ConfigParam",
    "document": "Document",
    "capability": "Capability",
    "feature": "Feature",
    "workflow": "Workflow",
    "file": "File",
    "schema": "Schema",
    "tag": "Tag",
}
# Edge type mapping from KG to SuperRoot relationship types
KG_TO_SUPERROOT_EDGE = {
    "contains": "CONTAINS",
    "provides": "PROVIDES",
    "configures": "CONFIGURES",
    "documents": "DOCUMENTED_BY",
    "depends_on": "DEPENDS_ON",
    "implements": "IMPLEMENTS",
}
def load_knowledge_graph(kg_path: Path) -> dict[str, Any]:
    """Load and validate the knowledge graph YAML file."""
    if not kg_path.exists():
        print(f"Error: Knowledge graph file not found: {kg_path}")
        sys.exit(1)
    with open(kg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        print(f"Error: Empty knowledge graph file: {kg_path}")
        sys.exit(1)
    return data
def project_node(node: dict[str, Any]) -> dict[str, Any]:
    """Project a KG node to SuperRoot entity format."""
    kg_type = node.get("type", "Unknown")
    superroot_type = KG_TO_SUPERROOT_TYPE.get(kg_type, kg_type)
    entity = {
        "id": node.get("id", "unknown"),
        "type": superroot_type,
        "label": node.get("label", node.get("id", "unknown")),
    }
    # Add optional fields
    if "path" in node:
        entity["path"] = node["path"]
    if "description" in node:
        entity["description"] = node["description"]
    if "tags" in node:
        entity["tags"] = node["tags"]
    # Project properties
    if "properties" in node:
        props = node["properties"]
        entity["meta"] = {
            "source": "knowledge-graph",
        }
        if "label_zh" in props:
            entity["meta"]["label_zh"] = props["label_zh"]
    return entity
def project_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Project a KG edge to SuperRoot relationship format."""
    kg_type = edge.get("type", "unknown")
    superroot_type = KG_TO_SUPERROOT_EDGE.get(kg_type, kg_type.upper())
    relationship = {
        "source": edge.get("source", "unknown"),
        "target": edge.get("target", "unknown"),
        "type": superroot_type,
    }
    if "label" in edge:
        relationship["label"] = edge["label"]
    if "weight" in edge:
        relationship["weight"] = edge["weight"]
    return relationship
def generate_superroot_entities(kg_data: dict[str, Any]) -> dict[str, Any]:
    """Generate SuperRoot entity definitions from knowledge graph."""
    nodes = kg_data.get("nodes", [])
    edges = kg_data.get("edges", [])
    # Project entities
    entities = [project_node(node) for node in nodes]
    # Project relationships
    relationships = [project_edge(edge) for edge in edges]
    # Group entities by type for summary
    entities_by_type: dict[str, list] = {}
    for entity in entities:
        entity_type = entity["type"]
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    # Build output structure
    output = {
        "$schema": "https://schema.superroot.kn/entities/v1",
        "version": "1.0.0",
        "id": "unmanned-island-superroot-entities",
        "title": "Unmanned Island System - SuperRoot Entities",
        "description": "SuperRoot-compatible entity definitions projected from knowledge graph",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": {
            "knowledge_graph": kg_data.get("id", "unknown"),
            "generated_at": kg_data.get("generated_at"),
        },
        "entities": entities,
        "relationships": relationships,
        "summary": {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entities_by_type": {
                entity_type: len(entity_list)
                for entity_type, entity_list in sorted(entities_by_type.items())
            },
        },
        "meta": {
            "generator": "project_to_superroot.py",
            "version": "1.0.0",
        },
    }
    return output
def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Project Knowledge Graph entities to SuperRoot format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_to_superroot.py --kg docs/knowledge-graph.yaml --output docs/superroot-entities.yaml
  python project_to_superroot.py --kg docs/knowledge-graph.yaml --output - --verbose
        """,
    )
    parser.add_argument(
        "--kg",
        type=str,
        required=True,
        help="Path to knowledge graph YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for SuperRoot entities YAML (use '-' for stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()
    kg_path = Path(args.kg)
    if args.verbose:
        print(f"Loading knowledge graph: {kg_path}")
    # Load knowledge graph
    kg_data = load_knowledge_graph(kg_path)
    if args.verbose:
        nodes_count = len(kg_data.get("nodes", []))
        edges_count = len(kg_data.get("edges", []))
        print(f"  - Nodes: {nodes_count}")
        print(f"  - Edges: {edges_count}")
    # Generate SuperRoot entities
    superroot_data = generate_superroot_entities(kg_data)
    # Output
    yaml_output = yaml.dump(
        superroot_data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )
    if args.output == "-":
        print(yaml_output)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_output, encoding="utf-8")
        if args.verbose:
            print(f"✅ Generated: {output_path}")
            print(f"   - Entities: {superroot_data['summary']['total_entities']}")
            print(
                f"   - Relationships: {superroot_data['summary']['total_relationships']}"
            )
            print(
                f"   - Entity types: {', '.join(superroot_data['summary']['entities_by_type'].keys())}"
            )
if __name__ == "__main__":
    main()
