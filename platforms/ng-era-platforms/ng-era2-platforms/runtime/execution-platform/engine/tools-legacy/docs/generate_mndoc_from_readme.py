# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: generate_mndoc_from_readme
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Generate machine-native documentation (mndoc.yaml) from README.md.
This script extracts structured information from README.md and generates
a machine-consumable YAML file following the MN-DOC schema.
Usage:
  python tools/docs/generate_mndoc_from_readme.py \
    --readme README.md \
    --output docs/unmanned-island.mndoc.yaml
  python tools/docs/generate_mndoc_from_readme.py --help
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Any
# Try to import yaml, provide helpful error if not available
try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)
# Regular expressions for parsing README.md
TITLE_RE = re.compile(r"^#\s+(.*)", re.MULTILINE)
VERSION_BADGE_RE = re.compile(r"version-([\d\.]+)-")
SECTION_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
SECTION_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
# Subsystem detection keywords
SUBSYSTEM_KEYWORDS = {
    "machinenativenops": ["SynergyMesh", "Core Engine", "核心引擎"],
    "gl-platform.governance": ["Governance", "治理", "Schema"],
    "autonomous": ["Autonomous", "自主", "五骨架", "Framework"],
}
def extract_title(text: str) -> str:
    """Extract the main title from README."""
    match = TITLE_RE.search(text)
    if match:
        # Remove emoji and clean up
        title = match.group(1).strip()
        title = re.sub(r"[🏝️🚀✨]+", "", title).strip()
        return title
    return "Unmanned Island System"
def extract_version(text: str) -> str:
    """Extract version from badge in README."""
    match = VERSION_BADGE_RE.search(text)
    return match.group(1) if match else "1.0.0"
# Canonical tag mapping: maps various forms (Chinese, English, typos) to
# machine-friendly keys
TAG_CANON = {
    # AI variants
    "人工智慧": "ai",
    "人工智能": "ai",
    "AI": "ai",
    "決策引擎": "ai",
    # Automation variants
    "自動化": "automation",
    "automation": "automation",
    "自動": "automation",
    # Autonomous system variants
    "自主系統": "autonomous-system",
    "autonomous": "autonomous-system",
    "無人機": "autonomous-system",
    "自駕": "autonomous-system",
    # Cloud-native variants
    "雲原生": "cloud-native",
    "雲端原生": "cloud-native",
    "cloud-native": "cloud-native",
    "cloud native": "cloud-native",
    # Governance variants
    "治理": "gl-platform.governance",
    "gl-platform.governance": "gl-platform.governance",
    # Kubernetes variants
    "Kubernetes": "kubernetes",
    "K8s": "kubernetes",
    "k8s": "kubernetes",
    # MCP
    "MCP": "mcp",
    # Sigstore variants
    "Sigstore": "sigstore",
    "sigstore": "sigstore",
    "Cosign": "sigstore",
    "cosign": "sigstore",
    # SLSA variants (including typo fix)
    "SLSA": "slsa",
    "slsa": "slsa",
    "SLSSA": "slsa",  # Fix common typo
    # CI/CD variants
    "持續整合/持續交付": "ci-cd",
    "CI/CD": "ci-cd",
    "ci-cd": "ci-cd",
    "workflow": "ci-cd",
}
def extract_tags(text: str) -> list[str]:
    """Extract tags based on content keywords using canonical mapping."""
    tags: set[str] = set()
    for key, canon in TAG_CANON.items():
        if key in text:
            tags.add(canon)
    return sorted(tags)
def extract_languages(text: str) -> list[str]:
    """Detect languages used in the document."""
    languages = []
    # Check for Traditional Chinese
    if re.search(r"[\u4e00-\u9fff]", text):
        languages.append("zh-Hant")
    # Check for English
    if re.search(r"[a-zA-Z]{5,}", text):
        languages.append("en")
    return languages
def split_sections(text: str) -> list[dict[str, Any]]:
    """Split README into sections based on H2 headers."""
    sections = []
    # Find all H2 sections
    h2_matches = list(SECTION_H2_RE.finditer(text))
    for i, match in enumerate(h2_matches):
        start = match.start()
        end = h2_matches[i + 1].start() if i + 1 < len(h2_matches) else len(text)
        section_title = match.group(1).strip()
        section_content = text[start:end].strip()
        # Clean up title (remove emoji)
        clean_title = re.sub(r"[🌟🔷📁🚀🛠️🤝📚🔄📄🙏]+", "", section_title).strip()
        # Generate section ID from title
        section_id = re.sub(r"[^\w\s-]", "", clean_title.lower())
        section_id = re.sub(r"[\s]+", "-", section_id).strip("-")
        sections.append(
            {
                "id": section_id or f"section-{i}",
                "title": clean_title,
                "type": classify_section(clean_title),
                "content_preview": (
                    section_content[:200] + "..."
                    if len(section_content) > 200
                    else section_content
                ),
            }
        )
    return sections
def classify_section(title: str) -> str:
    """Classify section type based on title keywords."""
    title_lower = title.lower()
    if any(k in title_lower for k in ["概述", "overview", "系統"]):
        return "overview"
    if any(k in title_lower for k in ["架構", "architecture", "子系統", "subsystem"]):
        return "architecture"
    if any(k in title_lower for k in ["目錄", "directory", "結構", "structure"]):
        return "structure"
    if any(k in title_lower for k in ["快速", "quick", "開始", "start", "安裝"]):
        return "getting-started"
    if any(k in title_lower for k in ["功能", "feature", "核心"]):
        return "features"
    if any(k in title_lower for k in ["命令", "command", "互動"]):
        return "commands"
    if any(k in title_lower for k in ["文檔", "doc", "導航"]):
        return "documentation"
    if any(k in title_lower for k in ["ci", "cd", "workflow"]):
        return "cicd"
    if any(k in title_lower for k in ["授權", "license"]):
        return "license"
    if any(k in title_lower for k in ["致謝", "acknowledgement", "thanks"]):
        return "acknowledgements"
    return "content"
def detect_subsystems(text: str) -> list[dict[str, Any]]:
    """Detect and extract subsystem information."""
    subsystems = []
    for sys_id, keywords in SUBSYSTEM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                subsystem = {
                    "id": sys_id,
                    "detected": True,
                    "keywords_found": [k for k in keywords if k in text],
                }
                subsystems.append(subsystem)
                break
    return subsystems
def extract_summary(text: str) -> str:
    """Extract a summary from the README."""
    # Try to find content after title and badges but before first H2
    match = re.search(r"</div>\s*---\s*(.*?)(?=^##|\Z)", text, re.MULTILINE | re.DOTALL)
    if match:
        summary = match.group(1).strip()
        # Clean up
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return summary
    return "Enterprise-grade cloud-native intelligent automation platform."
def generate_mndoc(readme_path: Path, source_path: str = "README.md") -> dict[str, Any]:
    """Generate MN-DOC structure from README.md content."""
    text = readme_path.read_text(encoding="utf-8")
    mndoc = {
        "$schema": "https://schema.superroot.kn/mndoc/v1",
        "id": "unmanned-island",
        "kind": "SystemDoc",
        "path": source_path,
        "title": extract_title(text),
        "version": extract_version(text),
        "domain": "unmanned-island",
        "layer": "platform",
        "type": "overview",
        "owner": "unmanned-island",
        "status": "active",
        "languages": extract_languages(text),
        "tags": extract_tags(text),
        "summary": extract_summary(text),
        "sections": split_sections(text),
        "detected_subsystems": detect_subsystems(text),
        "meta": {
            "generated_by": "generate_mndoc_from_readme.py",
            "source": source_path,
        },
    }
    return mndoc
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate machine-native documentation (mndoc.yaml) from README.md"
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md file (default: README.md)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/unmanned-island.mndoc.yaml"),
        help="Output path for mndoc.yaml (default: docs/unmanned-island.mndoc.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output to stdout instead of writing to file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    args = parser.parse_args()
    if not args.readme.exists():
        print(f"Error: README file not found: {args.readme}")
        sys.exit(1)
    if args.verbose:
        print(f"Reading: {args.readme}")
    # Generate MN-DOC
    mndoc = generate_mndoc(args.readme, str(args.readme))
    # Output
    yaml_output = yaml.dump(
        mndoc,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=100,
    )
    if args.dry_run:
        print(yaml_output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_output, encoding="utf-8")
        print(f"✅ Generated: {args.output}")
        if args.verbose:
            print(f"   - Title: {mndoc['title']}")
            print(f"   - Version: {mndoc['version']}")
            print(f"   - Sections: {len(mndoc['sections'])}")
            print(f"   - Tags: {', '.join(mndoc['tags'])}")
if __name__ == "__main__":
    main()
