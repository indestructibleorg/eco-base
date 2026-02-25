#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: code_deduplicator
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
Machine-Native Code Deduplicator
Identifies and reports duplicate code patterns for refactoring.
"""
# MNGA-002: Import organization needs review
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
class CodeDeduplicator:
    """Identifies duplicate code patterns in Python files."""
    def __init__(self, min_lines: int = 5):
        self.min_lines = min_lines
        self.function_hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.class_hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.block_hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    def analyze_file(self, file_path: str) -> None:
        """Analyze a Python file for duplicate patterns."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            self._extract_functions(tree, file_path)
            self._extract_classes(tree, file_path)
        except (SyntaxError, UnicodeDecodeError):
            pass
    def _extract_functions(self, tree: ast.AST, file_path: str) -> None:
        """Extract function signatures and bodies."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Create normalized signature
                args = [arg.arg for arg in node.args.args]
                body_str = ast.dump(node)
                body_hash = hashlib.md5(body_str.encode(), usedforsecurity=False).hexdigest()[:12]
                self.function_hashes[body_hash].append({
                    "file": file_path,
                    "name": node.name,
                    "line": node.lineno,
                    "args": args,
                    "lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0
                })
    def _extract_classes(self, tree: ast.AST, file_path: str) -> None:
        """Extract class definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                class_str = ast.dump(node)
                class_hash = hashlib.md5(class_str.encode(), usedforsecurity=False).hexdigest()[:12]
                self.class_hashes[class_hash].append({
                    "file": file_path,
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods
                })
    def analyze_directory(self, dir_path: str) -> None:
        """Analyze all Python files in a directory."""
        path = Path(dir_path)
        for file_path in path.rglob("*.py"):
            if ".git" in str(file_path) or "archive" in str(file_path):
                continue
            self.analyze_file(str(file_path))
    def get_duplicates(self) -> Dict[str, Any]:
        """Get all identified duplicates."""
        duplicates = {
            "functions": [],
            "classes": [],
            "summary": {
                "duplicate_functions": 0,
                "duplicate_classes": 0,
                "total_duplicate_lines": 0
            }
        }
        # Find duplicate functions
        for hash_val, occurrences in self.function_hashes.items():
            if len(occurrences) > 1:
                total_lines = sum(o.get("lines", 0) for o in occurrences)
                duplicates["functions"].append({
                    "hash": hash_val,
                    "count": len(occurrences),
                    "total_lines": total_lines,
                    "occurrences": occurrences
                })
                duplicates["summary"]["duplicate_functions"] += len(occurrences) - 1
                duplicates["summary"]["total_duplicate_lines"] += total_lines - (total_lines // len(occurrences))
        # Find duplicate classes
        for hash_val, occurrences in self.class_hashes.items():
            if len(occurrences) > 1:
                duplicates["classes"].append({
                    "hash": hash_val,
                    "count": len(occurrences),
                    "occurrences": occurrences
                })
                duplicates["summary"]["duplicate_classes"] += len(occurrences) - 1
        # Sort by count
        duplicates["functions"].sort(key=lambda x: x["count"], reverse=True)
        duplicates["classes"].sort(key=lambda x: x["count"], reverse=True)
        return duplicates
    def generate_report(self) -> str:
        """Generate a human-readable report."""
        duplicates = self.get_duplicates()
        lines = []
        lines.append("=" * 60)
        lines.append("CODE DUPLICATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Duplicate functions: {duplicates['summary']['duplicate_functions']}")
        lines.append(f"Duplicate classes: {duplicates['summary']['duplicate_classes']}")
        lines.append(f"Estimated duplicate lines: {duplicates['summary']['total_duplicate_lines']}")
        lines.append("")
        if duplicates["functions"]:
            lines.append("-" * 40)
            lines.append("DUPLICATE FUNCTIONS")
            lines.append("-" * 40)
            for dup in duplicates["functions"][:10]:  # Top 10
                lines.append(f"\n[{dup['count']} occurrences, ~{dup['total_lines']} lines]")
                for occ in dup["occurrences"]:
                    lines.append(f"  - {occ['file']}:{occ['line']} {occ['name']}({', '.join(occ['args'])})")
        if duplicates["classes"]:
            lines.append("")
            lines.append("-" * 40)
            lines.append("DUPLICATE CLASSES")
            lines.append("-" * 40)
            for dup in duplicates["classes"][:10]:  # Top 10
                lines.append(f"\n[{dup['count']} occurrences]")
                for occ in dup["occurrences"]:
                    lines.append(f"  - {occ['file']}:{occ['line']} {occ['name']}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
def main():
    deduplicator = CodeDeduplicator()
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    deduplicator.analyze_directory(target)
    # Print report
    print(deduplicator.generate_report())
    # Export JSON
    duplicates = deduplicator.get_duplicates()
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "code_duplicates.json", "w") as f:
        json.dump(duplicates, f, indent=2)
    print("\nDetailed report saved to: reports/code_duplicates.json")
if __name__ == "__main__":
    main()