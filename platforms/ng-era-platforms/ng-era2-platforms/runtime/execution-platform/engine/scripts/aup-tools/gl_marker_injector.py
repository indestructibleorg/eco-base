#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: gl-marker-injector
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
GL Marker Injector - 自動注入 GL 治理標記
GL Unified Architecture Governance Framework Activated
"""
# MNGA-002: Import organization needs review
import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
class GLMarkerInjector:
    """GL 標記注入器"""
    # GL 層級映射
    GL_LAYER_MAP = {
        'docs': 'ECO-90-META',
        'src': 'ECO-30-EXEC',
        'config': 'ECO-10-POLICY',
        'schemas': 'ECO-20-SCHEMA',
        'tests': 'ECO-80-TEST',
        'scripts': 'ECO-70-TOOL',
        'gl-platform.gl-platform.governance': 'ECO-10-POLICY',
        'security': 'ECO-15-SECURITY',
        'namespaces-adk': 'ECO-00-NAMESPACE',
        'namespaces-mcp': 'ECO-00-NAMESPACE',
        'namespaces-sdk': 'ECO-00-NAMESPACE',
        'gl-platform.gl-platform.governance_layer': 'ECO-10-POLICY',
        'schema_system': 'ECO-20-SCHEMA',
        'security_layer': 'ECO-15-SECURITY',
        'taxonomy-core': 'ECO-25-TAXONOMY',
    }
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.modified_files = []
        self.skipped_files = []
        self.errors = []
    def determine_gl_layer(self, file_path: Path) -> str:
        """確定檔案的 GL 層級"""
        relative_path = file_path.relative_to(self.root_path)
        parts = relative_path.parts
        # 檢查路徑中的目錄
        for part in parts:
            if part in self.GL_LAYER_MAP:
                return self.GL_LAYER_MAP[part]
        # 根據檔案類型判斷
        suffix = file_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return 'ECO-10-POLICY'
        elif suffix == '.json':
            return 'ECO-20-SCHEMA'
        elif suffix in ['.ts', '.js', '.py']:
            return 'ECO-30-EXEC'
        elif suffix == '.md':
            return 'ECO-90-META'
        elif suffix == '.sh':
            return 'ECO-70-TOOL'
        return 'ECO-00-NAMESPACE'
    def generate_semantic_anchor(self, file_path: Path, gl_layer: str) -> str:
        """生成語意錨定 ID"""
        relative_path = file_path.relative_to(self.root_path)
        # 將路徑轉換為錨定 ID
        anchor_parts = []
        for part in relative_path.parts[:-1]:  # 排除檔名
            clean_part = re.sub(r'[^a-zA-Z0-9]', '', part.upper())[:8]
            if clean_part:
                anchor_parts.append(clean_part)
        # 添加檔名
        filename = file_path.stem.upper().replace('-', '').replace('_', '')[:12]
        anchor_parts.append(filename)
        layer_code = gl_layer.split('-')[1] if '-' in gl_layer else '00'
        return f"ECO-{layer_code}-{'_'.join(anchor_parts[-3:])}"
    def generate_module_path(self, file_path: Path) -> str:
        """生成模組路徑"""
        relative_path = file_path.relative_to(self.root_path)
        return f"ns-root/{'/'.join(relative_path.parts[:-1])}"
    def inject_yaml_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 YAML 檔案的 GL 標記"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        # 檢查是否已有標記
        if '@gl-layer' in content or 'gl_layer:' in content:
            return content, False
        # 生成標記區塊
        marker_block = f"""# GL Governance Markers
# @gl-layer {gl_layer}
# @gl-module {module_path}
# @gl-semantic-anchor {semantic_anchor}
# @gl-evidence-required false
# GL Unified Architecture Governance Framework Activated
"""
        # 如果檔案以 --- 開頭（YAML frontmatter），在其後插入
        if content.startswith('---'):
            # 找到第二個 ---
            second_dash = content.find('---', 3)
            if second_dash != -1:
                return content[:second_dash+3] + '\n' + marker_block + content[second_dash+3:], True
        return marker_block + content, True
    def inject_markdown_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 Markdown 檔案的 GL 標記"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        # 檢查是否已有標記
        if '@gl-layer' in content:
            return content, False
        # 生成標記區塊
        marker_block = f"""<!--
GL Governance Markers
@gl-layer {gl_layer}
@gl-module {module_path}
@gl-semantic-anchor {semantic_anchor}
@gl-evidence-required false
GL Unified Architecture Governance Framework Activated
-->
"""
        # 如果檔案以 --- 開頭（frontmatter），在其後插入
        if content.startswith('---'):
            second_dash = content.find('---', 3)
            if second_dash != -1:
                return content[:second_dash+4] + '\n' + marker_block + content[second_dash+4:], True
        return marker_block + content, True
    def inject_python_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 Python 檔案的 GL 標記"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        # 檢查是否已有標記
        if '@gl-layer' in content or '@gl_layer' in content:
            return content, False
        # 生成標記區塊
        marker_block = f'''"""
GL Governance Markers
@gl-layer {gl_layer}
@gl-module {module_path}
@gl-semantic-anchor {semantic_anchor}
@gl-evidence-required false
GL Unified Architecture Governance Framework Activated
"""
'''
        # 如果檔案以 shebang 開頭，在其後插入
        if content.startswith('#!'):
            newline_pos = content.find('\n')
            if newline_pos != -1:
                return content[:newline_pos+1] + marker_block + content[newline_pos+1:], True
        # 如果檔案以 docstring 開頭，替換或追加
        if content.startswith('"""') or content.startswith("'''"):
            # 找到 docstring 結束位置
            quote = content[:3]
            end_pos = content.find(quote, 3)
            if end_pos != -1:
                # 在現有 docstring 中添加標記
                existing_doc = content[3:end_pos]
                new_doc = f'''"""
{existing_doc.strip()}
GL Governance Markers
@gl-layer {gl_layer}
@gl-module {module_path}
@gl-semantic-anchor {semantic_anchor}
@gl-evidence-required false
GL Unified Architecture Governance Framework Activated
"""'''
                return new_doc + content[end_pos+3:], True
        return marker_block + content, True
    def inject_typescript_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 TypeScript 檔案的 GL 標記"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        # 檢查是否已有標記
        if '@gl-layer' in content:
            return content, False
        # 生成標記區塊
        marker_block = f"""/**
# GL Governance Markers
# @gl-layer {gl_layer}
# @gl-module {module_path}
# @gl-semantic-anchor {semantic_anchor}
# @gl-evidence-required false
"""
        return marker_block + content, True
    def inject_json_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 JSON 檔案的 GL 標記（通過添加 _gl_metadata 欄位）"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        try:
            data = json.loads(content)
            # 檢查是否已有標記
            if '_gl_metadata' in data or 'gl_layer' in data:
                return content, False
            if isinstance(data, dict):
                # 添加 GL metadata
                gl_metadata = {
                    "_gl_metadata": {
                        "gl_layer": gl_layer,
                        "gl_module": module_path,
                        "gl_semantic_anchor": semantic_anchor,
                        "gl_evidence_required": False,
                        "gl_charter": "GL Unified Architecture Governance Framework Activated"
                    }
                }
                # 將 GL metadata 放在最前面
                new_data = {**gl_metadata, **data}
                return json.dumps(new_data, indent=2, ensure_ascii=False), True
            return content, False
        except json.JSONDecodeError:
            return content, False
    def inject_shell_markers(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """注入 Shell 檔案的 GL 標記"""
        gl_layer = self.determine_gl_layer(file_path)
        semantic_anchor = self.generate_semantic_anchor(file_path, gl_layer)
        module_path = self.generate_module_path(file_path)
        # 檢查是否已有標記
        if '@gl-layer' in content:
            return content, False
        # 生成標記區塊
        marker_block = f"""# GL Governance Markers
# @gl-layer {gl_layer}
# @gl-module {module_path}
# @gl-semantic-anchor {semantic_anchor}
# @gl-evidence-required false
# GL Unified Architecture Governance Framework Activated
"""
        # 如果檔案以 shebang 開頭，在其後插入
        if content.startswith('#!'):
            newline_pos = content.find('\n')
            if newline_pos != -1:
                return content[:newline_pos+1] + '\n' + marker_block + content[newline_pos+1:], True
        return marker_block + content, True
    def inject_markers(self, file_path: Path) -> bool:
        """為單一檔案注入 GL 標記"""
        try:
            # 讀取檔案
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            suffix = file_path.suffix.lower()
            new_content = content
            modified = False
            # 根據檔案類型選擇注入方法
            if suffix in ['.yaml', '.yml']:
                new_content, modified = self.inject_yaml_markers(content, file_path)
            elif suffix == '.md':
                new_content, modified = self.inject_markdown_markers(content, file_path)
            elif suffix == '.py':
                new_content, modified = self.inject_python_markers(content, file_path)
            elif suffix in ['.ts', '.js']:
                new_content, modified = self.inject_typescript_markers(content, file_path)
            elif suffix == '.json':
                new_content, modified = self.inject_json_markers(content, file_path)
            elif suffix == '.sh':
                new_content, modified = self.inject_shell_markers(content, file_path)
            else:
                self.skipped_files.append(str(file_path))
                return False
            if modified:
                # 寫入檔案
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.modified_files.append(str(file_path))
                return True
            else:
                self.skipped_files.append(str(file_path))
                return False
        except Exception as e:
            self.errors.append({'file': str(file_path), 'error': str(e)})
            return False
    def run(self) -> Dict:
        """執行 GL 標記注入"""
        print(f"\n{'='*60}")
        print("GL Marker Injector")
        print("GL Unified Architecture Governance Framework Activated")
        print(f"{'='*60}\n")
        # 掃描檔案
        files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file():
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if 'node_modules' in file_path.parts:
                    continue
                if '__pycache__' in file_path.parts:
                    continue
                files.append(file_path)
        print(f"Found {len(files)} files to process\n")
        # 處理檔案
        for i, file_path in enumerate(files, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(files)}")
            self.inject_markers(file_path)
        # 生成報告
        report = {
            'execution_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_files': len(files),
            'modified_files': len(self.modified_files),
            'skipped_files': len(self.skipped_files),
            'errors': len(self.errors),
            'modified_list': self.modified_files,
            'error_list': self.errors
        }
        print(f"\n{'='*60}")
        print("INJECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files: {len(files)}")
        print(f"Modified: {len(self.modified_files)}")
        print(f"Skipped: {len(self.skipped_files)}")
        print(f"Errors: {len(self.errors)}")
        return report
def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/machine-native-ops/ns-root'
    injector = GLMarkerInjector(root_path)
    report = injector.run()
    # 保存報告
    report_path = Path(root_path).parent / 'aep-gl-platform.gl-platform.governance-audit' / 'reports' / 'ECO-MARKER-INJECTION-REPORT.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    return 0
if __name__ == '__main__':
    sys.exit(main())