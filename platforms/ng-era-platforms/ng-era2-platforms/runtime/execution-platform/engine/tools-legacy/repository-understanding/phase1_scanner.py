# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: phase1_scanner
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
第一階段：儲存庫掃描和知識庫建立
"""
# MNGA-002: Import organization needs review
import os
import json
import yaml
from pathlib import Path
from collections import defaultdict
from datetime import datetime
class RepositoryScanner:
    def __init__(self, root_path=None):
        # Default to repository root (3 levels up from this script's location)
        if root_path is None:
            script_dir = Path(__file__).parent.absolute()
            root_path = script_dir.parent.parent.parent  # workspace/tools/repository-understanding -> workspace -> root
        self.root_path = Path(root_path)
        self.knowledge_base = {
            'metadata': {
                'scan_date': datetime.now().isoformat(),
                'phase': 'Phase 1 - Initial Scan',
                'scanner_version': '1.0.0'
            },
            'directories': {},
            'files': {},
            'relationships': {},
            'statistics': {},
            'critical_files': [],
            'configurations': {}
        }
    def scan(self):
        """執行完整掃描"""
        print("🔍 開始掃描儲存庫...")
        print(f"📍 根目錄: {self.root_path.absolute()}")
        # 檢查根目錄是否存在
        if not self.root_path.exists():
            print(f"❌ 錯誤：根目錄不存在: {self.root_path}")
            return False
        # 掃描目錄結構
        self.scan_directories()
        # 掃描檔案
        self.scan_files()
        # 分析配置檔案
        self.analyze_configurations()
        # 建立關係圖
        self.build_relationships()
        # 生成統計
        self.generate_statistics()
        # 識別關鍵檔案
        self.identify_critical_files()
        print("✅ 掃描完成！")
        return True
    def scan_directories(self):
        """掃描目錄結構"""
        print("📁 掃描目錄結構...")
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            # 跳過隱藏目錄和某些系統目錄
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            if root_path != self.root_path:
                rel_path = root_path.relative_to(self.root_path)
                dir_info = {
                    'path': str(rel_path),
                    'absolute_path': str(root_path),
                    'parent': str(rel_path.parent) if rel_path.parent != Path('.') else 'root',
                    'subdirectories': dirs.copy(),
                    'file_count': len(files),
                    'purpose': self.classify_directory_purpose(root_path, dirs, files),
                    'depth': len(rel_path.parts)
                }
                self.knowledge_base['directories'][str(rel_path)] = dir_info
        print(f"✅ 掃描完成: {len(self.knowledge_base['directories'])} 個目錄")
    def classify_directory_purpose(self, dir_path, subdirs, files):
        """分類目錄用途"""
        dir_name = dir_path.name.lower()
        path_parts = str(dir_path).lower().split(os.sep)
        # 基於目錄名稱分類
        purpose_map = {
            'src': 'source_code',
            'source': 'source_code',
            'lib': 'source_code',
            'library': 'source_code',
            'bin': 'executable',
            'sbin': 'system_executable',
            'etc': 'configuration',
            'config': 'configuration',
            'configuration': 'configuration',
            'docs': 'documentation',
            'doc': 'documentation',
            'documentation': 'documentation',
            'test': 'testing',
            'tests': 'testing',
            'spec': 'testing',
            'specs': 'testing',
            'scripts': 'scripts',
            'tools': 'tools',
            'utils': 'utilities',
            'workspace': 'workspace',
            'work': 'workspace',
            'var': 'variable_data',
            'tmp': 'temporary',
            'temp': 'temporary',
            'opt': 'optional_software',
            'srv': 'service_data',
            'home': 'user_home',
            'root': 'root_home'
        }
        if dir_name in purpose_map:
            return purpose_map[dir_name]
        # 基於路徑分類
        if 'namespace' in path_parts:
            return 'namespace_organization'
        elif 'controlplane' in path_parts:
            return 'control_plane'
        elif 'gl-platform.governance' in path_parts:
            return 'gl-platform.governance'
        elif 'github' in path_parts:
            return 'ci_cd'
        elif '.github' in path_parts:
            return 'ci_cd'
        # 基於內容推斷
        if any(f.endswith('.py') for f in files):
            return 'python_code'
        elif any(f.endswith('.js') or f.endswith('.ts') for f in files):
            return 'javascript_code'
        elif any(f.endswith('.yaml') or f.endswith('.json') for f in files):
            return 'configuration'
        elif any(f.endswith('.md') for f in files):
            return 'documentation'
        return 'unknown'
    def scan_files(self):
        """掃描檔案"""
        print("📄 掃描檔案...")
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            root_path = Path(root)
            for file in files:
                if file.startswith('.'):
                    continue
                file_path = root_path / file
                rel_path = file_path.relative_to(self.root_path)
                try:
                    file_info = {
                        'name': file,
                        'path': str(rel_path),
                        'directory': str(rel_path.parent),
                        'extension': file_path.suffix.lower(),
                        'size': file_path.stat().st_size,
                        'type': self.classify_file_type(file_path.suffix),
                        'is_executable': os.access(file_path, os.X_OK),
                        'is_critical': self.is_critical_file(file_path)
                    }
                    self.knowledge_base['files'][str(rel_path)] = file_info
                    if file_info['is_critical']:
                        self.knowledge_base['critical_files'].append(str(rel_path))
                except Exception as e:
                    print(f"⚠️  警告：無法讀取檔案 {file_path}: {e}")
        print(f"✅ 掃描完成: {len(self.knowledge_base['files'])} 個檔案")
    def classify_file_type(self, extension):
        """分類檔案類型"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'react',
            '.tsx': 'react',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.env': 'environment',
            '.gitignore': 'git',
            '.dockerignore': 'docker',
            'dockerfile': 'docker',
            'makefile': 'make',
            '.cfg': 'config',
            '.conf': 'config',
            '.ini': 'config',
            '.lock': 'lock',
            '.sum': 'checksum',
            '.pem': 'certificate',
            '.key': 'key',
            '.crt': 'certificate'
        }
        return ext_map.get(extension.lower(), 'unknown')
    def is_critical_file(self, file_path):
        """判斷是否為關鍵檔案"""
        critical_patterns = [
            'makefile',
            'dockerfile',
            '.env',
            'secrets',
            'config',
            'bootstrap',
            'main',
            'init',
            'setup',
            'requirements',
            'package.json',
            'tsconfig',
            'pyproject'
        ]
        file_name = file_path.name.lower()
        return any(pattern in file_name for pattern in critical_patterns)
    def analyze_configurations(self):
        """分析配置檔案"""
        print("⚙️  分析配置檔案...")
        for file_path, file_info in self.knowledge_base['files'].items():
            if file_info['type'] in ['yaml', 'json', 'toml', 'config']:
                try:
                    full_path = self.root_path / file_path
                    if full_path.exists() and full_path.stat().st_size < 100000:  # 只分析小於100KB的檔案
                        config_data = self.parse_config_file(full_path, file_info['type'])
                        if config_data:
                            self.knowledge_base['configurations'][file_path] = config_data
                except Exception as e:
                    print(f"⚠️  警告：無法解析配置檔案 {file_path}: {e}")
        print(f"✅ 分析完成: {len(self.knowledge_base['configurations'])} 個配置檔案")
    def parse_config_file(self, file_path, file_type):
        """解析配置檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_type in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                elif file_type == 'json':
                    return json.load(f)
                else:
                    # 對於其他類型，只記錄基本資訊
                    return {
                        'type': file_type,
                        'size': file_path.stat().st_size,
                        'note': 'Content not parsed for safety'
                    }
        except Exception:
            return {
                'type': file_type,
                'error': 'Parse error',
                'note': 'File exists but could not be parsed'
            }
    def build_relationships(self):
        """建立檔案關係圖"""
        print("🔗 建立檔案關係圖...")
        # 建立目錄到檔案的關係
        for dir_path, dir_info in self.knowledge_base['directories'].items():
            dir_info['files'] = []
            for file_path, file_info in self.knowledge_base['files'].items():
                if file_info['directory'] == dir_path:
                    dir_info['files'].append(file_path)
        # 建立檔案類型統計
        type_counts = defaultdict(int)
        for file_info in self.knowledge_base['files'].values():
            type_counts[file_info['type']] += 1
        self.knowledge_base['relationships']['file_types'] = dict(type_counts)
        print("✅ 關係圖建立完成")
    def generate_statistics(self):
        """生成統計資訊"""
        print("📊 生成統計資訊...")
        stats = {
            'total_directories': len(self.knowledge_base['directories']),
            'total_files': len(self.knowledge_base['files']),
            'critical_files_count': len(self.knowledge_base['critical_files']),
            'configurations_count': len(self.knowledge_base['configurations']),
            'directory_purposes': defaultdict(int),
            'file_types': defaultdict(int),
            'total_size': 0
        }
        for dir_info in self.knowledge_base['directories'].values():
            stats['directory_purposes'][dir_info['purpose']] += 1
        for file_info in self.knowledge_base['files'].values():
            stats['file_types'][file_info['type']] += 1
            stats['total_size'] += file_info['size']
        # 轉換 defaultdict 為 dict
        stats['directory_purposes'] = dict(stats['directory_purposes'])
        stats['file_types'] = dict(stats['file_types'])
        self.knowledge_base['statistics'] = stats
        print("✅ 統計資訊生成完成")
    def identify_critical_files(self):
        """識別關鍵檔案"""
        print("🎯 識別關鍵檔案...")
        critical_categories = {
            'bootstrap': [],
            'configuration': [],
            'security': [],
            'build': [],
            'entry_points': []
        }
        for file_path, file_info in self.knowledge_base['files'].items():
            file_name = file_info['name'].lower()
            if 'bootstrap' in file_name or 'root' in file_name:
                critical_categories['bootstrap'].append(file_path)
            elif file_info['type'] in ['yaml', 'json', 'toml', 'config']:
                critical_categories['configuration'].append(file_path)
            elif 'secret' in file_name or 'key' in file_name or 'auth' in file_name:
                critical_categories['security'].append(file_path)
            elif file_name in ['makefile', 'dockerfile', 'build.sh']:
                critical_categories['build'].append(file_path)
            elif file_name in ['main.py', 'main.js', 'index.js', 'app.py']:
                critical_categories['entry_points'].append(file_path)
        self.knowledge_base['critical_files_by_category'] = critical_categories
        print(f"✅ 識別完成: {len(self.knowledge_base['critical_files'])} 個關鍵檔案")
    def save_knowledge_base(self, filename='knowledge_base.json'):
        """保存知識庫"""
        print(f"💾 保存知識庫到 {filename}...")
        # 清理無法序列化的對象
        clean_kb = self._clean_for_json(self.knowledge_base)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_kb, f, indent=2, ensure_ascii=False)
        print("✅ 知識庫已保存")
        return filename
    def _clean_for_json(self, obj):
        """清理對象以確保可以 JSON 序列化"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # 將其他類型轉換為字符串
            return str(obj)
    def generate_report(self, filename='phase1_report.md'):
        """生成報告"""
        print(f"📝 生成報告到 {filename}...")
        stats = self.knowledge_base['statistics']
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 第一階段：儲存庫掃描報告\n\n")
            f.write(f"**掃描日期**: {self.knowledge_base['metadata']['scan_date']}\n")
            f.write(f"**階段**: {self.knowledge_base['metadata']['phase']}\n\n")
            f.write("## 📊 統計摘要\n\n")
            f.write(f"- **總目錄數**: {stats['total_directories']}\n")
            f.write(f"- **總檔案數**: {stats['total_files']}\n")
            f.write(f"- **關鍵檔案數**: {stats['critical_files_count']}\n")
            f.write(f"- **配置檔案數**: {stats['configurations_count']}\n")
            f.write(f"- **總大小**: {stats['total_size'] / 1024:.2f} KB\n\n")
            f.write("## 📁 目錄用途分佈\n\n")
            f.write("| 用途 | 數量 |\n")
            f.write("|------|------|\n")
            for purpose, count in sorted(stats['directory_purposes'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {purpose} | {count} |\n")
            f.write("\n")
            f.write("## 📄 檔案類型分佈\n\n")
            f.write("| 類型 | 數量 |\n")
            f.write("|------|------|\n")
            for file_type, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {file_type} | {count} |\n")
            f.write("\n")
            f.write("## 🎯 關鍵檔案\n\n")
            f.write("### 按類別分類\n\n")
            for category, files in self.knowledge_base['critical_files_by_category'].items():
                f.write(f"#### {category.capitalize()}\n\n")
                for file_path in files[:10]:  # 只顯示前10個
                    f.write(f"- `{file_path}`\n")
                if len(files) > 10:
                    f.write(f"- ... 和其他 {len(files) - 10} 個檔案\n")
                f.write("\n")
        print("✅ 報告已生成")
        return filename
def main():
    """主程式"""
    print("="*60)
    print("🚀 第一階段：儲存庫掃描和知識庫建立")
    print("="*60 + "\n")
    # 創建掃描器
    scanner = RepositoryScanner()
    # 執行掃描
    if scanner.scan():
        # 保存知識庫
        kb_file = scanner.save_knowledge_base('knowledge_base.json')
        # 生成報告
        report_file = scanner.generate_report('phase1_report.md')
        print("\n" + "="*60)
        print("✅ 第一階段完成！")
        print("="*60)
        print(f"📁 知識庫: {kb_file}")
        print(f"📄 報告: {report_file}")
        print("="*60)
        return True
    else:
        print("❌ 掃描失敗")
        return False
if __name__ == '__main__':
    main()