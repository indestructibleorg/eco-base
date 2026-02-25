# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: repository_explorer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
儲存庫探索工具 - 系統性地掃描和分析儲存庫結構
"""
# MNGA-002: Import organization needs review
import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
class RepositoryExplorer:
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.repository_map = {
            'scan_date': datetime.now().isoformat(),
            'root_path': str(self.root_path.absolute()),
            'directories': {},
            'files': {},
            'structure': {},
            'statistics': {}
        }
    def scan_repository(self):
        """完整掃描儲存庫"""
        print(f"🔍 開始掃描儲存庫: {self.root_path}")
        # 掃描所有目錄和檔案
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            rel_path = root_path.relative_to(self.root_path)
            # 記錄目錄資訊
            dir_info = {
                'path': str(rel_path),
                'subdirectories': dirs,
                'file_count': len(files),
                'files': files
            }
            self.repository_map['directories'][str(rel_path)] = dir_info
            # 記錄檔案資訊
            for file in files:
                file_path = root_path / file
                rel_file_path = file_path.relative_to(self.root_path)
                file_info = {
                    'name': file,
                    'path': str(rel_file_path),
                    'size': file_path.stat().st_size if file_path.exists() else 0,
                    'extension': file_path.suffix,
                    'type': self.classify_file_type(file_path.suffix)
                }
                self.repository_map['files'][str(rel_file_path)] = file_info
        # 生成統計資訊
        self.generate_statistics()
        print(f"✅ 掃描完成: {len(self.repository_map['directories'])} 個目錄, {len(self.repository_map['files'])} 個檔案")
    def classify_file_type(self, extension):
        """分類檔案類型"""
        extension = extension.lower()
        type_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sh': 'shell',
            '.toml': 'toml',
            '.env': 'environment',
            '.gitignore': 'git',
            '': 'directory'
        }
        return type_map.get(extension, 'unknown')
    def generate_statistics(self):
        """生成統計資訊"""
        stats = {
            'total_directories': len(self.repository_map['directories']),
            'total_files': len(self.repository_map['files']),
            'file_types': defaultdict(int),
            'largest_files': [],
            'directories_by_depth': defaultdict(int)
        }
        # 統計檔案類型
        for file_path, file_info in self.repository_map['files'].items():
            stats['file_types'][file_info['type']] += 1
        # 統計目錄深度
        for dir_path in self.repository_map['directories'].keys():
            depth = dir_path.count(os.sep)
            stats['directories_by_depth'][depth] += 1
        self.repository_map['statistics'] = dict(stats)
    def analyze_structure(self):
        """分析儲存庫結構"""
        print("📊 分析儲存庫結構...")
        # 識別關鍵目錄
        key_directories = self.identify_key_directories()
        self.repository_map['structure']['key_directories'] = key_directories
        # 識別配置檔案
        config_files = self.identify_config_files()
        self.repository_map['structure']['config_files'] = config_files
        # 識別文件檔案
        doc_files = self.identify_documentation_files()
        self.repository_map['structure']['documentation_files'] = doc_files
    def identify_key_directories(self):
        """識別關鍵目錄"""
        key_dirs = {}
        for dir_path, dir_info in self.repository_map['directories'].items():
            path_parts = dir_path.split(os.sep)
            # 根據目錄名稱識別用途
            for part in path_parts:
                if part.lower() in ['src', 'source', 'lib', 'library']:
                    key_dirs[dir_path] = 'source_code'
                elif part.lower() in ['docs', 'doc', 'documentation']:
                    key_dirs[dir_path] = 'documentation'
                elif part.lower() in ['config', 'configuration', 'etc']:
                    key_dirs[dir_path] = 'configuration'
                elif part.lower() in ['test', 'tests', 'spec']:
                    key_dirs[dir_path] = 'testing'
                elif part.lower() in ['bin', 'sbin', 'scripts']:
                    key_dirs[dir_path] = 'scripts'
                elif part.lower() in ['workspace', 'work']:
                    key_dirs[dir_path] = 'workspace'
        return key_dirs
    def identify_config_files(self):
        """識別配置檔案"""
        config_files = []
        for file_path, file_info in self.repository_map['files'].items():
            if file_info['type'] in ['yaml', 'json', 'toml', 'environment']:
                config_files.append({
                    'path': file_path,
                    'type': file_info['type'],
                    'size': file_info['size']
                })
        return config_files
    def identify_documentation_files(self):
        """識別文件檔案"""
        doc_files = []
        for file_path, file_info in self.repository_map['files'].items():
            if file_info['type'] == 'markdown' or file_path.lower().endswith('readme'):
                doc_files.append({
                    'path': file_path,
                    'size': file_info['size']
                })
        return doc_files
    def generate_report(self, output_file='repository_map.json'):
        """生成報告"""
        print(f"📝 生成報告: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.repository_map, f, indent=2, ensure_ascii=False)
        print(f"✅ 報告已保存: {output_file}")
    def print_summary(self):
        """列印摘要"""
        stats = self.repository_map['statistics']
        print("\n" + "="*50)
        print("📊 儲存庫摘要")
        print("="*50)
        print(f"總目錄數: {stats['total_directories']}")
        print(f"總檔案數: {stats['total_files']}")
        print("\n檔案類型分佈:")
        for file_type, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {file_type}: {count}")
        print("="*50 + "\n")
def main():
    """主程式"""
    # 創建探索器
    explorer = RepositoryExplorer()
    # 執行掃描
    explorer.scan_repository()
    # 分析結構
    explorer.analyze_structure()
    # 列印摘要
    explorer.print_summary()
    # 生成報告
    explorer.generate_report('repository_map.json')
    # 生成人類可讀報告
    generate_human_readable_report(explorer)
def generate_human_readable_report(explorer):
    """生成人類可讀的報告"""
    report_file = 'repository_structure_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 儲存庫結構報告\n\n")
        f.write(f"**掃描日期**: {explorer.repository_map['scan_date']}\n")
        f.write(f"**根路徑**: {explorer.repository_map['root_path']}\n\n")
        # 統計資訊
        stats = explorer.repository_map['statistics']
        f.write("## 統計資訊\n\n")
        f.write(f"- **總目錄數**: {stats['total_directories']}\n")
        f.write(f"- **總檔案數**: {stats['total_files']}\n\n")
        f.write("### 檔案類型分佈\n\n")
        f.write("| 類型 | 數量 |\n")
        f.write("|------|------|\n")
        for file_type, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {file_type} | {count} |\n")
        f.write("\n")
        # 關鍵目錄
        f.write("## 關鍵目錄\n\n")
        key_dirs = explorer.repository_map['structure'].get('key_directories', {})
        for dir_path, purpose in key_dirs.items():
            f.write(f"- **{dir_path}**: {purpose}\n")
        f.write("\n")
        # 配置檔案
        f.write("## 配置檔案\n\n")
        config_files = explorer.repository_map['structure'].get('config_files', [])
        for config in config_files:
            f.write(f"- **{config['path']}** ({config['type']})\n")
        f.write("\n")
        # 文件檔案
        f.write("## 文件檔案\n\n")
        doc_files = explorer.repository_map['structure'].get('documentation_files', [])
        for doc in doc_files:
            f.write(f"- **{doc['path']}**\n")
        f.write("\n")
    print(f"✅ 人類可讀報告已生成: {report_file}")
if __name__ == '__main__':
    main()