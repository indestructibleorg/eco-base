# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: archive-tools
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
#
# @ECO-governed
# @ECO-layer: gl-platform.governance
# @ECO-semantic: phase3_visualizer
# @ECO-audit-trail: ../../engine/gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
第三階段：視覺化與查詢系統
"""
# MNGA-002: Import organization needs review
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
class KnowledgeVisualizer:
    def __init__(self, knowledge_base_path='knowledge_base.json'):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self.load_knowledge_base()
        self.queries_log = []
    def load_knowledge_base(self):
        """載入知識庫"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 無法載入知識庫: {e}")
            return None
    def query_file_context(self, file_path: str) -> Dict:
        """查詢檔案的完整上下文"""
        print(f"\n🔍 查詢檔案上下文: {file_path}")
        print("="*60)
        files = self.knowledge_base.get('files', {})
        directories = self.knowledge_base.get('directories', {})
        context = {
            'file_path': file_path,
            'exists': False,
            'type': None,
            'directory': None,
            'purpose': None,
            'dependencies': [],
            'affected_by': [],
            'risks': None,
            'size': None,
            'is_critical': False
        }
        if file_path in files:
            file_info = files[file_path]
            context.update({
                'exists': True,
                'type': file_info.get('type'),
                'directory': file_info.get('directory'),
                'size': file_info.get('size'),
                'is_critical': file_info.get('is_critical', False)
            })
            # 獲取目錄用途
            if file_info.get('directory') in directories:
                dir_info = directories[file_info.get('directory')]
                context['purpose'] = dir_info.get('purpose')
            # 查找依賴關係
            context['dependencies'] = self.find_dependencies(file_path)
            context['affected_by'] = self.find_affected_files(file_path)
            # 評估風險
            context['risks'] = self.assess_file_risk(file_path)
        self.queries_log.append({
            'timestamp': datetime.now().isoformat(),
            'query_type': 'file_context',
            'target': file_path,
            'result': context
        })
        return context
    def find_dependencies(self, file_path: str) -> List[str]:
        """查找檔案依賴"""
        dependencies = []
        file_info = self.knowledge_base.get('files', {}).get(file_path, {})
        # 簡單實現：基於目錄和檔案類型
        file_type = file_info.get('type', '')
        directory = file_info.get('directory', '')
        # 查找同目錄下的相關檔案
        files = self.knowledge_base.get('files', {})
        for other_file, other_info in files.items():
            if (other_info.get('directory') == directory and 
                other_file != file_path and
                other_info.get('type') == file_type):
                dependencies.append(other_file)
        return dependencies[:10]  # 限制返回數量
    def find_affected_files(self, file_path: str) -> List[str]:
        """查找受影響的檔案"""
        affected = []
        file_info = self.knowledge_base.get('files', {}).get(file_path, {})
        # 查找可能依賴此檔案的其他檔案
        directory = file_info.get('directory', '')
        files = self.knowledge_base.get('files', {})
        for other_file, other_info in files.items():
            if other_info.get('directory') == directory:
                affected.append(other_file)
        return affected[:10]
    def assess_file_risk(self, file_path: str) -> str:
        """評估檔案風險"""
        critical_categories = ['bootstrap', 'security', 'build', 'entry_points']
        for category in critical_categories:
            critical_files = self.knowledge_base.get('critical_files_by_category', {}).get(category, [])
            if any(file_path in f for f in critical_files):
                return f'high ({category})'
        return 'low'
    def query_directory_structure(self, directory_path: str) -> Dict:
        """查詢目錄結構"""
        print(f"\n🔍 查詢目錄結構: {directory_path}")
        print("="*60)
        directories = self.knowledge_base.get('directories', {})
        files = self.knowledge_base.get('files', {})
        structure = {
            'directory_path': directory_path,
            'exists': False,
            'purpose': None,
            'subdirectories': [],
            'files': [],
            'total_size': 0,
            'file_types': defaultdict(int)
        }
        if directory_path in directories:
            dir_info = directories[directory_path]
            structure.update({
                'exists': True,
                'purpose': dir_info.get('purpose'),
                'subdirectories': dir_info.get('subdirectories', [])
            })
            # 獲取該目錄下的所有檔案
            for file_path, file_info in files.items():
                if file_info.get('directory') == directory_path:
                    structure['files'].append({
                        'name': file_info.get('name'),
                        'type': file_info.get('type'),
                        'size': file_info.get('size'),
                        'is_critical': file_info.get('is_critical', False)
                    })
                    structure['total_size'] += file_info.get('size', 0)
                    structure['file_types'][file_info.get('type')] += 1
        self.queries_log.append({
            'timestamp': datetime.now().isoformat(),
            'query_type': 'directory_structure',
            'target': directory_path,
            'result': structure
        })
        return structure
    def search_by_pattern(self, pattern: str, search_type: str = 'name') -> List[Dict]:
        """根據模式搜尋"""
        print(f"\n🔍 搜尋: {pattern} (類型: {search_type})")
        print("="*60)
        results = []
        files = self.knowledge_base.get('files', {})
        directories = self.knowledge_base.get('directories', {})
        if search_type == 'name':
            # 搜尋檔案名稱
            for file_path, file_info in files.items():
                if pattern.lower() in file_info.get('name', '').lower():
                    results.append({
                        'type': 'file',
                        'path': file_path,
                        'name': file_info.get('name'),
                        'file_type': file_info.get('type'),
                        'is_critical': file_info.get('is_critical', False)
                    })
            # 搜尋目錄名稱
            for dir_path, dir_info in directories.items():
                if pattern.lower() in dir_path.lower():
                    results.append({
                        'type': 'directory',
                        'path': dir_path,
                        'purpose': dir_info.get('purpose'),
                        'file_count': dir_info.get('file_count', 0)
                    })
        elif search_type == 'type':
            # 搜尋特定類型檔案
            for file_path, file_info in files.items():
                if file_info.get('type') == pattern:
                    results.append({
                        'type': 'file',
                        'path': file_path,
                        'name': file_info.get('name'),
                        'size': file_info.get('size')
                    })
        elif search_type == 'purpose':
            # 搜尋特定用途的目錄
            for dir_path, dir_info in directories.items():
                if dir_info.get('purpose') == pattern:
                    results.append({
                        'type': 'directory',
                        'path': dir_path,
                        'file_count': dir_info.get('file_count', 0),
                        'subdirectories': dir_info.get('subdirectories', [])
                    })
        self.queries_log.append({
            'timestamp': datetime.now().isoformat(),
            'query_type': 'search',
            'pattern': pattern,
            'search_type': search_type,
            'results_count': len(results)
        })
        return results
    def generate_visualization_report(self, filename='phase3_report.md'):
        """生成視覺化報告"""
        print(f"\n📝 生成視覺化報告到 {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 第三階段：視覺化與查詢系統報告\n\n")
            f.write(f"**生成日期**: {datetime.now().isoformat()}\n")
            f.write("**階段**: Phase 3 - Visualization and Query System\n\n")
            f.write("## 📊 系統概覽\n\n")
            f.write(f"- **知識庫檔案數**: {len(self.knowledge_base.get('files', {}))}\n")
            f.write(f"- **知識庫目錄數**: {len(self.knowledge_base.get('directories', {}))}\n")
            f.write(f"- **查詢次數**: {len(self.queries_log)}\n\n")
            f.write("## 🔍 查詢功能\n\n")
            f.write("### 1. 檔案上下文查詢\n\n")
            f.write("```python\n")
            f.write("# 查詢檔案的完整上下文資訊\n")
            f.write("context = visualizer.query_file_context('path/to/file.py')\n")
            f.write("# 返回: 檔案類型、目錄用途、依賴關係、風險等級等\n")
            f.write("```\n\n")
            f.write("### 2. 目錄結構查詢\n\n")
            f.write("```python\n")
            f.write("# 查詢目錄的完整結構\n")
            f.write("structure = visualizer.query_directory_structure('path/to/directory')\n")
            f.write("# 返回: 目錄用途、子目錄、檔案列表、統計資訊等\n")
            f.write("```\n\n")
            f.write("### 3. 模式搜尋\n\n")
            f.write("```python\n")
            f.write("# 根據名稱搜尋\n")
            f.write("results = visualizer.search_by_pattern('config', search_type='name')\n")
            f.write("\n")
            f.write("# 根據類型搜尋\n")
            f.write("results = visualizer.search_by_pattern('python', search_type='type')\n")
            f.write("\n")
            f.write("# 根據用途搜尋\n")
            f.write("results = visualizer.search_by_pattern('configuration', search_type='purpose')\n")
            f.write("```\n\n")
            f.write("## 📈 統計資訊\n\n")
            # 檔案類型統計
            f.write("### 檔案類型分佈\n\n")
            file_types = defaultdict(int)
            for file_info in self.knowledge_base.get('files', {}).values():
                file_types[file_info.get('type', 'unknown')] += 1
            f.write("| 類型 | 數量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            total_files = len(self.knowledge_base.get('files', {}))
            for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_files * 100) if total_files > 0 else 0
                f.write(f"| {file_type} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
            # 目錄用途統計
            f.write("### 目錄用途分佈\n\n")
            dir_purposes = defaultdict(int)
            for dir_info in self.knowledge_base.get('directories', {}).values():
                dir_purposes[dir_info.get('purpose', 'unknown')] += 1
            f.write("| 用途 | 數量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            total_dirs = len(self.knowledge_base.get('directories', {}))
            for purpose, count in sorted(dir_purposes.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_dirs * 100) if total_dirs > 0 else 0
                f.write(f"| {purpose} | {count} | {percentage:.1f}% |\n")
            f.write("\n")
            f.write("## 🔗 檔案關係圖示例\n\n")
            f.write("系統能夠識別和分析檔案之間的關係：\n\n")
            f.write("- **依賴關係**: 哪些檔案依賴於當前檔案\n")
            f.write("- **影響範圍**: 修改當前檔案會影響哪些其他檔案\n")
            f.write("- **風險評估**: 基於檔案類型和用途的風險等級\n")
            f.write("- **關鍵性**: 標記系統關鍵檔案\n\n")
        print("✅ 視覺化報告已生成")
        return filename
    def print_file_context(self, context: Dict):
        """列印檔案上下文"""
        print(f"\n📄 檔案: {context['file_path']}")
        print(f"   狀態: {'✅ 存在' if context['exists'] else '❌ 不存在'}")
        if context['exists']:
            print(f"   類型: {context['type']}")
            print(f"   目錄: {context['directory']}")
            print(f"   用途: {context['purpose']}")
            print(f"   大小: {context['size']} bytes")
            print(f"   關鍵性: {'🔥 是' if context['is_critical'] else '❌ 否'}")
            print(f"   風險: {context['risks']}")
            print(f"   依賴: {len(context['dependencies'])} 個檔案")
            print(f"   影響: {len(context['affected_by'])} 個檔案")
def main():
    """主程式 - 演示視覺化與查詢系統"""
    print("="*60)
    print("🚀 第三階段：建立視覺化與查詢系統")
    print("="*60 + "\n")
    # 創建視覺化器
    visualizer = KnowledgeVisualizer()
    if not visualizer.knowledge_base:
        print("❌ 無法載入知識庫，請先執行第一階段")
        return False
    print("✅ 視覺化系統初始化成功")
    # 演示查詢功能
    print("\n" + "="*60)
    print("📊 演示查詢功能")
    print("="*60)
    # 1. 查詢檔案上下文
    context = visualizer.query_file_context('machine-native-ops/README.md')
    visualizer.print_file_context(context)
    # 2. 查詢目錄結構
    structure = visualizer.query_directory_structure('machine-native-ops')
    if structure['exists']:
        print(f"\n📁 目錄: {structure['directory_path']}")
        print(f"   用途: {structure['purpose']}")
        print(f"   檔案數: {len(structure['files'])}")
        print(f"   總大小: {structure['total_size']} bytes")
    # 3. 模式搜尋
    results = visualizer.search_by_pattern('config', search_type='name')
    print(f"\n🔍 搜尋 'config': 找到 {len(results)} 個結果")
    for result in results[:5]:
        icon = "📁" if result['type'] == 'directory' else "📄"
        print(f"   {icon} {result['path']}")
    # 4. 生成報告
    report_file = visualizer.generate_visualization_report('phase3_report.md')
    print("\n" + "="*60)
    print("✅ 第三階段完成！")
    print("="*60)
    print(f"📄 報告: {report_file}")
    print(f"📊 執行了 {len(visualizer.queries_log)} 次查詢")
    print("="*60)
    return True
if __name__ == '__main__':
    main()