#
# @ECO-governed
# @ECO-layer: gl-platform.gl-platform.governance
# @ECO-semantic: extract_conflicts
# @ECO-audit-trail: ../../engine/gl-platform.gl-platform.governance/GL_SEMANTIC_ANCHOR.json
#
#!/usr/bin/env python3
"""
提取並分析命名衝突
INSTANT 系統檢測到 750 個命名衝突，主要圍繞 "DEFINITION" 這個名稱
Author: SuperNinja AI Agent
Date: 2026-01-22
Version: 1.0.0
"""
# MNGA-002: Import organization needs review
import os
import re
import json
from datetime import datetime
def find_definition_conflicts():
    """查找所有包含 DEFINITION 的文件"""
    conflicts = {
        'yaml_files': [],
        'python_files': [],
        'other_files': [],
        'total_files': 0
    }
    print("🔍 開始掃描文件中的 DEFINITION 衝突...")
    # 掃描目錄
    for root, dirs, files in os.walk('.'):
        # 跳過 Git 和緩存目錄
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            filepath = os.path.join(root, file)
            # 處理 YAML 文件
            if file.endswith(('.yaml', '.yml')):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'DEFINITION' in content.upper():
                            conflicts['yaml_files'].append(filepath)
                except (IOError, OSError, UnicodeDecodeError):
                    continue
            # 處理 Python 文件
            elif file.endswith('.py'):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if 'DEFINITION' in content.upper():
                            conflicts['python_files'].append(filepath)
                except (IOError, OSError, UnicodeDecodeError):
                    continue
    conflicts['total_files'] = len(conflicts['yaml_files']) + len(conflicts['python_files'])
    print(f"✓ YAML 文件：{len(conflicts['yaml_files'])} 個")
    print(f"✓ Python 文件：{len(conflicts['python_files'])} 個")
    print(f"✓ 總計：{conflicts['total_files']} 個文件")
    return conflicts
def analyze_conflict_types(conflicts):
    """分析衝突類型"""
    print("\n🔍 分析衝突類型...")
    analysis = {
        'type_a_direct_conflicts': [],      # 直接衝突：頂級 DEFINITION 鍵
        'type_b_indirect_conflicts': [],    # 間接衝突：嵌套的 DEFINITION
        'type_c_potential_conflicts': [],   # 潛在衝突：註釋或字符串中的 DEFINITION
        'type_d_duplicate_definitions': []  # 重複定義：相同的 DEFINITION 內容
    }
    # 分析 YAML 文件
    for filepath in conflicts['yaml_files']:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # 檢查直接衝突（頂級鍵）
                    if re.match(r'^DEFINITION\s*:', stripped):
                        analysis['type_a_direct_conflicts'].append({
                            'file': filepath,
                            'line': i + 1,
                            'content': line.strip(),
                            'type': 'direct'
                        })
                    # 檢查間接衝突（嵌套）
                    elif re.search(r'\w+\.DEFINITION\s*:', stripped) or re.search(r'\w+[_-]DEFINITION\s*:', stripped):
                        analysis['type_b_indirect_conflicts'].append({
                            'file': filepath,
                            'line': i + 1,
                            'content': line.strip(),
                            'type': 'indirect'
                        })
                    # 檢查潛在衝突（註釋或字符串）
                    elif 'DEFINITION' in stripped.upper():
                        if '#' in stripped or '"' in stripped or "'" in stripped:
                            analysis['type_c_potential_conflicts'].append({
                                'file': filepath,
                                'line': i + 1,
                                'content': line.strip(),
                                'type': 'potential'
                            })
        except (IOError, OSError, UnicodeDecodeError):
            continue
    # 分析 Python 文件
    for filepath in conflicts['python_files']:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # 檢查類定義
                    if re.match(r'class\s+DEFINITION', stripped):
                        analysis['type_a_direct_conflicts'].append({
                            'file': filepath,
                            'line': i + 1,
                            'content': line.strip(),
                            'type': 'python_class'
                        })
                    # 檢查函數定義
                    elif re.match(r'def\s+DEFINITION', stripped):
                        analysis['type_a_direct_conflicts'].append({
                            'file': filepath,
                            'line': i + 1,
                            'content': line.strip(),
                            'type': 'python_function'
                        })
                    # 檢查變量定義
                    elif re.match(r'^DEFINITION\s*=', stripped):
                        analysis['type_c_potential_conflicts'].append({
                            'file': filepath,
                            'line': i + 1,
                            'content': line.strip(),
                            'type': 'python_variable'
                        })
        except (IOError, OSError, UnicodeDecodeError):
            continue
    print(f"✓ A 類（直接衝突）：{len(analysis['type_a_direct_conflicts'])} 個")
    print(f"✓ B 類（間接衝突）：{len(analysis['type_b_indirect_conflicts'])} 個")
    print(f"✓ C 類（潛在衝突）：{len(analysis['type_c_potential_conflicts'])} 個")
    return analysis
def generate_priority_list(analysis):
    """生成優先級清單"""
    print("\n🔍 生成優先級清單...")
    priority_list = {
        'p0_critical': [],      # 關鍵：直接衝突
        'p1_high': [],          # 高：間接衝突
        'p2_medium': [],        # 中：潛在衝突
        'p3_low': []            # 低：其他
    }
    # A 類衝突為 P0
    for conflict in analysis['type_a_direct_conflicts']:
        priority_list['p0_critical'].append(conflict)
    # B 類衝突為 P1
    for conflict in analysis['type_b_indirect_conflicts']:
        priority_list['p1_high'].append(conflict)
    # C 類衝突為 P2
    for conflict in analysis['type_c_potential_conflicts']:
        priority_list['p2_medium'].append(conflict)
    print(f"✓ P0（關鍵）：{len(priority_list['p0_critical'])} 個")
    print(f"✓ P1（高）：{len(priority_list['p1_high'])} 個")
    print(f"✓ P2（中）：{len(priority_list['p2_medium'])} 個")
    return priority_list
def save_report(conflicts, analysis, priority_list):
    """保存分析報告"""
    print("\n💾 保存分析報告...")
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'yaml_files': len(conflicts['yaml_files']),
            'python_files': len(conflicts['python_files']),
            'total_files': conflicts['total_files']
        },
        'analysis': {
            'type_a_direct_conflicts_count': len(analysis['type_a_direct_conflicts']),
            'type_b_indirect_conflicts_count': len(analysis['type_b_indirect_conflicts']),
            'type_c_potential_conflicts_count': len(analysis['type_c_potential_conflicts'])
        },
        'priority_list': {
            'p0_critical_count': len(priority_list['p0_critical']),
            'p1_high_count': len(priority_list['p1_high']),
            'p2_medium_count': len(priority_list['p2_medium'])
        },
        'details': {
            'conflicts': {
                'yaml_files': conflicts['yaml_files'],
                'python_files': conflicts['python_files']
            },
            'analysis': analysis,
            'priority_list': priority_list
        }
    }
    # 保存完整報告
    with open('conflict_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # 保存優先級清單（文本格式）
    with open('conflict_priority_list.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GL00-09 命名衝突優先級清單\n")
        f.write(f"生成時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"P0 - 關鍵（{len(priority_list['p0_critical'])} 個）\n")
        f.write("-" * 80 + "\n")
        for conflict in priority_list['p0_critical'][:50]:  # 只顯示前 50 個
            f.write(f"  {conflict['file']}:{conflict['line']} - {conflict['content'][:80]}\n")
        if len(priority_list['p0_critical']) > 50:
            f.write(f"  ... 還有 {len(priority_list['p0_critical']) - 50} 個\n")
        f.write("\n")
        f.write(f"P1 - 高（{len(priority_list['p1_high'])} 個）\n")
        f.write("-" * 80 + "\n")
        for conflict in priority_list['p1_high'][:50]:
            f.write(f"  {conflict['file']}:{conflict['line']} - {conflict['content'][:80]}\n")
        if len(priority_list['p1_high']) > 50:
            f.write(f"  ... 還有 {len(priority_list['p1_high']) - 50} 個\n")
        f.write("\n")
        f.write(f"P2 - 中（{len(priority_list['p2_medium'])} 個）\n")
        f.write("-" * 80 + "\n")
        for conflict in priority_list['p2_medium'][:50]:
            f.write(f"  {conflict['file']}:{conflict['line']} - {conflict['content'][:80]}\n")
        if len(priority_list['p2_medium']) > 50:
            f.write(f"  ... 還有 {len(priority_list['p2_medium']) - 50} 個\n")
    print("✓ 完整報告已保存到：conflict_analysis_report.json")
    print("✓ 優先級清單已保存到：conflict_priority_list.txt")
def main():
    """主函數"""
    print("=" * 80)
    print("GL00-09 命名衝突分析工具")
    print(f"執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    # 1. 查找衝突
    conflicts = find_definition_conflicts()
    # 2. 分析衝突類型
    analysis = analyze_conflict_types(conflicts)
    # 3. 生成優先級清單
    priority_list = generate_priority_list(analysis)
    # 4. 保存報告
    save_report(conflicts, analysis, priority_list)
    print()
    print("=" * 80)
    print("✓ 分析完成！")
    print("=" * 80)
    print()
    print("📊 總結：")
    print(f"  - 總文件數：{conflicts['total_files']} 個")
    print(f"  - YAML 文件：{len(conflicts['yaml_files'])} 個")
    print(f"  - Python 文件：{len(conflicts['python_files'])} 個")
    print(f"  - P0 關鍵衝突：{len(priority_list['p0_critical'])} 個")
    print(f"  - P1 高優先級：{len(priority_list['p1_high'])} 個")
    print(f"  - P2 中優先級：{len(priority_list['p2_medium'])} 個")
    print()
    print("📁 輸出文件：")
    print("  - conflict_analysis_report.json（完整報告）")
    print("  - conflict_priority_list.txt（優先級清單）")
    print()
if __name__ == '__main__':
    main()
