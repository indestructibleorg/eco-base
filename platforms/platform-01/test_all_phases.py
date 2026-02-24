#!/usr/bin/env python3
"""
全阶段整合测试脚本
测试 Phase 1、Phase 2、Phase 3 所有组件
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

print("="*70)
print("闭环系统全阶段整合测试")
print("="*70)

# 运行 Phase 1 & 2 测试
print("\n>>> 运行 Phase 1 & 2 测试...")
import test_integration

# 运行 Phase 3 测试
print("\n>>> 运行 Phase 3 测试...")
import test_phase3

print("\n" + "="*70)
print("🎉 所有阶段测试完成！")
print("="*70)
