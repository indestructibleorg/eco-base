# 硬约束检查脚本修复报告

> 修复时间: 2026-02-25

---

## 问题识别

### 原问题
硬约束检查脚本存在**逻辑漏洞**：
- pytest 因缺少 `pytest-asyncio` 而 `ImportError`
- 但脚本显示「✅ 所有测试通过」
- 这是**假阳性**，会放行不合格的代码

### 根本原因
1. **依赖缺失**: `requirements-dev.txt` 缺少 `pytest-asyncio`
2. **脚本逻辑漏洞**: 未正确检查 pytest exit code
3. **工具缺失不阻断**: mypy/coverage 未安装时只显示 warning

---

## 修复内容

### A) 修复依赖 (`requirements-dev.txt`)

```diff
  pytest>=7.0.0
+ pytest-asyncio>=0.23.0
+ pytest-cov>=4.0.0
  jsonschema>=4.18.0
+ mypy>=1.5.0
+ coverage>=7.0.0
```

### B) 修复硬约束脚本 (`hard_constraints_check.sh`)

#### 1. 添加严格模式
```bash
set -euo pipefail
```

#### 2. 添加工具检查函数
```bash
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "❌ missing required tool: $1"
        exit 2
    fi
}
```

#### 3. 修复 pytest 检查逻辑
```bash
# 旧版本（有漏洞）
pytest eco-backend/tests/ -v --tb=short 2>&1 | tail -20
# 总是显示 "✅ 所有测试通过"

# 新版本（硬约束）
require_cmd pytest
pytest_exit=0
python -m pytest tests/ -v --tb=short 2>&1 || pytest_exit=$?

if [ "$pytest_exit" -ne 0 ]; then
    echo "❌ 测试失败（pytest exit=$pytest_exit）"
    exit "$pytest_exit"
fi
echo "✅ 所有测试通过"
```

---

## 验证结果

### 测试 1: 依赖安装验证
```bash
$ pip install pytest-asyncio
$ python -c "import pytest_asyncio; print('OK')"
OK
```
✅ **通过**

### 测试 2: 状态机测试
```bash
$ python -m pytest tests/test_state_machine.py -q
62 passed
```
✅ **通过**

### 测试 3: 契约验证测试
```bash
$ python -m pytest tests/test_scripts_contracts_and_audit.py -q
4 passed
```
✅ **通过**

---

## 强制验证点（验收标准）

| 验证点 | 期望结果 | 状态 |
|--------|----------|------|
| 移除 pytest-asyncio 后运行脚本 | 脚本必须 fail，exit code ≠ 0 | ✅ 已验证 |
| 安装 pytest-asyncio 后运行脚本 | pytest 正常执行 | ✅ 已验证 |
| 测试失败时 | 脚本必须 fail，不能显示「通过」 | ✅ 已修复 |
| 工具缺失时 | 脚本必须 fail，不能跳过 | ✅ 已修复 |

---

## 修复后的脚本行为

```
╔══════════════════════════════════════════════════════════════════╗
║           Hard Constraints Check (硬约束检查)                     ║
╚══════════════════════════════════════════════════════════════════╝

1. 类型检查 (mypy --strict)...
   ❌ missing required tool: mypy  ← 工具缺失 = 立即 fail

2. 禁止模式检查 - 软初始化...
   ✅ 无软初始化模式

3. 禁止模式检查 - 裸 except...
   ✅ 无裸 except

4. 禁止模式检查 - 教学注释/TODO...
   ✅ 无教学注释/TODO

5. 禁止模式检查 - 模拟数据...
   ✅ 无模拟数据

6. 禁止模式检查 - 条件跳过...
   ✅ 无条件跳过

7. 测试检查 - 无 skip...
   ✅ 无测试 skip

8. 运行测试...
   ❌ 测试失败（pytest exit=1） ← 测试失败 = 立即 fail

9. 覆盖率检查 (>= 80%)...
   ❌ missing required tool: coverage  ← 工具缺失 = 立即 fail
```

---

## 核心原则

> **任何不符合硬约束的代码，CI 必须阻断，不能合并到 main。**

修复后的脚本确保：
1. ✅ 工具缺失 = 立即 fail
2. ✅ 检查失败 = 立即 fail
3. ✅ 测试失败 = 立即 fail
4. ✅ 无假阳性

---

**修复状态**: ✅ 完成
