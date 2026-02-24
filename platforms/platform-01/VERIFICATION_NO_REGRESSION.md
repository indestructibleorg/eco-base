# 硬约束验证与防退化体系

> **目标**: 确保硬闸不会退化，做到强制、可验证、可终结

---

## 1. 三层验证体系

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 守卫测试 (Guard Tests)                                  │
│  - 验证硬约束脚本本身的正确性                                      │
│  - 检测禁止模式                                                   │
│  - 验证退出码处理                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: 硬约束检查 (Hard Constraints Check)                     │
│  - 类型检查 (mypy)                                               │
│  - 禁止模式检测                                                   │
│  - 测试运行                                                       │
│  - 覆盖率检查                                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: CI/CD 门控 (CI Gate)                                    │
│  - GitHub Actions 工作流                                          │
│  - 强制阻断不合规代码                                             │
│  - 覆盖率门槛 >= 80%                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: 守卫测试 (`test_hard_constraints_guard.py`)

### 2.1 测试覆盖

| 测试类 | 测试数 | 验证内容 |
|--------|--------|----------|
| `TestHardConstraintsGuardBehavior` | 5 | 脚本行为正确性 |
| `TestProhibitedPatternsDetection` | 4 | 禁止模式检测 |
| `TestHardConstraintsExitCodes` | 2 | 退出码处理 |
| `TestRequirementsDev` | 4 | 依赖完整性 |
| `TestCIWorkflow` | 3 | CI 配置正确性 |
| `TestNoRegressionPatterns` | 2 | 防止退化模式 |

### 2.2 运行方式

```bash
# 运行守卫测试
python -m pytest tests/test_hard_constraints_guard.py -v

# 期望输出: 20 passed
```

---

## 3. Layer 2: 硬约束检查 (`hard_constraints_check.sh`)

### 3.1 检查项

| 步骤 | 检查内容 | 失败行为 |
|------|----------|----------|
| 1 | 类型检查 (mypy --strict) | exit 2 |
| 2 | 软初始化检测 | exit 1 |
| 3 | 裸 except 检测 | exit 1 |
| 4 | 教学注释/TODO 检测 | exit 1 |
| 5 | 模拟数据检测 | exit 1 |
| 6 | 条件跳过检测 | exit 1 |
| 7 | 测试 skip 检测 | exit 1 |
| 8 | 测试运行 | exit pytest_exit |
| 9 | 覆盖率检查 | exit coverage_exit |

### 3.2 运行方式

```bash
# 运行硬约束检查
bash hard_constraints_check.sh

# 期望: 全部通过，exit 0
# 失败: 立即 exit 非 0
```

---

## 4. Layer 3: CI/CD 门控

### 4.1 工作流结构

```yaml
# .github/workflows/hard-constraints.yml
jobs:
  hard-constraints:      # 类型检查 + 硬约束检查
  state-machine-tests:   # 状态机测试
  contract-tests:        # 契约验证测试
  guard-tests:           # 守卫测试（新增）
  gate:                  # 最终门控
    needs: [all above]
    if: always()
    # 任何失败都阻断合并
```

### 4.2 门控规则

```bash
# gate job 逻辑
if [[ "$hard_constraints" != "success" ]] || \
   [[ "$state_machine_tests" != "success" ]] || \
   [[ "$contract_tests" != "success" ]] || \
   [[ "$guard_tests" != "success" ]]; then
  echo "❌ Gate FAILED"
  exit 1
fi
```

---

## 5. 防止退化机制

### 5.1 代码层面

```python
# test_hard_constraints_guard.py::TestNoRegressionPatterns

def test_no_pipe_to_true_in_script(self):
    """禁止用 `|| true` 忽略错误"""
    # 防止: pytest ... || true

def test_no_echo_pipe_grep_for_result_check(self):
    """禁止用 echo | grep 检查结果"""
    # 防止: echo "$VAR" | grep ...
```

### 5.2 脚本层面

```bash
# hard_constraints_check.sh
set -euo pipefail  # 严格模式

require_cmd() {    # 工具缺失 = 立即 fail
  if ! command -v "$1"; then
    echo "❌ missing tool: $1"
    exit 2
  fi
}

# pytest 检查 - 必须捕获 exit code
pytest_exit=0
python -m pytest ... || pytest_exit=$?
if [ "$pytest_exit" -ne 0 ]; then
  exit "$pytest_exit"
fi
```

### 5.3 CI 层面

```yaml
# 新增 guard-tests job
  guard-tests:
    name: Guard Tests
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/test_hard_constraints_guard.py -v
```

---

## 6. 验收清单 (Acceptance Checklist)

### 6.1 开发环境验收

- [ ] `pip install -r requirements-dev.txt` 成功
- [ ] `python -m pytest tests/test_hard_constraints_guard.py` 20 passed
- [ ] `bash hard_constraints_check.sh` 通过

### 6.2 CI 环境验收

- [ ] PR 触发 CI 工作流
- [ ] 硬约束检查通过
- [ ] 守卫测试通过
- [ ] 状态机测试通过
- [ ] 契约验证通过
- [ ] 覆盖率 >= 80%

### 6.3 退化测试验收

- [ ] 故意引入软初始化 → CI 阻断
- [ ] 故意引入 TODO → CI 阻断
- [ ] 故意引入裸 except → CI 阻断
- [ ] 故意降低覆盖率 → CI 阻断

---

## 7. 强制验证流程

```bash
# 每次提交前必须运行
./verify_hard_constraints.sh
```

### 7.1 验证脚本

```bash
#!/bin/bash
# verify_hard_constraints.sh

echo "=== 强制验证流程 ==="

# 1. 守卫测试
echo "1. 运行守卫测试..."
pytest tests/test_hard_constraints_guard.py -q || exit 1

# 2. 硬约束检查
echo "2. 运行硬约束检查..."
bash hard_constraints_check.sh || exit 1

# 3. 状态机测试
echo "3. 运行状态机测试..."
pytest tests/test_state_machine.py -q || exit 1

# 4. 契约验证
echo "4. 运行契约验证..."
pytest tests/test_scripts_contracts_and_audit.py -q || exit 1

echo "✅ 全部验证通过"
```

---

## 8. 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `hard_constraints_check.sh` | 硬约束检查脚本 | ✅ |
| `tests/test_hard_constraints_guard.py` | 守卫测试 | ✅ |
| `.github/workflows/hard-constraints.yml` | CI 工作流 | ✅ |
| `requirements-dev.txt` | 开发依赖 | ✅ |
| `VERIFICATION_NO_REGRESSION.md` | 本文档 | ✅ |

---

## 9. 核心原则

> **任何不符合硬约束的代码，CI 必须阻断，不能合并到 main。**

### 不可妥协的底线

1. ✅ 守卫测试必须全部通过
2. ✅ 硬约束检查必须全部通过
3. ✅ 测试覆盖率必须 >= 80%
4. ✅ 无软初始化模式
5. ✅ 无教学注释/TODO
6. ✅ 无裸 except

---

**验证体系状态**: ✅ 已建立，可防止退化
