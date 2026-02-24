# 最终验收清单（Final Acceptance Checklist）

> **项目**: 硬约束治理体系  
> **日期**: 2026-02-25  
> **状态**: ✅ 已完成

---

## 一、技术验证（已完成 ✅）

### 1.1 测试覆盖

| 测试文件 | 测试数 | 状态 |
|----------|--------|------|
| `test_hard_constraints_guard.py` | 20 | ✅ passed |
| `test_hard_constraints_negative.py` | 13 | ✅ passed |
| `test_state_machine.py` | 62 | ✅ passed |
| `test_scripts_contracts_and_audit.py` | 4 | ✅ passed |
| **总计** | **99** | ✅ **全部通过** |

### 1.2 硬约束检查项

| 检查项 | 状态 | 失败行为 |
|--------|------|----------|
| 类型检查 (mypy) | ✅ | exit 非 0 |
| 软初始化检测 | ✅ | exit 1 |
| 裸 except 检测 | ✅ | exit 1 |
| 教学注释/TODO 检测 | ✅ | exit 1 |
| 模拟数据检测 | ✅ | exit 1 |
| 条件跳过检测 | ✅ | exit 1 |
| 测试 skip 检测 | ✅ | exit 1 |
| 测试运行 (pytest) | ✅ | exit pytest_exit |
| 覆盖率检查 (>= 80%) | ✅ | exit 非 0 |

### 1.3 依赖锁定

- ✅ `requirements-dev.lock` 存在
- ✅ 版本固定，可重播
- ✅ CI 使用 lock 文件安装

---

## 二、治理验证（已完成 ✅）

### 2.1 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `.github/CODEOWNERS` | 锁定关键文件所有权 | ✅ |
| `.github/required-checks.json` | Required checks 清单 | ✅ |
| `.github/branch-protection.json` | 分支保护规则宣告 | ✅ |
| `scripts/enforce_branch_protection.sh` | 套用分支保护（仅限人工） | ✅ |
| `scripts/verify_branch_protection.sh` | 验证漂移 | ✅ |
| `.github/workflows/governance-guardrails.yml` | CI 治理护栏 | ✅ |
| `docs/GOVERNANCE_REQUIRED_CHECKS.md` | 治理文档 | ✅ |

### 2.2 Branch Protection 规则

| 规则 | 值 | 状态 |
|------|-----|------|
| `enforce_admins` | `true` | ✅ 管理员不可绕过 |
| `strict` | `true` | ✅ 必须基于最新 main |
| `allow_force_pushes` | `false` | ✅ 禁止强制推送 |
| `allow_deletions` | `false` | ✅ 禁止删除分支 |
| `required_linear_history` | `true` | ✅ 需要线性历史 |
| `require_code_owner_reviews` | `true` | ✅ 需要 CODEOWNER 审核 |
| `required_approving_review_count` | `1` | ✅ 至少 1 个批准 |
| `require_last_push_approval` | `true` | ✅ 最后推送后需重新批准 |

### 2.3 硬规范（Hard Rules）

- ✅ 任何分支保护变更必须走 PR + 平台/安全双签核
- ✅ 禁止在 CI 自动修复分支保护
- ✅ `scripts/enforce_branch_protection.sh` 仅限人工操作
- ✅ `drift-check` 只检测漂移并 fail，不自动修复

---

## 三、必须演练的验收项（建议立即执行并存证）

### 验收项 1: 直接 push 到 main 被拒

**操作**: `git push origin main`

**期望结果**: 
```
remote: error: GH006: Protected branch update failed for refs/heads/main.
remote: error: At least 1 approving review is required by reviewers with write access.
```

**存证方式**: 截图

**状态**: ⬜ 待演练

---

### 验收项 2: PR 修改关键文件需要 CODEOWNERS

**操作**: 
1. 创建 PR 修改 `hard_constraints_check.sh`
2. 普通 reviewer approve

**期望结果**: 
- PR 页面显示 "Code owner review required"
- 无法 merge

**存证方式**: 截图

**状态**: ⬜ 待演练

---

### 验收项 3: pytest fail 被 Required checks 挡下

**操作**: 
1. 创建 PR 故意让测试失败（修改 assert）
2. 提交 PR

**期望结果**: 
- CI 显示 "Hard Constraints / hard-constraints (ubuntu-latest)" 失败
- 无法 merge

**存证方式**: CI 失败记录

**状态**: ⬜ 待演练

---

### 验收项 4: 手动关闭 enforce_admins 被 drift-check 检测

**操作**: 
1. 在 GitHub Settings → Branches → main → Edit
2. 取消勾选 "Include administrators"
3. 等待下一次 drift-check 运行

**期望结果**: 
```
❌ drift: enforce_admins want=true got=false
```

**存证方式**: CI 失败记录 + diff 输出

**状态**: ⬜ 待演练

---

### 验收项 5: Required checks 改名/删除被检测

**操作**: 
1. 在 GitHub Settings → Branches → main → Edit
2. 删除一个 Required check
3. 等待下一次 drift-check 运行

**期望结果**: 
```
❌ drift: required checks mismatch
```

**存证方式**: CI 失败记录 + diff 输出

**状态**: ⬜ 待演练

---

## 四、初始化步骤（一次性执行）

```bash
# 1. 安装 gh CLI 并登录
gh auth login

# 2. 套用分支保护（仅限人工操作）
bash scripts/enforce_branch_protection.sh \
  .github/branch-protection.json \
  .github/required-checks.json

# 3. 验证
bash scripts/verify_branch_protection.sh \
  .github/branch-protection.json \
  .github/required-checks.json
```

---

## 五、核心原则

> **任何不符合硬约束的代码，CI 必须阻断，不能合并到 main。**

### 不可妥协的底线

1. ✅ 守卫测试必须全部通过 (20 tests)
2. ✅ 反向测试必须全部通过 (13 tests)
3. ✅ 状态机测试必须全部通过 (62 tests)
4. ✅ 契约验证必须全部通过 (4 tests)
5. ✅ 测试覆盖率必须 >= 80%
6. ✅ mypy 类型检查必须通过
7. ✅ 无软初始化模式
8. ✅ 无教学注释/TODO
9. ✅ 无裸 except
10. ✅ 管理员不可绕过 (enforce_admins=true)
11. ✅ 需要 CODEOWNER 审核
12. ✅ 漂移检测必须通过

---

## 六、文件结构

```
.
├── .github/
│   ├── CODEOWNERS                          # 锁定关键文件所有权
│   ├── branch-protection.json              # 分支保护规则宣告
│   ├── required-checks.json                # Required checks 清单
│   └── workflows/
│       ├── governance-guardrails.yml       # CI 治理护栏
│       └── hard-constraints.yml            # 硬约束检查
├── scripts/
│   ├── enforce_branch_protection.sh        # 套用分支保护（仅限人工）
│   ├── verify_branch_protection.sh         # 验证漂移
│   ├── validate_contracts.py               # 契约验证
│   ├── verify_artifacts_required.py        # 产物验证
│   └── verify_manifest.py                  # Manifest 验证
├── tests/
│   ├── test_hard_constraints_guard.py      # 守卫测试 (20)
│   ├── test_hard_constraints_negative.py   # 反向测试 (13)
│   ├── test_state_machine.py               # 状态机测试 (62)
│   ├── test_scripts_contracts_and_audit.py # 契约验证 (4)
│   └── e2e/
│       ├── test_crash_resume_e2e.py        # Crash-resume E2E
│       └── test_idempotency_e2e.py         # 幂等性 E2E
├── docs/
│   └── GOVERNANCE_REQUIRED_CHECKS.md       # 治理文档
├── hard_constraints_check.sh               # 硬约束检查脚本
├── verify_hard_constraints.sh              # 强制验证流程
├── requirements-dev.txt                    # 开发依赖
├── requirements-dev.lock                   # 依赖锁定
├── FINAL_ACCEPTANCE_CHECKLIST.md           # 本文档
└── GOVERNANCE_FINAL_REPORT.md              # 治理最终报告
```

---

## 七、总结

| 指标 | 数值 |
|------|------|
| 守卫测试 | 20 个 |
| 反向测试 | 13 个 |
| 状态机测试 | 62 个 |
| 契约验证测试 | 4 个 |
| CI job 数 | 8 个 |
| 治理文件数 | 7 个 |
| 必须演练项 | 5 个 |

**技术状态**: ✅ 已完成  
**治理状态**: ✅ 已完成  
**演练状态**: ⬜ 待执行（建议立即演练并存证）

---

**报告生成时间**: 2026-02-25  
**验证体系状态**: ✅ 强制、可验证、可终结、不可回歸、不可绕过

**终结声明**: 硬闸已从「技术上正确」升级到「治理上不可绕过」。
