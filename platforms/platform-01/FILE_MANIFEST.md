# 文件清单（File Manifest）

> **生成时间**: 2026-02-25  
> **项目**: 闭环自愈系统 - 硬约束治理体系

---

## 一、治理文件（Governance）

| 文件 | 路径 | 作用 |
|------|------|------|
| CODEOWNERS | `.github/CODEOWNERS` | 锁定关键文件所有权 |
| branch-protection.json | `.github/branch-protection.json` | 分支保护规则宣告 |
| required-checks.json | `.github/required-checks.json` | Required checks 清单 |
| GOVERNANCE_REQUIRED_CHECKS.md | `docs/GOVERNANCE_REQUIRED_CHECKS.md` | 治理文档 |

---

## 二、CI/CD 工作流（Workflows）

| 文件 | 路径 | 作用 |
|------|------|------|
| governance-guardrails.yml | `.github/workflows/governance-guardrails.yml` | 治理护栏 + 漂移检测 |
| hard-constraints.yml | `.github/workflows/hard-constraints.yml` | 硬约束检查 |
| policy-check.yml | `.github/workflows/policy-check.yml` | PR 策略检查 |
| closed-loop-e2e.yml | `.github/workflows/closed-loop-e2e.yml` | E2E 测试 |

---

## 三、脚本（Scripts）

| 文件 | 路径 | 作用 | 权限 |
|------|------|------|------|
| hard_constraints_check.sh | `hard_constraints_check.sh` | 硬约束检查脚本 | 可执行 |
| verify_hard_constraints.sh | `verify_hard_constraints.sh` | 强制验证流程 | 可执行 |
| enforce_branch_protection.sh | `scripts/enforce_branch_protection.sh` | 套用分支保护 | 仅限人工 |
| verify_branch_protection.sh | `scripts/verify_branch_protection.sh` | 验证漂移 | CI 使用 |
| validate_contracts.py | `scripts/validate_contracts.py` | 契约验证 | - |
| verify_artifacts_required.py | `scripts/verify_artifacts_required.py` | 产物验证 | - |
| verify_manifest.py | `scripts/verify_manifest.py` | Manifest 验证 | - |
| _lib_hash.py | `scripts/_lib_hash.py` | 哈希工具库 | - |

---

## 四、测试（Tests）

| 文件 | 路径 | 测试数 | 说明 |
|------|------|--------|------|
| test_hard_constraints_guard.py | `tests/test_hard_constraints_guard.py` | 20 | 守卫测试 |
| test_hard_constraints_negative.py | `tests/test_hard_constraints_negative.py` | 13 | 反向测试 |
| test_state_machine.py | `tests/test_state_machine.py` | 62 | 状态机不变量测试 |
| test_scripts_contracts_and_audit.py | `tests/test_scripts_contracts_and_audit.py` | 4 | 契约验证测试 |
| test_crash_resume_e2e.py | `tests/e2e/test_crash_resume_e2e.py` | - | Crash-resume E2E |
| test_idempotency_e2e.py | `tests/e2e/test_idempotency_e2e.py` | - | 幂等性 E2E |
| test_verify_fail_e2e.py | `tests/e2e/test_verify_fail_e2e.py` | - | Verify-fail E2E |
| test_audit_e2e.py | `tests/e2e/test_audit_e2e.py` | - | Audit E2E |

**总计**: 99+ 测试

---

## 五、依赖（Dependencies）

| 文件 | 路径 | 说明 |
|------|------|------|
| requirements-dev.txt | `requirements-dev.txt` | 开发依赖声明 |
| requirements-dev.lock | `requirements-dev.lock` | 依赖锁定（可重播） |

---

## 六、契约（Contracts）

| 文件 | 路径 | 说明 |
|------|------|------|
| decision.schema.json | `contracts/decision.schema.json` | 决策契约 Schema |

---

## 七、报告文档（Reports）

| 文件 | 路径 | 说明 |
|------|------|------|
| PROJECT_COMPLETION_SUMMARY.md | `PROJECT_COMPLETION_SUMMARY.md` | 项目完成总结 |
| FINAL_ACCEPTANCE_CHECKLIST.md | `FINAL_ACCEPTANCE_CHECKLIST.md` | 最终验收清单 |
| GOVERNANCE_FINAL_REPORT.md | `GOVERNANCE_FINAL_REPORT.md` | 治理最终报告 |
| HARD_CONSTRAINTS_TERMINATION_REPORT.md | `HARD_CONSTRAINTS_TERMINATION_REPORT.md` | 硬约束终结报告 |
| VERIFICATION_NO_REGRESSION.md | `VERIFICATION_NO_REGRESSION.md` | 防退化验证 |
| HARD_CONSTRAINTS_DELIVERY.md | `HARD_CONSTRAINTS_DELIVERY.md` | 硬约束交付 |
| HARD_CONSTRAINTS_SPEC.md | `HARD_CONSTRAINTS_SPEC.md` | 硬约束规范 |
| IMPLEMENTATION_COMPLETION_REPORT.md | `IMPLEMENTATION_COMPLETION_REPORT.md` | 实现完成报告 |
| PLACEHOLDER_SCAN_REPORT.md | `PLACEHOLDER_SCAN_REPORT.md` | 占位符扫描报告 |
| ACCEPTANCE_KIT.md | `ACCEPTANCE_KIT.md` | 验收套件 |

---

## 八、核心代码（Core Code）

| 文件 | 路径 | 说明 |
|------|------|------|
| platform_integration_service.py | `eco-backend/app/services/platform_integration_service.py` | 平台集成服务（硬约束版） |
| approval_service.py | `eco-backend/app/closed_loop/governance/approval_service.py` | 审批服务 |
| forecast_engine.py | `eco-backend/app/closed_loop/capacity/forecast_engine.py` | 预测引擎 |
| rule_engine.py | `eco-backend/app/closed_loop/rules/rule_engine.py` | 规则引擎（集成审批） |
| state_store.py | `app/closed_loop/core/state_store.py` | 状态存储 |

---

## 九、统计

| 类别 | 数量 |
|------|------|
| 治理文件 | 4 |
| CI 工作流 | 4 |
| 脚本 | 8 |
| 测试 | 8 |
| 依赖文件 | 2 |
| 契约 | 1 |
| 报告文档 | 10+ |
| 核心代码 | 5+ |

---

## 十、关键指标

| 指标 | 数值 |
|------|------|
| 守卫测试 | 20 个 |
| 反向测试 | 13 个 |
| 状态机测试 | 62 个 |
| 契约验证测试 | 4 个 |
| CI job 数 | 8 个 |
| 硬约束检查项 | 9 项 |

---

**状态**: ✅ 全部完成
