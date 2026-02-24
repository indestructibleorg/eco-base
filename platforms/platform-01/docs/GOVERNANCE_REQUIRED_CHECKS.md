# Governance Required Checks（不可繞過）

本 repo 的 main 分支合併必須滿足：

## Required Status Checks（必須全部綠）

- `Hard Constraints / hard-constraints (ubuntu-latest)`
- `Hard Constraints / drift-check (ubuntu-latest)`
- `Hard Constraints / state-machine-tests (ubuntu-latest)`
- `Hard Constraints / contract-tests (ubuntu-latest)`
- `Hard Constraints / guard-tests (ubuntu-latest)`

## Required Reviews（必須滿足）

- 需要 Code Owner Review（CODEOWNERS）
- 至少 1 個 Approving Review
- Last push 之後需要重新批准（require_last_push_approval）

## Branch Protection（必須滿足）

- `enforce_admins = true`（管理員不可繞過）
- `strict status checks = true`（必須基於最新 main）
- `allow_force_pushes = false`
- `allow_deletions = false`
- `required_conversation_resolution = true`
- `required_linear_history = true`

## 文件清單

| 文件 | 作用 |
|------|------|
| `.github/CODEOWNERS` | 鎖住關鍵文件所有權 |
| `.github/required-checks.json` | Required checks 清單 |
| `.github/branch-protection.json` | 分支保護規則宣告 |
| `scripts/enforce_branch_protection.sh` | 套用分支保護 |
| `scripts/verify_branch_protection.sh` | 驗證漂移 |
| `.github/workflows/governance-guardrails.yml` | CI 治理護欄 |

## 初始化步驟

```bash
# 1. 安裝 gh CLI 並登錄
gh auth login

# 2. 套用分支保護
bash scripts/enforce_branch_protection.sh .github/branch-protection.json .github/required-checks.json

# 3. 驗證
bash scripts/verify_branch_protection.sh .github/branch-protection.json .github/required-checks.json
```

## 硬規範（Hard Rules）

### 1. 分支保護變更流程

> ⚠️ **任何分支保護變更必須走 PR + 平台/安全雙簽核**

- 禁止在 CI 自動修復分支保護（避免供應鏈風險把 admin 權限交給 runner）
- `scripts/enforce_branch_protection.sh` 的寫入權限只留給 **人工操作**
- `drift-check` 只負責 **檢測漂移並 fail**，不自動修復

### 2. 關鍵文件變更審核

以下文件變更必須經過 CODEOWNERS（`@platform-team` / `@security-team`）審核：

- `.github/workflows/*`
- `.github/branch-protection.json`
- `.github/required-checks.json`
- `hard_constraints_check.sh`
- `scripts/enforce_branch_protection.sh`
- `scripts/verify_branch_protection.sh`
- `requirements-dev.lock`
- `/tests/e2e/*`

## 不可繞過驗收清單（必須演練並存證）

| 驗收項 | 操作 | 期望結果 | 存證方式 |
|--------|------|----------|----------|
| 1 | 嘗試直接 push 到 main | 被拒絕 | 截圖 |
| 2 | PR 修改 `hard_constraints_check.sh` | 要求 CODEOWNERS approve | 截圖 |
| 3 | PR 刻意讓 pytest fail | Required checks 擋下來 | CI 失敗紀錄 |
| 4 | 在 Repo Settings 手動關閉 `enforce_admins` | `drift-check` fail | CI 失敗紀錄 + diff 輸出 |
| 5 | Required checks 改名/刪除 | `drift-check` fail | CI 失敗紀錄 + diff 輸出 |

## 漂移檢測

### 檢測內容

- `enforce_admins` 是否被關閉
- `require_code_owner_reviews` 是否被關閉
- `required_approving_review_count` 是否被降低
- `require_last_push_approval` 是否被關閉
- Required checks 列表是否被修改

### 檢測頻率

- 每次 PR 都檢測
- 每天定時檢測 (cron: `0 2 * * *`)

### 檢測行為

- **發現漂移 → CI fail**（不自動修復）
- 人工介入 → PR 修復 → 重新驗證
