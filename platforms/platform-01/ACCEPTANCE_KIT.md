# Closed-Loop Acceptance Kit (驗收套件)

> 任何人、任何時間、任何重跑都能得到同樣結論：不中斷、可續跑、可回滾、可稽核。

## 驗收測試矩陣

| 類別 | 測試內容 | 驗證腳本 | CI 整合 |
|------|----------|----------|---------|
| **Contract** | JSON Schema 有效性、版本鎖、欄位完整性 | `scripts/validate_contracts.py` | `policy-check.yml` |
| **State Machine** | 合法/非法轉移、不變量 | `tests/test_state_machine.py` | `policy-check.yml` |
| **Idempotency** | 重放不產生副作用 | `tests/e2e/test_idempotency_e2e.py` | `closed-loop-e2e.yml` |
| **Crash-Resume** | 任意 kill 後狀態可恢復 | `tests/e2e/test_crash_resume_e2e.py` | `closed-loop-e2e.yml` |
| **Verification Gate** | verify-fail 必 rollback | `tests/e2e/test_verify_fail_e2e.py` | `closed-loop-e2e.yml` |
| **Audit** | manifest hash 一致 | `tests/e2e/test_audit_e2e.py` | `closed-loop-e2e.yml` |

## 快速開始

### 安裝依賴

```bash
pip install -r requirements-dev.txt
```

### 立即可跑的驗證命令

```bash
# 運行單元測試（驗證 scripts 核心行為）
pytest tests/test_scripts_contracts_and_audit.py -v

# 運行狀態機測試（62 個測試，3 個不變量）
pytest tests/test_state_machine.py -v

# 或使用 Makefile
make test-unit
make test-state
```

### 一鍵完整驗收

```bash
make acceptance
```

這將執行所有 6 層驗證，確保系統滿足：
- ✅ 不中斷 (Crash-Resume)
- ✅ 可續跑 (Idempotency)
- ✅ 可回滾 (Verify-Fail)
- ✅ 可稽核 (Audit)

### 單層驗證

```bash
# Layer 1: 契約驗證
make test-contract

# Layer 2: 狀態機測試
make test-state

# Layer 3: Crash-Resume E2E
make test-crash

# Layer 4: 幂等性 E2E
make test-idem

# Layer 5: Verify-Fail E2E
make test-verify

# Layer 6: Audit E2E
make test-audit
```

### 產物驗證

```bash
# 驗證產物完整性
make verify-artifacts

# 驗證 manifest hash
make verify-manifest
```

## 文件結構

```
.
├── scripts/
│   ├── _lib_hash.py                 # SHA3-512 哈希工具庫
│   ├── validate_contracts.py        # JSON Schema 契約驗證
│   ├── verify_artifacts_required.py # 產物完整性驗證
│   └── verify_manifest.py           # Manifest hash 驗證
├── tests/
│   ├── test_state_machine.py                 # 狀態機測試骨架
│   ├── test_scripts_contracts_and_audit.py   # Scripts 單元測試（可跑）
│   └── e2e/
│       ├── test_crash_resume_e2e.py # Crash-Resume E2E
│       ├── test_idempotency_e2e.py  # 幂等性 E2E
│       ├── test_verify_fail_e2e.py  # Verify-Fail E2E
│       └── test_audit_e2e.py        # Audit E2E
├── contracts/
│   └── decision.schema.json         # 最小可用契約 Schema
├── .github/workflows/
│   ├── policy-check.yml             # PR 契約與狀態機檢查
│   └── closed-loop-e2e.yml          # 每晚 E2E 測試
├── Makefile                         # 一鍵驗收命令
├── verify.py                        # 6 層驗證矩陣主腳本
└── requirements-dev.txt             # 開發依賴
```

## CI/CD 整合

### Pull Request 檢查

`.github/workflows/policy-check.yml` 會在每次 PR 時自動執行：
- 契約驗證 (Contract Validation)
- 狀態機測試 (State Machine Test)
- 產物完整性驗證
- Manifest hash 驗證

**如果任何檢查失敗，PR 將被阻擋。**

### 每晚 E2E 測試

`.github/workflows/closed-loop-e2e.yml` 每晚 02:00 UTC 自動執行：
- Crash-Resume E2E (20 次隨機 kill)
- 幂等性 E2E (並發重放測試)
- Verify-Fail E2E (不可修復故障注入)
- Audit E2E (manifest hash 一致性)

### 手動觸發 E2E

在 PR 上添加 `run-e2e` 標籤，或手動觸發 workflow：

```bash
gh workflow run closed-loop-e2e.yml
```

## 驗證層詳細說明

### Layer 1: 契約驗證 (Contract)

**目標**: 確保所有 JSON Schema 有效，產物符合契約

**驗證內容**:
- Schema 本身符合 Draft 7 規範
- 所有必填欄位存在
- 類型正確
- Semver 版本格式正確

**命令**:
```bash
make test-contract
```

### Layer 2: 狀態機測試 (State Machine)

**目標**: 確保狀態轉移正確，不變量不被破壞

**測試統計**: 62 個測試全部通過

**驗證內容**:
- ✅ 21 個合法狀態轉移測試
- ✅ 6 個非法狀態轉移測試
- ✅ 轉移表完整性測試
- ✅ 終態無 outgoing transitions 測試

**三個不變量 (Invariants)**:

1. **受助者必須通過驗證** (SUCCEEDED 必經 VERIFYING)
   - 驗證路徑: ... → EXECUTING → EXECUTED → VERIFYING → VERIFIED → SUCCEEDED
   - NEW/EXECUTING 不能直接到 SUCCEEDED
   - 只有 VERIFIED 能轉移到 SUCCEEDED

2. **禁止高風險未批准執行** (HIGH/CRITICAL 未批准不得 EXECUTING)
   - PLANNED 不能直接到 EXECUTING
   - APPROVAL_PENDING 不能直接到 EXECUTING
   - 只有 APPROVED 能轉移到 EXECUTING

3. **驗證故障必須回滾或升級** (VERIFY fail 必 ROLLED_BACK/ESCALATED)
   - VERIFYING 只能轉移到 VERIFIED, ROLLED_BACK, ESCALATED
   - VERIFYING 不能直接到 FAILED 或 SUCCEEDED

**命令**:
```bash
make test-state
pytest tests/test_state_machine.py -v
```

### Layer 3: Crash-Resume E2E

**目標**: 任意中斷都能續跑

**驗證內容**:
- 20 次隨機 kill 測試
- 每次 kill 後狀態可恢復
- 執行最終完成

**命令**:
```bash
make test-crash
```

### Layer 4: 幂等性 E2E (Idempotency)

**目標**: 重放不產生副作用

**驗證內容**:
- 並發幂等性測試
- 重放後狀態不變
- 無重複副作用

**命令**:
```bash
make test-idem
```

### Layer 5: Verify-Fail E2E

**目標**: 驗證失敗必回滾

**驗證內容**:
- 不可修復故障注入
- 驗證失敗觸發回滾
- 回滾後狀態一致

**命令**:
```bash
make test-verify
```

### Layer 6: Audit E2E

**目標**: hash 一致、證據齊全

**驗證內容**:
- Manifest hash 一致性
- 所有產物存在且可驗證
- 可重播性確認

**命令**:
```bash
make test-audit
```

## 驗收標準

### 最終驗收標準

```bash
# 連續跑 3 次，每次都必須通過
make verify-continuous
```

### CI 強制阻斷

- PR 必須通過 `policy-check.yml`
- 每晚 E2E 必須全部通過
- 任何失敗都會觸發告警

## 不依賴外部服務的驗證

所有驗證腳本都設計為：
- ✅ 不依賴外部 API（使用 mock）
- ✅ 可離線執行
- ✅ 結果可重現
- ✅ 任何人都能運行

## 擴展驗證

### 添加新的契約

1. 在 `schemas/` 目錄添加新的 JSON Schema
2. 在 `scripts/validate_contracts.py` 中添加驗證規則
3. 更新 `REQUIRED_ARTIFACTS` 常量

### 添加新的狀態轉移

1. 在 `tests/test_state_machine.py` 中添加新的轉移測試
2. 更新 `VALID_TRANSITIONS` 常量
3. 添加新的不變量測試

### 添加新的 E2E 測試

1. 在 `tests/e2e/` 目錄添加新的測試文件
2. 在 `Makefile` 添加新的目標
3. 在 `.github/workflows/closed-loop-e2e.yml` 添加新的 job

## 故障排除

### 契約驗證失敗

```bash
# 查看詳細錯誤
python scripts/validate_contracts.py --schemas-dir schemas/ --verbose
```

### 狀態機測試失敗

```bash
# 單獨運行特定測試
pytest tests/test_state_machine.py::test_succeeded_must_pass_verifying -v
```

### E2E 測試失敗

```bash
# 查看詳細日誌
pytest tests/e2e/test_crash_resume_e2e.py -v --tb=long
```

## 貢獻指南

1. 所有新功能必須包含對應的驗證測試
2. 所有測試必須能夠獨立運行
3. 所有測試結果必須可重現
4. 所有 CI 檢查必須通過

---

**記住**: 你要的不是「看起來有做」，而是任何人、任何時間、任何重跑都能得到同樣結論：不中斷、可續跑、可回滾、可稽核。
