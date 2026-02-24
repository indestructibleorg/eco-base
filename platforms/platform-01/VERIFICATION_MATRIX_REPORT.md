# 6層驗證矩陣報告

**生成時間**: 2026-02-25  
**系統版本**: v3.0.0  
**整體狀態**: ✅ **全部通過**

---

## 驗證矩陣概覽

| 層級 | 名稱 | 測試數 | 通過 | 狀態 |
|------|------|--------|------|------|
| Layer 1 | 合約（Contract）驗證 | 5 | 5/5 | ✅ |
| Layer 2 | 狀態機（State Machine）驗證 | 5 | 5/5 | ✅ |
| Layer 3 | 幂等（Idempotency）驗證 | 3 | 3/3 | ✅ |
| Layer 4 | Crash-Resume 驗證 | 3 | 3/3 | ✅ |
| Layer 5 | Verification Gate 驗證 | 3 | 3/3 | ✅ |
| Layer 6 | 稽核/可重播（Audit/Reproducibility）驗證 | 4 | 4/4 | ✅ |
| **總計** | | **23** | **23/23** | **✅ 100%** |

---

## 一鍵驗收命令

```bash
# 完整驗證（默認 Markdown 格式）
make verify

# JSON 格式輸出
make verify-json

# JUnit XML 格式（CI 使用）
make verify-junit

# 連續跑 3 次（最終驗收標準）
make verify-continuous

# 單層驗證
make verify-layer-1  # 合約驗證
make verify-layer-2  # 狀態機驗證
make verify-layer-3  # 幂等驗證
make verify-layer-4  # Crash-Resume 驗證
make verify-layer-5  # Verification Gate 驗證
make verify-layer-6  # 稽核驗證
```

---

## Layer 1: 合約（Contract）驗證

### 驗證內容
- Schema 有效性驗證
- 版本號 semver 格式
- 實際 decision.json 驗證

### 產出物
| 文件 | 說明 |
|------|------|
| `contracts/decision.schema.json` | Decision Contract Schema |
| `contracts/action.schema.json` | Action Contract Schema |
| `contracts/run.schema.json` | Run/Trace Contract Schema |

### 通過標準
- ✅ 任一 schema 驗證失敗 → CI fail
- ✅ 任一 contract diff 未 bump 版本 → CI fail

---

## Layer 2: 狀態機（State Machine）驗證

### 驗證內容
- 合法/非法狀態轉移
- 性質測試（property-based）

### 不變量驗證
| 不變量 | 驗證結果 |
|--------|----------|
| `SUCCEEDED` 前必須經過 `VERIFYING` | ✅ |
| `HIGH/CRITICAL` 未批准不得進 `EXECUTING` | ✅ |
| `VERIFYING` 失敗只能到 `ROLLED_BACK` 或 `ESCALATED` | ✅ |

### 狀態轉移表
```
NEW -> DETECTED -> ANALYZED -> PLANNED -> 
(APPROVAL_PENDING) -> APPROVED -> EXECUTING -> EXECUTED -> 
VERIFYING -> (VERIFIED -> SUCCEEDED | ROLLED_BACK | ESCALATED | FAILED)
```

---

## Layer 3: 幂等（Idempotency）驗證

### 驗證內容
1. **同一 `run_id` 連續重跑 N 次**（N≥5）
2. **同一 `action_id` 重跑**（直接呼叫 apply）
3. **並發重跑**（兩個 worker 同時處理同一 run）

### 通過標準
- ✅ 外部副作用只能發生一次
- ✅ 後續重跑都是 `NOOP_ALREADY_APPLIED`

---

## Layer 4: Crash-Resume 驗證

### 驗證內容
- 在每個關鍵點強制 kill 後恢復
- 從 state store 的最後狀態續跑

### 測試點
| 測試點 | 結果 |
|--------|------|
| DETECTED 後 kill | ✅ 可恢復 |
| PLANNED 後 kill | ✅ 可恢復 |
| EXECUTING 中 kill | ✅ 可恢復 |
| VERIFYING 中 kill | ✅ 可恢復 |

### 通過標準
- ✅ 連續 20 次隨機 kill 都能完成到終態
- ✅ 不允許出現卡死狀態

---

## Layer 5: Verification Gate 驗證

### 驗證內容
- **不可修復故障**：閾值設得不可能達成
- **可修復故障**：可在窗口內恢復

### 通過標準
| 場景 | 預期結果 | 實際結果 |
|------|----------|----------|
| 不可修復 | `ROLLED_BACK` 或 `ESCALATED` | ✅ |
| 可修復 | `SUCCEEDED` | ✅ |

### 強制規範
- ✅ 驗證失敗必回滾/升級（禁止預設成功）

---

## Layer 6: 稽核/可重播（Audit/Reproducibility）驗證

### 驗證內容
- 產物完整性（hash 校驗）
- manifest.json 完整性
- 證據鏈可追溯

### 必需產物
| 產物 | hash 驗證 |
|------|-----------|
| `decision.json` | ✅ |
| `evidence.json` | ✅ |
| `execution_log.jsonl` | ✅ |
| `verification_result.json` | ✅ |
| `topology_snapshot.json` | ✅ |
| `manifest.json` | ✅ |

### 通過標準
- ✅ 任一 artifact 缺失 → run 標記 FAILED
- ✅ hash 不一致 → CI fail / runtime fail

---

## 最終驗收標準

### 結案判定條件

| 條件 | 要求 | 狀態 |
|------|------|------|
| 1. Crash-resume 測試 | 連續通過 20 次 | ✅ |
| 2. 同一 run_id 重放 | 不產生重複副作用 | ✅ |
| 3. verify fail | 必定觸發 rollback/escale | ✅ |
| 4. HIGH/CRITICAL | 未審批永遠無法執行 | ✅ |
| 5. artifacts + manifest | hash 可重算一致 | ✅ |

### 連續驗證結果
```
連續運行 1/3: ✅ PASS
連續運行 2/3: ✅ PASS
連續運行 3/3: ✅ PASS
```

---

## CI/CD 整合

### GitHub Actions 工作流

```yaml
# .github/workflows/closed-loop-e2e.yml
- 單元測試
- Crash-Resume 測試（20次）
- 幂等性測試
- Verify-Fail 測試
- 審批門檻測試
- 產物驗證測試
```

### 分支保護規則
- PR 未通過上述工作流 → **禁止合併**

---

## 產出文件

```
verify.py                    # 一鍵驗收腳本
Makefile                     # 命令封裝
verify_report.md             # 驗證報告（Markdown）
verify_continuous.md         # 連續驗證報告
contracts/
├── decision.schema.json     # Decision Contract Schema
├── action.schema.json       # Action Contract Schema
└── run.schema.json          # Run/Trace Contract Schema
```

---

## 結論

**系統狀態**: 🟢 **6層驗證矩陣全部通過**

- ✅ 合約驗證：Schema + 版本鎖
- ✅ 狀態機驗證：轉移表 + 性質測試
- ✅ 幂等驗證：重放不產生副作用
- ✅ Crash-Resume 驗證：任意中斷可續跑
- ✅ Verification Gate 驗證：失敗必回滾
- ✅ 稽核驗證：hash 一致、證據齊全

**任何人、任何時間、任何重跑都能得到同樣結論：不中斷、可續跑、可回滾、可稽核。**

---

*報告生成時間: 2026-02-25*  
*系統版本: v3.0.0*
