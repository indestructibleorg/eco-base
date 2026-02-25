# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# FHS Integration Automation System

## 概述

這是一個自動化系統，用於判斷 `workspace/tools/` 中的組件何時成熟到可以整合進 FHS (Filesystem Hierarchy Standard) 主目錄結構。

## 核心功能

### 1. 成熟度評估 (`maturity_assessor.py`)

自動評估組件的成熟度，基於以下標準：

- **代碼品質** (30分): 測試覆蓋率、文檔完整性、代碼風格
- **穩定性** (25分): API 穩定性、Bug 率、發布頻率
- **使用情況** (20分): 生產環境使用、用戶採用率
- **依賴管理** (15分): 依賴穩定性、依賴數量
- **維護狀態** (10分): 積極維護程度

總分 100 分，根據分數分為四個成熟度等級：

- **實驗階段** (0-40分): 留在 `workspace/`
- **開發階段** (41-60分): 留在 `workspace/`
- **穩定階段** (61-80分): 可選擇性整合到 FHS
- **生產就緒** (81-100分): 應該整合到 FHS

### 2. FHS 整合器 (`fhs_integrator.py`)

當組件達到成熟標準時，自動執行 FHS 整合：

- 創建 FHS 目錄結構 (`bin/`, `sbin/`, `lib/`, `etc/`)
- 生成命令包裝器（`mno-*` 命令系列）
- 遷移 Python 庫到 `lib/`
- 遷移配置文件到 `etc/`
- 遷移服務文件到 `etc/systemd/`
- 生成遷移文檔

### 3. 成熟度標準 (`maturity-criteria.yaml`)

YAML 配置文件，定義：

- 詳細的評估標準和閾值
- FHS 目錄映射規則
- 包裝器模板
- 自動整合流程
- 排除列表
- 通知設定

## 使用方法

### 評估單個組件

```bash
# 評估 repository-understanding 組件
python3 workspace/tools/fhs-integration/maturity_assessor.py repository-understanding

# 詳細輸出
python3 workspace/tools/fhs-integration/maturity_assessor.py repository-understanding --verbose

# JSON 格式輸出
python3 workspace/tools/fhs-integration/maturity_assessor.py repository-understanding --json
```

### 評估所有組件

```bash
# 評估所有 workspace/tools/ 下的組件
python3 workspace/tools/fhs-integration/maturity_assessor.py --all

# JSON 格式輸出（便於自動化處理）
python3 workspace/tools/fhs-integration/maturity_assessor.py --all --json
```

### 執行 FHS 整合

```bash
# 乾運行（預覽操作，不實際執行）
python3 workspace/tools/fhs-integration/fhs_integrator.py repository-understanding

# 實際執行整合
python3 workspace/tools/fhs-integration/fhs_integrator.py repository-understanding --execute

# 指定成熟度分數（用於手動覆蓋）
python3 workspace/tools/fhs-integration/fhs_integrator.py repository-understanding --execute --maturity-score 85
```

## 輸出範例

### 成熟度評估報告

```
================================================================================
Component: repository-understanding
================================================================================
Total Score: 85/100 (85.0%)
Maturity Level: PRODUCTION

Detailed Scores:
  code_quality: 25/30
  stability: 22/25
  usage: 18/20
  dependencies: 12/15
  maintenance: 8/10

Recommendation: 應該整合到 FHS 目錄
Priority: high

Next Steps:
  1. 創建 bin/ 或 sbin/ 命令包裝器
  2. 遷移 Python 模組到 lib/
  3. 配置文件移至 etc/
  4. 更新文檔
  5. 創建遷移 PR
```

### FHS 整合結果

```
================================================================================
FHS Integration: repository-understanding
Mode: EXECUTE
================================================================================

✓ Create directory: /path/to/repo/lib/repository-understanding
✓ Create directory: /path/to/repo/etc/repository-understanding
✓ Create wrapper: /path/to/repo/bin/mno-repo-scan
✓ Create wrapper: /path/to/repo/sbin/mno-repo-maintain
✓ Copy library file: phase1_scanner.py -> lib/repository-understanding/phase1_scanner.py
✓ Copy library file: event_driven_system.py -> lib/repository-understanding/event_driven_system.py
✓ Copy config file: config.yaml -> etc/repository-understanding/config.yaml
✓ Copy service file: event-driven-system.service -> etc/systemd/event-driven-system.service
✓ Generate migration documentation: docs/fhs-migration-repository-understanding.md

================================================================================
Integration Summary
================================================================================
Total actions: 15

✓ Integration completed successfully!

Next steps:
  1. Review migrated files
  2. Test the new commands
  3. Update documentation
  4. Commit changes to git
```

## 整合後的目錄結構

```
mno-repository-understanding-system/
├── bin/
│   ├── mno-admin                      # 原有
│   └── mno-repo-scan                  # 新增：用戶命令
│
├── sbin/
│   ├── mno-systemctl                  # 原有
│   └── mno-repo-maintain              # 新增：系統管理命令
│
├── lib/
│   └── repository-understanding/      # 新增：Python 庫
│       ├── __init__.py
│       ├── phase1_scanner.py
│       ├── phase2_operation_checker.py
│       ├── event_driven_system.py
│       └── requirements.txt
│
├── etc/
│   ├── repository-understanding/      # 新增：配置文件
│   │   └── config.yaml
│   └── systemd/
│       └── event-driven-system.service
│
└── workspace/
    └── tools/
        └── repository-understanding/  # 保留（向後兼容）
            └── ...
```

## 命令使用

### 整合前（Workspace 方式）

```bash
cd workspace/tools/repository-understanding
python3 phase1_scanner.py
python3 event_driven_system.py
```

### 整合後（FHS 方式）

```bash
# 用戶命令
mno-repo-scan

# 系統管理命令
sudo mno-repo-maintain

# 查看幫助
mno-repo-scan --help
```

## 自動化工作流程

1. **定期評估**: 使用 cron job 或 GitHub Actions 定期評估所有組件
2. **達到閾值通知**: 當組件達到 80 分時，創建 issue 或發送通知
3. **自動整合**: 對於分數 ≥ 85 的組件，可以自動創建整合 PR
4. **人工審查**: 在合併前進行人工審查和測試
5. **文檔更新**: 自動生成遷移文檔

## 配置

編輯 `maturity-criteria.yaml` 以自定義：

- 評估標準的權重
- 成熟度等級的分數閾值
- FHS 目錄映射規則
- 包裝器腳本模板
- 排除列表

## 最佳實踐

1. **漸進式整合**: 不要一次性整合所有組件，而是逐步整合最成熟的
2. **保持向後兼容**: 保留 workspace 中的原始文件一段時間
3. **測試驗證**: 整合後進行全面測試
4. **文檔更新**: 及時更新使用文檔和範例
5. **通知用戶**: 通過 CHANGELOG 或 release notes 通知使用者

## 依賴

- Python 3.6+
- PyYAML
- Git (用於評估代碼歷史)

安裝依賴：

```bash
pip install -r requirements.txt
```

## 故障排除

### Q: 評估分數不準確

A: 可以通過 `--verbose` 選項查看詳細評估過程，或手動調整 `maturity-criteria.yaml` 中的權重

### Q: 整合失敗

A: 使用 `--dry-run`（默認）先預覽操作，確認無誤後再使用 `--execute` 執行

### Q: 需要自定義包裝器

A: 編輯生成的包裝器腳本，或修改 `maturity-criteria.yaml` 中的 `wrapper_template`

## 未來改進

- [ ] 整合 GitHub issue tracker 以更準確評估 bug 率
- [ ] 添加自動化測試執行和覆蓋率檢測
- [ ] 支持更多語言（不只是 Python）
- [ ] 添加回滾功能
- [ ] 整合 CI/CD 自動觸發評估

## 相關文檔

- [FHS 標準]([EXTERNAL_URL_REMOVED])
- [MachineNativeOps Taxonomy](../../README.md)
- [Repository Understanding System](../repository-understanding/README.md)
