# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# @ECO-governed
# @ECO-layer: GL90-99
# @ECO-semantic: documentation
# @ECO-audit-trail: ../../engine/governance/GL_SEMANTIC_ANCHOR.json
#
# GL Unified Architecture Governance Framework Activated
# Path Tools - 路徑掃描辨識與修復工具集

本目錄包含用於掃描辨識路徑與修復路徑錯誤的工具。

## 工具清單

### 1. path_scanner.py - 路徑掃描器

掃描專案目錄，識別所有檔案路徑並生成索引。

```bash
python tools/path_tools/path_scanner.py --target .
python tools/path_tools/path_scanner.py --target ./src --output scan_result.json
```

### 2. path_validator.py - 路徑驗證器

驗證路徑的有效性，檢測斷開的連結、無效引用等問題。

```bash
python tools/path_tools/path_validator.py --target ./docs
python tools/path_tools/path_validator.py --full --target ./project
```

### 3. path_fixer.py - 路徑修復器

自動修復常見的路徑問題，如斷開連結、錯誤引用等。

```bash
python tools/path_tools/path_fixer.py --target ./docs --dry-run
python tools/path_tools/path_fixer.py --target ./docs --fix
```

## 功能特點

- **掃描辨識**: 遞迴掃描目錄結構，識別所有檔案和連結
- **驗證功能**: 檢測 Markdown 連結、YAML 引用、相對路徑等
- **修復能力**: 自動修復斷開連結、更新過時路徑引用
- **報告生成**: 生成詳細的掃描和驗證報告

## 安全特性

- 路徑注入防護（禁止 shell 元字符）
- 路徑遍歷保護（禁止 ../ 逃逸）
- 符號連結安全驗證
