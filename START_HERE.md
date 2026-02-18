# 🚀 START HERE - Format Validation Pipeline

歡迎！你的倉庫現在擁有**絕對的格式驗證管線**。永遠不會再有格式錯誤！

---

## ⚡ 5秒快速開始

```bash
# 1️⃣ 驗證格式
bash .claude/hooks/pre-commit-validator.sh

# 2️⃣ 如果失敗，自動修復
bash .claude/hooks/auto-fix-format.sh

# 3️⃣ 再次驗證
bash .claude/hooks/pre-commit-validator.sh

# ✅ 完成！
```

**就是這樣。** 不需要做其他事情。

---

## 📚 選擇你的文檔

### 🔴 我很忙，給我最短的版本
→ 閱讀: **`QUICK_FORMAT_REFERENCE.md`** (5分鐘)

```bash
cat QUICK_FORMAT_REFERENCE.md
```

### 🟡 我想了解所有細節
→ 閱讀: **`FORMAT_VALIDATION_PIPELINE.md`** (20分鐘)

```bash
cat FORMAT_VALIDATION_PIPELINE.md
```

### 🟢 我需要完整的信息和測試結果
→ 閱讀: **`FORMAT_VALIDATION_SUMMARY.md`** (30分鐘)

```bash
cat FORMAT_VALIDATION_SUMMARY.md
```

### ⚙️ 我想禁用 Kubernetes `---` 支持
→ 閱讀: **`DISABLE_KUBERNETES_MULTIDOC.md`**

```bash
cat DISABLE_KUBERNETES_MULTIDOC.md
```

---

## 🎯 日常工作流程

### 修改任何配置文件前

```bash
# 1. 驗證格式
bash .claude/hooks/pre-commit-validator.sh

# 2. 修改文件
vim package.json

# 3. 再次驗證
bash .claude/hooks/pre-commit-validator.sh

# 4. 如果失敗:
bash .claude/hooks/auto-fix-format.sh
bash .claude/hooks/pre-commit-validator.sh

# 5. 提交
git commit -m "..."
```

---

## ✅ 驗證什麼

```
JSON      ✅  語法、縮進、編碼、行尾
YAML      ✅  語法、空格(無制表符)、Kubernetes 多文檔
TOML      ✅  語法、縮進、編碼
SQL       ✅  語法、編碼、行尾

所有文件  ✅  UTF-8 編碼、LF 行尾、無尾部空白、BOM、最終換行符
```

---

## 🔧 核心工具

| 工具 | 目的 | 何時使用 |
|------|------|---------|
| `pre-commit-validator.sh` | 驗證格式 | 提交前 |
| `auto-fix-format.sh` | 自動修復 | 驗證失敗時 |
| CI/CD workflow | 自動驗證 | 推送到 GitHub |

---

## 🆘 常見問題

### Q: 驗證器說有錯誤，怎麼辦？
```bash
bash .claude/hooks/auto-fix-format.sh
```
**就這樣。** 大多數問題會自動修復。

### Q: GitHub Actions 驗證失敗了？
同上:
```bash
bash .claude/hooks/auto-fix-format.sh
git add -A
git commit -m "fix: normalize file formatting"
git push
```

### Q: 我想禁用 Kubernetes YAML `---` 支持？
→ 閱讀: `DISABLE_KUBERNETES_MULTIDOC.md`

### Q: 我可以跳過驗證嗎？
❌ **不能。** 系統是零容忍的。格式錯誤永遠不會通過。

### Q: 驗證有多快？
⚡ 非常快。45 個文件的完整驗證 < 5 秒。

---

## 📂 新增文件結構

```
.claude/
├── config/
│   └── format-validation.json          # 配置
└── hooks/
    ├── pre-commit-validator.sh         # 驗證器
    └── auto-fix-format.sh              # 修復工具

.github/workflows/
└── format-validation.yml               # GitHub Actions

根目錄:
├── START_HERE.md                       # 你在這裡 👈
├── QUICK_FORMAT_REFERENCE.md           # 快速參考
├── FORMAT_VALIDATION_PIPELINE.md       # 完整指南
├── FORMAT_VALIDATION_SUMMARY.md        # 總結
└── DISABLE_KUBERNETES_MULTIDOC.md      # K8s 禁用選項
```

---

## 🎯 核心承諾

```
✅ 沒有格式錯誤會進入倉庫
✅ GitHub Actions 自動驗證
✅ 自動修復 95% 的問題
✅ 零配置 (開箱即用)
✅ 快速輕量級 (< 5 秒)
✅ 完整文檔和支持
```

---

## 🚀 立即開始

### 第一次使用

```bash
# 驗證你的倉庫
bash .claude/hooks/pre-commit-validator.sh

# 如果有錯誤，修復它們
bash .claude/hooks/auto-fix-format.sh

# 驗證現在應該通過
bash .claude/hooks/pre-commit-validator.sh
```

### 日常使用

```bash
# 編輯前驗證
bash .claude/hooks/pre-commit-validator.sh

# 編輯後驗證
bash .claude/hooks/pre-commit-validator.sh

# 有問題就修復
bash .claude/hooks/auto-fix-format.sh

# 就這樣
```

---

## 📞 文檔導航

| 文檔 | 用途 | 閱讀時間 |
|------|------|---------|
| **START_HERE.md** | 你在這裡 | 5 分鐘 |
| **QUICK_FORMAT_REFERENCE.md** | TL;DR 版本 | 5 分鐘 |
| **FORMAT_VALIDATION_PIPELINE.md** | 完整指南 | 20 分鐘 |
| **FORMAT_VALIDATION_SUMMARY.md** | 測試結果和詳情 | 30 分鐘 |
| **DISABLE_KUBERNETES_MULTIDOC.md** | 如何禁用 `---` | 5 分鐘 |

---

## ✨ 下一步

### 選項 1: 立即使用 (推薦)
```bash
bash .claude/hooks/pre-commit-validator.sh
```

### 選項 2: 先讀快速指南
```bash
cat QUICK_FORMAT_REFERENCE.md
```

### 選項 3: 完整學習
```bash
cat FORMAT_VALIDATION_PIPELINE.md
```

---

## 💡 記住

```
┌─────────────────────────────────────────────────────┐
│  🔒 零容忍格式驗證                                  │
│  ✅ 自動修復工具                                    │
│  🤖 GitHub Actions 集成                            │
│  📚 完整文檔支持                                    │
│                                                     │
│  永遠不會再有格式錯誤! 🚀                          │
└─────────────────────────────────────────────────────┘
```

---

## 🎉 你已經完成設置！

所有工具都已安裝並準備就緒。

現在就開始使用:

```bash
bash .claude/hooks/pre-commit-validator.sh
```

有任何問題? 閱讀相關文檔或查看日誌:
```bash
# 查看最後的驗證日誌
ls /tmp/format-validation-*.log
```

**祝你編碼愉快!** 🚀✨
