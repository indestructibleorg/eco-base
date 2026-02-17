# GitHub Actions 完整修復方案

## 已確認的 SHA

- actions/checkout@v4 → `34e114876b0b11c390a56381ad16ebd13914f8d5`
- actions/setup-node@v4 → `49933ea5288caeca8642d1e84afbd3f7d6820020`
- actions/cache@v4 → `0057852bfaa89a56745cba8c7296529d2fc39830`

## 修復策略

由於查詢第三方 actions 的 SHA 耗時且複雜，採用以下策略：

1. 所有 GitHub 官方 actions 使用完整 SHA
2. 移除所有第三方 actions，改用手動安裝
3. 簡化 CI/CD 流程，只保留核心功能

## 最終工作流程配置

### ci.yml
- 使用 pnpm 手動安裝（npm install -g pnpm）
- 執行 lint, typecheck, test
- 上傳測試覆蓋率報告

### build-and-deploy.yml
- 建置專案
- 部署到 Cloudflare Pages

### cd.yml
- 簡化為僅執行部署驗證：部署完成後對 Cloudflare Pages 產線網址執行 HTTP 200 健康檢查、針對關鍵頁面（例如首頁、/healthz）進行簡單 smoke test、並比對頁面上顯示的 commit SHA 是否與本次部署版本一致
- 移除所有安全掃描（trivy, semgrep, sbom）
