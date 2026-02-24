# 平台適配器映射對照表

本文件提供希臘字母適配器與實際第三方平台的映射關係（僅供內部參考）。

## 能力領域與適配器總覽

| 能力領域 | 適配器數量 | 希臘字母標識 |
|---------|-----------|-------------|
| 數據持久化 | 2 | Alpha, Beta |
| 認知計算 | 3 | Gamma, Delta, Epsilon |
| 代碼工程 | 3 | Zeta, Eta, Theta |
| 協作通信 | 2 | Iota, Kappa |
| 視覺設計 | 2 | Lambda, Mu |
| 知識管理 | 2 | Nu, Xi |
| 部署交付 | 3 | Omicron, Pi, Rho |
| 學習教育 | 3 | Sigma, Tau, Upsilon |

**總計: 20個適配器，覆蓋30+第三方平台能力**

## 詳細映射表

### 數據持久化 (Data Persistence)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Alpha | PostgreSQL + 實時訂閱 | 查詢、變更、訂閱、向量搜索 |
| Beta | 分散式數據庫 | 查詢、分片、零停機遷移 |

### 認知計算 (Cognitive Computing)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Gamma | 多模型聚合平台 | 生成、流式、函數調用、嵌入、多模態 |
| Delta | 企業級推理模型 | 長上下文、代碼、推理 |
| Epsilon | 開源推理模型 | 生成、推理、代碼、成本效益 |

### 代碼工程 (Code Engineering)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Zeta | AI 驅動代碼編輯器 | 補全、解釋、重構、審查、測試生成 |
| Eta | 代碼審查與 CI 自動化 | 安全審查、性能分析、風格檢查 |
| Theta | 雲端 IDE | 補全、協作編程、測試生成 |

### 協作通信 (Collaboration)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Iota | 團隊即時通訊 | 消息、頻道、摘要、工作流 |
| Kappa | 代碼託管與協作 | PR、Issue、代碼搜索、Actions |

### 視覺設計 (Visual Design)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Lambda | 雲端協作設計工具 | 組件庫、導出、AI生成、檢視、原型 |
| Mu | 原生設計工具 | 組件庫、導出、檢視 |

### 知識管理 (Knowledge Management)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Nu | 多功能筆記平台 | 文檔創建、查詢、導出 |
| Xi | Git 驅動文檔平台 | Git 同步、導出、版本控制 |

### 部署交付 (Deployment)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Omicron | 前端部署平台 | 構建、部署、預覽、回滾 |
| Pi | 構建加速平台 | 遠程構建、緩存、並行 |
| Rho | 基礎設施即代碼平台 | Plan、Apply、狀態管理 |

### 學習教育 (Learning)

| 適配器 | 平台類型 | 核心能力 |
|--------|---------|---------|
| Sigma | 互動式程式學習 | 學習路徑、練習提交、提示、進度追蹤 |
| Tau | 雲端 IDE 學習 | 課程、代碼運行、協作 |
| Upsilon | 前端實驗平台 | 代碼展示、分享、協作 |

## 環境變數配置清單

```bash
# 數據持久化
ALPHA_URL=
ALPHA_ANON_KEY=
ALPHA_SERVICE_KEY=
BETA_HOST=
BETA_USERNAME=
BETA_PASSWORD=
BETA_DATABASE=

# 認知計算
GAMMA_API_KEY=
DELTA_API_KEY=
EPSILON_API_KEY=
EPSILON_BASE_URL=

# 代碼工程
ZETA_API_KEY=
ETA_API_KEY=
THETA_API_KEY=

# 協作通信
IOTA_BOT_TOKEN=
KAPPA_TOKEN=

# 視覺設計
LAMBDA_TOKEN=
MU_TOKEN=

# 知識管理
NU_TOKEN=
XI_TOKEN=

# 部署交付
OMICRON_TOKEN=
OMICRON_TEAM_ID=
PI_TOKEN=
PI_ORG_ID=
RHO_TOKEN=
RHO_ORG_ID=

# 學習教育
SIGMA_API_KEY=
TAU_API_KEY=
UPSILON_API_KEY=
```

## 使用建議

1. **按需初始化**: 只初始化需要的適配器，避免不必要的資源消耗
2. **配置驗證**: 使用 `validate_config()` 函數驗證配置完整性
3. **健康檢查**: 定期執行 `health_check()` 監控適配器狀態
4. **錯誤處理**: 所有操作返回 `OperationResult`，請檢查 `success` 字段
5. **上下文傳遞**: 使用 `CapabilityContext` 傳遞請求追蹤信息
