# Eco-Backend

Eco-Platform 生產級企業後端服務

## 功能特性

- **認證授權**: JWT 令牌認證、API 密鑰管理
- **第三方平台集成**: 20+ 適配器覆蓋 30+ 平台
- **能力領域**: 數據持久化、認知計算、代碼工程、協作通信、視覺設計、知識管理、部署交付、學習教育
- **企業級特性**: 限流、熔斷、健康檢查、監控指標
- **異步支持**: 基於 FastAPI 和 SQLAlchemy 2.0 異步
- **容器化部署**: Docker 和 Kubernetes 支持

## 項目結構

```
eco-backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/      # API 端點
│   │       └── router.py       # 路由聚合
│   ├── core/
│   │   ├── config.py           # 配置管理
│   │   ├── exceptions.py       # 異常處理
│   │   ├── logging.py          # 日誌配置
│   │   └── security.py         # 安全相關
│   ├── db/
│   │   └── base.py             # 數據庫基礎
│   ├── models/                 # 數據庫模型
│   ├── schemas/                # Pydantic 模型
│   └── main.py                 # 應用入口
├── deployments/
│   ├── docker/                 # Docker 配置
│   └── k8s/                    # Kubernetes 配置
├── migrations/                 # 數據庫遷移
├── tests/                      # 測試套件
└── requirements.txt            # 依賴列表
```

## 快速開始

### 本地開發

1. 安裝依賴

```bash
pip install -r requirements.txt
```

2. 配置環境變數

```bash
cp .env.example .env
# 編輯 .env 文件填入配置
```

3. 運行數據庫遷移

```bash
alembic upgrade head
```

4. 啟動服務

```bash
uvicorn app.main:app --reload
```

### Docker 部署

```bash
cd deployments/docker
docker-compose up -d
```

### Kubernetes 部署

```bash
kubectl apply -f deployments/k8s/
```

## API 文檔

啟動服務後訪問:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 能力領域 API

### 認知計算

```bash
# 文本生成
POST /api/cognitive/generate
{
  "prompt": "Explain quantum computing",
  "provider": "gamma-cognitive"
}

# 文本嵌入
POST /api/cognitive/embed
{
  "texts": ["Hello", "World"],
  "provider": "gamma-cognitive"
}
```

### 數據持久化

```bash
# 查詢數據
POST /api/data/query
{
  "table": "users",
  "filters": {"status": "active"},
  "provider": "alpha-persistence"
}

# 向量搜索
POST /api/data/vector-search
{
  "table": "documents",
  "vector": [0.1, 0.2, ...],
  "top_k": 10
}
```

### 代碼工程

```bash
# 代碼補全
POST /api/code/complete
{
  "code": "def fibonacci(n):",
  "language": "python",
  "provider": "zeta-code"
}

# 代碼審查
POST /api/code/review
{
  "code": "...",
  "language": "python",
  "review_type": "security"
}
```

## 測試

```bash
# 運行測試
pytest

# 運行測試並生成覆蓋率報告
pytest --cov=app --cov-report=html
```

## 環境變數

| 變數 | 說明 | 默認值 |
|-----|------|--------|
| `ENVIRONMENT` | 運行環境 | `production` |
| `DEBUG` | 調試模式 | `false` |
| `DATABASE_URL` | 數據庫連接URL | - |
| `REDIS_URL` | Redis連接URL | - |
| `SECRET_KEY` | JWT密鑰 | - |

## 提供者配置

參見 `.env.example` 了解所有第三方平台配置選項。

## 許可證

MIT
