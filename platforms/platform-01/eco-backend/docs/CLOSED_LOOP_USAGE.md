# 閉環系統使用手冊

## 概述

閉環系統實現了 MAPE-K 自動化循環：
- **Monitor (監控)**: 收集指標和事件
- **Analyze (分析)**: 檢測異常
- **Plan (規劃)**: 評估規則和決策
- **Execute (執行)**: 執行修復動作
- **Knowledge (知識)**: 存儲歷史和經驗

## 快速開始

### 1. 初始化閉環系統

```python
from app.closed_loop import init_closed_loop, ClosedLoopConfig, ClosedLoopMode

# 創建配置
config = ClosedLoopConfig(
    mode=ClosedLoopMode.FULL_AUTO,  # 全自動模式
    detection_interval_seconds=30,   # 檢測間隔
    max_concurrent_remediations=3,   # 最大並發修復
    global_cooldown_minutes=5        # 全局冷卻時間
)

# 初始化並啟動
controller = init_closed_loop(config)
await controller.start()
```

### 2. 運行模式

| 模式 | 說明 | 適用場景 |
|-----|------|---------|
| `MANUAL` | 只檢測不執行 | 開發測試 |
| `SEMI_AUTO` | 需要人工確認 | 生產初期 |
| `FULL_AUTO` | 全自動執行 | 穩定生產 |

### 3. 添加自定義規則

#### 檢測規則

```python
from app.closed_loop import (
    DetectionRule, AnomalyType, Severity
)

# 創建檢測規則
rule = DetectionRule(
    name="high_cpu_detection",
    metric_name="cpu_usage_percent",
    anomaly_type=AnomalyType.THRESHOLD,
    severity=Severity.WARNING,
    threshold_max=80.0,  # CPU 超過 80% 觸發
    cooldown_minutes=10
)

# 註冊規則
controller.add_detection_rule(rule)
```

#### 修復規則

```python
from app.closed_loop import (
    Rule, Action, Condition, ConditionOperator
)

# 創建修復規則
remediation_rule = Rule(
    name="scale_on_high_cpu",
    description="Scale up when CPU is high",
    priority=100,
    condition=Condition(
        field="anomaly.rule_name",
        operator=ConditionOperator.EQ,
        value="high_cpu_detection"
    ),
    actions=[
        Action(
            action_type="scale_up",
            parameters={
                "namespace": "default",
                "deployment": "eco-backend",
                "replicas_delta": 2
            }
        )
    ],
    cooldown_minutes=10
)

# 添加規則
controller.add_remediation_rule(remediation_rule)
```

### 4. 從 YAML 加載規則

```python
# 讀取 YAML 文件
with open("rules.yaml", "r") as f:
    yaml_content = f.read()

# 加載規則
count = controller.load_rules_from_yaml(yaml_content)
print(f"Loaded {count} rules")
```

YAML 格式示例：

```yaml
rules:
  - name: my_rule
    description: My custom rule
    enabled: true
    priority: 100
    cooldown_minutes: 5
    
    condition:
      field: status
      operator: "=="
      value: error
    
    actions:
      - type: clear_cache
        parameters:
          pattern: "*"
        delay_seconds: 0
        require_approval: false
```

### 5. 手動觸發規則

```python
# 手動執行規則
results = await controller.manual_trigger("my_rule")

for result in results:
    print(f"Action: {result.action_type.name}")
    print(f"Status: {result.status.value}")
    print(f"Message: {result.message}")
```

### 6. 獲取系統狀態

```python
# 獲取狀態
status = controller.get_status()

print(f"State: {status['state']}")
print(f"Mode: {status['mode']}")
print(f"Active Remediations: {status['active_remediations']}")
print(f"Detection Rules: {status['detection_rules']}")
print(f"Remediation Rules: {status['remediation_rules']}")
```

### 7. 獲取 Prometheus 指標

```python
# 導出 Prometheus 格式指標
metrics = controller.get_metrics()
print(metrics)
```

輸出示例：

```
# HELP closed_loop_anomaly_detections_total Total number of anomaly detections
# TYPE closed_loop_anomaly_detections_total counter
closed_loop_anomaly_detections_total{metric_name="cpu_usage_percent",anomaly_type="THRESHOLD",severity="warning"} 5

# HELP closed_loop_mttd_seconds Mean Time To Detection
# TYPE closed_loop_mttd_seconds gauge
closed_loop_mttd_seconds 0.523

# HELP closed_loop_mttr_seconds Mean Time To Remediation
# TYPE closed_loop_mttr_seconds gauge
closed_loop_mttr_seconds 4.125
```

## 高級用法

### 自定義異常檢測

```python
from app.closed_loop import StatisticalDetector, TrendDetector

# 統計檢測
data = [10, 11, 9, 10.5, 100]  # 100 是異常
anomalies = StatisticalDetector.detect_sigma(data)
print(anomalies)  # [False, False, False, False, True]

# 趨勢檢測
data = [10] * 10 + [12, 14, 16, 18, 20]  # 上升趨勢
changes = TrendDetector.detect_trend_change(data)
print(changes)  # 趨勢變化點
```

### 自定義修復動作

```python
from app.closed_loop import (
    RemediationActionHandler, RemediationAction, 
    RemediationResult, RemediationType, RemediationStatus
)

class CustomHandler(RemediationActionHandler):
    async def validate(self, action: RemediationAction) -> bool:
        return "param" in action.parameters
    
    async def execute(self, action: RemediationAction) -> RemediationResult:
        # 執行自定義邏輯
        return RemediationResult(
            action_id=str(uuid.uuid4()),
            action_type=RemediationType.RESTART_POD,
            status=RemediationStatus.SUCCESS,
            message="Custom action executed",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
    
    async def rollback(self, action: RemediationAction) -> RemediationResult:
        # 回滾邏輯
        pass

# 註冊處理器
controller.remediator.register_handler(
    RemediationType.RESTART_POD, 
    CustomHandler()
)
```

### 事件回調

```python
# 註冊異常回調
def on_anomaly(anomaly):
    print(f"Anomaly detected: {anomaly.metric_name}")
    # 發送通知等

controller.on_anomaly(on_anomaly)

# 註冊修復回調
def on_remediation(result):
    print(f"Remediation completed: {result.status}")
    # 記錄日誌等

controller.on_remediation(on_remediation)
```

## 配置參數

### ClosedLoopConfig

| 參數 | 類型 | 默認值 | 說明 |
|-----|------|--------|------|
| `mode` | ClosedLoopMode | `SEMI_AUTO` | 運行模式 |
| `detection_interval_seconds` | int | 30 | 檢測間隔（秒） |
| `max_concurrent_remediations` | int | 3 | 最大並發修復數 |
| `global_cooldown_minutes` | int | 5 | 全局冷卻時間（分鐘） |
| `enable_metrics` | bool | True | 啟用指標收集 |
| `enable_events` | bool | True | 啟用事件發布 |

### DetectionRule

| 參數 | 類型 | 說明 |
|-----|------|------|
| `name` | str | 規則名稱 |
| `metric_name` | str | 監控指標名稱 |
| `anomaly_type` | AnomalyType | 異常類型 |
| `severity` | Severity | 嚴重程度 |
| `threshold_min` | float | 最小閾值 |
| `threshold_max` | float | 最大閾值 |
| `sigma_multiplier` | float | 3-sigma 乘數 |
| `cooldown_minutes` | int | 冷卻時間 |

### Rule (修復規則)

| 參數 | 類型 | 說明 |
|-----|------|------|
| `name` | str | 規則名稱 |
| `description` | str | 規則描述 |
| `priority` | int | 優先級（數字越小越高） |
| `condition` | Condition | 觸發條件 |
| `actions` | List[Action] | 執行動作 |
| `cooldown_minutes` | int | 冷卻時間 |

## 最佳實踐

1. **從手動模式開始**: 先在 `MANUAL` 模式下測試規則
2. **逐步過渡**: 確認規則有效後切換到 `SEMI_AUTO`
3. **設置合理的冷卻時間**: 防止頻繁觸發
4. **監控指標**: 持續關注 MTTD/MTTR 等指標
5. **定期審查規則**: 根據運行情況調整規則

## 故障排除

### 規則不觸發

- 檢查規則是否啟用 (`enabled: true`)
- 檢查條件是否正確
- 檢查冷卻時間是否已過

### 修復失敗

- 檢查處理器是否註冊
- 檢查參數是否正確
- 查看日誌獲取詳細錯誤

### 指標不更新

- 確認 `enable_metrics` 為 True
- 檢查指標導出端點
