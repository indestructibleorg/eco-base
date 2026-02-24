# Phase 2: 智能閉環詳細企劃

## 執行摘要

Phase 2 將在 Phase 1 基礎閉環的基礎上，引入智能分析和決策能力，實現：
- 根因分析引擎 (RCA)
- 智能告警路由
- 容量預測與自動調整
- 決策工作流引擎

**預期目標**: MTTD < 30秒, MTTR < 2分鐘, 自動修復率 > 80%, 誤報率 < 8%

---

## 一、根因分析引擎 (RCA Engine)

### 1.1 功能概述

自動分析異常事件的根本原因，減少誤報，提高修復準確率。

### 1.2 核心能力

| 能力 | 描述 | 實現方式 |
|-----|------|---------|
| 事件關聯 | 將相關異常事件分組 | 時間窗口 + 屬性匹配 |
| 依賴追蹤 | 追蹤異常傳播路徑 | 服務拓撲圖 |
| 根因識別 | 識別根本原因 | 貝葉斯網絡 + 決策樹 |
| RCA 報告 | 生成分析報告 | 結構化輸出 |

### 1.3 技術架構

```
┌─────────────────────────────────────────────────────────────────┐
│                     RCA Engine                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Event        │───▶│ Correlation  │───▶│ Root Cause   │     │
│  │ Collector    │    │ Analyzer     │    │ Identifier   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Knowledge Base                          │      │
│  │  - 歷史 RCA  - 依賴圖譜  - 模式庫  - 規則庫         │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 實現步驟 (Week 5-6)

#### Week 5 Day 1-2: 事件收集器

**文件**: `closed_loop/rca/event_collector.py`

**功能**:
- 收集異常事件
- 標準化事件格式
- 事件去重
- 事件存儲

```python
@dataclass
class Event:
    id: str
    timestamp: datetime
    source: str
    type: str
    severity: str
    attributes: Dict[str, Any]
    correlation_id: Optional[str] = None

class EventCollector:
    def collect(self, event: Event) -> None
    def get_events(self, start: datetime, end: datetime) -> List[Event]
    def find_duplicates(self, event: Event, window_seconds: int) -> List[Event]
```

#### Week 5 Day 3-4: 關聯分析器

**文件**: `closed_loop/rca/correlation_analyzer.py`

**功能**:
- 時間窗口關聯
- 屬性匹配
- 因果推斷

```python
class CorrelationAnalyzer:
    def find_correlated_events(
        self,
        target_event: Event,
        window_seconds: int = 300
    ) -> List[EventGroup]
    
    def calculate_correlation_score(
        self,
        event1: Event,
        event2: Event
    ) -> float
```

#### Week 5 Day 5 - Week 6 Day 2: 根因識別器

**文件**: `closed_loop/rca/root_cause_identifier.py`

**算法**:
- 貝葉斯網絡推理
- 決策樹分析
- 圖遍歷算法

```python
class RootCauseIdentifier:
    def identify_root_cause(
        self,
        event_group: EventGroup
    ) -> RootCauseResult
    
    def build_causal_graph(
        self,
        events: List[Event]
    ) -> CausalGraph
```

#### Week 6 Day 3-4: RCA 報告生成器

**文件**: `closed_loop/rca/report_generator.py`

```python
class RCAResult:
    root_cause: str
    confidence: float
    evidence: List[str]
    affected_services: List[str]
    recommended_actions: List[str]
    timeline: List[Event]

class ReportGenerator:
    def generate_report(self, result: RCAResult) -> Dict[str, Any]
    def export_to_markdown(self, result: RCAResult) -> str
```

#### Week 6 Day 5: 整合與測試

**文件**: `closed_loop/rca/__init__.py`

```python
class RCAEngine:
    def analyze(self, event: Event) -> RCAResult
    def batch_analyze(self, events: List[Event]) -> List[RCAResult]
```

---

## 二、智能告警路由

### 2.1 功能概述

智能分類、聚合和路由告警，減少告警疲勞，確保關鍵問題得到及時處理。

### 2.2 核心能力

| 能力 | 描述 | 實現方式 |
|-----|------|---------|
| 告警分類 | 自動分類告警 | 機器學習 + 規則 |
| 告警聚合 | 合併相似告警 | 模式匹配 |
| 智能路由 | 動態路由決策 | 決策樹 |
| 升級策略 | 自動升級 | 時間/條件觸發 |

### 2.3 技術架構

```
┌─────────────────────────────────────────────────────────────────┐
│                   Alert Router                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Classify │──▶│ Aggregate│──▶│  Route   │──▶| Escalate │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Routing Rules                           │      │
│  │  - 團隊映射  - 時間規則  - 嚴重程度  - 渠道偏好     │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 實現步驟 (Week 7-8)

#### Week 7 Day 1-2: 告警分類器

**文件**: `closed_loop/alert/classifier.py`

```python
class AlertClassifier:
    def classify(self, alert: Alert) -> AlertCategory
    def train(self, labeled_alerts: List[Tuple[Alert, AlertCategory]]) -> None
    
class AlertCategory(Enum):
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
```

#### Week 7 Day 3-4: 告警聚合器

**文件**: `closed_loop/alert/aggregator.py`

```python
class AlertAggregator:
    def aggregate(self, alerts: List[Alert]) -> List[AlertGroup]
    def should_aggregate(self, alert1: Alert, alert2: Alert) -> bool
    
class AlertGroup:
    id: str
    alerts: List[Alert]
    pattern: str
    count: int
    first_seen: datetime
    last_seen: datetime
```

#### Week 7 Day 5 - Week 8 Day 2: 路由引擎

**文件**: `closed_loop/alert/router.py`

```python
class RoutingRule:
    conditions: List[Condition]
    target: NotificationTarget
    priority: int

class AlertRouter:
    def route(self, alert: Alert) -> List[NotificationTarget]
    def add_rule(self, rule: RoutingRule) -> None
    def evaluate_rules(self, alert: Alert) -> List[RoutingRule]
```

#### Week 8 Day 3-4: 升級策略

**文件**: `closed_loop/alert/escalation.py`

```python
class EscalationPolicy:
    levels: List[EscalationLevel]
    escalation_intervals: List[int]  # minutes

class EscalationManager:
    def check_escalation(self, alert: Alert) -> Optional[EscalationLevel]
    def escalate(self, alert: Alert, level: EscalationLevel) -> None
```

#### Week 8 Day 5: 整合與測試

**文件**: `closed_loop/alert/__init__.py`

```python
class SmartAlertRouter:
    def process(self, alert: Alert) -> RoutingResult
    def process_batch(self, alerts: List[Alert]) -> List[RoutingResult]
```

---

## 三、容量預測與自動調整

### 3.1 功能概述

基於歷史數據預測負載，提前進行容量調整，避免性能問題。

### 3.2 核心能力

| 能力 | 描述 | 實現方式 |
|-----|------|---------|
| 負載預測 | 預測未來負載 | Holt-Winters / Prophet |
| 預擴容 | 提前擴容 | 預測驅動 |
| 成本優化 | 最小化成本 | 約束優化 |
| KEDA 集成 | 與 KEDA 聯動 | API 集成 |

### 3.3 技術架構

```
┌─────────────────────────────────────────────────────────────────┐
│              Capacity Manager                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Forecast │──▶│  Plan    │──▶│ Optimize │──▶│ Execute  │   │
│  │ Engine   │   │ Capacity │   │  Cost    │   │  Scale   │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Prediction Models                       │      │
│  │  - Holt-Winters  - Prophet  - LSTM  - Linear Reg    │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 實現步驟 (Week 9-10)

#### Week 9 Day 1-2: 預測引擎

**文件**: `closed_loop/capacity/forecast_engine.py`

```python
class ForecastEngine:
    def forecast(
        self,
        metric_name: str,
        horizon_minutes: int = 60
    ) -> ForecastResult
    
    def select_model(self, metric_name: str) -> ForecastModel

class HoltWintersModel:
    def fit(self, data: List[float]) -> None
    def predict(self, steps: int) -> List[float]

class ProphetModel:
    def fit(self, df: pd.DataFrame) -> None
    def predict(self, periods: int) -> pd.DataFrame
```

#### Week 9 Day 3-4: 容量規劃器

**文件**: `closed_loop/capacity/planner.py`

```python
class CapacityPlanner:
    def plan(
        self,
        forecast: ForecastResult,
        constraints: CapacityConstraints
    ) -> ScalingPlan
    
class ScalingPlan:
    actions: List[ScalingAction]
    timeline: List[datetime]
    expected_cost: float
```

#### Week 9 Day 5 - Week 10 Day 2: 成本優化器

**文件**: `closed_loop/capacity/optimizer.py`

```python
class CostOptimizer:
    def optimize(
        self,
        plan: ScalingPlan,
        budget: float
    ) -> ScalingPlan
    
    def calculate_cost(
        self,
        replicas: int,
        duration_hours: float
    ) -> float
```

#### Week 10 Day 3-4: KEDA 集成

**文件**: `closed_loop/capacity/keda_integration.py`

```python
class KEDAIntegration:
    def create_scaled_object(
        self,
        name: str,
        namespace: str,
        triggers: List[ScaleTrigger]
    ) -> None
    
    def update_triggers(
        self,
        name: str,
        namespace: str,
        triggers: List[ScaleTrigger]
    ) -> None
```

#### Week 10 Day 5: 整合與測試

**文件**: `closed_loop/capacity/__init__.py`

```python
class CapacityManager:
    def predict_and_scale(self) -> ScalingResult
    def enable_auto_scaling(self, config: AutoScalingConfig) -> None
```

---

## 四、決策工作流引擎

### 4.1 功能概述

可視化定義和執行複雜的決策流程，支持人工審批節點。

### 4.2 核心能力

| 能力 | 描述 | 實現方式 |
|-----|------|---------|
| 工作流定義 | 可視化定義 | YAML/JSON DSL |
| 狀態機 | 管理流程狀態 | 狀態機引擎 |
| 人工審批 | 支持人工節點 | 通知 + 回調 |
| 執行歷史 | 記錄執行過程 | 持久化存儲 |

### 4.3 技術架構

```
┌─────────────────────────────────────────────────────────────────┐
│              Workflow Engine                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │  Define  │──▶│  Parse   │──▶│ Execute  │──▶| Monitor  │   │
│  │ Workflow │   │  DSL     │   │  State   │   │  Status  │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Node Types                              │      │
│  │  - Condition  - Action  - Approval  - Parallel      │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 實現步驟 (Week 11-12)

#### Week 11 Day 1-2: 工作流定義 DSL

**文件**: `closed_loop/workflow/dsl.py`

```python
@dataclass
class WorkflowDefinition:
    name: str
    version: str
    start_node: str
    nodes: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge]

class WorkflowNode:
    id: str
    type: NodeType  # CONDITION, ACTION, APPROVAL, PARALLEL
    config: Dict[str, Any]

class WorkflowParser:
    def parse_yaml(self, yaml_content: str) -> WorkflowDefinition
    def parse_json(self, json_content: str) -> WorkflowDefinition
    def validate(self, definition: WorkflowDefinition) -> bool
```

#### Week 11 Day 3-4: 狀態機引擎

**文件**: `closed_loop/workflow/state_machine.py`

```python
class WorkflowState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StateMachine:
    def transition(
        self,
        instance: WorkflowInstance,
        event: StateEvent
    ) -> None
    
    def get_valid_transitions(
        self,
        current_state: WorkflowState
    ) -> List[WorkflowState]
```

#### Week 11 Day 5 - Week 12 Day 2: 執行引擎

**文件**: `closed_loop/workflow/executor.py`

```python
class WorkflowExecutor:
    def execute(
        self,
        definition: WorkflowDefinition,
        context: Dict[str, Any]
    ) -> WorkflowInstance
    
    def resume(
        self,
        instance_id: str,
        approval_result: ApprovalResult
    ) -> None
    
    def cancel(self, instance_id: str) -> None

class WorkflowInstance:
    id: str
    definition: WorkflowDefinition
    state: WorkflowState
    current_node: str
    context: Dict[str, Any]
    history: List[ExecutionStep]
```

#### Week 12 Day 3-4: 人工審批節點

**文件**: `closed_loop/workflow/approval.py`

```python
class ApprovalNode:
    approvers: List[str]
    timeout_minutes: int
    escalation_policy: EscalationPolicy

class ApprovalManager:
    def request_approval(
        self,
        instance_id: str,
        node_id: str,
        approvers: List[str]
    ) -> ApprovalRequest
    
    def submit_approval(
        self,
        request_id: str,
        approver: str,
        decision: ApprovalDecision
    ) -> None
```

#### Week 12 Day 5: 整合與測試

**文件**: `closed_loop/workflow/__init__.py`

```python
class WorkflowEngine:
    def register_workflow(self, definition: WorkflowDefinition) -> None
    def start_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any]
    ) -> WorkflowInstance
    def get_instance(self, instance_id: str) -> WorkflowInstance
```

---

## 五、Phase 2 整合架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Phase 2: 智能閉環                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Monitor Layer                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │  │ Prometheus │  │   Loki     │  │   Jaeger   │  │   Events   │   │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Analyze Layer (NEW)                          │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │  │  Anomaly   │  │    RCA     │  │  Forecast  │  │  Classify  │   │   │
│  │  │  Detector  │  │   Engine   │  │   Engine   │  │   Alert    │   │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Plan Layer (NEW)                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │  │   Rule     │  │  Capacity  │  │  Workflow  │  │   Route    │   │   │
│  │  │  Engine    │  │  Planner   │  │   Engine   │  │   Alert    │   │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Execute Layer                                │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │  │  Restart   │  │   Scale    │  │  Switch    │  │  Workflow  │   │   │
│  │  │    Pod     │  │   Up/Down  │  │  Provider  │  │  Execute   │   │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Knowledge Base                                 │   │
│  │  - RCA History  - Forecast Models  - Workflow Defs  - Alert Rules   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 六、實施時間表

| 週次 | 任務 | 輸出 |
|-----|------|------|
| Week 5 | 事件收集器 + 關聯分析器 | `event_collector.py`, `correlation_analyzer.py` |
| Week 6 | 根因識別器 + 報告生成器 | `root_cause_identifier.py`, `report_generator.py` |
| Week 7 | 告警分類器 + 聚合器 | `classifier.py`, `aggregator.py` |
| Week 8 | 路由引擎 + 升級策略 | `router.py`, `escalation.py` |
| Week 9 | 預測引擎 + 容量規劃器 | `forecast_engine.py`, `planner.py` |
| Week 10 | 成本優化器 + KEDA 集成 | `optimizer.py`, `keda_integration.py` |
| Week 11 | 工作流 DSL + 狀態機 | `dsl.py`, `state_machine.py` |
| Week 12 | 執行引擎 + 審批節點 | `executor.py`, `approval.py` |

---

## 七、預期效果

| 指標 | Phase 1 | Phase 2 目標 | 提升 |
|-----|---------|-------------|------|
| MTTD | 1 min | 30 sec | 50% |
| MTTR | 5 min | 2 min | 60% |
| 自動修復率 | 60% | 80% | +20% |
| 誤報率 | 15% | 8% | -47% |
| 人工介入率 | 40% | 20% | -50% |

---

## 八、風險與緩解

| 風險 | 可能性 | 影響 | 緩解措施 |
|-----|--------|------|---------|
| RCA 準確率不足 | 中 | 高 | 持續訓練模型，人工反饋 |
| 預測模型不準確 | 中 | 中 | 多模型集成，A/B 測試 |
| 工作流複雜度 | 高 | 中 | 提供可視化工具 |
| 性能開銷 | 中 | 中 | 異步處理，緩存優化 |

---

**版本**: v1.0  
**日期**: 2025-02-24  
**作者**: AI Architecture Assistant
