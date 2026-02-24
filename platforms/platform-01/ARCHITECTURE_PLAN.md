# 自治闭环系统架构规划方案 v3.0

## 1. 系统总体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              自治闭环系统 v3.0                                   │
│                        Autonomous Closed-Loop System                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        API Gateway / 入口层                              │   │
│  │         (REST API / gRPC / WebSocket / Event Stream)                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Phase 3: 自治闭环 (Autonomous)                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │ 自学习优化    │ │ 多目标决策    │ │  知识图谱     │ │ 预测性修复    │   │   │
│  │  │   Engine     │ │  Optimizer   │ │   System     │ │   System     │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                      │   │
│  │  │ 跨系统协同    │ │ 人机协作界面  │ │ 专家知识系统  │                      │   │
│  │  │Orchestration │ │  Human-AI    │ │   Expert     │                      │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Phase 2: 智能闭环 (Intelligent)                      │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │   RCA引擎    │ │  智能告警    │ │  容量管理    │ │  工作流引擎   │   │   │
│  │  │  (根因分析)   │ │   路由系统    │ │ (预测+规划)  │ │              │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Phase 1: 基础闭环 (Foundation)                       │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │ MAPE-K控制器 │ │  异常检测器   │ │  自动修复器   │ │   规则引擎    │   │   │
│  │  │ Controller   │ │   Detector   │ │  Remediator  │ │  Rule Engine │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │   │
│  │  ┌──────────────┐                                                          │   │
│  │  │  闭环指标    │                                                          │   │
│  │  │   Metrics    │                                                          │   │
│  │  └──────────────┘                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        数据层 / Data Layer                               │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │   │
│  │  │  时序数据库   │ │   图数据库    │ │   向量数据库  │ │   事件存储    │   │   │
│  │  │  (Metrics)   │ │  (Knowledge) │ │  (Embeddings)│ │   (Events)   │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块详细设计

### 2.1 Phase 1: 基础闭环层

#### 2.1.1 MAPE-K 控制器
```python
class ClosedLoopController:
    """
    MAPE-K 循环实现
    - Monitor: 监控指标数据
    - Analyze: 异常检测分析
    - Plan: 规则评估与决策
    - Execute: 执行修复动作
    - Knowledge: 记录执行结果
    """
    
    核心接口:
    - process_metric(metric_name, value) -> LoopContext
    - start() / stop()
    - get_status() -> Dict
```

#### 2.1.2 异常检测器
```python
class AnomalyDetector:
    """
    多算法异常检测
    - 统计方法: Z-Score, IQR
    - 机器学习方法: Isolation Forest
    - 深度学习方法: LSTM Autoencoder
    """
    
    检测算法:
    - STATISTICAL_ZSCORE
    - STATISTICAL_IQR
    - ML_ISOLATION_FOREST
    - ML_LOCAL_OUTLIER_FACTOR
    - DEEP_LSTM
```

#### 2.1.3 自动修复器
```python
class AutoRemediator:
    """
    自动修复执行器
    - 重启服务
    - 扩容/缩容
    - 切换流量
    - 自定义脚本
    """
    
    修复类型:
    - RESTART_SERVICE
    - SCALE_UP / SCALE_DOWN
    - SWITCH_TRAFFIC
    - CLEAR_CACHE
    - CUSTOM_SCRIPT
```

#### 2.1.4 规则引擎
```python
class RuleEngine:
    """
    条件-动作规则引擎
    - 支持复杂条件组合
    - 优先级排序
    - 动态规则加载
    """
    
    规则结构:
    - condition: 触发条件
    - action: 执行动作
    - priority: 优先级
    - enabled: 启用状态
```

---

### 2.2 Phase 2: 智能闭环层

#### 2.2.1 RCA 引擎
```python
class RootCauseAnalyzer:
    """
    根因分析引擎
    - 事件关联分析
    - 拓扑传播分析
    - 时序因果推断
    """
    
    组件:
    - EventCollector: 事件收集
    - CorrelationAnalyzer: 关联分析
    - RootCauseIdentifier: 根因定位
    - ReportGenerator: 报告生成
```

#### 2.2.2 智能告警路由
```python
class SmartAlertRouter:
    """
    智能告警路由系统
    - 告警聚合降噪
    - 智能路由决策
    - 值班表集成
    """
    
    路由策略:
    - 基于严重级别
    - 基于服务归属
    - 基于时间/值班表
    - 基于历史响应
```

#### 2.2.3 容量管理
```python
class CapacityManager:
    """
    容量预测与规划
    - 时序预测 (Prophet, ARIMA)
    - 容量规划算法
    - 自动扩缩容建议
    """
    
    预测模型:
    - PROPHET (Facebook)
    - ARIMA
    - LSTM
    - XGBOOST
```

#### 2.2.4 工作流引擎
```python
class WorkflowEngine:
    """
    决策工作流引擎
    - 审批工作流
    - 条件分支
    - 并行执行
    """
    
    工作流类型:
    - 自动执行
    - 人工审批
    - 混合模式
```

---

### 2.3 Phase 3: 自治闭环层

#### 2.3.1 自学习优化引擎
```python
class SelfLearningOptimizer:
    """
    自学习优化引擎
    - PPO 强化学习策略优化
    - 贝叶斯超参数优化
    - 在线学习更新
    """
    
    组件:
    - RLPOLICYLearner: PPO策略学习
    - BayesianOptimizer: 贝叶斯优化
    - EffectEvaluator: 效果评估
    - OnlineLearner: 在线学习
```

#### 2.3.2 多目标决策优化
```python
class MultiObjectiveOptimizer:
    """
    多目标决策优化
    - 成本模型
    - 风险评估
    - NSGA-II Pareto优化
    """
    
    优化目标:
    - 最小化成本
    - 最小化风险
    - 最大化性能
    - 最大化可用性
```

#### 2.3.3 知识图谱系统
```python
class KnowledgeGraphSystem:
    """
    知识图谱系统
    - 实体抽取
    - 关系构建
    - GNN图神经网络推理
    - 查询接口
    """
    
    组件:
    - EntityExtractor: 实体抽取
    - RelationBuilder: 关系构建
    - GNNEngine: GNN推理
    - QueryInterface: 查询接口
```

#### 2.3.4 预测性修复系统
```python
class PredictiveRemediationSystem:
    """
    预测性修复系统
    - 故障预测
    - 影响分析
    - 预修复规划
    """
    
    组件:
    - FailurePredictor: 故障预测
    - ImpactAnalyzer: 影响分析
    - PrepairPlanner: 预修复规划
```

#### 2.3.5 跨系统协同
```python
class CrossSystemOrchestration:
    """
    跨系统协同
    - 拓扑构建
    - 协同决策
    - 级联控制
    """
    
    组件:
    - TopologyBuilder: 拓扑构建
    - ConsensusEngine: 协同决策
    - CascadeController: 级联控制
```

#### 2.3.6 人机协作界面
```python
class HumanAIInterface:
    """
    人机协作界面
    - XAI可解释AI
    - 审批工作流
    - 专家知识系统
    """
    
    组件:
    - XAIExplainer: 可解释AI
    - ApprovalWorkflow: 审批工作流
    - ExpertKnowledge: 专家知识
```

---

## 3. 数据流架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据流图                                        │
└─────────────────────────────────────────────────────────────────────────────┘

    指标数据                    事件数据                    告警数据
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│  Metric      │           │   Event      │           │   Alert      │
│  Ingestion   │           │  Ingestion   │           │  Ingestion   │
└──────┬───────┘           └──────┬───────┘           └──────┬───────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          消息队列 (Kafka/Redis)                          │
└─────────────────────────────────────────────────────────────────────────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│   Phase 1    │           │   Phase 2    │           │   Phase 3    │
│  异常检测     │ ────────> │   RCA分析    │ ────────> │  知识推理    │
│  自动修复     │           │   告警路由   │           │  预测修复    │
└──────┬───────┘           └──────┬───────┘           └──────┬───────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据存储层                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ 时序DB   │  │ 图DB     │  │ 向量DB   │  │ 事件存储 │                │
│  │InfluxDB │  │Neo4j    │  │Milvus   │  │EventStore│                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 接口设计

### 4.1 核心接口

```python
# 指标处理接口
POST /api/v1/metrics
{
    "metric_name": "cpu_usage",
    "value": 85.5,
    "timestamp": "2024-01-15T10:30:00Z",
    "tags": {
        "service": "api-gateway",
        "host": "server-01"
    }
}

# 事件上报接口
POST /api/v1/events
{
    "event_type": "service_restart",
    "source": "kubernetes",
    "severity": "warning",
    "message": "Service pod restarted",
    "metadata": {...}
}

# 告警接口
POST /api/v1/alerts
{
    "title": "High CPU Usage",
    "description": "CPU usage exceeded 80%",
    "severity": "critical",
    "source": "monitoring-system"
}

# 状态查询接口
GET /api/v1/status
{
    "system_status": "healthy",
    "phase1": {...},
    "phase2": {...},
    "phase3": {...}
}

# 指标查询接口
GET /api/v1/metrics/summary
{
    "health_score": 0.85,
    "mttr": 120,
    "anomaly_count": 5,
    "remediation_success_rate": 0.92
}
```

---

## 5. 部署架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              部署架构                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kubernetes Cluster                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Ingress Controller                          │   │
│  │                     (Nginx / Traefik / Istio)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         API Gateway Service                          │   │
│  │                    (3 replicas, load balanced)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Core Services (微服务)                          │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │ Controller │ │  Detector  │ │ Remediator │ │ RuleEngine │       │   │
│  │  │  (2 reps)  │ │  (3 reps)  │ │  (2 reps)  │ │  (2 reps)  │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │    RCA     │ │   Alert    │ │  Capacity  │ │  Workflow  │       │   │
│  │  │  (2 reps)  │ │  (2 reps)  │ │  (2 reps)  │ │  (2 reps)  │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │  Learning  │ │ Optimizer  │ │ Knowledge  │ │ Predictive │       │   │
│  │  │  (2 reps)  │ │  (2 reps)  │ │  (2 reps)  │ │  (2 reps)  │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          Data Layer                                  │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │ InfluxDB   │ │  Neo4j     │ │  Milvus    │ │ PostgreSQL │       │   │
│  │  │ (Stateful) │ │ (Stateful) │ │ (Stateful) │ │ (Stateful) │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  │  ┌────────────┐ ┌────────────┐                                      │   │
│  │  │   Redis    │ │   Kafka    │                                      │   │
│  │  │  (Cache)   │ │  (Queue)   │                                      │   │
│  │  └────────────┘ └────────────┘                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 依赖映射

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              模块依赖图                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1 (基础层)
├── detector
│   └── numpy, scipy, scikit-learn, torch
├── remediator
│   └── aiohttp, kubernetes (client)
├── rules
│   └── (pure python)
├── metrics
│   └── prometheus_client
└── core
    └── detector, remediator, rules, metrics

Phase 2 (智能层)
├── rca
│   ├── core
│   └── networkx, pandas
├── alert
│   ├── core
│   └── aiohttp
├── capacity
│   ├── core
│   └── prophet, pandas, numpy
└── workflow
    ├── core
    └── (pure python)

Phase 3 (自治层)
├── learning
│   ├── core, metrics
│   └── torch, optuna, numpy
├── optimizer
│   ├── core, metrics
│   └── deap, numpy
├── knowledge
│   ├── core, rca
│   └── torch_geometric, neo4j
├── predictive
│   ├── core, knowledge
│   └── torch, sklearn
├── orchestration
│   ├── core, predictive
│   └── networkx
└── human
    ├── core, knowledge
    └── shap, lime
```

---

## 7. 配置文件

### 7.1 系统配置
```yaml
# config/system.yaml
system:
  name: "autonomous-closed-loop"
  version: "3.0.0"
  environment: "production"
  
logging:
  level: "INFO"
  format: "json"
  output: "stdout"

metrics:
  enabled: true
  backend: "prometheus"
  interval: 15

phases:
  phase1:
    enabled: true
  phase2:
    enabled: true
  phase3:
    enabled: true
```

### 7.2 组件配置
```yaml
# config/components.yaml
detector:
  algorithms:
    - STATISTICAL_ZSCORE
    - ML_ISOLATION_FOREST
  sensitivity: 0.95
  window_size: 100

remediator:
  max_concurrent: 5
  timeout: 300
  dry_run: false

rule_engine:
  auto_reload: true
  max_rules: 1000

rca:
  correlation_window: 300
  max_depth: 5

alert_router:
  aggregation_window: 60
  deduplication_ttl: 300

capacity:
  forecast_horizon: 24
  models:
    - PROPHET
    - ARIMA
```

---

## 8. 实施路线图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              实施路线图                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: 基础闭环 (Week 1-4)
├── Week 1: 核心控制器 + 异常检测器
├── Week 2: 自动修复器 + 规则引擎
├── Week 3: 闭环指标 + 集成测试
└── Week 4: 性能优化 + 文档

Phase 2: 智能闭环 (Week 5-8)
├── Week 5: RCA引擎 (事件收集 + 关联分析)
├── Week 6: 根因定位 + 报告生成
├── Week 7: 智能告警路由 + 容量管理
├── Week 8: 工作流引擎 + 集成测试

Phase 3: 自治闭环 (Week 9-16)
├── Week 9-10: 自学习优化引擎 (PPO + 贝叶斯)
├── Week 11-12: 多目标决策优化 (NSGA-II)
├── Week 13-14: 知识图谱系统 (GNN)
├── Week 15-16: 预测性修复 + 跨系统协同
└── Week 17-18: 人机协作界面 + 专家知识

部署上线 (Week 19-20)
├── Week 19: 生产环境部署
└── Week 20: 灰度发布 + 监控
```

---

## 9. 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 异常检测延迟 | < 100ms | 单指标检测 |
| 修复执行延迟 | < 5s | 简单修复动作 |
| RCA分析时间 | < 30s | 完整根因分析 |
| 告警路由延迟 | < 50ms | 路由决策 |
| 容量预测时间 | < 1s | 单次预测 |
| 系统吞吐量 | > 10K TPS | 指标处理 |
| 可用性 | > 99.9% | 系统可用性 |

---

## 10. 安全考虑

- API认证: JWT Token
- 访问控制: RBAC
- 数据加密: TLS 1.3
- 审计日志: 全量记录
- 沙箱执行: 修复动作隔离

---

---

## 11. 强制治理规范（避免断尾/重演）

> 以下8节为**必填章节**，缺任何一项，后续自动化都会出现「中断、不可重播、不可验证、无法回滚」

---

### 11.1 系统边界与非目标（In/Out of Scope）

#### 11.1.1 系统边界（In Scope）

| 功能域 | 范围说明 |
|--------|----------|
| 异常检测 | 基于指标/日志/事件的异常识别与分类 |
| 根因分析 | 拓扑传播分析、时序因果推断、关联挖掘 |
| 决策规划 | 基于规则/ML/优化的修复策略生成 |
| 自动执行 | 低风险修复动作的自动执行（重启、扩缩容、切流） |
| 人工审批 | 高风险动作的强制审批流程 |
| 闭环验证 | 修复后指标验证与自动回滚 |
| 知识沉淀 | 事件-决策-结果的关联存储与查询 |

#### 11.1.2 非目标（Out of Scope）

| 功能域 | 说明 |
|--------|------|
| 基础设施变更 | 不直接操作网络设备、物理服务器 |
| 数据修复 | 不涉及数据库数据修复/回滚 |
| 安全事件响应 | 安全入侵检测与响应由专门系统处理 |
| 未审批的高风险变更 | **绝对禁止**自动执行未审批的高风险变更 |

#### 11.1.3 强制红线

```yaml
forbidden_operations:
  - 未审批的生产环境配置变更
  - 未验证的数据库操作
  - 影响范围超过阈值的全局操作
  - 无回滚方案的高风险变更
```

---

### 11.2 责任分层（Layered Responsibility）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           责任分层架构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 6: Verification Layer (验证层)                                       │
│  ├── 修复后指标验证                                                         │
│  ├── SLO/SLA 合规检查                                                       │
│  └── 自动回滚触发                                                           │
│                                                                             │
│  Layer 5: Executor Layer (执行层)                                           │
│  ├── 动作执行 (apply)                                                       │
│  ├── 回滚执行 (rollback)                                                    │
│  └── 执行状态跟踪                                                           │
│                                                                             │
│  Layer 4: Controller Layer (控制层)                                         │
│  ├── 状态机管理                                                             │
│  ├── 流程编排                                                               │
│  └── 失败处理策略                                                           │
│                                                                             │
│  Layer 3: Planner Layer (规划层)                                            │
│  ├── 修复策略生成                                                           │
│  ├── 风险评估                                                               │
│  └── 回滚方案生成                                                           │
│                                                                             │
│  Layer 2: Knowledge Layer (知识层)                                          │
│  ├── 实体抽取与关系构建                                                     │
│  ├── 拓扑推理                                                               │
│  └── 历史案例匹配                                                           │
│                                                                             │
│  Layer 1: Detector Layer (检测层)                                           │
│  ├── 异常检测                                                               │
│  ├── 根因候选生成                                                           │
│  └── 证据收集                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 11.2.1 层间契约

| 层 | 输入 | 输出 | 契约要求 |
|----|------|------|----------|
| Detector | 指标/日志/事件 | Anomaly + Evidence | 必须包含置信度与影响范围 |
| Knowledge | Anomaly + Evidence | 拓扑上下文 + 历史案例 | 必须包含版本与快照 |
| Planner | 知识上下文 | ActionPlan + Risk + Rollback | 必须包含回滚方案 |
| Controller | ActionPlan | 编排执行 + 状态跟踪 | 必须幂等、可重放 |
| Executor | Action | 执行结果 + 状态 | 必须支持 apply/rollback/verify |
| Verification | 执行结果 | 验证结果 + 回滚触发 | 必须基于 SLO 门檻 |

---

### 11.3 决策契约（Decision Contract）

每个决策必须输出标准化的 `decision.json`：

```json
{
  "decision_id": "dec_20240225103000_abc123",
  "trace_id": "trace_abc123",
  "timestamp": "2024-02-25T10:30:00Z",
  "version": {
    "schema": "1.0.0",
    "model": "rca_v2.1.0",
    "topology": "topo_v20240225",
    "rules": "rules_v1.5.0"
  },
  "evidence": {
    "anomaly": {
      "anomaly_id": "anom_001",
      "type": "cpu_high",
      "severity": "critical",
      "confidence": 0.95,
      "affected_services": ["api-gateway", "order-service"]
    },
    "root_causes": [
      {
        "cause": "connection_pool_exhausted",
        "confidence": 0.88,
        "evidence": ["metric_001", "log_002"]
      }
    ],
    "input_snapshot_hash": "sha256:abc123..."
  },
  "risk": {
    "level": "high",
    "score": 0.85,
    "factors": [
      {
        "name": "blast_radius",
        "impact": "3_services",
        "weight": 0.4
      }
    ],
    "requires_approval": true
  },
  "actions": [
    {
      "action_id": "act_001",
      "type": "restart_service",
      "target": "order-service",
      "params": {"graceful": true},
      "estimated_duration": 30,
      "order": 1
    }
  ],
  "rollback": {
    "enabled": true,
    "actions": [
      {
        "action_id": "rollback_act_001",
        "type": "restore_config",
        "target": "order-service",
        "backup_ref": "backup_20240225102900"
      }
    ],
    "auto_trigger": {
      "enabled": true,
      "conditions": [
        {"metric": "error_rate", "threshold": "> 0.05", "duration": "5m"}
      ]
    }
  },
  "verify": {
    "enabled": true,
    "checks": [
      {
        "metric": "cpu_usage",
        "expected": "< 70%",
        "duration": "3m"
      },
      {
        "metric": "error_rate",
        "expected": "< 0.01",
        "duration": "5m"
      }
    ],
    "timeout": 600
  },
  "approval": {
    "required": true,
    "level": "L2",
    "approvers": ["oncall_db", "sre_lead"],
    "timeout": 1800,
    "escalation": {
      "enabled": true,
      "after": "15m",
      "to": "vp_engineering"
    }
  },
  "metadata": {
    "source": "closed_loop_controller",
    "triggered_by": "anomaly_detector",
    "correlation_id": "corr_xyz789"
  }
}
```

#### 11.3.1 必填字段检查清单

| 字段 | 必填 | 说明 |
|------|------|------|
| decision_id | ✅ | 全局唯一标识 |
| trace_id | ✅ | 用于全链路追踪 |
| version | ✅ | 模型/规则/拓扑版本 |
| evidence | ✅ | 决策证据链 |
| risk | ✅ | 风险评估与审批要求 |
| actions | ✅ | 执行动作列表 |
| rollback | ✅ | 回滚方案（必须存在，可为空） |
| verify | ✅ | 验证方案 |
| approval | 条件 | 高风险必须填写 |

---

### 11.4 幂等与状态机

#### 11.4.1 Action 幂等设计

```python
class IdempotentAction:
    """
    每个 Action 必须实现幂等接口
    """
    
    def __init__(self, action_id: str, params: Dict):
        self.action_id = action_id
        self.params = params
        self.state = ActionState.PENDING
    
    def apply(self) -> ActionResult:
        """
        执行动作 - 必须幂等
        相同 action_id 多次调用结果一致
        """
        # 1. 检查是否已执行
        if self._is_executed():
            return self._get_previous_result()
        
        # 2. 执行动作
        result = self._do_apply()
        
        # 3. 记录执行状态
        self._record_execution(result)
        
        return result
    
    def rollback(self) -> ActionResult:
        """
        回滚动作 - 必须幂等
        """
        if not self._is_executed():
            return ActionResult.skipped("Not executed")
        
        if self._is_rolled_back():
            return self._get_rollback_result()
        
        return self._do_rollback()
    
    def verify(self) -> VerificationResult:
        """
        验证动作效果
        """
        return self._do_verify()
    
    def _is_executed(self) -> bool:
        """检查是否已执行 - 基于 action_id 查询状态存储"""
        pass
```

#### 11.4.2 状态机定义

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Action 状态机                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│    │ PENDING │───>│APPROVED │───>│EXECUTING│───>│SUCCESS  │                │
│    └────┬────┘    └─────────┘    └────┬────┘    └────┬────┘                │
│         │                              │              │                     │
│         │                              │              ▼                     │
│         │                              │         ┌─────────┐                │
│         │                              │         │VERIFIED │                │
│         │                              │         └────┬────┘                │
│         │                              │              │                     │
│         │                              ▼              │                     │
│         │                         ┌─────────┐         │                     │
│         │                         │ FAILED  │─────────┘                     │
│         │                         └────┬────┘                               │
│         │                              │                                    │
│         │                              ▼                                    │
│         │                         ┌─────────┐                               │
│         └────────────────────────>│REJECTED │                               │
│                                   └─────────┘                               │
│                                                                             │
│    ┌─────────┐    ┌─────────┐    ┌─────────┐                               │
│    │SUCCESS  │───>│ROLLBACK │───>│ROLLED   │                               │
│    │(verify  │    │INITIATED│    │BACK     │                               │
│    │ failed) │    └─────────┘    └─────────┘                               │
│    └─────────┘                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| 状态 | 说明 | 可转移 |
|------|------|--------|
| PENDING | 待审批 | APPROVED, REJECTED |
| APPROVED | 已审批 | EXECUTING |
| EXECUTING | 执行中 | SUCCESS, FAILED |
| SUCCESS | 执行成功 | VERIFIED, ROLLBACK_INITIATED |
| FAILED | 执行失败 | ROLLBACK_INITIATED |
| VERIFIED | 验证通过 | (终态) |
| ROLLBACK_INITIATED | 回滚中 | ROLLED_BACK |
| ROLLED_BACK | 已回滚 | (终态) |
| REJECTED | 已拒绝 | (终态) |

---

### 11.5 强制 Gate（Policy + Human Approval）

#### 11.5.1 风险分级与审批策略

```yaml
risk_levels:
  critical:
    score_range: [0.8, 1.0]
    requires_approval: true
    approval_level: L3  # VP级别
    auto_execute: false
    examples:
      - 全局配置变更
      - 数据库操作
      - 核心服务下线
      
  high:
    score_range: [0.6, 0.8)
    requires_approval: true
    approval_level: L2  # SRE Lead
    auto_execute: false
    examples:
      - 多服务重启
      - 大规模扩缩容
      - 流量切换
      
  medium:
    score_range: [0.3, 0.6)
    requires_approval: true
    approval_level: L1  # On-call
    auto_execute: false
    examples:
      - 单服务重启
      - 缓存清理
      
  low:
    score_range: [0, 0.3)
    requires_approval: false
    approval_level: null
    auto_execute: true
    examples:
      - 日志清理
      - 健康检查
      - 指标采集
```

#### 11.5.2 审批工作流

```python
class ApprovalGate:
    """
    强制审批门檻
    """
    
    def evaluate(self, decision: Decision) -> GateResult:
        """
        评估决策是否需要审批
        """
        # 1. 计算风险分数
        risk_score = self._calculate_risk(decision)
        
        # 2. 确定风险等级
        risk_level = self._get_risk_level(risk_score)
        
        # 3. 检查是否需要审批
        if risk_level.requires_approval:
            # 4. 创建审批请求
            approval_request = self._create_approval_request(
                decision=decision,
                risk_level=risk_level
            )
            
            # 5. 发送审批通知
            self._send_approval_notification(approval_request)
            
            return GateResult.pending_approval(approval_request)
        
        # 6. 低风险直接通过
        return GateResult.approved(auto_execute=True)
    
    def check_approval_status(self, decision_id: str) -> ApprovalStatus:
        """
        检查审批状态
        """
        # 未批准 = 禁止执行
        pass
```

#### 11.5.3 强制规则

| 规则 | 说明 | 违反后果 |
|------|------|----------|
| R1 | 高风险操作必须审批 | 未审批自动拦截 |
| R2 | 审批超时自动升级 | 15分钟无响应升级至上级 |
| R3 | 审批记录永久保存 | 用于审计与追溯 |
| R4 | 紧急情况下可强制绕过 | 需事后24小时内补审批 |

---

### 11.6 闭环验证（Verification Gate）

#### 11.6.1 验证策略

```python
class VerificationGate:
    """
    闭环验证门檻
    验证不通过 = 自动回滚或升级人工
    """
    
    def verify(self, execution_result: ExecutionResult) -> VerificationResult:
        """
        执行后验证
        """
        decision = execution_result.decision
        
        # 1. 收集验证指标
        metrics = self._collect_verification_metrics(decision.verify.checks)
        
        # 2. 评估验证结果
        results = []
        for check in decision.verify.checks:
            result = self._evaluate_check(check, metrics)
            results.append(result)
        
        # 3. 综合判断
        all_passed = all(r.passed for r in results)
        
        if all_passed:
            return VerificationResult.success(results)
        
        # 4. 验证失败 - 触发回滚
        return VerificationResult.failed(
            results=results,
            trigger_rollback=True
        )
    
    def _evaluate_check(self, check: VerifyCheck, metrics: Dict) -> CheckResult:
        """
        评估单个检查项
        """
        actual_value = metrics.get(check.metric)
        expected = check.expected
        
        # 解析期望值表达式
        passed = self._evaluate_expression(actual_value, expected)
        
        return CheckResult(
            metric=check.metric,
            expected=expected,
            actual=actual_value,
            passed=passed
        )
```

#### 11.6.2 验证检查清单

| 检查项 | 说明 | 失败处理 |
|--------|------|----------|
| 错误率 | error_rate < threshold | 回滚 |
| 延迟 | p99_latency < threshold | 回滚 |
| CPU使用率 | cpu_usage < threshold | 回滚 |
| 可用性 | availability > threshold | 回滚 |
| 业务指标 | business_metric 正常 | 人工介入 |

#### 11.6.3 强制规范

```yaml
verification:
  enabled: true  # 必须启用
  timeout: 600   # 验证超时时间（秒）
  
  # 验证失败处理策略
  on_failure:
    strategy: "rollback"  # rollback | escalate | alert
    auto_rollback: true   # 自动回滚
    escalation_delay: 300 # 升级延迟（秒）
    
  # 验证通过标准
  pass_criteria:
    all_checks_pass: true      # 所有检查通过
    min_pass_ratio: 1.0        # 最低通过比例
    consecutive_period: 180    # 持续正常时间（秒）
```

---

### 11.7 可观测性与证据链（Audit Trail）

#### 11.7.1 证据链架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            证据链架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Evidence                    Decision Evidence                        │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │ 原始指标数据     │              │ 决策上下文       │                       │
│  │ 日志/追踪       │─────────────>│ 推理过程        │                       │
│  │ 事件流          │              │ 规则匹配        │                       │
│  └─────────────────┘              └────────┬────────┘                       │
│                                            │                                │
│                                            ▼                                │
│  Execution Evidence              Verification Evidence                      │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │ 执行动作         │              │ 验证指标        │                       │
│  │ 执行结果        │<─────────────│ 验证结论        │                       │
│  │ 状态变更        │              │ 回滚记录        │                       │
│  └─────────────────┘              └─────────────────┘                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         证据链存储                                   │   │
│  │  - 不可篡改（WORM存储）                                              │   │
│  │  - 版本化（每次变更生成新版本）                                      │   │
│  │  - 可哈希验证（完整性校验）                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 11.7.2 证据固化要求

| 证据类型 | 存储位置 | 保留期限 | 格式 |
|----------|----------|----------|------|
| 决策记录 | PostgreSQL | 7年 | JSON |
| 执行日志 | Elasticsearch | 1年 | JSON |
| 指标快照 | InfluxDB | 90天 | TimeSeries |
| 拓扑快照 | Neo4j | 7年 | Graph |
| 审计日志 | S3 (WORM) | 7年 | Parquet |

#### 11.7.3 证据查询接口

```python
class AuditTrail:
    """
    证据链查询
    """
    
    def get_decision_evidence(self, decision_id: str) -> DecisionEvidence:
        """
        获取决策证据
        """
        pass
    
    def get_execution_evidence(self, action_id: str) -> ExecutionEvidence:
        """
        获取执行证据
        """
        pass
    
    def verify_evidence_integrity(self, evidence_id: str) -> bool:
        """
        验证证据完整性
        """
        # 计算哈希并比对
        pass
```

---

### 11.8 故障域与降级策略

#### 11.8.1 故障域划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            故障域架构                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      全局故障域 (Global)                             │   │
│  │  - 配置中心、API Gateway、消息队列                                   │   │
│  │  - 故障影响：整个系统不可用                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      区域故障域 (Zone)                               │   │
│  │  - Phase 控制器、数据存储                                            │   │
│  │  - 故障影响：单个 Phase 不可用                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      模块故障域 (Module)                             │   │
│  │  - 检测器、修复器、规则引擎等                                        │   │
│  │  - 故障影响：单个功能不可用                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      实例故障域 (Instance)                           │   │
│  │  - 单个 Pod/进程                                                     │   │
│  │  - 故障影响：单个实例不可用                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 11.8.2 降级策略

| 故障域 | 故障场景 | 降级策略 |
|--------|----------|----------|
| 全局 | 配置中心不可用 | 使用本地缓存配置，禁止变更操作 |
| 全局 | 消息队列不可用 | 降级为同步调用，限流保护 |
| 区域 | Phase 3 不可用 | 降级至 Phase 2 决策 |
| 区域 | Phase 2 不可用 | 降级至 Phase 1 规则引擎 |
| 模块 | 检测器不可用 | 告警升级，人工介入 |
| 模块 | 修复器不可用 | 仅检测不修复，告警通知 |
| 实例 | 单实例故障 | 流量切换至健康实例 |

#### 11.8.3 重试策略

```python
class RetryPolicy:
    """
    重试策略
    """
    
    # 指数退避
    exponential_backoff:
      initial_delay: 1s
      max_delay: 60s
      multiplier: 2
      max_retries: 5
    
    # 熔断器
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 30s
      half_open_max_calls: 3
```

#### 11.8.4 停机保护

```yaml
kill_switch:
  enabled: true
  
  # 触发条件
  triggers:
    - metric: "error_rate"
      threshold: "> 0.5"
      duration: "2m"
    - metric: "system_availability"
      threshold: "< 0.5"
      duration: "1m"
  
  # 保护动作
  actions:
    - stop_auto_remediation    # 停止自动修复
    - alert_critical           # 紧急告警
    - escalate_manual          # 升级人工
```

---

*文档版本: 3.0.0*  
*最后更新: 2026-02-25*
