#!/usr/bin/env python3
"""
全阶段 Bug 检测测试脚本
运行 Phase 1~3 所有组件，捕获所有错误
"""

import sys
import os
import traceback
import warnings

warnings.filterwarnings('always')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

print("="*80)
print("闭环系统全阶段 Bug 检测测试")
print("="*80)

errors = []
warnings_list = []

def log_error(phase, module, test_name, error):
    """记录错误"""
    error_info = {
        'phase': phase,
        'module': module,
        'test': test_name,
        'error': str(error),
        'traceback': traceback.format_exc()
    }
    errors.append(error_info)
    print(f"  ❌ ERROR: {test_name}")
    print(f"     {str(error)[:200]}")

def log_warning(message):
    """记录警告"""
    warnings_list.append(message)
    print(f"  ⚠️  WARNING: {message}")

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# =============================================================================
# Phase 1 测试
# =============================================================================
print_section("Phase 1: 基础闭环系统测试")

# 1.1 异常检测引擎
print("\n[1.1] 异常检测引擎测试")
try:
    from app.closed_loop.detector.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    
    # 测试数据
    data = [10, 12, 11, 13, 12, 100, 11, 12]
    result = detector.detect(data, 'spike')
    print(f"  ✅ 异常检测成功: {result['type']}")
except Exception as e:
    log_error("Phase 1", "detector", "异常检测引擎", e)

# 1.2 自动修复引擎
print("\n[1.2] 自动修复引擎测试")
try:
    from app.closed_loop.remediator.remediator import AutoRemediator
    remediator = AutoRemediator()
    
    result = remediator.execute_action('restart', 'test_service', {'timeout': 60})
    print(f"  ✅ 修复执行: {result['status']}")
except Exception as e:
    log_error("Phase 1", "remediator", "自动修复引擎", e)

# 1.3 规则引擎
print("\n[1.3] 规则引擎测试")
try:
    from app.closed_loop.rules.rule_engine import RuleEngine
    engine = RuleEngine()
    
    rule = {
        'id': 'test_rule',
        'condition': {'metric': 'cpu', 'operator': '>', 'threshold': 80},
        'action': 'scale_up'
    }
    engine.add_rule_from_dict(rule)
    
    metrics = {'cpu': 85}
    triggered = engine.evaluate_rules(metrics)
    print(f"  ✅ 规则评估: {'触发' if triggered else '未触发'}")
except Exception as e:
    log_error("Phase 1", "rule_engine", "规则引擎", e)

# 1.4 闭环指标
print("\n[1.4] 闭环指标测试")
try:
    from app.closed_loop.metrics.closed_loop_metrics import ClosedLoopMetrics
    metrics = ClosedLoopMetrics()
    
    metrics.increment_counter('test_counter', labels={'service': 'test'})
    metrics.set_gauge('test_gauge', 42, labels={'service': 'test'})
    print(f"  ✅ 指标记录成功")
except Exception as e:
    log_error("Phase 1", "metrics", "闭环指标", e)

# 1.5 闭环控制器
print("\n[1.5] 闭环控制器测试")
try:
    from app.closed_loop.core.controller import ClosedLoopController
    controller = ClosedLoopController()
    
    status = controller.get_status()
    print(f"  ✅ 控制器状态: {status['status']}")
except Exception as e:
    log_error("Phase 1", "controller", "闭环控制器", e)

# =============================================================================
# Phase 2 测试
# =============================================================================
print_section("Phase 2: 智能闭环系统测试")

# 2.1 RCA 引擎 - 事件收集器
print("\n[2.1] RCA 事件收集器测试")
try:
    from app.closed_loop.rca.event_collector import EventCollector
    collector = EventCollector()
    
    event = {
        'id': 'test_event',
        'type': 'log_error',
        'source': 'test_service',
        'timestamp': __import__('datetime').datetime.now()
    }
    collector.collect_event(event)
    print(f"  ✅ 事件收集成功")
except Exception as e:
    log_error("Phase 2", "rca", "事件收集器", e)

# 2.2 RCA 引擎 - 关联分析器
print("\n[2.2] RCA 关联分析器测试")
try:
    from app.closed_loop.rca.correlation_analyzer import CorrelationAnalyzer
    analyzer = CorrelationAnalyzer()
    
    events = [
        {'id': 'e1', 'type': 'error', 'service': 'svc1', 'timestamp': __import__('datetime').datetime.now()},
        {'id': 'e2', 'type': 'error', 'service': 'svc2', 'timestamp': __import__('datetime').datetime.now()},
    ]
    result = analyzer.analyze_temporal_correlation(events)
    print(f"  ✅ 关联分析成功: {len(result)} 个相关事件")
except Exception as e:
    log_error("Phase 2", "rca", "关联分析器", e)

# 2.3 RCA 引擎 - 根因识别器
print("\n[2.3] RCA 根因识别器测试")
try:
    from app.closed_loop.rca.root_cause_identifier import RootCauseIdentifier
    identifier = RootCauseIdentifier()
    
    events = [{'type': 'error', 'service': 'test', 'metric': 'cpu', 'value': 90}]
    causes = identifier.identify_root_causes_bayesian(events)
    print(f"  ✅ 根因识别成功: {len(causes)} 个可能原因")
except Exception as e:
    log_error("Phase 2", "rca", "根因识别器", e)

# 2.4 RCA 引擎 - 报告生成器
print("\n[2.4] RCA 报告生成器测试")
try:
    from app.closed_loop.rca.report_generator import ReportGenerator
    generator = ReportGenerator(None, None, None)  # 参数为 None
    
    root_causes = [{'cause': 'code_bug', 'confidence': 0.9}]
    report = generator.generate_report('incident_001', root_causes, {})
    print(f"  ✅ 报告生成成功: {report['report_id']}")
except Exception as e:
    log_error("Phase 2", "rca", "报告生成器", e)

# 2.5 智能告警路由
print("\n[2.5] 智能告警路由测试")
try:
    import asyncio
    from app.closed_loop.alert.router import SmartAlertRouter
    router = SmartAlertRouter()
    
    alert = {'name': 'HighCPU', 'severity': 'critical', 'service': 'test'}
    team = asyncio.run(router.route_alert(alert))
    print(f"  ✅ 告警路由成功: {team}")
except Exception as e:
    log_error("Phase 2", "alert", "智能告警路由", e)

# 2.6 容量预测引擎
print("\n[2.6] 容量预测引擎测试")
try:
    from app.closed_loop.capacity.forecast_engine import ForecastEngine
    engine = ForecastEngine()
    
    data = [100, 110, 120, 130, 140]
    forecast = engine.forecast(data, model='linear', periods=3)
    print(f"  ✅ 预测成功: {len(forecast)} 个预测点")
except Exception as e:
    log_error("Phase 2", "capacity", "容量预测引擎", e)

# 2.7 容量规划器
print("\n[2.7] 容量规划器测试")
try:
    from app.closed_loop.capacity.planner import CapacityPlanner
    planner = CapacityPlanner()
    
    forecast = {'values': [100, 110, 120], 'upper': [110, 120, 130]}
    plans = planner.generate_plans('test_service', forecast, {})
    print(f"  ✅ 容量规划成功: {len(plans)} 个计划")
except Exception as e:
    log_error("Phase 2", "capacity", "容量规划器", e)

# 2.8 工作流引擎
print("\n[2.8] 工作流引擎测试")
try:
    from app.closed_loop.workflow.engine import WorkflowEngine
    wf_engine = WorkflowEngine()
    
    workflow_def = {
        'id': 'test_workflow',
        'states': ['start', 'end'],
        'transitions': [{'from': 'start', 'to': 'end', 'event': 'complete'}]
    }
    instance = wf_engine.create_workflow(workflow_def)
    print(f"  ✅ 工作流创建成功: {instance['instance_id']}")
except Exception as e:
    log_error("Phase 2", "workflow", "工作流引擎", e)

# =============================================================================
# Phase 3 测试
# =============================================================================
print_section("Phase 3: 自治闭环系统测试")

# 3.1 自学习优化引擎 - PPO
print("\n[3.1] PPO 策略学习器测试")
try:
    from app.closed_loop.learning.rl_policy_learner import PPOPolicyLearner
    ppo = PPOPolicyLearner(config={'population_size': 10})
    
    metrics = {'cpu_usage': 85, 'memory_usage': 70, 'error_rate': 0.05, 'latency_p99': 1500, 'qps': 5000}
    recommendation = ppo.get_action_recommendation(metrics)
    print(f"  ✅ PPO 推荐成功: {recommendation['action']}")
except Exception as e:
    log_error("Phase 3", "learning", "PPO策略学习器", e)

# 3.2 贝叶斯优化器
print("\n[3.2] 贝叶斯优化器测试")
try:
    from app.closed_loop.learning.bayesian_optimizer import BayesianOptimizer
    optimizer = BayesianOptimizer()
    optimizer.add_parameter('threshold', 'continuous', (0.0, 1.0))
    
    # 先随机采样
    for _ in range(3):
        params = {'threshold': __import__('numpy').random.uniform(0, 1)}
        optimizer._update_observation(params, 0.5)
    
    # 贝叶斯优化
    params = optimizer.suggest_next_point()
    print(f"  ✅ 贝叶斯优化成功: {params}")
except Exception as e:
    log_error("Phase 3", "learning", "贝叶斯优化器", e)

# 3.3 效果评估器
print("\n[3.3] 效果评估器测试")
try:
    from app.closed_loop.learning.effect_evaluator import EffectEvaluator
    evaluator = EffectEvaluator()
    
    before = {'error_rate': 0.1, 'latency_p99': 2000, 'availability': 0.95}
    after = {'error_rate': 0.02, 'latency_p99': 500, 'availability': 0.99}
    result = evaluator.evaluate_decision('dec_001', 'restart', before, after, 120, 50)
    print(f"  ✅ 效果评估成功: 评分={result['score']:.2f}")
except Exception as e:
    log_error("Phase 3", "learning", "效果评估器", e)

# 3.4 成本模型
print("\n[3.4] 成本模型测试")
try:
    from app.closed_loop.optimizer.cost_model import CostModelBuilder
    cost_model = CostModelBuilder()
    
    result = cost_model.calculate_total_cost(
        action='restart', action_params={},
        resource_usage={'compute': {'cpu_cores': 4, 'memory_gb': 16, 'duration_minutes': 10}},
        incident_impact={'downtime_minutes': 5, 'severity': 'medium'},
        decision_context={}
    )
    print(f"  ✅ 成本计算成功: ${result['total_cost']:.2f}")
except Exception as e:
    log_error("Phase 3", "optimizer", "成本模型", e)

# 3.5 风险评估引擎
print("\n[3.5] 风险评估引擎测试")
try:
    from app.closed_loop.optimizer.risk_engine import RiskEngine
    risk_engine = RiskEngine(service_graph={'svc_a': ['svc_b']})
    
    assessment = risk_engine.assess_risk('svc_a', {'cpu_high': True, 'error_spike': True})
    print(f"  ✅ 风险评估成功: {assessment.overall_risk.value}")
except Exception as e:
    log_error("Phase 3", "optimizer", "风险评估引擎", e)

# 3.6 NSGA-II 优化器
print("\n[3.6] NSGA-II 优化器测试")
try:
    from app.closed_loop.optimizer.pareto_optimizer import NSGAIIOptimizer
    nsga2 = NSGAIIOptimizer(config={'population_size': 20, 'generations': 10})
    
    nsga2.add_decision_variable('x1', 0, 1)
    nsga2.add_decision_variable('x2', 0, 1)
    nsga2.add_objective('minimize_cost', 'minimize')
    nsga2.set_evaluation_function(lambda v: {'minimize_cost': v['x1'] * 100})
    
    result = nsga2.optimize()
    print(f"  ✅ NSGA-II 优化成功: {len(result['pareto_front'])} 个Pareto解")
except Exception as e:
    log_error("Phase 3", "optimizer", "NSGA-II优化器", e)

# 3.7 实体抽取器
print("\n[3.7] 实体抽取器测试")
try:
    from app.closed_loop.knowledge.entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    
    log_entry = {'message': 'Service order-service calling user-service failed', 'source': 'logs'}
    entities = extractor.extract_from_log(log_entry)
    print(f"  ✅ 实体抽取成功: {len(entities)} 个实体")
except Exception as e:
    log_error("Phase 3", "knowledge", "实体抽取器", e)

# 3.8 关系构建器
print("\n[3.8] 关系构建器测试")
try:
    from app.closed_loop.knowledge.relation_builder import RelationBuilder
    builder = RelationBuilder()
    
    logs = [{'service': 'svc1', 'message': 'Calling svc2', 'timestamp': __import__('datetime').datetime.now()}]
    edges = builder.build_relations(entities=[], logs=logs)
    print(f"  ✅ 关系构建成功: {len(edges)} 个关系")
except Exception as e:
    log_error("Phase 3", "knowledge", "关系构建器", e)

# 3.9 GNN 推理引擎
print("\n[3.9] GNN 推理引擎测试")
try:
    from app.closed_loop.knowledge.gnn_engine import GraphReasoningEngine
    gnn = GraphReasoningEngine(num_features=6, hidden_dim=32)
    
    nodes = [{'id': 'n1', 'type': 'service', 'properties': {}}]
    edges = [{'source_id': 'n1', 'target_id': 'n2', 'relation_type': 'CALLS', 'properties': {}}]
    gnn.add_nodes(nodes)
    gnn.add_edges(edges)
    
    print(f"  ✅ GNN 引擎初始化成功")
except Exception as e:
    log_error("Phase 3", "knowledge", "GNN推理引擎", e)

# 3.10 知识查询接口
print("\n[3.10] 知识查询接口测试")
try:
    from app.closed_loop.knowledge.query_interface import KnowledgeQueryInterface
    query = KnowledgeQueryInterface()
    
    nodes = [{'id': 'svc1', 'type': 'service', 'properties': {}}]
    edges = [{'source_id': 'svc1', 'target_id': 'svc2', 'relation_type': 'CALLS', 'properties': {}}]
    query.set_knowledge_graph(nodes, edges)
    
    result = query.query("find services", 'natural')
    print(f"  ✅ 知识查询成功: {result.total_count} 个结果")
except Exception as e:
    log_error("Phase 3", "knowledge", "知识查询接口", e)

# 3.11 故障预测器
print("\n[3.11] 故障预测器测试")
try:
    from app.closed_loop.predictive.failure_predictor import FailurePredictor
    predictor = FailurePredictor(model_type='lstm')
    
    metrics_history = []
    for i in range(30):
        metrics_history.append({
            'cpu_usage': 50 + i * 2, 'memory_usage': 60 + i,
            'error_rate': 0.01 + i * 0.002, 'latency_p99': 500 + i * 50,
            'qps': 3000, 'disk_usage': 40
        })
    
    predictions = predictor.predict('test_svc', metrics_history, [1, 6])
    print(f"  ✅ 故障预测成功: {len(predictions)} 个预测")
except Exception as e:
    log_error("Phase 3", "predictive", "故障预测器", e)

# 3.12 影响分析器
print("\n[3.12] 影响分析器测试")
try:
    from app.closed_loop.predictive.impact_analyzer import ImpactAnalyzer
    analyzer = ImpactAnalyzer(topology={'a': ['b'], 'b': ['c']})
    
    impact = analyzer.analyze_impact('a', 'partial_degradation', 30)
    print(f"  ✅ 影响分析成功: 分数={impact['overall_impact_score']:.2f}")
except Exception as e:
    log_error("Phase 3", "predictive", "影响分析器", e)

# 3.13 预修复规划器
print("\n[3.13] 预修复规划器测试")
try:
    from app.closed_loop.predictive.prepair_planner import PrepairPlanner
    planner = PrepairPlanner()
    
    predictions = [{'service_id': 'svc1', 'failure_probability': 0.8, 'horizon_hours': 6, 'failure_type': 'cpu'}]
    impact_analyses = {'svc1': {'overall_severity': 'high', 'overall_impact_score': 0.7}}
    
    plan = planner.create_plan(predictions, impact_analyses)
    print(f"  ✅ 预修复规划成功: {len(plan.maintenance_windows)} 个维护窗口")
except Exception as e:
    log_error("Phase 3", "predictive", "预修复规划器", e)

# 3.14 拓扑构建器
print("\n[3.14] 拓扑构建器测试")
try:
    from app.closed_loop.orchestration.topology_builder import TopologyBuilder
    builder = TopologyBuilder()
    
    topology = builder.build_topology(['kubernetes'])
    print(f"  ✅ 拓扑构建成功: {topology['node_count']} 节点, {topology['edge_count']} 边")
except Exception as e:
    log_error("Phase 3", "orchestration", "拓扑构建器", e)

# 3.15 协同决策引擎
print("\n[3.15] 协同决策引擎测试")
try:
    from app.closed_loop.orchestration.consensus_engine import ConsensusEngine, VoteType
    engine = ConsensusEngine(node_id='test')
    
    proposal = engine.propose('restart', 'svc1', {})
    engine.vote(proposal.proposal_id, VoteType.APPROVE, voter='v1')
    engine.vote(proposal.proposal_id, VoteType.APPROVE, voter='v2')
    engine.vote(proposal.proposal_id, VoteType.APPROVE, voter='v3')
    
    decision = engine.get_decision(f"dec_{proposal.proposal_id}")
    print(f"  ✅ 协同决策成功: {decision.status.value}")
except Exception as e:
    log_error("Phase 3", "orchestration", "协同决策引擎", e)

# 3.16 级联控制器
print("\n[3.16] 级联控制器测试")
try:
    from app.closed_loop.orchestration.cascade_controller import CascadeController
    controller = CascadeController(topology={'a': ['b']})
    
    controller.register_service('svc1')
    result = controller.handle_failure('svc1', 'timeout', {'isolate': True, 'degrade': True})
    print(f"  ✅ 级联控制成功: {len(result['actions_taken'])} 个动作")
except Exception as e:
    log_error("Phase 3", "orchestration", "级联控制器", e)

# 3.17 XAI 解释器
print("\n[3.17] XAI 解释器测试")
try:
    from app.closed_loop.human.xai_explainer import XAIExplainer
    explainer = XAIExplainer()
    
    features = {'cpu_usage': 85, 'memory_usage': 70, 'error_rate': 0.08, 'latency_p99': 2000, 'qps': 5000}
    explanation = explainer.explain_decision('dec_001', 'scale_up', features, 0.85)
    print(f"  ✅ XAI 解释成功: {len(explanation.feature_importance)} 个特征重要性")
except Exception as e:
    log_error("Phase 3", "human", "XAI解释器", e)

# 3.18 审批工作流引擎
print("\n[3.18] 审批工作流引擎测试")
try:
    from app.closed_loop.human.approval_workflow import ApprovalWorkflowEngine, ApprovalUrgency
    engine = ApprovalWorkflowEngine()
    
    request = engine.create_request('restart', 'svc1', {}, 'user1', ApprovalUrgency.HIGH)
    result = engine.approve(request.request_id, 'approver1')
    print(f"  ✅ 审批工作流成功: {request.status.value}")
except Exception as e:
    log_error("Phase 3", "human", "审批工作流引擎", e)

# 3.19 专家知识系统
print("\n[3.19] 专家知识系统测试")
try:
    from app.closed_loop.human.expert_knowledge import ExpertKnowledgeSystem
    eks = ExpertKnowledgeSystem()
    
    context = {'cpu_usage': 85, 'memory_usage': 95, 'error_rate': 0.15}
    recommendations = eks.get_recommendations(context)
    print(f"  ✅ 专家知识系统成功: {len(recommendations)} 个推荐")
except Exception as e:
    log_error("Phase 3", "human", "专家知识系统", e)

# =============================================================================
# 总结
# =============================================================================
print_section("测试结果总结")

print(f"\n总测试数: 19")
print(f"错误数: {len(errors)}")
print(f"警告数: {len(warnings_list)}")

if errors:
    print("\n" + "="*80)
    print("发现的错误列表:")
    print("="*80)
    for i, err in enumerate(errors, 1):
        print(f"\n[{i}] Phase: {err['phase']}, Module: {err['module']}, Test: {err['test']}")
        print(f"    Error: {err['error'][:300]}")
        print(f"    Traceback (last 5 lines):")
        tb_lines = err['traceback'].strip().split('\n')[-5:]
        for line in tb_lines:
            print(f"      {line}")
else:
    print("\n✅ 未发现任何错误！所有测试通过！")

if warnings_list:
    print("\n" + "="*80)
    print("警告列表:")
    print("="*80)
    for w in warnings_list:
        print(f"  - {w}")

print("\n" + "="*80)
print("Bug 检测测试完成")
print("="*80)
