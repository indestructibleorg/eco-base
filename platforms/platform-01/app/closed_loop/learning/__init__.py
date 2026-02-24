"""
自学习优化引擎

提供以下功能:
- 强化学习策略学习 (PPO)
- 贝叶斯参数优化
- 效果评估与 A/B 测试
- 在线学习与模型更新
"""

from .rl_policy_learner import (
    PPOPolicyLearner,
    ActorNetwork,
    CriticNetwork,
    StateEncoder,
    RewardCalculator,
    Experience,
    ExperienceReplayBuffer
)

from .bayesian_optimizer import (
    BayesianOptimizer,
    MultiObjectiveBayesianOptimizer,
    GaussianProcess,
    ParameterSpace,
    AcquisitionFunction
)

from .effect_evaluator import (
    EffectEvaluator,
    ABTestFramework,
    CausalInferenceAnalyzer,
    LongTermEffectTracker,
    DecisionRecord,
    OutcomeMetrics,
    Experiment,
    ExperimentType,
    EvaluationStatus
)

from .online_learner import (
    OnlineLearner,
    IncrementalLearner,
    ConceptDriftDetector,
    ModelVersionManager,
    ModelVersion
)

__all__ = [
    # RL Policy Learner
    'PPOPolicyLearner',
    'ActorNetwork',
    'CriticNetwork',
    'StateEncoder',
    'RewardCalculator',
    'Experience',
    'ExperienceReplayBuffer',
    
    # Bayesian Optimizer
    'BayesianOptimizer',
    'MultiObjectiveBayesianOptimizer',
    'GaussianProcess',
    'ParameterSpace',
    'AcquisitionFunction',
    
    # Effect Evaluator
    'EffectEvaluator',
    'ABTestFramework',
    'CausalInferenceAnalyzer',
    'LongTermEffectTracker',
    'DecisionRecord',
    'OutcomeMetrics',
    'Experiment',
    'ExperimentType',
    'EvaluationStatus',
    
    # Online Learner
    'OnlineLearner',
    'IncrementalLearner',
    'ConceptDriftDetector',
    'ModelVersionManager',
    'ModelVersion'
]
