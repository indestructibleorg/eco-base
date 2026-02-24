"""
知识图谱系统

提供以下功能:
- 实体抽取与识别
- 关系构建
- 图神经网络推理
- 知识查询接口
"""

from .entity_extractor import (
    EntityExtractor,
    Entity,
    EntityType,
    PatternMatcher
)

from .relation_builder import (
    RelationBuilder,
    Relationship,
    DependencyDiscoverer,
    CausalityAnalyzer,
    SimilarityCalculator
)

from .gnn_engine import (
    GNNReasoner,
    GraphAttentionLayer,
    KnowledgeGraphEmbedder,
    GraphReasoningEngine
)

from .query_interface import (
    KnowledgeQueryInterface,
    CypherQueryParser,
    NaturalLanguageQueryParser,
    QueryResult
)

__all__ = [
    # Entity Extractor
    'EntityExtractor',
    'Entity',
    'EntityType',
    'PatternMatcher',
    
    # Relation Builder
    'RelationBuilder',
    'Relationship',
    'DependencyDiscoverer',
    'CausalityAnalyzer',
    'SimilarityCalculator',
    
    # GNN Engine
    'GNNReasoner',
    'GraphAttentionLayer',
    'KnowledgeGraphEmbedder',
    'GraphReasoningEngine',
    
    # Query Interface
    'KnowledgeQueryInterface',
    'CypherQueryParser',
    'NaturalLanguageQueryParser',
    'QueryResult'
]
