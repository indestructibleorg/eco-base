"""
知识查询接口
支持 Cypher 查询和自然语言查询
"""

import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    results: List[Dict[str, Any]]
    execution_time_ms: float
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': self.results,
            'execution_time_ms': self.execution_time_ms,
            'total_count': self.total_count
        }


class CypherQueryParser:
    """Cypher 查询解析器"""
    
    def __init__(self):
        # 简化的 Cypher 语法支持
        self.patterns = {
            'match_node': r'MATCH\s*\((\w+):?(\w*)\)',
            'match_rel': r'MATCH\s*\((\w+)\)-\[:?(\w*)\]->\((\w+)\)',
            'where': r'WHERE\s+(.+?)(?:RETURN|LIMIT|ORDER|$)',
            'return': r'RETURN\s+(.+?)(?:LIMIT|ORDER|$)',
            'limit': r'LIMIT\s+(\d+)',
        }
        
    def parse(self, query: str) -> Dict[str, Any]:
        """解析 Cypher 查询"""
        parsed = {
            'type': 'unknown',
            'nodes': [],
            'relationships': [],
            'where_clause': None,
            'return_clause': None,
            'limit': None
        }
        
        query_upper = query.upper()
        
        # 解析 MATCH 节点
        node_matches = re.finditer(self.patterns['match_node'], query, re.IGNORECASE)
        for match in node_matches:
            parsed['nodes'].append({
                'variable': match.group(1),
                'label': match.group(2) if match.group(2) else None
            })
        
        # 解析 MATCH 关系
        rel_matches = re.finditer(self.patterns['match_rel'], query, re.IGNORECASE)
        for match in rel_matches:
            parsed['relationships'].append({
                'source': match.group(1),
                'type': match.group(2) if match.group(2) else None,
                'target': match.group(3)
            })
        
        # 解析 WHERE
        where_match = re.search(self.patterns['where'], query, re.IGNORECASE)
        if where_match:
            parsed['where_clause'] = where_match.group(1).strip()
        
        # 解析 RETURN
        return_match = re.search(self.patterns['return'], query, re.IGNORECASE)
        if return_match:
            parsed['return_clause'] = return_match.group(1).strip()
        
        # 解析 LIMIT
        limit_match = re.search(self.patterns['limit'], query, re.IGNORECASE)
        if limit_match:
            parsed['limit'] = int(limit_match.group(1))
        
        # 确定查询类型
        if 'MATCH' in query_upper:
            parsed['type'] = 'match'
        
        return parsed


class NaturalLanguageQueryParser:
    """自然语言查询解析器"""
    
    def __init__(self):
        # 查询模板
        self.templates = [
            {
                'pattern': r'(?:find|get|show|list)\s+(?:all\s+)?(?:the\s+)?(\w+)\s+(?:that|which)\s+(?:depend\s+on|call)\s+(\w+)',
                'query_type': 'find_dependencies',
                'extractor': lambda m: {'target_type': m.group(1), 'source': m.group(2)}
            },
            {
                'pattern': r'(?:find|get|show|list)\s+(?:all\s+)?(?:the\s+)?(\w+)\s+(?:related\s+to|connected\s+to|for)\s+(\w+)',
                'query_type': 'find_related',
                'extractor': lambda m: {'entity_type': m.group(1), 'entity': m.group(2)}
            },
            {
                'pattern': r'(?:what|which)\s+(\w+)\s+(?:is|are)\s+(?:affected\s+by|impacted\s+by)\s+(\w+)',
                'query_type': 'find_affected',
                'extractor': lambda m: {'target_type': m.group(1), 'source': m.group(2)}
            },
            {
                'pattern': r'(?:find|get|show)\s+(?:the\s+)?root\s+cause\s+(?:of\s+)?(\w+)',
                'query_type': 'find_root_cause',
                'extractor': lambda m: {'target': m.group(1)}
            },
            {
                'pattern': r'(?:find|get|show)\s+(?:the\s+)?similar\s+(\w+)\s+(?:to\s+)?(\w+)',
                'query_type': 'find_similar',
                'extractor': lambda m: {'entity_type': m.group(1), 'entity': m.group(2)}
            },
            {
                'pattern': r'(?:what|which)\s+(\w+)\s+(?:has|have)\s+(?:the\s+)?most\s+(\w+)',
                'query_type': 'find_max',
                'extractor': lambda m: {'entity_type': m.group(1), 'metric': m.group(2)}
            },
        ]
        
    def parse(self, query: str) -> Optional[Dict[str, Any]]:
        """解析自然语言查询"""
        query_lower = query.lower()
        
        for template in self.templates:
            match = re.search(template['pattern'], query_lower)
            if match:
                return {
                    'query_type': template['query_type'],
                    'parameters': template['extractor'](match),
                    'original_query': query
                }
        
        return None


class KnowledgeQueryInterface:
    """
    知识查询接口主类
    支持 Cypher 查询和自然语言查询
    """
    
    def __init__(self, knowledge_graph: Any = None):
        self.knowledge_graph = knowledge_graph
        self.cypher_parser = CypherQueryParser()
        self.nl_parser = NaturalLanguageQueryParser()
        
        # 查询执行器
        self.query_executors: Dict[str, Callable] = {
            'match': self._execute_match,
            'find_dependencies': self._execute_find_dependencies,
            'find_related': self._execute_find_related,
            'find_affected': self._execute_find_affected,
            'find_root_cause': self._execute_find_root_cause,
            'find_similar': self._execute_find_similar,
            'find_max': self._execute_find_max,
        }
        
        # 内存中的知识图谱 (简化实现)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        
    def set_knowledge_graph(self, nodes: List[Dict], relationships: List[Dict]):
        """设置知识图谱数据"""
        for node in nodes:
            self.nodes[node.get('id')] = node
        self.relationships = relationships
        
    def query(self, query: str, query_type: str = 'auto') -> QueryResult:
        """
        执行查询
        
        Args:
            query: 查询字符串
            query_type: 'cypher', 'natural', 或 'auto'
        
        Returns:
            查询结果
        """
        import time
        start_time = time.time()
        
        if query_type == 'auto':
            # 自动检测查询类型
            if query.strip().upper().startswith('MATCH'):
                query_type = 'cypher'
            else:
                query_type = 'natural'
        
        if query_type == 'cypher':
            results = self._execute_cypher(query)
        else:
            results = self._execute_natural(query)
        
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            query=query,
            results=results,
            execution_time_ms=execution_time,
            total_count=len(results)
        )
    
    def _execute_cypher(self, query: str) -> List[Dict[str, Any]]:
        """执行 Cypher 查询"""
        parsed = self.cypher_parser.parse(query)
        
        executor = self.query_executors.get(parsed['type'])
        if executor:
            return executor(parsed)
        
        return []
    
    def _execute_natural(self, query: str) -> List[Dict[str, Any]]:
        """执行自然语言查询"""
        parsed = self.nl_parser.parse(query)
        
        if parsed:
            executor = self.query_executors.get(parsed['query_type'])
            if executor:
                return executor(parsed['parameters'])
        
        return [{'error': 'Could not parse query', 'query': query}]
    
    def _execute_match(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行 MATCH 查询"""
        results = []
        
        # 简化的 MATCH 实现
        if parsed['relationships']:
            for rel in self.relationships:
                match = True
                
                # 检查关系类型
                if parsed['relationships']:
                    rel_type = parsed['relationships'][0].get('type')
                    if rel_type and rel.get('relation_type') != rel_type:
                        match = False
                
                if match:
                    source_node = self.nodes.get(rel.get('source_id'), {})
                    target_node = self.nodes.get(rel.get('target_id'), {})
                    
                    result = {
                        'source': source_node,
                        'relationship': rel,
                        'target': target_node
                    }
                    results.append(result)
        
        elif parsed['nodes']:
            # 只查询节点
            label = parsed['nodes'][0].get('label')
            for node_id, node in self.nodes.items():
                if label is None or node.get('type') == label:
                    results.append({'node': node})
        
        # 应用 LIMIT
        if parsed['limit']:
            results = results[:parsed['limit']]
        
        return results
    
    def _execute_find_dependencies(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找依赖"""
        source = params.get('source', '').lower()
        target_type = params.get('target_type', '').lower()
        
        results = []
        for rel in self.relationships:
            if rel.get('source_id', '').lower() == source:
                if rel.get('relation_type') in ['DEPENDS_ON', 'CALLS']:
                    target = self.nodes.get(rel.get('target_id'))
                    if target and (not target_type or target.get('type') == target_type):
                        results.append({
                            'source': source,
                            'relationship': rel.get('relation_type'),
                            'target': target
                        })
        
        return results
    
    def _execute_find_related(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相关实体"""
        entity = params.get('entity', '').lower()
        entity_type = params.get('entity_type', '').lower()
        
        results = []
        visited = set()
        
        # BFS 查找相关实体
        queue = [entity]
        visited.add(entity)
        
        while queue:
            current = queue.pop(0)
            
            for rel in self.relationships:
                other = None
                if rel.get('source_id', '').lower() == current:
                    other = rel.get('target_id')
                elif rel.get('target_id', '').lower() == current:
                    other = rel.get('source_id')
                
                if other and other not in visited:
                    visited.add(other)
                    other_node = self.nodes.get(other)
                    
                    if other_node and (not entity_type or other_node.get('type') == entity_type):
                        results.append({
                            'entity': other_node,
                            'relationship': rel.get('relation_type'),
                            'connected_via': current
                        })
                    
                    queue.append(other)
        
        return results
    
    def _execute_find_affected(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找受影响的实体"""
        source = params.get('source', '').lower()
        target_type = params.get('target_type', '').lower()
        
        # 使用传播算法
        affected = self._propagate_impact([source], max_depth=3)
        
        results = []
        for entity_id in affected:
            if entity_id != source:
                node = self.nodes.get(entity_id)
                if node and (not target_type or node.get('type') == target_type):
                    results.append({'affected_entity': node})
        
        return results
    
    def _execute_find_root_cause(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找根因"""
        target = params.get('target', '').lower()
        
        # 反向追踪可能的原因
        causes = []
        
        for rel in self.relationships:
            if rel.get('target_id', '').lower() == target:
                if rel.get('relation_type') in ['CAUSES', 'GRANGER_CAUSES']:
                    source = self.nodes.get(rel.get('source_id'))
                    if source:
                        causes.append({
                            'potential_cause': source,
                            'confidence': rel.get('confidence', 0.5),
                            'relation_type': rel.get('relation_type')
                        })
        
        # 按置信度排序
        causes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return causes
    
    def _execute_find_similar(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找相似实体"""
        entity = params.get('entity', '').lower()
        entity_type = params.get('entity_type', '').lower()
        
        results = []
        
        for rel in self.relationships:
            if rel.get('relation_type') == 'SIMILAR_TO':
                other = None
                if rel.get('source_id', '').lower() == entity:
                    other = rel.get('target_id')
                elif rel.get('target_id', '').lower() == entity:
                    other = rel.get('source_id')
                
                if other:
                    other_node = self.nodes.get(other)
                    if other_node and (not entity_type or other_node.get('type') == entity_type):
                        results.append({
                            'similar_entity': other_node,
                            'similarity_score': rel.get('properties', {}).get('similarity_score', 0)
                        })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results
    
    def _execute_find_max(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找最大值"""
        entity_type = params.get('entity_type', '').lower()
        metric = params.get('metric', '').lower()
        
        # 简化的实现
        candidates = []
        for node_id, node in self.nodes.items():
            if not entity_type or node.get('type') == entity_type:
                value = node.get('properties', {}).get(metric, 0)
                candidates.append((node, value))
        
        # 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [{'entity': c[0], metric: c[1]} for c in candidates[:10]]
    
    def _propagate_impact(self, sources: List[str], max_depth: int = 3) -> set:
        """传播影响"""
        affected = set(sources)
        current_level = set(sources)
        
        for _ in range(max_depth):
            next_level = set()
            for entity in current_level:
                for rel in self.relationships:
                    if rel.get('source_id', '').lower() == entity:
                        target = rel.get('target_id', '').lower()
                        if target not in affected:
                            affected.add(target)
                            next_level.add(target)
            current_level = next_level
        
        return affected
    
    def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取子图"""
        nodes = set()
        edges = []
        
        entity_id = entity_id.lower()
        current_level = {entity_id}
        nodes.add(entity_id)
        
        for _ in range(depth):
            next_level = set()
            for entity in current_level:
                for rel in self.relationships:
                    if rel.get('source_id', '').lower() == entity:
                        target = rel.get('target_id', '').lower()
                        nodes.add(target)
                        next_level.add(target)
                        edges.append(rel)
                    elif rel.get('target_id', '').lower() == entity:
                        source = rel.get('source_id', '').lower()
                        nodes.add(source)
                        next_level.add(source)
                        edges.append(rel)
            current_level = next_level
        
        return {
            'center': entity_id,
            'depth': depth,
            'nodes': [self.nodes.get(n, {'id': n}) for n in nodes],
            'edges': edges
        }
    
    def explain_query(self, query: str) -> Dict[str, Any]:
        """解释查询"""
        # 解析 Cypher
        cypher_parsed = self.cypher_parser.parse(query)
        
        # 解析自然语言
        nl_parsed = self.nl_parser.parse(query)
        
        return {
            'original_query': query,
            'cypher_parsed': cypher_parsed,
            'natural_language_parsed': nl_parsed,
            'explanation': self._generate_explanation(cypher_parsed, nl_parsed)
        }
    
    def _generate_explanation(self, cypher: Dict, nl: Dict) -> str:
        """生成查询解释"""
        if nl:
            query_types = {
                'find_dependencies': '查找依赖关系',
                'find_related': '查找相关实体',
                'find_affected': '查找受影响的实体',
                'find_root_cause': '查找根因',
                'find_similar': '查找相似实体',
                'find_max': '查找最大值'
            }
            return f"这是一个{query_types.get(nl['query_type'], '未知')}查询"
        
        if cypher.get('type') == 'match':
            return "这是一个 MATCH 查询，用于匹配图中的节点和关系"
        
        return "无法识别查询类型"
