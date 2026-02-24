"""
专家知识集成系统
规则注入、案例库、反馈学习
"""

import hashlib
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpertRule:
    """专家规则"""
    rule_id: str
    name: str
    condition: str
    action: str
    confidence: float
    author: str
    created_at: datetime
    usage_count: int
    success_rate: float
    enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'condition': self.condition,
            'action': self.action,
            'confidence': self.confidence,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'usage_count': self.usage_count,
            'success_rate': self.success_rate,
            'enabled': self.enabled
        }


@dataclass
class CaseStudy:
    """案例"""
    case_id: str
    title: str
    description: str
    symptoms: List[str]
    root_cause: str
    solution: str
    outcome: str
    tags: List[str]
    created_by: str
    created_at: datetime
    relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'title': self.title,
            'description': self.description,
            'symptoms': self.symptoms,
            'root_cause': self.root_cause,
            'solution': self.solution,
            'outcome': self.outcome,
            'tags': self.tags,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'relevance_score': self.relevance_score
        }


@dataclass
class FeedbackRecord:
    """反馈记录"""
    feedback_id: str
    decision_id: str
    rating: int  # 1-5
    comments: str
    expected_action: str
    actual_action: str
    timestamp: datetime
    user: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feedback_id': self.feedback_id,
            'decision_id': self.decision_id,
            'rating': self.rating,
            'comments': self.comments,
            'expected_action': self.expected_action,
            'actual_action': self.actual_action,
            'timestamp': self.timestamp.isoformat(),
            'user': self.user
        }


class RuleInjector:
    """规则注入器"""
    
    def __init__(self):
        self.rules: Dict[str, ExpertRule] = {}
        self.rule_categories: Dict[str, List[str]] = defaultdict(list)
        
    def add_rule(self, name: str, condition: str, action: str,
                author: str, confidence: float = 0.9) -> ExpertRule:
        """添加规则"""
        rule_id = f"rule_{hashlib.md5(f"{name}_{condition}".encode()).hexdigest()[:12]}"
        
        rule = ExpertRule(
            rule_id=rule_id,
            name=name,
            condition=condition,
            action=action,
            confidence=confidence,
            author=author,
            created_at=datetime.now(),
            usage_count=0,
            success_rate=1.0,
            enabled=True
        )
        
        self.rules[rule_id] = rule
        
        # 分类
        category = self._categorize_rule(condition)
        self.rule_categories[category].append(rule_id)
        
        logger.info(f"Added expert rule: {rule_id} - {name}")
        
        return rule
    
    def _categorize_rule(self, condition: str) -> str:
        """规则分类"""
        condition_lower = condition.lower()
        
        if 'cpu' in condition_lower:
            return 'cpu'
        elif 'memory' in condition_lower or 'mem' in condition_lower:
            return 'memory'
        elif 'error' in condition_lower:
            return 'error'
        elif 'latency' in condition_lower:
            return 'latency'
        elif 'disk' in condition_lower:
            return 'disk'
        else:
            return 'general'
    
    def evaluate_rule(self, rule_id: str, context: Dict[str, Any]) -> bool:
        """评估规则条件"""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return False
        
        # 简化的条件评估
        condition = rule.condition.lower()
        
        # CPU 条件
        if 'cpu >' in condition:
            threshold = self._extract_threshold(condition)
            return context.get('cpu_usage', 0) > threshold
        
        # 内存条件
        if 'memory >' in condition:
            threshold = self._extract_threshold(condition)
            return context.get('memory_usage', 0) > threshold
        
        # 错误率条件
        if 'error_rate >' in condition:
            threshold = self._extract_threshold(condition)
            return context.get('error_rate', 0) > threshold
        
        # 延迟条件
        if 'latency >' in condition:
            threshold = self._extract_threshold(condition)
            return context.get('latency_p99', 0) > threshold
        
        return False
    
    def _extract_threshold(self, condition: str) -> float:
        """从条件中提取阈值"""
        import re
        match = re.search(r'>\s*(\d+\.?\d*)', condition)
        if match:
            return float(match.group(1))
        return 0
    
    def find_applicable_rules(self, context: Dict[str, Any]) -> List[ExpertRule]:
        """查找适用的规则"""
        applicable = []
        
        for rule in self.rules.values():
            if rule.enabled and self.evaluate_rule(rule.rule_id, context):
                applicable.append(rule)
        
        # 按置信度和成功率排序
        applicable.sort(key=lambda r: (r.confidence * r.success_rate), reverse=True)
        
        return applicable
    
    def update_rule_performance(self, rule_id: str, success: bool):
        """更新规则性能"""
        rule = self.rules.get(rule_id)
        if rule:
            rule.usage_count += 1
            
            # 更新成功率 (指数移动平均)
            alpha = 0.1
            success_val = 1.0 if success else 0.0
            rule.success_rate = (1 - alpha) * rule.success_rate + alpha * success_val
    
    def get_rules_by_category(self, category: str) -> List[ExpertRule]:
        """按类别获取规则"""
        rule_ids = self.rule_categories.get(category, [])
        return [self.rules[rid] for rid in rule_ids if rid in self.rules]


class CaseLibrary:
    """案例库"""
    
    def __init__(self):
        self.cases: Dict[str, CaseStudy] = {}
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.symptom_index: Dict[str, List[str]] = defaultdict(list)
        
    def add_case(self, title: str, description: str, symptoms: List[str],
                root_cause: str, solution: str, outcome: str,
                tags: List[str], created_by: str) -> CaseStudy:
        """添加案例"""
        case_id = f"case_{hashlib.md5(title.encode()).hexdigest()[:12]}"
        
        case = CaseStudy(
            case_id=case_id,
            title=title,
            description=description,
            symptoms=symptoms,
            root_cause=root_cause,
            solution=solution,
            outcome=outcome,
            tags=tags,
            created_by=created_by,
            created_at=datetime.now(),
            relevance_score=1.0
        )
        
        self.cases[case_id] = case
        
        # 建立索引
        for tag in tags:
            self.tag_index[tag.lower()].append(case_id)
        
        for symptom in symptoms:
            self.symptom_index[symptom.lower()].append(case_id)
        
        logger.info(f"Added case study: {case_id} - {title}")
        
        return case
    
    def search_cases(self, query: str = None, tags: List[str] = None,
                    symptoms: List[str] = None, limit: int = 10) -> List[CaseStudy]:
        """搜索案例"""
        candidates = set()
        
        # 标签搜索
        if tags:
            for tag in tags:
                candidates.update(self.tag_index.get(tag.lower(), []))
        
        # 症状搜索
        if symptoms:
            for symptom in symptoms:
                candidates.update(self.symptom_index.get(symptom.lower(), []))
        
        # 如果没有指定条件，返回所有案例
        if not tags and not symptoms:
            candidates = set(self.cases.keys())
        
        # 计算相关性
        results = []
        for case_id in candidates:
            case = self.cases.get(case_id)
            if case:
                relevance = self._calculate_relevance(case, query, tags, symptoms)
                case.relevance_score = relevance
                results.append(case)
        
        # 排序
        results.sort(key=lambda c: c.relevance_score, reverse=True)
        
        return results[:limit]
    
    def _calculate_relevance(self, case: CaseStudy, query: str = None,
                            tags: List[str] = None, symptoms: List[str] = None) -> float:
        """计算相关性"""
        score = 0.0
        
        # 标签匹配
        if tags:
            matching_tags = sum(1 for t in tags if t.lower() in [ct.lower() for ct in case.tags])
            score += matching_tags * 0.3
        
        # 症状匹配
        if symptoms:
            matching_symptoms = sum(1 for s in symptoms if s.lower() in [cs.lower() for cs in case.symptoms])
            score += matching_symptoms * 0.5
        
        # 文本匹配
        if query:
            query_lower = query.lower()
            if query_lower in case.title.lower():
                score += 0.2
            if query_lower in case.description.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def get_similar_cases(self, case_id: str, limit: int = 5) -> List[CaseStudy]:
        """获取相似案例"""
        case = self.cases.get(case_id)
        if not case:
            return []
        
        return self.search_cases(tags=case.tags, symptoms=case.symptoms, limit=limit)


class FeedbackLearner:
    """反馈学习器"""
    
    def __init__(self):
        self.feedback_records: Dict[str, FeedbackRecord] = {}
        self.decision_feedback: Dict[str, List[str]] = defaultdict(list)
        
    def record_feedback(self, decision_id: str, rating: int,
                       comments: str, expected_action: str,
                       actual_action: str, user: str) -> FeedbackRecord:
        """记录反馈"""
        feedback_id = f"fb_{decision_id}_{int(datetime.now().timestamp())}"
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            decision_id=decision_id,
            rating=rating,
            comments=comments,
            expected_action=expected_action,
            actual_action=actual_action,
            timestamp=datetime.now(),
            user=user
        )
        
        self.feedback_records[feedback_id] = feedback
        self.decision_feedback[decision_id].append(feedback_id)
        
        logger.info(f"Recorded feedback for decision {decision_id}: rating={rating}")
        
        return feedback
    
    def get_feedback_for_decision(self, decision_id: str) -> List[FeedbackRecord]:
        """获取决策的反馈"""
        feedback_ids = self.decision_feedback.get(decision_id, [])
        return [self.feedback_records[fid] for fid in feedback_ids if fid in self.feedback_records]
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """分析反馈"""
        if not self.feedback_records:
            return {'error': 'No feedback records'}
        
        ratings = [f.rating for f in self.feedback_records.values()]
        
        # 计算统计
        avg_rating = sum(ratings) / len(ratings)
        
        # 满意度分布
        satisfaction_dist = {
            'very_satisfied': sum(1 for r in ratings if r == 5),
            'satisfied': sum(1 for r in ratings if r == 4),
            'neutral': sum(1 for r in ratings if r == 3),
            'dissatisfied': sum(1 for r in ratings if r == 2),
            'very_dissatisfied': sum(1 for r in ratings if r == 1)
        }
        
        # 常见反馈主题
        common_issues = self._extract_common_issues()
        
        return {
            'total_feedback': len(ratings),
            'average_rating': avg_rating,
            'satisfaction_distribution': satisfaction_dist,
            'common_issues': common_issues,
            'improvement_suggestions': self._extract_suggestions()
        }
    
    def _extract_common_issues(self) -> List[str]:
        """提取常见问题"""
        # 简化的关键词提取
        issues = []
        
        for feedback in self.feedback_records.values():
            if feedback.rating <= 2:
                comments = feedback.comments.lower()
                if 'slow' in comments or 'delay' in comments:
                    issues.append('slow_response')
                if 'wrong' in comments or 'incorrect' in comments:
                    issues.append('incorrect_action')
                if 'miss' in comments:
                    issues.append('missed_detection')
        
        return list(set(issues))
    
    def _extract_suggestions(self) -> List[str]:
        """提取改进建议"""
        suggestions = []
        
        for feedback in self.feedback_records.values():
            if 'should' in feedback.comments.lower():
                suggestions.append(feedback.comments)
        
        return suggestions[:10]


class ExpertKnowledgeSystem:
    """
    专家知识系统主类
    整合规则注入、案例库、反馈学习
    """
    
    def __init__(self):
        self.rule_injector = RuleInjector()
        self.case_library = CaseLibrary()
        self.feedback_learner = FeedbackLearner()
        
        # 初始化默认规则
        self._init_default_rules()
        
    def _init_default_rules(self):
        """初始化默认规则"""
        default_rules = [
            {
                'name': 'High CPU Auto-Scale',
                'condition': 'cpu_usage > 80',
                'action': 'scale_up',
                'confidence': 0.9
            },
            {
                'name': 'Memory Leak Restart',
                'condition': 'memory_usage > 90',
                'action': 'restart',
                'confidence': 0.85
            },
            {
                'name': 'High Error Rate Rollback',
                'condition': 'error_rate > 0.1',
                'action': 'rollback',
                'confidence': 0.8
            },
            {
                'name': 'High Latency Circuit Break',
                'condition': 'latency_p99 > 5000',
                'action': 'circuit_break',
                'confidence': 0.75
            }
        ]
        
        for rule in default_rules:
            self.rule_injector.add_rule(
                name=rule['name'],
                condition=rule['condition'],
                action=rule['action'],
                author='system',
                confidence=rule['confidence']
            )
    
    def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取推荐"""
        recommendations = []
        
        # 查找适用的规则
        applicable_rules = self.rule_injector.find_applicable_rules(context)
        
        for rule in applicable_rules[:3]:  # 返回前3个
            recommendations.append({
                'type': 'rule',
                'source': rule.name,
                'action': rule.action,
                'confidence': rule.confidence * rule.success_rate,
                'rule_id': rule.rule_id
            })
        
        # 查找相似案例
        symptoms = self._context_to_symptoms(context)
        similar_cases = self.case_library.search_cases(symptoms=symptoms, limit=3)
        
        for case in similar_cases:
            recommendations.append({
                'type': 'case',
                'source': case.title,
                'action': case.solution,
                'confidence': case.relevance_score,
                'case_id': case.case_id
            })
        
        # 排序
        recommendations.sort(key=lambda r: r['confidence'], reverse=True)
        
        return recommendations
    
    def _context_to_symptoms(self, context: Dict[str, Any]) -> List[str]:
        """将上下文转换为症状"""
        symptoms = []
        
        if context.get('cpu_usage', 0) > 80:
            symptoms.append('high_cpu')
        
        if context.get('memory_usage', 0) > 80:
            symptoms.append('high_memory')
        
        if context.get('error_rate', 0) > 0.05:
            symptoms.append('high_error_rate')
        
        if context.get('latency_p99', 0) > 1000:
            symptoms.append('high_latency')
        
        return symptoms
    
    def learn_from_feedback(self):
        """从反馈学习"""
        analysis = self.feedback_learner.analyze_feedback()
        
        # 更新规则性能
        for feedback in self.feedback_learner.feedback_records.values():
            if feedback.rating >= 4:
                # 正面反馈 - 增强相关规则
                pass  # 实际实现需要关联反馈和规则
            elif feedback.rating <= 2:
                # 负面反馈 - 削弱相关规则
                pass
        
        return analysis
    
    def add_expert_rule(self, name: str, condition: str, action: str,
                       author: str, confidence: float = 0.9) -> ExpertRule:
        """添加专家规则"""
        return self.rule_injector.add_rule(name, condition, action, author, confidence)
    
    def add_case_study(self, title: str, description: str, symptoms: List[str],
                      root_cause: str, solution: str, outcome: str,
                      tags: List[str], created_by: str) -> CaseStudy:
        """添加案例"""
        return self.case_library.add_case(
            title, description, symptoms, root_cause, 
            solution, outcome, tags, created_by
        )
    
    def record_feedback(self, decision_id: str, rating: int,
                       comments: str, expected_action: str,
                       actual_action: str, user: str) -> FeedbackRecord:
        """记录反馈"""
        return self.feedback_learner.record_feedback(
            decision_id, rating, comments, expected_action, actual_action, user
        )
