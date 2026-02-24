"""
多目标 Pareto 优化器
使用 NSGA-II 算法进行多目标优化
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecisionVariable:
    """决策变量"""
    name: str
    lower_bound: float
    upper_bound: float
    type: str = 'continuous'  # continuous, integer
    
    def random_value(self) -> float:
        """生成随机值"""
        if self.type == 'integer':
            return random.randint(int(self.lower_bound), int(self.upper_bound))
        return random.uniform(self.lower_bound, self.upper_bound)
    
    def clip(self, value: float) -> float:
        """裁剪到有效范围"""
        clipped = max(self.lower_bound, min(self.upper_bound, value))
        if self.type == 'integer':
            return int(round(clipped))
        return clipped


@dataclass
class Objective:
    """优化目标"""
    name: str
    direction: str  # minimize, maximize
    weight: float = 1.0
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """归一化目标值"""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        if self.direction == 'minimize':
            return 1 - normalized  # 反转，使越大越好
        return normalized


class Individual:
    """NSGA-II 个体"""
    
    def __init__(self, variables: Dict[str, float]):
        self.variables = variables
        self.objectives: Dict[str, float] = {}
        self.normalized_objectives: Dict[str, float] = {}
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.dominated_count: int = 0
        self.dominating_set: List['Individual'] = []
        
    def dominates(self, other: 'Individual') -> bool:
        """检查是否支配另一个体"""
        better_in_one = False
        for obj_name, value in self.objectives.items():
            other_value = other.objectives.get(obj_name, 0)
            if value > other_value:
                better_in_one = True
            elif value < other_value:
                return False
        return better_in_one
    
    def copy(self) -> 'Individual':
        """复制个体"""
        new_ind = Individual(self.variables.copy())
        new_ind.objectives = self.objectives.copy()
        new_ind.normalized_objectives = self.normalized_objectives.copy()
        return new_ind


class NSGAIIOptimizer:
    """
    NSGA-II 多目标优化算法
    
    优化目标:
    - 最小化修复成本
    - 最小化修复时间 (MTTR)
    - 最小化风险
    - 最大化修复质量
    - 最大化操作安全性
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 算法参数
        self.population_size = self.config.get('population_size', 100)
        self.generations = self.config.get('generations', 200)
        self.crossover_prob = self.config.get('crossover_prob', 0.9)
        self.mutation_prob = self.config.get('mutation_prob', 0.1)
        self.tournament_size = self.config.get('tournament_size', 2)
        
        # 决策变量和目标
        self.decision_variables: Dict[str, DecisionVariable] = {}
        self.objectives: Dict[str, Objective] = {}
        
        # 评估函数
        self.evaluation_function: Optional[Callable] = None
        
        # 种群
        self.population: List[Individual] = []
        self.pareto_fronts: List[List[Individual]] = []
        
    def add_decision_variable(self, name: str, lower: float, upper: float, 
                             var_type: str = 'continuous'):
        """添加决策变量"""
        self.decision_variables[name] = DecisionVariable(
            name=name,
            lower_bound=lower,
            upper_bound=upper,
            type=var_type
        )
        
    def add_objective(self, name: str, direction: str = 'minimize', weight: float = 1.0):
        """添加优化目标"""
        self.objectives[name] = Objective(
            name=name,
            direction=direction,
            weight=weight
        )
        
    def set_evaluation_function(self, func: Callable[[Dict[str, float]], Dict[str, float]]):
        """设置评估函数"""
        self.evaluation_function = func
        
    def initialize_population(self):
        """初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            variables = {
                name: var.random_value()
                for name, var in self.decision_variables.items()
            }
            individual = Individual(variables)
            self._evaluate(individual)
            self.population.append(individual)
            
    def _evaluate(self, individual: Individual):
        """评估个体"""
        if self.evaluation_function:
            objectives = self.evaluation_function(individual.variables)
            individual.objectives = objectives
            
    def non_dominated_sort(self) -> List[List[Individual]]:
        """非支配排序"""
        fronts = [[]]
        
        for p in self.population:
            p.dominated_count = 0
            p.dominating_set = []
            
            for q in self.population:
                if p == q:
                    continue
                    
                if p.dominates(q):
                    p.dominating_set.append(q)
                elif q.dominates(p):
                    p.dominated_count += 1
                    
            if p.dominated_count == 0:
                p.rank = 0
                fronts[0].append(p)
                
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            
            for p in fronts[i]:
                for q in p.dominating_set:
                    q.dominated_count -= 1
                    if q.dominated_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
                        
            i += 1
            fronts.append(next_front)
            
        return fronts[:-1]  # 移除空的前沿
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
            
        for ind in front:
            ind.crowding_distance = 0
            
        for obj_name in self.objectives.keys():
            # 按目标值排序
            front.sort(key=lambda x: x.objectives.get(obj_name, 0))
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_values = [ind.objectives.get(obj_name, 0) for ind in front]
            f_max = max(obj_values)
            f_min = min(obj_values)
            
            if f_max == f_min:
                continue
                
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives.get(obj_name, 0) - 
                           front[i - 1].objectives.get(obj_name, 0))
                front[i].crowding_distance += distance / (f_max - f_min)
                
    def tournament_selection(self) -> Individual:
        """锦标赛选择"""
        selected = random.sample(self.population, self.tournament_size)
        
        # 按 rank 和 crowding_distance 排序
        selected.sort(key=lambda x: (x.rank, -x.crowding_distance))
        
        return selected[0].copy()
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作 (SBX - Simulated Binary Crossover)"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
            
        child1_vars = {}
        child2_vars = {}
        
        eta = 20  # SBX 分布指数
        
        for name, var in self.decision_variables.items():
            x1 = parent1.variables[name]
            x2 = parent2.variables[name]
            
            if random.random() <= 0.5:
                if abs(x1 - x2) > 1e-14:
                    if x1 < x2:
                        y1, y2 = x1, x2
                    else:
                        y1, y2 = x2, x1
                        
                    beta = 1.0 + (2.0 * (y1 - var.lower_bound) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    rand = random.random()
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                        
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (var.upper_bound - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                        
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    c1 = var.clip(c1)
                    c2 = var.clip(c2)
                    
                    if random.random() <= 0.5:
                        child1_vars[name] = c2
                        child2_vars[name] = c1
                    else:
                        child1_vars[name] = c1
                        child2_vars[name] = c2
                else:
                    child1_vars[name] = x1
                    child2_vars[name] = x2
            else:
                child1_vars[name] = x1
                child2_vars[name] = x2
                
        child1 = Individual(child1_vars)
        child2 = Individual(child2_vars)
        
        return child1, child2
    
    def mutate(self, individual: Individual):
        """变异操作 (多项式变异)"""
        eta_m = 20  # 变异分布指数
        
        for name, var in self.decision_variables.items():
            if random.random() <= self.mutation_prob:
                x = individual.variables[name]
                delta1 = (x - var.lower_bound) / (var.upper_bound - var.lower_bound)
                delta2 = (var.upper_bound - x) / (var.upper_bound - var.lower_bound)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow
                
                x = x + delta_q * (var.upper_bound - var.lower_bound)
                individual.variables[name] = var.clip(x)
                
    def environmental_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """环境选择"""
        self.population = combined_population
        fronts = self.non_dominated_sort()
        
        new_population = []
        front_idx = 0
        
        while len(new_population) + len(fronts[front_idx]) <= self.population_size:
            self.calculate_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break
                
        # 如果还需要更多个体，按拥挤距离选择
        if len(new_population) < self.population_size and front_idx < len(fronts):
            self.calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: -x.crowding_distance)
            remaining = self.population_size - len(new_population)
            new_population.extend(fronts[front_idx][:remaining])
            
        return new_population
    
    def optimize(self, callback: Callable = None) -> Dict[str, Any]:
        """执行优化"""
        logger.info(f"Starting NSGA-II optimization: {self.generations} generations")
        
        # 初始化
        self.initialize_population()
        
        for generation in range(self.generations):
            # 非支配排序
            fronts = self.non_dominated_sort()
            
            # 计算拥挤距离
            for front in fronts:
                self.calculate_crowding_distance(front)
                
            # 生成子代
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                
                self._evaluate(child1)
                self._evaluate(child2)
                
                offspring.append(child1)
                offspring.append(child2)
                
            # 环境选择
            combined = self.population + offspring[:self.population_size]
            self.population = self.environmental_selection(combined)
            
            # 回调
            if callback and generation % 10 == 0:
                callback(generation, self.get_pareto_front())
                
        # 最终排序
        self.pareto_fronts = self.non_dominated_sort()
        
        return {
            'pareto_front': self.get_pareto_front(),
            'generations': self.generations,
            'population_size': self.population_size,
            'num_solutions': len(self.get_pareto_front())
        }
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """获取 Pareto 前沿"""
        if not self.pareto_fronts:
            return []
            
        front = self.pareto_fronts[0]
        
        return [
            {
                'variables': ind.variables,
                'objectives': ind.objectives,
                'rank': ind.rank,
                'crowding_distance': ind.crowding_distance
            }
            for ind in front
        ]


class DecisionRecommender:
    """决策推荐器"""
    
    def __init__(self, optimizer: NSGAIIOptimizer):
        self.optimizer = optimizer
        
    def recommend(self, preferences: Dict[str, float] = None) -> Dict[str, Any]:
        """根据偏好推荐决策"""
        pareto_front = self.optimizer.get_pareto_front()
        
        if not pareto_front:
            return {"error": "No solutions available"}
        
        if preferences is None:
            # 默认选择拥挤距离最大的 (多样性最好)
            best = max(pareto_front, key=lambda x: x['crowding_distance'])
        else:
            # 根据偏好加权选择
            def score_solution(sol):
                score = 0
                for obj_name, weight in preferences.items():
                    score += weight * sol['objectives'].get(obj_name, 0)
                return score
                
            best = max(pareto_front, key=score_solution)
        
        return {
            'recommended_solution': best,
            'all_solutions': pareto_front,
            'preference_used': preferences,
            'num_alternatives': len(pareto_front)
        }
    
    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """分析权衡关系"""
        pareto_front = self.optimizer.get_pareto_front()
        
        if not pareto_front:
            return {"error": "No solutions available"}
        
        # 计算目标间的相关性
        objectives = list(self.optimizer.objectives.keys())
        tradeoffs = {}
        
        for i, obj1 in enumerate(objectives):
            for obj2 in objectives[i+1:]:
                values1 = [sol['objectives'].get(obj1, 0) for sol in pareto_front]
                values2 = [sol['objectives'].get(obj2, 0) for sol in pareto_front]
                
                # 计算相关系数
                if len(values1) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    tradeoffs[f"{obj1}_vs_{obj2}"] = {
                        'correlation': correlation,
                        'relationship': 'tradeoff' if correlation < -0.5 else 'aligned' if correlation > 0.5 else 'neutral'
                    }
        
        return {
            'tradeoffs': tradeoffs,
            'num_objectives': len(objectives),
            'num_solutions': len(pareto_front)
        }
