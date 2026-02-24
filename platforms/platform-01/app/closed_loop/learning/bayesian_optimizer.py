"""
贝叶斯参数优化器
使用高斯过程代理模型和采集函数进行高效参数优化
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """采集函数类型"""
    EI = "expected_improvement"  # 期望改进
    UCB = "upper_confidence_bound"  # 上置信界
    PI = "probability_of_improvement"  # 改进概率
    POI = "probability_of_improvement_with_xi"  # 带探索的改进概率


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[float, float]  # 对于连续/离散参数
    choices: List[Any] = None  # 对于分类参数
    
    def sample(self) -> Union[float, int, str]:
        """从参数空间随机采样"""
        if self.type == 'continuous':
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.type == 'discrete':
            return np.random.randint(self.bounds[0], self.bounds[1] + 1)
        elif self.type == 'categorical':
            return np.random.choice(self.choices)
        else:
            raise ValueError(f"Unknown parameter type: {self.type}")
    
    def normalize(self, value: Union[float, int, str]) -> float:
        """将参数值归一化到 [0, 1]"""
        if self.type in ['continuous', 'discrete']:
            return (value - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        elif self.type == 'categorical':
            return self.choices.index(value) / len(self.choices)
    
    def denormalize(self, normalized_value: float) -> Union[float, int, str]:
        """将归一化值还原为参数值"""
        if self.type == 'continuous':
            return normalized_value * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        elif self.type == 'discrete':
            return int(round(normalized_value * (self.bounds[1] - self.bounds[0]) + self.bounds[0]))
        elif self.type == 'categorical':
            idx = int(round(normalized_value * (len(self.choices) - 1)))
            return self.choices[min(idx, len(self.choices) - 1)]


class GaussianProcess:
    """高斯过程代理模型"""
    
    def __init__(self, kernel: str = 'rbf', noise: float = 1e-5):
        self.kernel = kernel
        self.noise = noise
        self.X = None  # 观测点
        self.y = None  # 观测值
        self.K = None  # 核矩阵
        self.L = None  # Cholesky 分解
        self.alpha = None
        
    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF (高斯) 核函数"""
        sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return np.exp(-0.5 * sqdist / length_scale**2)
    
    def matern_kernel(self, x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0, nu: float = 2.5) -> np.ndarray:
        """Matern 核函数"""
        dist = cdist(x1, x2, metric='euclidean')
        if nu == 2.5:
            sqrt5_dist = np.sqrt(5) * dist / length_scale
            return (1 + sqrt5_dist + 5 * dist**2 / (3 * length_scale**2)) * np.exp(-sqrt5_dist)
        elif nu == 1.5:
            sqrt3_dist = np.sqrt(3) * dist / length_scale
            return (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
        else:
            return np.exp(-dist / length_scale)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合高斯过程"""
        self.X = X
        self.y = y
        
        # 计算核矩阵
        if self.kernel == 'rbf':
            K = self.rbf_kernel(X, X)
        elif self.kernel == 'matern':
            K = self.matern_kernel(X, X)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # 添加噪声项
        self.K = K + self.noise * np.eye(len(X))
        
        # Cholesky 分解
        try:
            self.L = np.linalg.cholesky(self.K)
        except np.linalg.LinAlgError:
            # 如果 Cholesky 失败，添加更多正则化
            self.K += 1e-3 * np.eye(len(X))
            self.L = np.linalg.cholesky(self.K)
        
        # 计算 alpha = K^{-1} * y
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        
    def predict(self, X_test: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """预测均值和方差"""
        if self.X is None:
            raise ValueError("Model not fitted yet")
        
        # 计算测试点与训练点的核
        if self.kernel == 'rbf':
            K_s = self.rbf_kernel(self.X, X_test)
            K_ss = self.rbf_kernel(X_test, X_test)
        elif self.kernel == 'matern':
            K_s = self.matern_kernel(self.X, X_test)
            K_ss = self.matern_kernel(X_test, X_test)
        
        # 预测均值: mu = K_s^T * alpha
        mu = np.dot(K_s.T, self.alpha)
        
        if return_std:
            # 预测方差: var = K_ss - K_s^T * K^{-1} * K_s
            v = np.linalg.solve(self.L, K_s)
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 0))
            return mu, std
        
        return mu, None


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    应用场景:
    - 异常检测阈值优化
    - 告警聚合参数调优
    - 修复策略参数优化
    - 容量规划参数优化
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.param_spaces: Dict[str, ParameterSpace] = {}
        self.gp = GaussianProcess(
            kernel=self.config.get('kernel', 'rbf'),
            noise=self.config.get('noise', 1e-5)
        )
        acquisition_str = self.config.get('acquisition', 'EI')
        if isinstance(acquisition_str, str):
            self.acquisition = AcquisitionFunction[acquisition_str]
        else:
            self.acquisition = acquisition_str
        
        # 观测数据
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.param_history: List[Dict[str, Any]] = []
        
        # 优化状态
        self.best_value = float('-inf')
        self.best_params = None
        self.iteration = 0
        
        # 探索参数
        self.xi = self.config.get('xi', 0.01)  # 探索参数
        self.kappa = self.config.get('kappa', 2.0)  # UCB 参数
        
    def add_parameter(self, name: str, param_type: str, 
                      bounds: Tuple[float, float] = None,
                      choices: List[Any] = None):
        """添加参数到优化空间"""
        self.param_spaces[name] = ParameterSpace(
            name=name,
            type=param_type,
            bounds=bounds or (0, 1),
            choices=choices
        )
        logger.info(f"Added parameter: {name} ({param_type})")
        
    def params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """将参数字典转换为数组"""
        return np.array([
            self.param_spaces[name].normalize(params[name])
            for name in sorted(self.param_spaces.keys())
        ])
    
    def array_to_params(self, array: np.ndarray) -> Dict[str, Any]:
        """将数组转换为参数字典"""
        params = {}
        for i, name in enumerate(sorted(self.param_spaces.keys())):
            params[name] = self.param_spaces[name].denormalize(array[i])
        return params
    
    def acquisition_function(self, x: np.ndarray) -> float:
        """计算采集函数值"""
        if len(self.X_observed) == 0:
            return 0.0
        
        x = x.reshape(1, -1)
        mu, sigma = self.gp.predict(x, return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        if self.acquisition == AcquisitionFunction.EI:
            # 期望改进
            if sigma == 0:
                return 0.0
            mu_opt = np.max(self.y_observed)
            imp = mu - mu_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei  # 最小化
            
        elif self.acquisition == AcquisitionFunction.PI:
            # 改进概率
            if sigma == 0:
                return 0.0
            mu_opt = np.max(self.y_observed)
            pi = norm.cdf((mu - mu_opt - self.xi) / sigma)
            return -pi
            
        elif self.acquisition == AcquisitionFunction.UCB:
            # 上置信界
            return -(mu + self.kappa * sigma)
            
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def suggest_next_point(self) -> Dict[str, Any]:
        """建议下一个采样点"""
        if len(self.X_observed) < 2:
            # 随机采样
            params = {name: space.sample() for name, space in self.param_spaces.items()}
            return params
        
        # 拟合 GP 模型
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # 多起点优化
        best_x = None
        best_acq = float('inf')
        
        n_restarts = self.config.get('n_restarts', 10)
        for _ in range(n_restarts):
            # 随机起点
            x0 = np.random.rand(len(self.param_spaces))
            
            # 使用 L-BFGS-B 优化
            bounds = [(0, 1)] * len(self.param_spaces)
            result = minimize(
                self.acquisition_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return self.array_to_params(best_x)
    
    def optimize(self, objective_func: Callable[[Dict[str, Any]], float],
                 n_iterations: int = 100,
                 n_initial_points: int = 5) -> Dict[str, Any]:
        """
        执行贝叶斯优化
        
        Args:
            objective_func: 目标函数，输入参数字典，输出评分
            n_iterations: 总迭代次数
            n_initial_points: 初始随机采样点数
        """
        logger.info(f"Starting Bayesian optimization: {n_iterations} iterations")
        
        # 初始随机采样
        for i in range(n_initial_points):
            params = {name: space.sample() for name, space in self.param_spaces.items()}
            value = objective_func(params)
            self._update_observation(params, value)
            logger.info(f"Initial point {i+1}/{n_initial_points}: {params} -> {value:.4f}")
        
        # 贝叶斯优化迭代
        for i in range(n_iterations - n_initial_points):
            self.iteration += 1
            
            # 拟合高斯过程
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
            
            # 建议下一个点
            next_params = self.suggest_next_point()
            
            # 评估目标函数
            value = objective_func(next_params)
            
            # 更新观测
            self._update_observation(next_params, value)
            
            logger.info(f"Iteration {self.iteration}/{n_iterations - n_initial_points}: "
                       f"{next_params} -> {value:.4f} (best: {self.best_value:.4f})")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'iterations': self.iteration,
            'history': self.param_history
        }
    
    def _update_observation(self, params: Dict[str, Any], value: float):
        """更新观测数据"""
        x = self.params_to_array(params)
        self.X_observed.append(x)
        self.y_observed.append(value)
        self.param_history.append({
            'params': params,
            'value': value,
            'iteration': self.iteration
        })
        
        if value > self.best_value:
            self.best_value = value
            self.best_params = params.copy()
    
    def parallel_optimize(self, objective_func: Callable[[Dict[str, Any]], float],
                         n_iterations: int = 100,
                         n_workers: int = 4) -> Dict[str, Any]:
        """并行贝叶斯优化"""
        logger.info(f"Starting parallel Bayesian optimization: {n_iterations} iterations, {n_workers} workers")
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # 初始随机采样
            futures = []
            for i in range(min(n_workers, n_iterations)):
                params = {name: space.sample() for name, space in self.param_spaces.items()}
                future = executor.submit(objective_func, params)
                futures.append((params, future))
            
            for params, future in futures:
                value = future.result()
                self._update_observation(params, value)
        
        # 继续顺序优化
        return self.optimize(objective_func, n_iterations, n_initial_points=0)


class MultiObjectiveBayesianOptimizer:
    """多目标贝叶斯优化器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.param_spaces: Dict[str, ParameterSpace] = {}
        self.objectives: List[str] = []
        self.gps: Dict[str, GaussianProcess] = {}
        
        self.X_observed: List[np.ndarray] = []
        self.y_observed: Dict[str, List[float]] = {}
        
        self.pareto_front: List[Dict[str, Any]] = []
        
    def add_objective(self, name: str):
        """添加优化目标"""
        self.objectives.append(name)
        self.y_observed[name] = []
        self.gps[name] = GaussianProcess()
        
    def add_parameter(self, name: str, param_type: str, 
                      bounds: Tuple[float, float] = None,
                      choices: List[Any] = None):
        """添加参数"""
        self.param_spaces[name] = ParameterSpace(
            name=name,
            type=param_type,
            bounds=bounds or (0, 1),
            choices=choices
        )
        
    def is_dominated(self, point: Dict[str, float], other: Dict[str, float]) -> bool:
        """检查 point 是否被 other 支配"""
        not_worse = all(point[obj] <= other[obj] for obj in self.objectives)
        strictly_better = any(point[obj] < other[obj] for obj in self.objectives)
        return not_worse and strictly_better
    
    def update_pareto_front(self, params: Dict[str, Any], objectives: Dict[str, float]):
        """更新 Pareto 前沿"""
        new_point = {**params, **objectives}
        
        # 检查是否被现有前沿支配
        for point in self.pareto_front:
            if self.is_dominated(new_point, point):
                return
        
        # 移除被新点支配的点
        self.pareto_front = [
            point for point in self.pareto_front 
            if not self.is_dominated(point, new_point)
        ]
        
        # 添加新点
        self.pareto_front.append(new_point)
        
    def suggest_next_point(self) -> Dict[str, Any]:
        """建议下一个采样点 (使用期望超体积改进)"""
        # 简化为随机采样
        return {name: space.sample() for name, space in self.param_spaces.items()}
