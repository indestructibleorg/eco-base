"""
强化学习策略学习器
基于 PPO (Proximal Policy Optimization) 算法
用于学习最优修复策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """经验样本"""
    state: np.ndarray
    action: int
    action_prob: float
    reward: float
    next_state: np.ndarray
    done: bool
    value: float
    
    
class ActorNetwork(nn.Module):
    """Actor 网络 - 输出动作概率分布"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        action_logits = self.action_head(features)
        return torch.softmax(action_logits, dim=-1)


class CriticNetwork(nn.Module):
    """Critic 网络 - 评估状态价值"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value


class StateEncoder:
    """状态编码器 - 将多维指标转换为状态向量"""
    
    def __init__(self):
        self.feature_names = [
            # 系统指标
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
            # 应用指标
            'qps', 'latency_p99', 'error_rate', 'throughput',
            # 历史决策
            'prev_action_1', 'prev_action_2', 'prev_action_3',
            # 环境因素
            'hour_of_day', 'day_of_week', 'load_trend'
        ]
        self.normalization_stats = {}
        
    def encode(self, metrics: Dict[str, float], history: List[str] = None, 
               context: Dict[str, Any] = None) -> np.ndarray:
        """将指标编码为状态向量"""
        state = []
        
        # 系统指标
        state.append(metrics.get('cpu_usage', 0.0) / 100.0)
        state.append(metrics.get('memory_usage', 0.0) / 100.0)
        state.append(metrics.get('disk_usage', 0.0) / 100.0)
        state.append(min(metrics.get('network_io', 0.0) / 1e9, 1.0))  # Normalize to GB
        
        # 应用指标
        state.append(min(metrics.get('qps', 0.0) / 10000.0, 1.0))
        state.append(min(metrics.get('latency_p99', 0.0) / 5000.0, 1.0))  # Normalize to 5s
        state.append(min(metrics.get('error_rate', 0.0) * 10, 1.0))  # Scale up
        state.append(min(metrics.get('throughput', 0.0) / 1000.0, 1.0))
        
        # 历史决策编码
        action_map = {'restart': 0.1, 'rollback': 0.2, 'scale_up': 0.3, 
                      'scale_down': 0.4, 'config_update': 0.5, 'none': 0.0}
        history = history or ['none', 'none', 'none']
        for i in range(3):
            action = history[i] if i < len(history) else 'none'
            state.append(action_map.get(action, 0.0))
        
        # 环境因素
        now = datetime.now()
        state.append(now.hour / 24.0)  # Hour normalized
        state.append(now.weekday() / 7.0)  # Day normalized
        state.append(metrics.get('load_trend', 0.0))  # -1 to 1
        
        return np.array(state, dtype=np.float32)
    
    def update_normalization(self, data: List[Dict[str, float]]):
        """更新归一化统计信息"""
        for feature in self.feature_names:
            values = [d.get(feature, 0.0) for d in data if feature in d]
            if values:
                self.normalization_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values) + 1e-8
                }


class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'mttr_improvement': 0.35,
            'cost_saving': 0.25,
            'stability': 0.25,
            'penalty': 0.15
        }
        self.baseline_mttr = 300  # 5 minutes
        self.baseline_cost = 100  # $100 per incident
        
    def calculate(self, state: Dict[str, float], action: str, 
                  next_state: Dict[str, float], execution_result: Dict[str, Any]) -> float:
        """计算奖励值"""
        reward = 0.0
        
        # MTTR 改善奖励
        mttr = execution_result.get('mttr', self.baseline_mttr)
        mttr_improvement = (self.baseline_mttr - mttr) / self.baseline_mttr
        reward += self.weights['mttr_improvement'] * mttr_improvement
        
        # 成本节约奖励
        cost = execution_result.get('cost', self.baseline_cost)
        cost_saving = (self.baseline_cost - cost) / self.baseline_cost
        reward += self.weights['cost_saving'] * cost_saving
        
        # 稳定性奖励
        error_before = state.get('error_rate', 0.0)
        error_after = next_state.get('error_rate', 0.0)
        stability = (error_before - error_after) * 10  # Scale up
        reward += self.weights['stability'] * stability
        
        # 惩罚项
        penalty = 0.0
        if execution_result.get('status') == 'failed':
            penalty += 1.0
        if execution_result.get('side_effects'):
            penalty += len(execution_result['side_effects']) * 0.3
        reward -= self.weights['penalty'] * penalty
        
        return reward


class PPOPolicyLearner:
    """
    PPO 策略学习器
    
    状态空间: 系统指标、应用指标、历史决策、环境因素
    动作空间: restart, rollback, scale_up, scale_down, config_update, wait
    """
    
    ACTIONS = ['restart', 'rollback', 'scale_up', 'scale_down', 'config_update', 'wait']
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state_dim = 14  # 状态向量维度
        self.action_dim = len(self.ACTIONS)
        
        # 神经网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.critic = CriticNetwork(self.state_dim)
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.get('actor_lr', 3e-4)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.get('critic_lr', 1e-3)
        )
        
        # 超参数
        self.gamma = self.config.get('gamma', 0.99)  # 折扣因子
        self.gae_lambda = self.config.get('gae_lambda', 0.95)  # GAE lambda
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)  # PPO clip
        self.value_coef = self.config.get('value_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 10)
        
        # 经验缓冲区
        self.experience_buffer: List[Experience] = []
        self.buffer_size = self.config.get('buffer_size', 2048)
        
        # 辅助组件
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator()
        
        # 训练状态
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            action_prob = action_probs[0][action].item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            action_prob = action_probs[0][action].item()
            
        return action, action_prob, value.item()
    
    def store_experience(self, experience: Experience):
        """存储经验"""
        self.experience_buffer.append(experience)
        
        # 缓冲区满时触发训练
        if len(self.experience_buffer) >= self.buffer_size:
            self.train()
            
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool]) -> Tuple[List[float], List[float]]:
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def train(self) -> Dict[str, float]:
        """训练策略网络"""
        if len(self.experience_buffer) < self.batch_size:
            return {}
        
        # 准备训练数据
        states = torch.FloatTensor([e.state for e in self.experience_buffer])
        actions = torch.LongTensor([e.action for e in self.experience_buffer])
        old_probs = torch.FloatTensor([e.action_prob for e in self.experience_buffer])
        rewards = [e.reward for e in self.experience_buffer]
        values = [e.value for e in self.experience_buffer]
        dones = [e.done for e in self.experience_buffer]
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮训练
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        
        for epoch in range(self.epochs):
            # 生成随机批次
            indices = torch.randperm(len(self.experience_buffer))
            
            for start in range(0, len(self.experience_buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor 更新
                action_probs = self.actor(batch_states)
                dist = Categorical(action_probs)
                new_probs = dist.probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
                
                ratio = new_probs / (batch_old_probs + 1e-8)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                entropy = dist.entropy().mean()
                actor_loss -= self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic 更新
                batch_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # 清空缓冲区
        self.experience_buffer.clear()
        self.training_step += 1
        
        n_updates = self.epochs * (len(states) // self.batch_size + 1)
        
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'training_step': self.training_step
        }
    
    def get_action_recommendation(self, metrics: Dict[str, float], 
                                   history: List[str] = None) -> Dict[str, Any]:
        """获取动作推荐"""
        state = self.state_encoder.encode(metrics, history)
        action_idx, prob, value = self.select_action(state, deterministic=True)
        
        return {
            'action': self.ACTIONS[action_idx],
            'confidence': prob,
            'expected_value': value,
            'all_actions': [
                {'action': a, 'probability': p} 
                for a, p in zip(self.ACTIONS, self.actor(torch.FloatTensor(state).unsqueeze(0))[0].tolist())
            ]
        }
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        logger.info(f"Model loaded from {path}")


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return np.random.choice(list(self.buffer), batch_size, replace=False).tolist()
    
    def __len__(self):
        return len(self.buffer)
