"""
图神经网络推理引擎
使用 GNN 进行知识图谱推理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """图注意力层 (GAT)"""
    
    def __init__(self, in_features: int, out_features: int, 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 权重矩阵
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, out_features // num_heads))
        
        # 注意力参数
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * (out_features // num_heads), 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, in_features]
            adj: 邻接矩阵 [N, N]
        
        Returns:
            输出特征 [N, out_features]
        """
        N = x.size(0)
        
        # 多头注意力
        outputs = []
        for head in range(self.num_heads):
            # 线性变换
            h = torch.matmul(x, self.W[head])  # [N, out_features // num_heads]
            
            # 计算注意力系数
            attn_input = torch.cat([
                h.unsqueeze(1).expand(N, N, -1),
                h.unsqueeze(0).expand(N, N, -1)
            ], dim=-1)  # [N, N, 2 * out_features // num_heads]
            
            e = F.leaky_relu(torch.matmul(attn_input, self.a[head]).squeeze(-1))  # [N, N]
            
            # 掩码非邻居
            e = e.masked_fill(adj == 0, float('-inf'))
            
            # Softmax
            alpha = F.softmax(e, dim=1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # 聚合
            out = torch.matmul(alpha, h)  # [N, out_features // num_heads]
            outputs.append(out)
        
        # 拼接多头输出
        output = torch.cat(outputs, dim=-1)  # [N, out_features]
        return F.elu(output)


class GNNReasoner(nn.Module):
    """
    图神经网络推理引擎
    
    功能:
    - 节点分类
    - 链接预测
    - 异常检测
    - 传播预测
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 128, 
                 num_classes: int = 5, num_heads: int = 4):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # GAT 层
        self.gat1 = GraphAttentionLayer(num_features, hidden_dim, num_heads)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
        self.gat3 = GraphAttentionLayer(hidden_dim, hidden_dim // 2, num_heads)
        
        # 节点分类头
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # 链接预测头
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 异常检测头
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, num_features]
            adj: 邻接矩阵 [N, N]
        
        Returns:
            包含各种预测结果的字典
        """
        # GAT 编码
        h1 = self.gat1(x, adj)
        h1 = F.dropout(h1, p=0.3, training=self.training)
        
        h2 = self.gat2(h1, adj)
        h2 = F.dropout(h2, p=0.3, training=self.training)
        
        h3 = self.gat3(h2, adj)
        
        # 节点分类
        node_logits = self.node_classifier(h3)
        
        # 异常检测
        anomaly_scores = self.anomaly_detector(h3)
        
        return {
            'embeddings': h3,
            'node_logits': node_logits,
            'anomaly_scores': anomaly_scores
        }
    
    def predict_links(self, node_i: torch.Tensor, node_j: torch.Tensor) -> torch.Tensor:
        """
        预测两个节点之间是否存在链接
        
        Args:
            node_i: 节点 i 的嵌入
            node_j: 节点 j 的嵌入
        
        Returns:
            链接概率
        """
        combined = torch.cat([node_i, node_j], dim=-1)
        return torch.sigmoid(self.link_predictor(combined))
    
    def predict_propagation(self, x: torch.Tensor, adj: torch.Tensor,
                           source_nodes: List[int], steps: int = 3) -> torch.Tensor:
        """
        预测故障传播
        
        Args:
            x: 节点特征
            adj: 邻接矩阵
            source_nodes: 故障源节点
            steps: 传播步数
        
        Returns:
            各节点的故障概率
        """
        # 获取节点嵌入
        with torch.no_grad():
            result = self.forward(x, adj)
            embeddings = result['embeddings']
        
        N = x.size(0)
        fault_prob = torch.zeros(N)
        fault_prob[source_nodes] = 1.0
        
        # 传播模拟
        for _ in range(steps):
            # 邻居影响
            neighbor_influence = torch.matmul(adj.float(), fault_prob.unsqueeze(1)).squeeze()
            
            # 考虑节点特征
            node_susceptibility = torch.sigmoid(torch.sum(embeddings, dim=1))
            
            # 更新故障概率
            new_prob = fault_prob + 0.3 * neighbor_influence * node_susceptibility
            fault_prob = torch.clamp(new_prob, 0, 1)
        
        return fault_prob


class KnowledgeGraphEmbedder:
    """知识图谱嵌入器"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.node_features: Dict[str, np.ndarray] = {}
        
    def add_node(self, node_id: str, features: Dict[str, Any]):
        """添加节点"""
        # 将属性转换为特征向量
        feature_vector = self._features_to_vector(features)
        self.node_features[node_id] = feature_vector
        
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """将特征字典转换为向量"""
        # 简化的特征编码
        vector = []
        
        # 数值特征
        vector.append(features.get('cpu_usage', 0) / 100)
        vector.append(features.get('memory_usage', 0) / 100)
        vector.append(features.get('error_rate', 0))
        vector.append(min(features.get('latency', 0) / 5000, 1))
        
        # 类别特征编码 (one-hot)
        entity_type = features.get('type', 'unknown')
        type_encoding = self._encode_type(entity_type)
        vector.extend(type_encoding)
        
        # 填充到固定维度
        while len(vector) < self.embedding_dim:
            vector.append(0)
        
        return np.array(vector[:self.embedding_dim])
    
    def _encode_type(self, entity_type: str) -> List[float]:
        """编码实体类型"""
        types = ['service', 'database', 'cache', 'error', 'warning', 'unknown']
        encoding = [0.0] * len(types)
        if entity_type in types:
            encoding[types.index(entity_type)] = 1.0
        return encoding
    
    def get_node_tensor(self, node_ids: List[str]) -> torch.Tensor:
        """获取节点特征张量"""
        features = [self.node_features.get(node_id, np.zeros(self.embedding_dim)) 
                   for node_id in node_ids]
        return torch.FloatTensor(np.array(features))
    
    def build_adjacency_matrix(self, edges: List[Tuple[str, str]], 
                              node_ids: List[str]) -> torch.Tensor:
        """构建邻接矩阵"""
        N = len(node_ids)
        node_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        adj = torch.zeros(N, N)
        
        for source, target in edges:
            if source in node_idx and target in node_idx:
                i, j = node_idx[source], node_idx[target]
                adj[i, j] = 1
                adj[j, i] = 1  # 无向图
        
        # 添加自环
        for i in range(N):
            adj[i, i] = 1
        
        return adj


class GraphReasoningEngine:
    """
    图推理引擎主类
    整合 GNN 模型和知识图谱操作
    """
    
    def __init__(self, num_features: int = 64, hidden_dim: int = 128):
        self.gnn = GNNReasoner(num_features, hidden_dim)
        self.embedder = KnowledgeGraphEmbedder(num_features)
        
        self.node_id_map: Dict[str, int] = {}
        self.reverse_node_map: Dict[int, str] = {}
        self.edges: List[Tuple[str, str]] = []
        
    def add_nodes(self, nodes: List[Dict[str, Any]]):
        """添加节点"""
        for node in nodes:
            node_id = node.get('id')
            features = node.get('properties', {})
            features['type'] = node.get('type', 'unknown')
            
            self.embedder.add_node(node_id, features)
            
            if node_id not in self.node_id_map:
                idx = len(self.node_id_map)
                self.node_id_map[node_id] = idx
                self.reverse_node_map[idx] = node_id
    
    def add_edges(self, edges: List[Dict[str, Any]]):
        """添加边"""
        for edge in edges:
            source = edge.get('source_id')
            target = edge.get('target_id')
            if source and target:
                self.edges.append((source, target))
    
    def classify_nodes(self) -> Dict[str, Any]:
        """节点分类"""
        if not self.node_id_map:
            return {"error": "No nodes available"}
        
        node_ids = list(self.node_id_map.keys())
        x = self.embedder.get_node_tensor(node_ids)
        adj = self.embedder.build_adjacency_matrix(self.edges, node_ids)
        
        with torch.no_grad():
            result = self.gnn(x, adj)
            logits = result['node_logits']
            predictions = torch.argmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)
        
        class_names = ['normal', 'warning', 'error', 'critical', 'unknown']
        
        results = {}
        for i, node_id in enumerate(node_ids):
            results[node_id] = {
                'predicted_class': class_names[predictions[i].item()],
                'confidence': probs[i].max().item(),
                'class_probabilities': {
                    class_names[j]: probs[i, j].item() 
                    for j in range(len(class_names))
                }
            }
        
        return results
    
    def predict_links(self, node_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """预测链接"""
        if not self.node_id_map:
            return []
        
        node_ids = list(self.node_id_map.keys())
        x = self.embedder.get_node_tensor(node_ids)
        adj = self.embedder.build_adjacency_matrix(self.edges, node_ids)
        
        with torch.no_grad():
            result = self.gnn(x, adj)
            embeddings = result['embeddings']
        
        predictions = []
        for source, target in node_pairs:
            if source in self.node_id_map and target in self.node_id_map:
                source_idx = self.node_id_map[source]
                target_idx = self.node_id_map[target]
                
                source_emb = embeddings[source_idx]
                target_emb = embeddings[target_idx]
                
                prob = self.gnn.predict_links(source_emb, target_emb).item()
                
                predictions.append({
                    'source': source,
                    'target': target,
                    'link_probability': prob,
                    'predicted': prob > 0.5
                })
        
        return predictions
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常节点"""
        if not self.node_id_map:
            return []
        
        node_ids = list(self.node_id_map.keys())
        x = self.embedder.get_node_tensor(node_ids)
        adj = self.embedder.build_adjacency_matrix(self.edges, node_ids)
        
        with torch.no_grad():
            result = self.gnn(x, adj)
            anomaly_scores = result['anomaly_scores']
        
        anomalies = []
        for i, node_id in enumerate(node_ids):
            score = anomaly_scores[i].item()
            if score > 0.5:
                anomalies.append({
                    'node_id': node_id,
                    'anomaly_score': score,
                    'severity': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low'
                })
        
        # 按异常分数排序
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return anomalies
    
    def predict_fault_propagation(self, fault_sources: List[str], 
                                  steps: int = 3) -> Dict[str, Any]:
        """预测故障传播"""
        if not self.node_id_map:
            return {"error": "No nodes available"}
        
        node_ids = list(self.node_id_map.keys())
        x = self.embedder.get_node_tensor(node_ids)
        adj = self.embedder.build_adjacency_matrix(self.edges, node_ids)
        
        # 获取源节点索引
        source_indices = [self.node_id_map[s] for s in fault_sources if s in self.node_id_map]
        
        if not source_indices:
            return {"error": "No valid fault sources"}
        
        # 预测传播
        fault_probs = self.gnn.predict_propagation(x, adj, source_indices, steps)
        
        propagation = {}
        for i, node_id in enumerate(node_ids):
            prob = fault_probs[i].item()
            if prob > 0.1:  # 只返回有意义的概率
                propagation[node_id] = {
                    'fault_probability': prob,
                    'is_source': node_id in fault_sources
                }
        
        return {
            'fault_sources': fault_sources,
            'propagation_steps': steps,
            'affected_nodes': propagation,
            'total_affected': len([p for p in propagation.values() if p['fault_probability'] > 0.5])
        }
    
    def find_similar_nodes(self, node_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """查找相似节点"""
        if node_id not in self.node_id_map:
            return []
        
        node_ids = list(self.node_id_map.keys())
        x = self.embedder.get_node_tensor(node_ids)
        adj = self.embedder.build_adjacency_matrix(self.edges, node_ids)
        
        with torch.no_grad():
            result = self.gnn(x, adj)
            embeddings = result['embeddings']
        
        target_idx = self.node_id_map[node_id]
        target_emb = embeddings[target_idx]
        
        # 计算相似度
        similarities = []
        for i, other_id in enumerate(node_ids):
            if other_id != node_id:
                other_emb = embeddings[i]
                sim = F.cosine_similarity(target_emb.unsqueeze(0), 
                                         other_emb.unsqueeze(0)).item()
                similarities.append({
                    'node_id': other_id,
                    'similarity': sim
                })
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
