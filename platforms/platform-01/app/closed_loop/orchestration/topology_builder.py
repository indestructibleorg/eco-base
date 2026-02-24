"""
拓扑构建器
自动发现服务和构建依赖拓扑
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceNode:
    """服务节点"""
    id: str
    name: str
    service_type: str
    namespace: str
    labels: Dict[str, str]
    endpoints: List[str]
    health_status: str
    last_seen: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.service_type,
            'namespace': self.namespace,
            'labels': self.labels,
            'endpoints': self.endpoints,
            'health_status': self.health_status,
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class TopologyEdge:
    """拓扑边"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]
    discovered_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'properties': self.properties,
            'discovered_at': self.discovered_at.isoformat()
        }


class KubernetesDiscovery:
    """Kubernetes 服务发现"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
        self.discovered_services: Dict[str, ServiceNode] = {}
        
    def discover_services(self) -> List[ServiceNode]:
        """从 Kubernetes 发现服务"""
        services = []
        
        # 尝试从 Kubernetes API 获取服务
        k8s_services = self._fetch_kubernetes_services()
        
        # 如果无法获取，使用配置的服务列表
        if not k8s_services:
            k8s_services = self._get_configured_services()
        
        for svc_data in k8s_services:
            node = ServiceNode(
                id=f"{svc_data.get('namespace', 'default')}/{svc_data['name']}",
                name=svc_data['name'],
                service_type=svc_data.get('type', 'service'),
                namespace=svc_data.get('namespace', 'default'),
                labels=svc_data.get('labels', {}),
                endpoints=svc_data.get('endpoints', []),
                health_status=svc_data.get('health_status', 'unknown'),
                last_seen=datetime.now()
            )
            services.append(node)
            self.discovered_services[node.id] = node
        
        return services
    
    def _fetch_kubernetes_services(self) -> List[Dict[str, Any]]:
        """从 Kubernetes API 获取服务列表"""
        try:
            # 尝试导入 kubernetes 客户端
            from kubernetes import client, config
            
            # 加载配置
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            
            v1 = client.CoreV1Api()
            
            # 获取所有服务
            k8s_services = v1.list_service_for_all_namespaces()
            
            services = []
            for svc in k8s_services.items:
                # 获取 endpoints
                try:
                    endpoints = v1.read_namespaced_endpoints(
                        name=svc.metadata.name,
                        namespace=svc.metadata.namespace
                    )
                    endpoint_addresses = []
                    if endpoints.subsets:
                        for subset in endpoints.subsets:
                            if subset.addresses:
                                for addr in subset.addresses:
                                    port = subset.ports[0].port if subset.ports else 80
                                    endpoint_addresses.append(f"{addr.ip}:{port}")
                except Exception:
                    endpoint_addresses = []
                
                services.append({
                    'name': svc.metadata.name,
                    'namespace': svc.metadata.namespace,
                    'labels': svc.metadata.labels or {},
                    'endpoints': endpoint_addresses,
                    'type': 'service',
                    'health_status': 'healthy' if endpoint_addresses else 'unhealthy',
                })
            
            return services
            
        except ImportError:
            # kubernetes 客户端未安装
            return []
        except Exception as e:
            # 其他错误
            return []
    
    def _get_configured_services(self) -> List[Dict[str, Any]]:
        """从配置获取服务列表"""
        # 从环境变量或配置文件读取服务列表
        import os
        import json
        
        services_config = os.environ.get('TOPOLOGY_SERVICES', '')
        if services_config:
            try:
                return json.loads(services_config)
            except json.JSONDecodeError:
                pass
        
        # 返回默认服务列表（用于开发和测试）
        return [
            {
                'name': 'user-service',
                'namespace': 'default',
                'labels': {'app': 'user', 'tier': 'backend'},
                'endpoints': [],
                'type': 'service',
                'health_status': 'unknown',
            },
            {
                'name': 'order-service',
                'namespace': 'default',
                'labels': {'app': 'order', 'tier': 'backend'},
                'endpoints': [],
                'type': 'service',
                'health_status': 'unknown',
            },
            {
                'name': 'payment-service',
                'namespace': 'default',
                'labels': {'app': 'payment', 'tier': 'backend'},
                'endpoints': [],
                'type': 'service',
                'health_status': 'unknown',
            }
        ]
    
    def discover_dependencies(self) -> List[TopologyEdge]:
        """从 Kubernetes 发现依赖"""
        edges = []
        
        # 尝试从 Kubernetes 获取依赖关系
        k8s_deps = self._fetch_kubernetes_dependencies()
        
        # 如果无法获取，使用配置的依赖关系
        if not k8s_deps:
            k8s_deps = self._get_configured_dependencies()
        
        for source, target in k8s_deps:
            edge = TopologyEdge(
                source=source,
                target=target,
                relation_type='DEPENDS_ON',
                properties={'source': 'kubernetes_config'},
                discovered_at=datetime.now()
            )
            edges.append(edge)
        
        return edges
    
    def _fetch_kubernetes_dependencies(self) -> List[tuple]:
        """从 Kubernetes 获取依赖关系"""
        # 实际实现需要分析 Kubernetes 资源
        # 例如：从 Deployment 的 env 变量、ConfigMap 等提取依赖
        return []
    
    def _get_configured_dependencies(self) -> List[tuple]:
        """从配置获取依赖关系"""
        import os
        import json
        
        deps_config = os.environ.get('TOPOLOGY_DEPENDENCIES', '')
        if deps_config:
            try:
                return [tuple(d) for d in json.loads(deps_config)]
            except json.JSONDecodeError:
                pass
        
        # 返回默认依赖关系（用于开发和测试）
        return [
            ('default/order-service', 'default/user-service'),
            ('default/order-service', 'default/payment-service'),
            ('default/payment-service', 'default/user-service'),
        ]


class ServiceMeshDiscovery:
    """Service Mesh 服务发现"""
    
    def __init__(self, mesh_type: str = 'istio'):
        self.mesh_type = mesh_type
        
    def discover_from_traces(self, traces: List[Dict[str, Any]]) -> List[TopologyEdge]:
        """从分布式追踪发现依赖"""
        edges = []
        call_counts = defaultdict(lambda: defaultdict(int))
        latencies = defaultdict(list)
        
        for trace in traces:
            spans = trace.get('spans', [])
            span_map = {span['id']: span for span in spans}
            
            for span in spans:
                parent_id = span.get('parent_id')
                if parent_id and parent_id in span_map:
                    parent = span_map[parent_id]
                    source = parent.get('service', '')
                    target = span.get('service', '')
                    
                    if source and target and source != target:
                        call_counts[source][target] += 1
                        latencies[(source, target)].append(span.get('duration', 0))
        
        # 生成边
        for source, targets in call_counts.items():
            for target, count in targets.items():
                avg_latency = sum(latencies[(source, target)]) / len(latencies[(source, target)])
                
                edge = TopologyEdge(
                    source=source,
                    target=target,
                    relation_type='CALLS',
                    properties={
                        'call_count': count,
                        'avg_latency_ms': avg_latency,
                        'source': 'distributed_tracing'
                    },
                    discovered_at=datetime.now()
                )
                edges.append(edge)
        
        return edges


class LogBasedDiscovery:
    """基于日志的服务发现"""
    
    def __init__(self):
        self.service_patterns = [
            r'service[:\s]+([a-z][a-z0-9-]*)',
            r'calling\s+([a-z][a-z0-9-]*)',
            r'request\s+to\s+([a-z][a-z0-9-]*)',
        ]
        
    def discover_from_logs(self, logs: List[Dict[str, Any]]) -> List[TopologyEdge]:
        """从日志发现依赖"""
        edges = []
        call_sequences = []
        
        # 按时间排序
        sorted_logs = sorted(logs, key=lambda x: x.get('timestamp', datetime.now()))
        
        # 分析调用序列
        for i, log in enumerate(sorted_logs):
            service = log.get('service', '')
            message = log.get('message', '')
            
            # 查找调用模式
            for pattern in self.service_patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    called_service = match.group(1)
                    if called_service and called_service != service:
                        call_sequences.append((service, called_service, log.get('timestamp')))
        
        # 统计调用关系
        call_counts = defaultdict(lambda: defaultdict(int))
        for source, target, _ in call_sequences:
            call_counts[source][target] += 1
        
        # 生成边
        for source, targets in call_counts.items():
            for target, count in targets.items():
                edge = TopologyEdge(
                    source=source,
                    target=target,
                    relation_type='CALLS',
                    properties={
                        'call_count': count,
                        'source': 'log_analysis'
                    },
                    discovered_at=datetime.now()
                )
                edges.append(edge)
        
        return edges


class TopologyBuilder:
    """
    拓扑构建器主类
    整合多种发现机制
    """
    
    def __init__(self):
        self.k8s_discovery = KubernetesDiscovery()
        self.mesh_discovery = ServiceMeshDiscovery()
        self.log_discovery = LogBasedDiscovery()
        
        self.nodes: Dict[str, ServiceNode] = {}
        self.edges: Dict[str, TopologyEdge] = {}
        
    def build_topology(self, sources: List[str] = None) -> Dict[str, Any]:
        """
        构建完整拓扑
        
        Args:
            sources: 数据源列表 ['kubernetes', 'mesh', 'logs']
        
        Returns:
            完整拓扑
        """
        sources = sources or ['kubernetes', 'mesh', 'logs']
        
        # 发现节点
        if 'kubernetes' in sources:
            k8s_services = self.k8s_discovery.discover_services()
            for svc in k8s_services:
                self.nodes[svc.id] = svc
        
        # 发现边
        all_edges = []
        
        if 'kubernetes' in sources:
            k8s_edges = self.k8s_discovery.discover_dependencies()
            all_edges.extend(k8s_edges)
        
        if 'mesh' in sources:
            # 模拟追踪数据
            mock_traces = self._generate_mock_traces()
            mesh_edges = self.mesh_discovery.discover_from_traces(mock_traces)
            all_edges.extend(mesh_edges)
        
        if 'logs' in sources:
            # 模拟日志数据
            mock_logs = self._generate_mock_logs()
            log_edges = self.log_discovery.discover_from_logs(mock_logs)
            all_edges.extend(log_edges)
        
        # 合并边
        for edge in all_edges:
            edge_id = f"{edge.source}:{edge.target}"
            if edge_id in self.edges:
                # 更新现有边
                existing = self.edges[edge_id]
                existing.properties.update(edge.properties)
            else:
                self.edges[edge_id] = edge
        
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges.values()],
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'built_at': datetime.now().isoformat()
        }
    
    def _generate_mock_traces(self) -> List[Dict[str, Any]]:
        """生成模拟追踪数据"""
        return [
            {
                'id': 'trace_001',
                'spans': [
                    {'id': 'span_1', 'parent_id': None, 'service': 'gateway', 'duration': 100},
                    {'id': 'span_2', 'parent_id': 'span_1', 'service': 'order-service', 'duration': 50},
                    {'id': 'span_3', 'parent_id': 'span_2', 'service': 'user-service', 'duration': 20},
                    {'id': 'span_4', 'parent_id': 'span_2', 'service': 'payment-service', 'duration': 30},
                ]
            }
        ]
    
    def _generate_mock_logs(self) -> List[Dict[str, Any]]:
        """生成模拟日志数据"""
        return [
            {'service': 'order-service', 'message': 'Calling user-service for authentication', 'timestamp': datetime.now()},
            {'service': 'order-service', 'message': 'Request to payment-service for processing', 'timestamp': datetime.now()},
            {'service': 'payment-service', 'message': 'Calling user-service to get user info', 'timestamp': datetime.now()},
        ]
    
    def get_service_dependencies(self, service_id: str, 
                                 direction: str = 'downstream') -> List[str]:
        """获取服务依赖"""
        dependencies = []
        
        for edge in self.edges.values():
            if direction == 'downstream' and edge.source == service_id:
                dependencies.append(edge.target)
            elif direction == 'upstream' and edge.target == service_id:
                dependencies.append(edge.source)
        
        return dependencies
    
    def find_critical_path(self, source: str, target: str) -> List[str]:
        """查找关键路径"""
        # BFS
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == target:
                return path
            
            for edge in self.edges.values():
                if edge.source == current and edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))
        
        return []
    
    def detect_cycles(self) -> List[List[str]]:
        """检测拓扑中的循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            
            for edge in self.edges.values():
                if edge.source == node:
                    neighbor = edge.target
                    if neighbor not in visited:
                        dfs(neighbor, path + [neighbor])
                    elif neighbor in rec_stack:
                        # 发现循环
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node in self.nodes:
            if node not in visited:
                dfs(node, [node])
        
        return cycles
    
    def get_topology_stats(self) -> Dict[str, Any]:
        """获取拓扑统计"""
        # 计算入度和出度
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in self.edges.values():
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1
        
        # 找出中心节点
        max_in_degree = max(in_degree.values()) if in_degree else 0
        central_nodes = [n for n, d in in_degree.items() if d == max_in_degree]
        
        # 找出叶子节点
        leaf_nodes = [n for n in self.nodes if out_degree.get(n, 0) == 0]
        
        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'avg_degree': len(self.edges) / len(self.nodes) if self.nodes else 0,
            'central_nodes': central_nodes,
            'leaf_nodes': leaf_nodes,
            'cycles_detected': len(self.detect_cycles())
        }
