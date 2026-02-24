"""
实体抽取器
从日志、事件、告警中提取实体信息
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型"""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    GATEWAY = "gateway"
    SERVER = "server"
    CONTAINER = "container"
    POD = "pod"
    VM = "vm"
    NETWORK = "network"
    ERROR = "error"
    WARNING = "warning"
    ANOMALY = "anomaly"
    DEPLOYMENT = "deployment"
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE = "scale"
    CONFIG_UPDATE = "config_update"


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    type: EntityType
    properties: Dict[str, Any]
    source: str
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'properties': self.properties,
            'source': self.source,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class PatternMatcher:
    """模式匹配器"""
    
    def __init__(self):
        # 服务名称模式
        self.service_patterns = [
            r'([a-z][a-z0-9-]*(?:-service)?)',
            r'service[:\s]+([a-z][a-z0-9-]*)',
            r'microservice[:\s]+([a-z][a-z0-9-]*)',
        ]
        
        # 数据库模式
        self.database_patterns = [
            r'(mysql|postgresql|mongodb|redis|elasticsearch|cassandra)[-\d]*',
            r'database[:\s]+([a-z][a-z0-9-]*)',
            r'db[:\s]+([a-z][a-z0-9-]*)',
        ]
        
        # IP 地址模式
        self.ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        # 容器/Pod 模式
        self.container_patterns = [
            r'container[:\s]+([a-z][a-z0-9-]*)',
            r'pod[:\s/]+([a-z][a-z0-9-]*)',
            r'deployment[:\s/]+([a-z][a-z0-9-]*)',
        ]
        
        # 错误模式
        self.error_patterns = [
            r'(Exception|Error)[:\s]+([A-Za-z]+)',
            r'error[:\s]+([a-z_]+)',
            r'failed[:\s]+([a-z_]+)',
        ]
        
    def extract_services(self, text: str) -> List[Dict[str, Any]]:
        """提取服务名称"""
        services = []
        for pattern in self.service_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if match.groups() else match.group(0)
                if len(name) > 2:  # 过滤短名称
                    services.append({
                        'name': name.lower(),
                        'match': match.group(0),
                        'position': match.span()
                    })
        return services
    
    def extract_databases(self, text: str) -> List[Dict[str, Any]]:
        """提取数据库"""
        databases = []
        for pattern in self.database_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if match.groups() else match.group(0)
                databases.append({
                    'name': name.lower(),
                    'match': match.group(0),
                    'position': match.span()
                })
        return databases
    
    def extract_ips(self, text: str) -> List[Dict[str, Any]]:
        """提取 IP 地址"""
        ips = []
        matches = re.finditer(self.ip_pattern, text)
        for match in matches:
            ips.append({
                'ip': match.group(0),
                'position': match.span()
            })
        return ips
    
    def extract_containers(self, text: str) -> List[Dict[str, Any]]:
        """提取容器/Pod"""
        containers = []
        for pattern in self.container_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if match.groups() else match.group(0)
                containers.append({
                    'name': name.lower(),
                    'match': match.group(0),
                    'position': match.span()
                })
        return containers
    
    def extract_errors(self, text: str) -> List[Dict[str, Any]]:
        """提取错误信息"""
        errors = []
        for pattern in self.error_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if match.groups() else match.group(0)
                errors.append({
                    'name': name,
                    'match': match.group(0),
                    'position': match.span()
                })
        return errors


class EntityExtractor:
    """
    实体抽取器
    从各种数据源中提取实体
    """
    
    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.entity_cache: Dict[str, Entity] = {}
        
        # 已知实体库 (用于实体链接)
        self.known_entities: Dict[str, Dict[str, Any]] = {}
        
    def register_known_entity(self, name: str, entity_type: EntityType, 
                             properties: Dict[str, Any]):
        """注册已知实体"""
        self.known_entities[name.lower()] = {
            'type': entity_type,
            'properties': properties
        }
        
    def extract_from_log(self, log_entry: Dict[str, Any]) -> List[Entity]:
        """从日志中提取实体"""
        entities = []
        
        message = log_entry.get('message', '')
        source = log_entry.get('source', 'unknown')
        timestamp = log_entry.get('timestamp', datetime.now())
        
        # 提取服务
        services = self.pattern_matcher.extract_services(message)
        for svc in services:
            entity = self._create_entity(
                name=svc['name'],
                entity_type=EntityType.SERVICE,
                source=source,
                confidence=0.8,
                timestamp=timestamp,
                properties={'extracted_from': 'log', 'match': svc['match']}
            )
            entities.append(entity)
        
        # 提取数据库
        databases = self.pattern_matcher.extract_databases(message)
        for db in databases:
            entity_type = self._infer_database_type(db['name'])
            entity = self._create_entity(
                name=db['name'],
                entity_type=entity_type,
                source=source,
                confidence=0.7,
                timestamp=timestamp,
                properties={'extracted_from': 'log', 'match': db['match']}
            )
            entities.append(entity)
        
        # 提取错误
        errors = self.pattern_matcher.extract_errors(message)
        for err in errors:
            entity = self._create_entity(
                name=err['name'],
                entity_type=EntityType.ERROR,
                source=source,
                confidence=0.6,
                timestamp=timestamp,
                properties={'extracted_from': 'log', 'match': err['match']}
            )
            entities.append(entity)
        
        # 提取容器
        containers = self.pattern_matcher.extract_containers(message)
        for container in containers:
            entity = self._create_entity(
                name=container['name'],
                entity_type=EntityType.CONTAINER,
                source=source,
                confidence=0.75,
                timestamp=timestamp,
                properties={'extracted_from': 'log', 'match': container['match']}
            )
            entities.append(entity)
        
        return entities
    
    def extract_from_event(self, event: Dict[str, Any]) -> List[Entity]:
        """从事件中提取实体"""
        entities = []
        
        event_type = event.get('type', '')
        event_data = event.get('data', {})
        timestamp = event.get('timestamp', datetime.now())
        
        # 根据事件类型提取不同实体
        if event_type == 'deployment':
            service = event_data.get('service', '')
            if service:
                entity = self._create_entity(
                    name=service,
                    entity_type=EntityType.SERVICE,
                    source='deployment_event',
                    confidence=0.95,
                    timestamp=timestamp,
                    properties={
                        'version': event_data.get('version'),
                        'deployer': event_data.get('deployer'),
                        'event_type': 'deployment'
                    }
                )
                entities.append(entity)
                
        elif event_type == 'alert':
            service = event_data.get('service', '')
            if service:
                entity = self._create_entity(
                    name=service,
                    entity_type=EntityType.SERVICE,
                    source='alert_event',
                    confidence=0.9,
                    timestamp=timestamp,
                    properties={
                        'alert_name': event_data.get('alert_name'),
                        'severity': event_data.get('severity'),
                        'event_type': 'alert'
                    }
                )
                entities.append(entity)
                
        elif event_type == 'scaling':
            service = event_data.get('service', '')
            if service:
                entity = self._create_entity(
                    name=service,
                    entity_type=EntityType.SERVICE,
                    source='scaling_event',
                    confidence=0.9,
                    timestamp=timestamp,
                    properties={
                        'scale_direction': event_data.get('direction'),
                        'replicas': event_data.get('replicas'),
                        'event_type': 'scaling'
                    }
                )
                entities.append(entity)
        
        return entities
    
    def extract_from_metrics(self, metrics: Dict[str, Any]) -> List[Entity]:
        """从指标中提取实体"""
        entities = []
        
        service = metrics.get('service', '')
        metric_name = metrics.get('metric', '')
        timestamp = metrics.get('timestamp', datetime.now())
        
        if service:
            # 检测异常类型
            entity_type = EntityType.ANOMALY
            if 'error' in metric_name.lower():
                entity_type = EntityType.ERROR
            elif 'latency' in metric_name.lower() or 'response_time' in metric_name.lower():
                entity_type = EntityType.WARNING
            
            entity = self._create_entity(
                name=f"{service}_{metric_name}_anomaly",
                entity_type=entity_type,
                source='metrics',
                confidence=metrics.get('confidence', 0.7),
                timestamp=timestamp,
                properties={
                    'service': service,
                    'metric': metric_name,
                    'value': metrics.get('value'),
                    'threshold': metrics.get('threshold')
                }
            )
            entities.append(entity)
        
        return entities
    
    def _create_entity(self, name: str, entity_type: EntityType, 
                      source: str, confidence: float,
                      timestamp: datetime, properties: Dict[str, Any]) -> Entity:
        """创建实体"""
        # 生成唯一 ID
        entity_id = self._generate_entity_id(name, entity_type)
        
        # 实体链接
        linked_properties = self._link_entity(name, properties)
        
        entity = Entity(
            id=entity_id,
            name=name,
            type=entity_type,
            properties=linked_properties,
            source=source,
            confidence=confidence,
            timestamp=timestamp
        )
        
        # 缓存实体
        self.entity_cache[entity_id] = entity
        
        return entity
    
    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """生成实体 ID"""
        unique_string = f"{entity_type.value}:{name.lower()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def _link_entity(self, name: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """实体链接 - 关联已知实体"""
        linked = properties.copy()
        
        known = self.known_entities.get(name.lower())
        if known:
            linked['linked_to'] = known
            
        return linked
    
    def _infer_database_type(self, name: str) -> EntityType:
        """推断数据库类型"""
        name_lower = name.lower()
        if 'redis' in name_lower or 'cache' in name_lower:
            return EntityType.CACHE
        elif 'kafka' in name_lower or 'queue' in name_lower or 'rabbit' in name_lower:
            return EntityType.QUEUE
        else:
            return EntityType.DATABASE
    
    def disambiguate_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体消歧"""
        # 按名称和类型分组
        grouped = defaultdict(list)
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            grouped[key].append(entity)
        
        # 每组选择置信度最高的
        disambiguated = []
        for key, group in grouped.items():
            best = max(group, key=lambda e: e.confidence)
            disambiguated.append(best)
        
        return disambiguated
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """通过 ID 获取实体"""
        return self.entity_cache.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """通过类型获取实体"""
        return [e for e in self.entity_cache.values() if e.type == entity_type]
    
    def get_all_entities(self) -> List[Entity]:
        """获取所有实体"""
        return list(self.entity_cache.values())
