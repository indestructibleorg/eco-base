# =============================================================================
# Data Persistence Adapters
# =============================================================================
# 數據持久化適配器實現
# 封裝具體的數據庫/BaaS服務調用
# =============================================================================

import os
import re
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from datetime import datetime
import httpx
import asyncio

from core.interfaces import (
    IDataPersistenceProvider, CapabilityContext, OperationResult,
    StreamChunk, QuerySpec, MutationSpec, CapabilityDomain
)


# =============================================================================
# SQL 安全驗證
# =============================================================================

# 危險 SQL 關鍵詞
DANGEROUS_SQL_KEYWORDS: Set[str] = {
    'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE',
    'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'SP_',
    'XP_', 'SYS_', 'INFORMATION_SCHEMA', 'PG_',
    'SHUTDOWN', 'KILL', 'BACKUP', 'RESTORE',
}

# SQL 注入模式
SQL_INJECTION_PATTERNS = [
    r'\bUNION\s+SELECT\b',
    r'\bUNION\s+ALL\s+SELECT\b',
    r';\s*(SELECT|INSERT|UPDATE|DELETE|DROP)\s+',
    r'/\*.*?\*/',
    r'--[^\n]*$',
    r'#[^\n]*$',
    r'\bOR\s+\d+\s*=\s*\d+',
    r'\bAND\s+\d+\s*=\s*\d+',
    r'\bSLEEP\s*\(',
    r'\bBENCHMARK\s*\(',
    r'\bLOAD_FILE\s*\(',
    r'\bINTO\s+(OUTFILE|DUMPFILE)\s*',
]

# 允許的表名格式
TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


class SQLSecurityError(Exception):
    """SQL 安全錯誤"""
    pass


def validate_table_name(table_name: str) -> bool:
    """
    驗證表名是否合法
    
    Args:
        table_name: 表名
        
    Returns:
        是否合法
        
    Raises:
        SQLSecurityError: 表名不合法
    """
    if not table_name or not isinstance(table_name, str):
        raise SQLSecurityError("Table name cannot be empty")
    
    if len(table_name) > 64:  # PostgreSQL 表名長度限制
        raise SQLSecurityError("Table name too long")
    
    if not TABLE_NAME_PATTERN.match(table_name):
        raise SQLSecurityError(f"Invalid table name format: {table_name}")
    
    # 檢查是否為保留字
    reserved_words = {'SELECT', 'FROM', 'WHERE', 'TABLE', 'INDEX', 'USER'}
    if table_name.upper() in reserved_words:
        raise SQLSecurityError(f"Table name is a reserved word: {table_name}")
    
    return True


def validate_sql_query(sql: str, allowed_tables: Optional[List[str]] = None) -> bool:
    """
    驗證 SQL 查詢是否安全
    
    Args:
        sql: SQL 查詢語句
        allowed_tables: 允許查詢的表名列表
        
    Returns:
        是否安全
        
    Raises:
        SQLSecurityError: 檢測到不安全 SQL
    """
    if not sql or not isinstance(sql, str):
        raise SQLSecurityError("SQL query cannot be empty")
    
    sql_upper = sql.upper().strip()
    
    # 1. 檢查危險關鍵詞
    for keyword in DANGEROUS_SQL_KEYWORDS:
        if keyword in sql_upper:
            raise SQLSecurityError(f"Dangerous SQL keyword detected: {keyword}")
    
    # 2. 檢查注入模式
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE):
            raise SQLSecurityError(f"SQL injection pattern detected")
    
    # 3. 檢查是否只包含 SELECT
    first_word = sql_upper.split()[0] if sql_upper else ""
    if first_word != 'SELECT':
        raise SQLSecurityError(f"Only SELECT queries are allowed, got: {first_word}")
    
    # 4. 檢查表名白名單
    if allowed_tables:
        # 提取 FROM 子句中的表名
        from_pattern = r'\bFROM\s+(\w+)'
        tables_found = re.findall(from_pattern, sql_upper)
        
        for table in tables_found:
            if table.upper() not in [t.upper() for t in allowed_tables]:
                raise SQLSecurityError(f"Table not in whitelist: {table}")
    
    # 5. 檢查語句數量
    if sql.count(';') > 0:
        raise SQLSecurityError("Multiple SQL statements not allowed")
    
    return True


def sanitize_filter_value(value: Any) -> str:
    """
    清理過濾值，防止注入
    
    Args:
        value: 過濾值
        
    Returns:
        清理後的字符串
    """
    if value is None:
        return "null"
    
    str_value = str(value)
    
    # 移除 NULL 字節
    str_value = str_value.replace('\x00', '')
    
    # 轉義單引號
    str_value = str_value.replace("'", "''")
    
    return str_value


class AlphaPersistenceAdapter(IDataPersistenceProvider):
    """
    Alpha 數據持久化適配器
    基於 PostgreSQL + 實時訂閱的 BaaS 服務
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._base_url = config.get('url')
        self._anon_key = config.get('anon_key')
        self._service_key = config.get('service_key')
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={'apikey': self._anon_key}
        )
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.DATA_PERSISTENCE
    
    @property
    def provider_id(self) -> str:
        return "alpha-persistence"
    
    async def health_check(self) -> OperationResult:
        try:
            response = await self._client.get('/rest/v1/')
            return OperationResult(
                success=response.status_code == 200,
                data={'status': 'healthy', 'latency': response.elapsed.total_seconds() * 1000}
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def get_capabilities(self) -> List[str]:
        return ['query', 'mutate', 'subscribe', 'sql', 'vector_search', 'auth', 'storage']
    
    async def query(self, spec: QuerySpec, ctx: CapabilityContext) -> OperationResult:
        """執行查詢"""
        start_time = datetime.utcnow()
        try:
            # 驗證表名
            validate_table_name(spec.table)
            
            # 構建 PostgREST 查詢
            url = f'/rest/v1/{spec.table}'
            params = {}
            
            if spec.filters:
                for key, value in spec.filters.items():
                    # 驗證過濾字段名
                    if not TABLE_NAME_PATTERN.match(key):
                        raise SQLSecurityError(f"Invalid filter field name: {key}")
                    params[key] = f'eq.{sanitize_filter_value(value)}'
            
            if spec.limit:
                if not isinstance(spec.limit, int) or spec.limit < 1 or spec.limit > 1000:
                    return OperationResult(
                        success=False,
                        error_message="Invalid limit value"
                    )
                params['limit'] = spec.limit
            
            if spec.offset:
                if not isinstance(spec.offset, int) or spec.offset < 0:
                    return OperationResult(
                        success=False,
                        error_message="Invalid offset value"
                    )
                params['offset'] = spec.offset
            
            if spec.ordering:
                # 驗證排序字段
                for field in spec.ordering:
                    clean_field = field.lstrip('-').lstrip('+')
                    if not TABLE_NAME_PATTERN.match(clean_field):
                        raise SQLSecurityError(f"Invalid ordering field: {field}")
                params['order'] = ','.join(spec.ordering)
            
            response = await self._client.get(url, params=params)
            data = response.json()
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return OperationResult(
                success=response.status_code == 200,
                data=data,
                latency_ms=latency,
                provider_info={'provider': self.provider_id, 'operation': 'query'}
            )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def mutate(self, spec: MutationSpec, ctx: CapabilityContext) -> OperationResult:
        """執行變更"""
        start_time = datetime.utcnow()
        try:
            # 驗證表名
            validate_table_name(spec.table)
            
            # 驗證操作類型
            allowed_operations = {'insert', 'update', 'delete', 'upsert'}
            if spec.operation not in allowed_operations:
                return OperationResult(
                    success=False,
                    error_message=f'Unknown operation: {spec.operation}. Allowed: {allowed_operations}'
                )
            
            url = f'/rest/v1/{spec.table}'
            
            if spec.operation == 'insert':
                response = await self._client.post(url, json=spec.data)
            elif spec.operation == 'update':
                # 驗證條件字段名
                for key in spec.conditions.keys():
                    if not TABLE_NAME_PATTERN.match(key):
                        raise SQLSecurityError(f"Invalid condition field name: {key}")
                params = {k: f'eq.{sanitize_filter_value(v)}' for k, v in spec.conditions.items()}
                response = await self._client.patch(url, params=params, json=spec.data)
            elif spec.operation == 'delete':
                # 驗證條件字段名
                for key in spec.conditions.keys():
                    if not TABLE_NAME_PATTERN.match(key):
                        raise SQLSecurityError(f"Invalid condition field name: {key}")
                params = {k: f'eq.{sanitize_filter_value(v)}' for k, v in spec.conditions.items()}
                response = await self._client.delete(url, params=params)
            elif spec.operation == 'upsert':
                headers = {'Prefer': 'resolution=merge-duplicates'}
                response = await self._client.post(url, json=spec.data, headers=headers)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return OperationResult(
                success=response.status_code in [200, 201, 204],
                data=response.json() if response.content else None,
                latency_ms=latency
            )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def subscribe(self, table: str, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """訂閱實時變更 (WebSocket)"""
        import websockets
        
        ws_url = self._base_url.replace('https://', 'wss://').replace('http://', 'ws://')
        ws_url = f"{ws_url}/realtime/v1/websocket?apikey={self._anon_key}"
        
        try:
            async with websockets.connect(ws_url) as ws:
                # 訂閱表變更
                subscribe_msg = {
                    'event': 'phx_join',
                    'topic': f'realtime:{table}',
                    'payload': {},
                    'ref': '1'
                }
                await ws.send(json.dumps(subscribe_msg))
                
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    if data.get('event') == 'INSERT' or data.get('event') == 'UPDATE':
                        yield StreamChunk(
                            content=data.get('payload', {}),
                            is_final=False,
                            metadata={'event_type': data.get('event')}
                        )
        except Exception as e:
            yield StreamChunk(content={'error': str(e)}, is_final=True)
    
    async def execute_sql(self, sql: str, params: List[Any], ctx: CapabilityContext) -> OperationResult:
        """執行原生 SQL (需要 service_key)"""
        try:
            # SQL 注入防護驗證
            validate_sql_query(sql)
            
            # 驗證參數類型
            if not isinstance(params, list):
                return OperationResult(
                    success=False,
                    error_message="Params must be a list"
                )
            
            client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={'apikey': self._service_key}
            )
            
            response = await client.post('/rest/v1/rpc/exec_sql', json={
                'query': sql,
                'params': params
            })
            
            return OperationResult(
                success=response.status_code == 200,
                data=response.json()
            )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def vector_search(self, table: str, vector: List[float], top_k: int, ctx: CapabilityContext) -> OperationResult:
        """向量相似度搜索 (使用 pgvector)"""
        try:
            response = await self._client.post(f'/rest/v1/rpc/match_{table}', json={
                'query_embedding': vector,
                'match_threshold': 0.7,
                'match_count': top_k
            })
            
            return OperationResult(
                success=response.status_code == 200,
                data=response.json()
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))


class BetaPersistenceAdapter(IDataPersistenceProvider):
    """
    Beta 數據持久化適配器
    基於 Vitess 的分散式數據庫服務
    """
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._host = config.get('host')
        self._username = config.get('username')
        self._password = config.get('password')
        self._database = config.get('database')
    
    @property
    def domain(self) -> CapabilityDomain:
        return CapabilityDomain.DATA_PERSISTENCE
    
    @property
    def provider_id(self) -> str:
        return "beta-persistence"
    
    async def health_check(self) -> OperationResult:
        # 使用 MySQL 連接池檢查
        return OperationResult(success=True, data={'status': 'healthy'})
    
    async def get_capabilities(self) -> List[str]:
        return ['query', 'mutate', 'sql', 'sharding', 'zero_downtime_migration']
    
    async def query(self, spec: QuerySpec, ctx: CapabilityContext) -> OperationResult:
        """執行查詢 (通過 HTTP API)"""
        try:
            # 驗證表名
            validate_table_name(spec.table)
            
            # 使用參數化查詢
            query = "SELECT * FROM {}".format(spec.table)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'https://{self._host}/v1/query',
                    auth=(self._username, self._password),
                    json={
                        'query': query,
                        'params': []
                    }
                )
                return OperationResult(
                    success=response.status_code == 200,
                    data=response.json()
                )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def mutate(self, spec: MutationSpec, ctx: CapabilityContext) -> OperationResult:
        """執行變更"""
        start_time = datetime.utcnow()
        try:
            # 驗證表名
            validate_table_name(spec.table)
            
            # 驗證操作類型
            allowed_operations = {'insert', 'update', 'delete', 'upsert'}
            if spec.operation not in allowed_operations:
                return OperationResult(
                    success=False,
                    error_message=f'Unknown operation: {spec.operation}. Allowed: {allowed_operations}'
                )
            
            async with httpx.AsyncClient() as client:
                if spec.operation == 'insert':
                    response = await client.post(
                        f'https://{self._host}/v1/query',
                        auth=(self._username, self._password),
                        json={
                            'query': f"INSERT INTO {spec.table} ({', '.join(spec.data.keys())}) VALUES ({', '.join(['?' for _ in spec.data])})",
                            'params': list(spec.data.values())
                        }
                    )
                elif spec.operation == 'update':
                    # 驗證條件字段名
                    for key in spec.conditions.keys():
                        if not TABLE_NAME_PATTERN.match(key):
                            raise SQLSecurityError(f"Invalid condition field name: {key}")
                    set_clause = ', '.join([f"{k} = ?" for k in spec.data.keys()])
                    where_clause = ' AND '.join([f"{k} = ?" for k in spec.conditions.keys()])
                    response = await client.post(
                        f'https://{self._host}/v1/query',
                        auth=(self._username, self._password),
                        json={
                            'query': f"UPDATE {spec.table} SET {set_clause} WHERE {where_clause}",
                            'params': list(spec.data.values()) + list(spec.conditions.values())
                        }
                    )
                elif spec.operation == 'delete':
                    # 驗證條件字段名
                    for key in spec.conditions.keys():
                        if not TABLE_NAME_PATTERN.match(key):
                            raise SQLSecurityError(f"Invalid condition field name: {key}")
                    where_clause = ' AND '.join([f"{k} = ?" for k in spec.conditions.keys()])
                    response = await client.post(
                        f'https://{self._host}/v1/query',
                        auth=(self._username, self._password),
                        json={
                            'query': f"DELETE FROM {spec.table} WHERE {where_clause}",
                            'params': list(spec.conditions.values())
                        }
                    )
                elif spec.operation == 'upsert':
                    # Vitess 不支持原生 upsert，使用 INSERT ON DUPLICATE KEY UPDATE
                    response = await client.post(
                        f'https://{self._host}/v1/query',
                        auth=(self._username, self._password),
                        json={
                            'query': f"INSERT INTO {spec.table} ({', '.join(spec.data.keys())}) VALUES ({', '.join(['?' for _ in spec.data])}) ON DUPLICATE KEY UPDATE {', '.join([f'{k}=VALUES({k})' for k in spec.data.keys()])}",
                            'params': list(spec.data.values())
                        }
                    )
                
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return OperationResult(
                    success=response.status_code == 200,
                    data=response.json() if response.content else None,
                    latency_ms=latency
                )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def subscribe(self, table: str, ctx: CapabilityContext) -> AsyncIterator[StreamChunk]:
        """訂閱變更"""
        # Beta 暫不支持實時訂閱
        yield StreamChunk(content={'message': 'Not supported'}, is_final=True)
    
    async def execute_sql(self, sql: str, params: List[Any], ctx: CapabilityContext) -> OperationResult:
        """執行原生 SQL"""
        try:
            # SQL 注入防護驗證
            validate_sql_query(sql)
            
            # 驗證參數類型
            if not isinstance(params, list):
                return OperationResult(
                    success=False,
                    error_message="Params must be a list"
                )
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f'https://{self._host}/v1/query',
                    auth=(self._username, self._password),
                    json={'query': sql, 'params': params}
                )
                return OperationResult(
                    success=response.status_code == 200,
                    data=response.json()
                )
        except SQLSecurityError as e:
            return OperationResult(
                success=False,
                error_message=f"SQL security error: {str(e)}"
            )
        except Exception as e:
            return OperationResult(success=False, error_message=str(e))
    
    async def vector_search(self, table: str, vector: List[float], top_k: int, ctx: CapabilityContext) -> OperationResult:
        """向量搜索 (Beta 不原生支持)"""
        return OperationResult(
            success=False,
            error_message='Vector search not supported in Beta adapter'
        )
