# =============================================================================
# Security Utilities
# =============================================================================
# 安全相關工具函數
# =============================================================================

import re
from typing import List, Dict, Any, Optional, Set
from app.core.logging import get_logger

logger = get_logger("security_utils")


# SQL 注入檢測模式
SQL_INJECTION_PATTERNS = [
    # 聯合查詢
    r'\bUNION\s+SELECT\b',
    r'\bUNION\s+ALL\s+SELECT\b',
    # 堆疊查詢
    r';\s*SELECT\s+',
    r';\s*INSERT\s+',
    r';\s*UPDATE\s+',
    r';\s*DELETE\s+',
    r';\s*DROP\s+',
    # 註釋
    r'/\*.*?\*/',
    r'--.*?$',
    r'#.*?$',
    # 布爾盲注
    r'\bOR\s+\d+\s*=\s*\d+',
    r'\bAND\s+\d+\s*=\s*\d+',
    # 時間盲注
    r'\bSLEEP\s*\(',
    r'\bBENCHMARK\s*\(',
    # 錯誤注入
    r'\bEXTRACTVALUE\s*\(',
    r'\bUPDATEXML\s*\(',
    # 其他危險函數
    r'\bLOAD_FILE\s*\(',
    r'\bINTO\s+OUTFILE\s*',
    r'\bINTO\s+DUMPFILE\s*',
]

# 危險 SQL 關鍵詞
DANGEROUS_SQL_KEYWORDS: Set[str] = {
    'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE',
    'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'SP_',
    'XP_', 'SYS_', 'INFORMATION_SCHEMA', 'PG_',
}

# 允許的 SQL 操作
ALLOWED_SQL_OPERATIONS: Set[str] = {'SELECT'}


class SQLInjectionError(Exception):
    """SQL 注入檢測錯誤"""
    pass


def validate_sql_query(sql: str, allowed_tables: Optional[List[str]] = None) -> bool:
    """
    驗證 SQL 查詢是否安全
    
    Args:
        sql: SQL 查詢語句
        allowed_tables: 允許查詢的表名列表
        
    Returns:
        是否安全
        
    Raises:
        SQLInjectionError: 檢測到 SQL 注入
    """
    if not sql or not isinstance(sql, str):
        raise SQLInjectionError("SQL query cannot be empty")
    
    # 轉換為大寫進行檢查
    sql_upper = sql.upper().strip()
    
    # 1. 檢查危險關鍵詞
    for keyword in DANGEROUS_SQL_KEYWORDS:
        if keyword in sql_upper:
            logger.warning(
                "sql_injection_detected",
                reason=f"Dangerous keyword: {keyword}",
                sql_preview=sql[:100]
            )
            raise SQLInjectionError(f"Dangerous SQL keyword detected: {keyword}")
    
    # 2. 檢查注入模式
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE):
            logger.warning(
                "sql_injection_detected",
                reason=f"Pattern match: {pattern}",
                sql_preview=sql[:100]
            )
            raise SQLInjectionError(f"SQL injection pattern detected: {pattern}")
    
    # 3. 檢查是否只包含允許的操作
    # 提取 SQL 操作類型
    first_word = sql_upper.split()[0] if sql_upper else ""
    if first_word not in ALLOWED_SQL_OPERATIONS:
        logger.warning(
            "sql_injection_detected",
            reason=f"Disallowed operation: {first_word}",
            sql_preview=sql[:100]
        )
        raise SQLInjectionError(f"SQL operation not allowed: {first_word}")
    
    # 4. 檢查表名白名單
    if allowed_tables:
        # 提取查詢中的表名
        table_pattern = r'\bFROM\s+(\w+)'
        tables_found = re.findall(table_pattern, sql_upper)
        
        for table in tables_found:
            if table.upper() not in [t.upper() for t in allowed_tables]:
                logger.warning(
                    "sql_injection_detected",
                    reason=f"Table not in whitelist: {table}",
                    sql_preview=sql[:100]
                )
                raise SQLInjectionError(f"Table not allowed: {table}")
    
    # 5. 檢查語句數量 (防止堆疊查詢)
    statement_count = sql.count(';') + 1
    if statement_count > 1:
        logger.warning(
            "sql_injection_detected",
            reason="Multiple statements detected",
            sql_preview=sql[:100]
        )
        raise SQLInjectionError("Multiple SQL statements not allowed")
    
    logger.debug("sql_query_validated", sql_preview=sql[:100])
    return True


def sanitize_sql_input(value: Any) -> str:
    """
    清理 SQL 輸入值
    
    Args:
        value: 輸入值
        
    Returns:
        清理後的字符串
    """
    if value is None:
        return "NULL"
    
    # 轉換為字符串
    str_value = str(value)
    
    # 移除危險字符
    dangerous_chars = [
        '\x00',  # NULL 字節
        '\x1a',  # SUB 字符
        '\x08',  # 退格
    ]
    
    for char in dangerous_chars:
        str_value = str_value.replace(char, '')
    
    # 轉義單引號
    str_value = str_value.replace("'", "''")
    
    return str_value


# =============================================================================
# 敏感信息過濾
# =============================================================================

# 敏感字段名稱 (大小寫不敏感)
SENSITIVE_FIELD_NAMES: Set[str] = {
    # 認證相關
    'password', 'passwd', 'pwd', 'pass',
    'token', 'access_token', 'refresh_token', 'auth_token',
    'api_key', 'apikey', 'secret', 'secret_key', 'private_key',
    'credential', 'credentials', 'auth',
    # 個人信息
    'ssn', 'social_security', 'credit_card', 'cvv', 'pin',
    # 其他敏感信息
    'authorization', 'cookie', 'session',
}

# 敏感字段模式
SENSITIVE_FIELD_PATTERNS = [
    r'.*password.*',
    r'.*secret.*',
    r'.*token.*',
    r'.*key.*',
    r'.*credential.*',
    r'.*auth.*',
]

# 敏感值模式
SENSITIVE_VALUE_PATTERNS = [
    r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$',  # JWT
    r'^sk-[a-zA-Z0-9]{48}$',  # OpenAI API Key
    r'^ghp_[a-zA-Z0-9]{36}$',  # GitHub PAT
    r'^xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*',  # Slack Token
]


def is_sensitive_field(field_name: str) -> bool:
    """
    檢查字段名是否敏感
    
    Args:
        field_name: 字段名稱
        
    Returns:
        是否敏感
    """
    field_lower = field_name.lower()
    
    # 精確匹配
    if field_lower in SENSITIVE_FIELD_NAMES:
        return True
    
    # 模式匹配
    for pattern in SENSITIVE_FIELD_PATTERNS:
        if re.match(pattern, field_lower, re.IGNORECASE):
            return True
    
    return False


def is_sensitive_value(value: str) -> bool:
    """
    檢查值是否為敏感信息
    
    Args:
        value: 值
        
    Returns:
        是否敏感
    """
    if not isinstance(value, str):
        return False
    
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if re.match(pattern, value):
            return True
    
    return False


def mask_sensitive_value(value: Any, mask: str = '***') -> Any:
    """
    遮罩敏感值
    
    Args:
        value: 原始值
        mask: 遮罩字符串
        
    Returns:
        遮罩後的值
    """
    if value is None:
        return None
    
    if isinstance(value, str):
        if is_sensitive_value(value):
            return mask
        # 長敏感值只顯示前後部分
        if len(value) > 20:
            return f"{value[:4]}...{value[-4:]}"
    
    return value


def sanitize_payload(payload: Dict[str, Any], depth: int = 0, max_depth: int = 5) -> Dict[str, Any]:
    """
    清理請求/響應數據中的敏感信息
    
    Args:
        payload: 數據字典
        depth: 當前深度
        max_depth: 最大深度
        
    Returns:
        清理後的數據
    """
    if depth > max_depth:
        return {"...": "max depth reached"}
    
    if not isinstance(payload, dict):
        if isinstance(payload, list):
            return [sanitize_payload(item, depth + 1, max_depth) if isinstance(item, dict) else item 
                    for item in payload]
        return payload
    
    result = {}
    for key, value in payload.items():
        # 檢查字段名是否敏感
        if is_sensitive_field(key):
            result[key] = '***'
        # 檢查值是否敏感
        elif isinstance(value, str) and is_sensitive_value(value):
            result[key] = '***'
        # 遞歸處理嵌套字典
        elif isinstance(value, dict):
            result[key] = sanitize_payload(value, depth + 1, max_depth)
        # 處理列表
        elif isinstance(value, list):
            result[key] = [
                sanitize_payload(item, depth + 1, max_depth) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    清理 HTTP 頭中的敏感信息
    
    Args:
        headers: HTTP 頭字典
        
    Returns:
        清理後的頭字典
    """
    sensitive_headers = {
        'authorization', 'cookie', 'x-api-key', 'x-auth-token',
        'proxy-authorization', 'set-cookie',
    }
    
    result = {}
    for key, value in headers.items():
        if key.lower() in sensitive_headers:
            result[key] = '***'
        else:
            result[key] = value
    
    return result


# =============================================================================
# 輸入驗證
# =============================================================================

def validate_prompt(prompt: str, max_length: int = 10000) -> str:
    """
    驗證提示文本
    
    Args:
        prompt: 提示文本
        max_length: 最大長度
        
    Returns:
        驗證後的文本
        
    Raises:
        ValueError: 驗證失敗
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt cannot be empty")
    
    if len(prompt) > max_length:
        raise ValueError(f"Prompt exceeds maximum length of {max_length}")
    
    # 檢查惡意內容模式
    malicious_patterns = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',  # JavaScript 協議
        r'on\w+\s*=',  # 事件處理器
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
            raise ValueError("Prompt contains potentially malicious content")
    
    return prompt


def validate_table_name(table_name: str, allowed_tables: Optional[List[str]] = None) -> str:
    """
    驗證表名
    
    Args:
        table_name: 表名
        allowed_tables: 允許的表名列表
        
    Returns:
        驗證後的表名
        
    Raises:
        ValueError: 驗證失敗
    """
    if not table_name or not isinstance(table_name, str):
        raise ValueError("Table name cannot be empty")
    
    # 表名只能包含字母、數字、下劃線
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        raise ValueError("Invalid table name format")
    
    # 檢查白名單
    if allowed_tables and table_name not in allowed_tables:
        raise ValueError(f"Table '{table_name}' is not in allowed list")
    
    return table_name


def validate_code_input(code: str, max_length: int = 10000) -> str:
    """
    驗證代碼輸入
    
    Args:
        code: 代碼字符串
        max_length: 最大長度
        
    Returns:
        驗證後的代碼
        
    Raises:
        ValueError: 驗證失敗
    """
    if not code or not isinstance(code, str):
        raise ValueError("Code cannot be empty")
    
    if len(code) > max_length:
        raise ValueError(f"Code exceeds maximum length of {max_length}")
    
    # 檢查危險模式
    dangerous_patterns = [
        r'\bexec\s*\(',
        r'\beval\s*\(',
        r'\bos\.system\s*\(',
        r'\bsubprocess\.',
        r'__import__\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            logger.warning(
                "dangerous_code_pattern_detected",
                pattern=pattern,
                code_preview=code[:100]
            )
            raise ValueError("Code contains potentially dangerous patterns")
    
    return code
