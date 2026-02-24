# =============================================================================
# Encryption Utilities
# =============================================================================

import base64
import os
from typing import Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("encryption")


class EncryptionManager:
    """加密管理器"""
    
    def __init__(self):
        self._key = self._derive_key()
        self._fernet = Fernet(self._key)
    
    def _derive_key(self) -> bytes:
        """從SECRET_KEY派生加密密鑰"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"eco-backend-salt",  # 生產環境應使用隨機salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(settings.SECRET_KEY.encode()))
        return key
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """加密數據"""
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self._fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密數據"""
        try:
            encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self._fernet.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            logger.error("decryption_failed", error=str(e))
            raise
    
    def encrypt_dict(self, data: dict) -> str:
        """加密字典"""
        import json
        return self.encrypt(json.dumps(data))
    
    def decrypt_dict(self, encrypted_data: str) -> dict:
        """解密字典"""
        import json
        return json.loads(self.decrypt(encrypted_data))


# 全局加密管理器實例
encryption_manager = EncryptionManager()


def encrypt_provider_config(config: dict) -> str:
    """加密提供者配置"""
    return encryption_manager.encrypt_dict(config)


def decrypt_provider_config(encrypted_config: str) -> dict:
    """解密提供者配置"""
    return encryption_manager.decrypt_dict(encrypted_config)


def generate_secure_token(length: int = 32) -> str:
    """生成安全令牌"""
    return base64.urlsafe_b64encode(os.urandom(length)).decode()[:length]


def hash_sensitive_data(data: str) -> str:
    """哈希敏感數據"""
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()
