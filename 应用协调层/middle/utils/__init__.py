"""
工具函数模块
"""

from .logging_utils import setup_logging, get_logger
from .cache_utils import CacheManager
from .query_classifier import get_query_classifier

# 简单的验证函数
def validate_query(query: str) -> bool:
    """验证查询是否有效"""
    return isinstance(query, str) and len(query.strip()) > 0

def validate_config(config: dict) -> bool:
    """验证配置是否有效"""
    return isinstance(config, dict)

__all__ = [
    "setup_logging",
    "get_logger", 
    "CacheManager",
    "validate_query",
    "validate_config",
    "get_query_classifier"
]