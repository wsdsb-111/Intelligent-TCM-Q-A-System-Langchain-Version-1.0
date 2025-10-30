"""
缓存工具
提供内存缓存和持久化缓存功能
"""

import time
import json
import hashlib
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timedelta
import threading
import logging


class CacheItem:
    """缓存项"""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        """
        初始化缓存项
        
        Args:
            value: 缓存值
            ttl: 生存时间（秒），None表示永不过期
        """
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """访问缓存项"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }


class MemoryCache:
    """内存缓存"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存项数量
            default_ttl: 默认生存时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheItem] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """生成缓存键"""
        if isinstance(key, dict):
            # 对字典进行排序后生成哈希
            sorted_key = json.dumps(key, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(sorted_key.encode('utf-8')).hexdigest()
        return str(key)
    
    def get(self, key: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值或None
        """
        cache_key = self._generate_key(key)
        
        with self.lock:
            if cache_key not in self.cache:
                return None
            
            item = self.cache[cache_key]
            
            if item.is_expired():
                del self.cache[cache_key]
                return None
            
            item.access()
            return item.value
    
    def set(self, key: Union[str, Dict[str, Any]], value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），None使用默认值
            
        Returns:
            是否设置成功
        """
        cache_key = self._generate_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        
        with self.lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                self._evict_oldest()
            
            self.cache[cache_key] = CacheItem(value, ttl)
            return True
    
    def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        cache_key = self._generate_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                return True
            return False
    
    def clear(self):
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
    
    def _evict_oldest(self):
        """驱逐最旧的缓存项"""
        if not self.cache:
            return
        
        # 找到最久未访问的项
        oldest_key = min(self.cache.keys(), 
                        key=lambda k: self.cache[k].last_accessed)
        del self.cache[oldest_key]
    
    def cleanup_expired(self):
        """清理过期的缓存项"""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_items = len(self.cache)
            expired_items = sum(1 for item in self.cache.values() if item.is_expired())
            
            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "active_items": total_items - expired_items,
                "max_size": self.max_size,
                "usage_ratio": total_items / self.max_size if self.max_size > 0 else 0
            }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        """初始化缓存管理器"""
        self.caches: Dict[str, MemoryCache] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_cache(self, name: str, max_size: int = 1000, default_ttl: Optional[int] = None) -> MemoryCache:
        """
        获取或创建缓存实例
        
        Args:
            name: 缓存名称
            max_size: 最大缓存项数量
            default_ttl: 默认生存时间
            
        Returns:
            缓存实例
        """
        if name not in self.caches:
            self.caches[name] = MemoryCache(max_size, default_ttl)
        return self.caches[name]
    
    def clear_all(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()
    
    def cleanup_all(self):
        """清理所有过期的缓存项"""
        for cache in self.caches.values():
            cache.cleanup_expired()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有缓存的统计信息"""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats


# 全局缓存管理器实例
_global_cache_manager = None

def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager

def cached(ttl: Optional[int] = None, cache_name: str = "default"):
    """
    缓存装饰器
    
    Args:
        ttl: 生存时间（秒）
        cache_name: 缓存名称
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        cache = get_cache_manager().get_cache(cache_name)
        
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            
            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator
