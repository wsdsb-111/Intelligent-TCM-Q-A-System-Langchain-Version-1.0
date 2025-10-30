"""
性能优化系统
提供查询结果缓存机制、连接池和资源管理、并发处理优化
"""

import asyncio
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import gc

from ..models.data_models import RetrievalResult, FusedResult
from ..utils.logging_utils import get_logger, StructuredLogger, PerformanceLogger


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: datetime
    ttl: int  # 生存时间（秒）
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    original_query: str = ""


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000  # 最大条目数
    max_memory_mb: int = 100  # 最大内存使用（MB）
    default_ttl: int = 3600  # 默认生存时间（秒）
    cleanup_interval: int = 300  # 清理间隔（秒）
    enable_compression: bool = False  # 是否启用压缩


@dataclass
class ConnectionPoolConfig:
    """连接池配置"""
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: int = 30
    idle_timeout: int = 300
    max_lifetime: int = 3600
    retry_attempts: int = 3


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存大小
        """
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key in self.cache:
                # 移动到末尾（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self.cache:
                # 更新现有值
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 删除最久未使用的项
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """获取所有键"""
        with self._lock:
            return list(self.cache.keys())


class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化查询缓存
        
        Args:
            config: 缓存配置
        """
        self.config = config or CacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.memory_usage = 0
        self._lock = threading.RLock()
        self.logger = StructuredLogger("query_cache")
        self.perf_logger = PerformanceLogger("query_cache")
        
        # 启动清理任务
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _generate_cache_key(self, query: str, config: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 将查询和配置组合生成唯一键
        key_data = f"{query}:{sorted(config.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """计算对象大小（字节）"""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            return 1024  # 默认大小
    
    def _start_cleanup_task(self):
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                    self._cleanup_expired()
                    self._cleanup_memory()
                except Exception as e:
                    self.logger.error(f"缓存清理任务失败: {e}")
        
        # 在后台运行清理任务
        loop = asyncio.get_event_loop()
        self._cleanup_task = loop.create_task(cleanup_loop())
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        with self._lock:
            now = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if now - entry.timestamp > timedelta(seconds=entry.ttl):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache.pop(key)
                self.memory_usage -= entry.size_bytes
                self.logger.debug(f"清理过期缓存: {key}")
            
            if expired_keys:
                self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def _cleanup_memory(self):
        """清理内存使用"""
        with self._lock:
            max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
            
            if self.memory_usage <= max_memory_bytes:
                return
            
            # 按访问时间排序，删除最久未访问的项
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_access
            )
            
            removed_count = 0
            for key, entry in sorted_entries:
                if self.memory_usage <= max_memory_bytes * 0.8:  # 清理到80%
                    break
                
                del self.cache[key]
                self.memory_usage -= entry.size_bytes
                removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"内存清理，删除了 {removed_count} 个缓存项")
    
    def get(self, query: str, config: Dict[str, Any]) -> Optional[List[FusedResult]]:
        """获取缓存结果"""
        key = self._generate_cache_key(query, config)
        
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查是否过期
                if datetime.now() - entry.timestamp > timedelta(seconds=entry.ttl):
                    del self.cache[key]
                    self.memory_usage -= entry.size_bytes
                    return None
                
                # 更新访问信息
                entry.access_count += 1
                entry.last_access = datetime.now()
                
                self.perf_logger.log_retrieval_performance(
                    query=f"cache_hit:{query[:20]}",
                    response_time=0.001,  # 缓存命中很快
                    result_count=len(entry.value) if isinstance(entry.value, list) else 1,
                    modules_used=["cache"],
                    fusion_method="cache"
                )
                
                self.logger.debug(f"缓存命中: {key}")
                return entry.value
            
            return None
    
    def put(self, query: str, config: Dict[str, Any], results: List[FusedResult], ttl: Optional[int] = None) -> None:
        """存储缓存结果"""
        key = self._generate_cache_key(query, config)
        ttl = ttl or self.config.default_ttl
        
        with self._lock:
            # 计算大小
            size_bytes = self._calculate_size(results)
            
            # 检查内存限制
            if self.memory_usage + size_bytes > self.config.max_memory_mb * 1024 * 1024:
                self._cleanup_memory()
            
            # 检查数量限制
            if len(self.cache) >= self.config.max_size:
                # 删除最久未访问的项
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
                old_entry = self.cache.pop(oldest_key)
                self.memory_usage -= old_entry.size_bytes
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=results,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes,
                original_query=query
            )
            
            self.cache[key] = entry
            self.memory_usage += size_bytes
            
            self.logger.debug(f"缓存存储: {key}, 大小: {size_bytes} bytes")
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """使缓存失效"""
        with self._lock:
            if pattern is None:
                # 清空所有缓存
                count = len(self.cache)
                self.cache.clear()
                self.memory_usage = 0
                return count
            else:
                # 删除匹配模式的缓存
                keys_to_delete = []
                for key, entry in self.cache.items():
                    # 检查原始查询是否匹配模式
                    if pattern in entry.original_query:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    entry = self.cache.pop(key)
                    self.memory_usage -= entry.size_bytes
                return len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_access = sum(entry.access_count for entry in self.cache.values())
            avg_access = total_access / len(self.cache) if self.cache else 0
            
            return {
                "cache_size": len(self.cache),
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "total_access_count": total_access,
                "average_access_count": avg_access,
                "hit_rate": 0.0,  # 需要外部计算
                "max_size": self.config.max_size,
                "max_memory_mb": self.config.max_memory_mb
            }


class ConnectionPool:
    """连接池管理器"""
    
    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        """
        初始化连接池
        
        Args:
            config: 连接池配置
        """
        self.config = config or ConnectionPoolConfig()
        self.connections: List[Any] = []
        self.available_connections: List[Any] = []
        self.busy_connections: set = set()
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
        
        # 连接创建函数
        self._connection_factory: Optional[Callable] = None
    
    def set_connection_factory(self, factory: Callable):
        """设置连接创建工厂函数"""
        self._connection_factory = factory
    
    def _create_connection(self) -> Any:
        """创建新连接"""
        if self._connection_factory:
            return self._connection_factory()
        return None
    
    def get_connection(self) -> Optional[Any]:
        """获取可用连接"""
        with self._lock:
            # 尝试从可用连接中获取
            if self.available_connections:
                connection = self.available_connections.pop()
                self.busy_connections.add(connection)
                return connection
            
            # 如果连接数未达到上限，创建新连接
            if len(self.connections) < self.config.max_connections:
                connection = self._create_connection()
                if connection:
                    self.connections.append(connection)
                    self.busy_connections.add(connection)
                    return connection
            
            return None
    
    def return_connection(self, connection: Any):
        """归还连接"""
        with self._lock:
            if connection in self.busy_connections:
                self.busy_connections.remove(connection)
                self.available_connections.append(connection)
    
    def close_connection(self, connection: Any):
        """关闭连接"""
        with self._lock:
            if connection in self.connections:
                self.connections.remove(connection)
            if connection in self.busy_connections:
                self.busy_connections.remove(connection)
            if connection in self.available_connections:
                self.available_connections.remove(connection)
            
            # 这里应该调用连接的实际关闭方法
            if hasattr(connection, 'close'):
                connection.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._lock:
            return {
                "total_connections": len(self.connections),
                "available_connections": len(self.available_connections),
                "busy_connections": len(self.busy_connections),
                "max_connections": self.config.max_connections,
                "min_connections": self.config.min_connections
            }


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        """初始化资源管理器"""
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.process_pools: Dict[str, ProcessPoolExecutor] = {}
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def get_thread_pool(self, name: str, max_workers: int = 4) -> ThreadPoolExecutor:
        """获取线程池"""
        with self._lock:
            if name not in self.thread_pools:
                self.thread_pools[name] = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"{name}_pool"
                )
                self.logger.info(f"创建线程池: {name}, 工作线程数: {max_workers}")
            
            return self.thread_pools[name]
    
    def get_process_pool(self, name: str, max_workers: int = 2) -> ProcessPoolExecutor:
        """获取进程池"""
        with self._lock:
            if name not in self.process_pools:
                self.process_pools[name] = ProcessPoolExecutor(
                    max_workers=max_workers
                )
                self.logger.info(f"创建进程池: {name}, 工作进程数: {max_workers}")
            
            return self.process_pools[name]
    
    def get_connection_pool(self, name: str, config: Optional[ConnectionPoolConfig] = None) -> ConnectionPool:
        """获取连接池"""
        with self._lock:
            if name not in self.connection_pools:
                self.connection_pools[name] = ConnectionPool(config)
                self.logger.info(f"创建连接池: {name}")
            
            return self.connection_pools[name]
    
    def shutdown_all(self):
        """关闭所有资源"""
        with self._lock:
            # 关闭线程池
            for name, pool in self.thread_pools.items():
                pool.shutdown(wait=True)
                self.logger.info(f"关闭线程池: {name}")
            
            # 关闭进程池
            for name, pool in self.process_pools.items():
                pool.shutdown(wait=True)
                self.logger.info(f"关闭进程池: {name}")
            
            # 关闭连接池
            for name, pool in self.connection_pools.items():
                for connection in pool.connections:
                    pool.close_connection(connection)
                self.logger.info(f"关闭连接池: {name}")
            
            self.thread_pools.clear()
            self.process_pools.clear()
            self.connection_pools.clear()
        
        self.logger.info("所有资源已关闭")


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, 
                 cache_config: Optional[CacheConfig] = None,
                 enable_caching: bool = True,
                 enable_connection_pooling: bool = True):
        """
        初始化性能优化器
        
        Args:
            cache_config: 缓存配置
            enable_caching: 是否启用缓存
            enable_connection_pooling: 是否启用连接池
        """
        self.enable_caching = enable_caching
        self.enable_connection_pooling = enable_connection_pooling
        
        # 初始化组件
        self.query_cache = QueryCache(cache_config) if enable_caching else None
        self.resource_manager = ResourceManager()
        
        # 性能统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        
        # 日志记录器
        self.logger = StructuredLogger("performance_optimizer")
        self.perf_logger = PerformanceLogger("performance_optimizer")
    
    def get_cached_result(self, query: str, config: Dict[str, Any]) -> Optional[List[FusedResult]]:
        """获取缓存结果"""
        if not self.enable_caching or not self.query_cache:
            return None
        
        result = self.query_cache.get(query, config)
        if result is not None:
            self.cache_hits += 1
            self.logger.debug(f"缓存命中: {query[:50]}...")
        else:
            self.cache_misses += 1
        
        return result
    
    def cache_result(self, query: str, config: Dict[str, Any], results: List[FusedResult], ttl: Optional[int] = None):
        """缓存结果"""
        if not self.enable_caching or not self.query_cache:
            return
        
        self.query_cache.put(query, config, results, ttl)
        self.logger.debug(f"缓存存储: {query[:50]}...")
    
    def get_thread_pool(self, name: str, max_workers: int = 4) -> ThreadPoolExecutor:
        """获取线程池"""
        return self.resource_manager.get_thread_pool(name, max_workers)
    
    def get_process_pool(self, name: str, max_workers: int = 2) -> ProcessPoolExecutor:
        """获取进程池"""
        return self.resource_manager.get_process_pool(name, max_workers)
    
    def get_connection_pool(self, name: str, config: Optional[ConnectionPoolConfig] = None) -> ConnectionPool:
        """获取连接池"""
        return self.resource_manager.get_connection_pool(name, config)
    
    async def execute_parallel(self, 
                             tasks: List[Callable],
                             max_workers: int = 4) -> List[Any]:
        """并行执行任务"""
        if not tasks:
            return []
        
        # 使用线程池执行任务
        thread_pool = self.get_thread_pool("parallel_execution", max_workers)
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                futures.append(task())
            else:
                future = loop.run_in_executor(thread_pool, task)
                futures.append(future)
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"并行任务执行失败: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """使缓存失效"""
        if not self.enable_caching or not self.query_cache:
            return 0
        
        count = self.query_cache.invalidate(pattern)
        self.logger.info(f"缓存失效: {count} 项")
        return count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }
        
        if self.query_cache:
            stats["cache_stats"] = self.query_cache.get_stats()
        
        stats["resource_stats"] = {
            "thread_pools": len(self.resource_manager.thread_pools),
            "process_pools": len(self.resource_manager.process_pools),
            "connection_pools": len(self.resource_manager.connection_pools)
        }
        
        return stats
    
    def shutdown(self):
        """关闭性能优化器"""
        if self.query_cache and self.query_cache._cleanup_task:
            self.query_cache._cleanup_task.cancel()
        
        self.resource_manager.shutdown_all()
        self.logger.info("性能优化器已关闭")


# 全局实例
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


# 缓存装饰器
def cached(ttl: int = 3600, key_func: Optional[Callable] = None):
    """缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            if not optimizer.enable_caching:
                return await func(*args, **kwargs)
            
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = optimizer.get_cached_result(cache_key, {})
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            optimizer.cache_result(cache_key, {}, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# 并行执行装饰器
def parallel_execution(max_workers: int = 4):
    """并行执行装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            # 如果参数是任务列表，并行执行
            if args and isinstance(args[0], list):
                tasks = args[0]
                return await optimizer.execute_parallel(tasks, max_workers)
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


__all__ = [
    "PerformanceOptimizer",
    "QueryCache",
    "ConnectionPool",
    "ResourceManager",
    "LRUCache",
    "CacheConfig",
    "ConnectionPoolConfig",
    "get_performance_optimizer",
    "cached",
    "parallel_execution"
]
