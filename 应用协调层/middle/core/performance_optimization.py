"""
性能优化模块
包含缓存机制、内存监控和性能调优功能
"""

import time
import hashlib
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass
import logging

from middle.models.data_models import RetrievalResult, FusedResult
# 评分策略已移除，使用简单的评分方法
class SimpleScoringStrategy:
    """简单的评分策略"""
    def calculate_score(self, query: str, result: dict) -> float:
        """计算简单评分"""
        return result.get('score', 0.5)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_access: datetime = None
    
    def __post_init__(self):
        if self.last_access is None:
            self.last_access = self.timestamp


class LRUCache:
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存生存时间(秒)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存条目是否过期"""
        return datetime.now() - entry.timestamp > timedelta(seconds=self.ttl_seconds)
    
    def _cleanup_expired(self):
        """清理过期的缓存条目"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.logger.debug(f"清理过期缓存: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    # 更新访问信息
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    # 移动到末尾（最近使用）
                    self.cache.move_to_end(key)
                    return entry.value
                else:
                    # 过期则删除
                    del self.cache[key]
            
            return None
    
    def put(self, key: str, value: Any) -> None:
        """存储缓存值"""
        with self.lock:
            self._cleanup_expired()
            
            # 如果缓存已满，删除最久未使用的条目
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.logger.debug(f"缓存已满，删除最久未使用条目: {oldest_key}")
            
            # 创建新的缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now()
            )
            
            self.cache[key] = entry
            self.logger.debug(f"缓存存储: {key}")
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.logger.info("缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            self._cleanup_expired()
            
            if not self.cache:
                return {
                    'size': 0,
                    'max_size': self.max_size,
                    'hit_rate': 0.0,
                    'total_access': 0
                }
            
            total_access = sum(entry.access_count for entry in self.cache.values())
            hit_rate = total_access / len(self.cache) if self.cache else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_access': total_access,
                'oldest_entry': min(entry.timestamp for entry in self.cache.values()).isoformat(),
                'newest_entry': max(entry.timestamp for entry in self.cache.values()).isoformat()
            }


class ScoringCache:
    """评分计算缓存"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 1800):
        """
        初始化评分缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存生存时间(秒)
        """
        self.cache = LRUCache(max_size, ttl_seconds)
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, query: str, content: str, strategy_type: str, metadata_hash: str) -> str:
        """生成缓存键"""
        key_data = f"{query}|{content}|{strategy_type}|{metadata_hash}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def _hash_metadata(self, metadata: Dict[str, Any]) -> str:
        """生成元数据哈希"""
        # 只对影响评分的关键字段进行哈希
        key_fields = ['avg_doc_length', 'term_frequency', 'total_documents', 
                     'similarity', 'entities', 'relationships']
        
        filtered_metadata = {k: v for k, v in metadata.items() if k in key_fields}
        metadata_str = str(sorted(filtered_metadata.items()))
        return hashlib.md5(metadata_str.encode('utf-8')).hexdigest()
    
    def get_score(self, query: str, content: str, strategy_type: str, metadata: Dict[str, Any]) -> Optional[float]:
        """获取缓存的评分"""
        metadata_hash = self._hash_metadata(metadata)
        key = self._generate_key(query, content, strategy_type, metadata_hash)
        
        cached_score = self.cache.get(key)
        if cached_score is not None:
            self.logger.debug(f"缓存命中: {strategy_type} - {query[:20]}...")
            return cached_score
        
        return None
    
    def put_score(self, query: str, content: str, strategy_type: str, metadata: Dict[str, Any], score: float) -> None:
        """存储评分到缓存"""
        metadata_hash = self._hash_metadata(metadata)
        key = self._generate_key(query, content, strategy_type, metadata_hash)
        
        self.cache.put(key, score)
        self.logger.debug(f"缓存存储: {strategy_type} - {query[:20]}... -> {score:.4f}")
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache.get_stats()


class MemoryMonitor:
    """内存使用监控"""
    
    def __init__(self, warning_threshold_mb: int = 500, critical_threshold_mb: int = 1000):
        """
        初始化内存监控
        
        Args:
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 严重阈值(MB)
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # 转换为字节
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    
    def check_memory_status(self) -> str:
        """检查内存状态"""
        memory_usage = self.get_memory_usage()
        rss_bytes = memory_usage['rss_mb'] * 1024 * 1024
        
        if rss_bytes >= self.critical_threshold:
            return 'critical'
        elif rss_bytes >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def start_monitoring(self, interval_seconds: int = 30):
        """开始内存监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"内存监控已启动，间隔: {interval_seconds}秒")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("内存监控已停止")
    
    def _monitor_loop(self, interval_seconds: int):
        """监控循环"""
        while self.monitoring:
            try:
                memory_usage = self.get_memory_usage()
                status = self.check_memory_status()
                
                # 记录内存历史
                self.memory_history.append({
                    'timestamp': datetime.now(),
                    'rss_mb': memory_usage['rss_mb'],
                    'status': status
                })
                
                # 只保留最近100条记录
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # 根据状态记录日志
                if status == 'critical':
                    self.logger.critical(f"内存使用严重: {memory_usage['rss_mb']:.1f}MB")
                elif status == 'warning':
                    self.logger.warning(f"内存使用警告: {memory_usage['rss_mb']:.1f}MB")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"内存监控错误: {e}")
                time.sleep(interval_seconds)
    
    def get_memory_history(self) -> List[Dict[str, Any]]:
        """获取内存使用历史"""
        return self.memory_history.copy()
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """强制垃圾回收"""
        before_memory = self.get_memory_usage()
        
        # 执行垃圾回收
        collected = gc.collect()
        
        after_memory = self.get_memory_usage()
        
        memory_freed = before_memory['rss_mb'] - after_memory['rss_mb']
        
        self.logger.info(f"垃圾回收完成: 回收对象 {collected} 个, 释放内存 {memory_freed:.1f}MB")
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': memory_freed,
            'before_memory_mb': before_memory['rss_mb'],
            'after_memory_mb': after_memory['rss_mb']
        }


class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, enable_caching: bool = True, enable_monitoring: bool = True):
        """
        初始化性能优化器
        
        Args:
            enable_caching: 是否启用缓存
            enable_monitoring: 是否启用监控
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化缓存
        if enable_caching:
            self.scoring_cache = ScoringCache()
            self.logger.info("评分缓存已启用")
        else:
            self.scoring_cache = None
        
        # 初始化内存监控
        if enable_monitoring:
            self.memory_monitor = MemoryMonitor()
            self.memory_monitor.start_monitoring()
            self.logger.info("内存监控已启用")
        else:
            self.memory_monitor = None
        
        # 性能统计
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'total_time': 0.0,
            'memory_cleanups': 0
        }
    
    def optimize_scoring_strategy(self, strategy: ScoringStrategy) -> ScoringStrategy:
        """优化评分策略，添加缓存支持"""
        if not self.scoring_cache:
            return strategy
        
        original_calculate_score = strategy.calculate_score
        
        def cached_calculate_score(query: str, content: str, metadata: Dict[str, Any]) -> float:
            # 尝试从缓存获取
            cached_score = self.scoring_cache.get_score(
                query, content, strategy.__class__.__name__, metadata
            )
            
            if cached_score is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_score
            
            # 缓存未命中，计算评分
            self.performance_stats['cache_misses'] += 1
            score = original_calculate_score(query, content, metadata)
            
            # 存储到缓存
            self.scoring_cache.put_score(
                query, content, strategy.__class__.__name__, metadata, score
            )
            
            return score
        
        # 替换方法
        strategy.calculate_score = cached_calculate_score
        self.logger.info(f"已为 {strategy.__class__.__name__} 添加缓存支持")
        
        return strategy
    
    def optimize_rrf_calculation(self, results_by_source: Dict[str, List[RetrievalResult]], 
                                top_k: int) -> List[FusedResult]:
        """优化RRF计算"""
        start_time = time.time()
        
        # 检查内存状态
        if self.memory_monitor:
            status = self.memory_monitor.check_memory_status()
            if status == 'critical':
                self.logger.warning("内存使用严重，执行垃圾回收")
                gc_result = self.memory_monitor.force_garbage_collection()
                self.performance_stats['memory_cleanups'] += 1
        
        # 执行RRF计算（这里可以添加更多优化逻辑）
        # 例如：预排序、批量处理等
        
        execution_time = time.time() - start_time
        self.performance_stats['total_time'] += execution_time
        self.performance_stats['total_queries'] += 1
        
        return []  # 这里应该返回实际的RRF计算结果
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'performance_stats': self.performance_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 添加缓存统计
        if self.scoring_cache:
            report['cache_stats'] = self.scoring_cache.get_stats()
        
        # 添加内存统计
        if self.memory_monitor:
            report['memory_usage'] = self.memory_monitor.get_memory_usage()
            report['memory_status'] = self.memory_monitor.check_memory_status()
            report['memory_history'] = self.memory_monitor.get_memory_history()[-10:]  # 最近10条
        
        # 计算性能指标
        if self.performance_stats['total_queries'] > 0:
            report['avg_query_time'] = self.performance_stats['total_time'] / self.performance_stats['total_queries']
            report['queries_per_second'] = self.performance_stats['total_queries'] / max(self.performance_stats['total_time'], 0.001)
        
        if self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'] > 0:
            report['cache_hit_rate'] = self.performance_stats['cache_hits'] / (
                self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            )
        
        return report
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self.scoring_cache:
            self.scoring_cache.clear()
            self.logger.info("缓存已清空")
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'total_time': 0.0,
            'memory_cleanups': 0
        }
        self.logger.info("性能统计已重置")
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
        
        if self.scoring_cache:
            self.scoring_cache.clear()
        
        self.logger.info("性能优化器已清理")


# 全局性能优化器实例
_global_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize_scoring_strategy(strategy: ScoringStrategy) -> ScoringStrategy:
    """优化评分策略的便捷函数"""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_scoring_strategy(strategy)


def get_performance_report() -> Dict[str, Any]:
    """获取性能报告的便捷函数"""
    optimizer = get_performance_optimizer()
    return optimizer.get_performance_report()


def clear_performance_cache() -> None:
    """清空性能缓存的便捷函数"""
    optimizer = get_performance_optimizer()
    optimizer.clear_cache()
