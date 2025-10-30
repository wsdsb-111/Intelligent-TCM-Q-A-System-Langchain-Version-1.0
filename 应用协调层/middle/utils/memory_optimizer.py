"""
内存和资源优化模块

提供内存使用监控、资源清理和回收、大批量查询处理优化等功能。
"""

import gc
import os
import psutil
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from ..utils.logging_utils import get_logger, StructuredLogger, PerformanceLogger
from ..models.data_models import FusedResult


@dataclass
class MemoryConfig:
    """内存配置"""
    max_memory_mb: int = 512  # 最大内存使用（MB）
    gc_threshold_mb: int = 400  # 垃圾回收阈值（MB）
    cleanup_interval: int = 60  # 清理间隔（秒）
    batch_size: int = 100  # 批处理大小
    max_batch_memory_mb: int = 50  # 单批最大内存（MB）
    resource_cleanup_timeout: int = 30  # 资源清理超时（秒）


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    process_memory_mb: float
    gc_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceInfo:
    """资源信息"""
    resource_id: str
    resource_type: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    cleanup_func: Optional[Callable] = None


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化内存监控器
        
        Args:
            config: 内存配置
        """
        self.config = config or MemoryConfig()
        self.logger = StructuredLogger("memory_monitor")
        self.perf_logger = PerformanceLogger("memory_monitor")
        
        # 内存历史记录
        self.memory_history: deque = deque(maxlen=100)
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 垃圾回收统计
        self._gc_counts = [0] * 3  # 0代、1代、2代
        
    def start_monitoring(self):
        """开始内存监控"""
        with self._lock:
            if not self._monitoring:
                self._monitoring = True
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    daemon=True
                )
                self._monitor_thread.start()
                self.logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        with self._lock:
            if self._monitoring:
                self._monitoring = False
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)
                self.logger.info("内存监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                self.memory_history.append(stats)
                
                # 检查是否需要垃圾回收
                if stats.used_memory_mb > self.config.gc_threshold_mb:
                    self._trigger_gc()
                
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                self.logger.error(f"内存监控错误: {e}")
                time.sleep(5)
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        # 系统内存信息
        memory = psutil.virtual_memory()
        
        # 进程内存信息
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 垃圾回收统计
        try:
            gc_counts = [gc.get_count(i) for i in range(3)]
        except TypeError:
            # Python 3.13+ 中 gc.get_count() 不接受参数，返回元组
            gc_count_result = gc.get_count()
            if isinstance(gc_count_result, tuple):
                gc_counts = list(gc_count_result)
            else:
                gc_counts = [gc_count_result] * 3
        
        stats = MemoryStats(
            total_memory_mb=memory.total / 1024 / 1024,
            used_memory_mb=memory.used / 1024 / 1024,
            available_memory_mb=memory.available / 1024 / 1024,
            memory_percent=memory.percent,
            process_memory_mb=process_memory,
            gc_count=sum(gc_counts)
        )
        
        return stats
    
    def _trigger_gc(self):
        """触发垃圾回收"""
        self.logger.info("触发垃圾回收")
        
        # 记录回收前的统计
        try:
            before_counts = [gc.get_count(i) for i in range(3)]
        except TypeError:
            gc_count_result = gc.get_count()
            if isinstance(gc_count_result, tuple):
                before_counts = list(gc_count_result)
            else:
                before_counts = [gc_count_result] * 3
        
        # 执行垃圾回收
        collected = gc.collect()
        
        # 记录回收后的统计
        try:
            after_counts = [gc.get_count(i) for i in range(3)]
        except TypeError:
            gc_count_result = gc.get_count()
            if isinstance(gc_count_result, tuple):
                after_counts = list(gc_count_result)
            else:
                after_counts = [gc_count_result] * 3
        
        self.logger.info(
            f"垃圾回收完成，回收对象: {collected}, "
            f"回收前: {before_counts}, 回收后: {after_counts}"
        )
        
        # 记录性能日志
        self.perf_logger.log_retrieval_performance(
            query="gc_collection",
            response_time=0.001,  # 垃圾回收很快
            result_count=collected,
            modules_used=["gc"],
            fusion_method="gc"
        )
    
    def get_memory_trend(self, minutes: int = 10) -> List[MemoryStats]:
        """获取内存趋势"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [stats for stats in self.memory_history if stats.timestamp >= cutoff_time]
    
    def is_memory_pressure(self) -> bool:
        """检查是否有内存压力"""
        stats = self.get_memory_stats()
        return (
            stats.memory_percent > 80 or
            stats.process_memory_mb > self.config.max_memory_mb
        )


class ResourceManager:
    """资源管理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化资源管理器
        
        Args:
            config: 内存配置
        """
        self.config = config or MemoryConfig()
        self.logger = StructuredLogger("resource_manager")
        
        # 资源注册表
        self.resources: Dict[str, ResourceInfo] = {}
        self.resource_refs: Dict[str, weakref.ref] = {}
        
        # 清理任务
        self._cleanup_tasks: Set[str] = set()
        self._lock = threading.RLock()
        
        # 启动清理任务
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        size_bytes: int,
        cleanup_func: Optional[Callable] = None
    ) -> None:
        """
        注册资源
        
        Args:
            resource_id: 资源ID
            resource_type: 资源类型
            size_bytes: 资源大小（字节）
            cleanup_func: 清理函数
        """
        with self._lock:
            resource_info = ResourceInfo(
                resource_id=resource_id,
                resource_type=resource_type,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                cleanup_func=cleanup_func
            )
            
            self.resources[resource_id] = resource_info
            self.logger.debug(f"注册资源: {resource_id}, 类型: {resource_type}, 大小: {size_bytes} bytes")
    
    def unregister_resource(self, resource_id: str) -> bool:
        """
        注销资源
        
        Args:
            resource_id: 资源ID
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if resource_id in self.resources:
                resource_info = self.resources.pop(resource_id)
                
                # 执行清理函数
                if resource_info.cleanup_func:
                    try:
                        resource_info.cleanup_func()
                    except Exception as e:
                        self.logger.error(f"资源清理失败 {resource_id}: {e}")
                
                self.logger.debug(f"注销资源: {resource_id}")
                return True
            
            return False
    
    def access_resource(self, resource_id: str) -> bool:
        """
        访问资源（更新访问时间）
        
        Args:
            resource_id: 资源ID
            
        Returns:
            资源是否存在
        """
        with self._lock:
            if resource_id in self.resources:
                self.resources[resource_id].last_accessed = datetime.now()
                self.resources[resource_id].access_count += 1
                return True
            
            return False
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        with self._lock:
            total_resources = len(self.resources)
            total_size = sum(r.size_bytes for r in self.resources.values())
            
            # 按类型统计
            type_stats = defaultdict(lambda: {"count": 0, "size": 0})
            for resource in self.resources.values():
                type_stats[resource.resource_type]["count"] += 1
                type_stats[resource.resource_type]["size"] += resource.size_bytes
            
            return {
                "total_resources": total_resources,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / 1024 / 1024,
                "type_stats": dict(type_stats),
                "oldest_resource": min(
                    (r.created_at for r in self.resources.values()),
                    default=None
                ),
                "newest_resource": max(
                    (r.created_at for r in self.resources.values()),
                    default=None
                )
            }
    
    def cleanup_unused_resources(self, max_age_minutes: int = 30) -> int:
        """
        清理未使用的资源
        
        Args:
            max_age_minutes: 最大年龄（分钟）
            
        Returns:
            清理的资源数量
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            resources_to_cleanup = []
            
            for resource_id, resource_info in self.resources.items():
                if resource_info.last_accessed < cutoff_time:
                    resources_to_cleanup.append(resource_id)
            
            cleaned_count = 0
            for resource_id in resources_to_cleanup:
                if self.unregister_resource(resource_id):
                    cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"清理了 {cleaned_count} 个未使用的资源")
            
            return cleaned_count
    
    def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                # 定期清理未使用的资源
                self.cleanup_unused_resources()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                self.logger.error(f"资源清理循环错误: {e}")
                time.sleep(5)


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化批处理器
        
        Args:
            config: 内存配置
        """
        self.config = config or MemoryConfig()
        self.logger = StructuredLogger("batch_processor")
        
        # 批处理统计
        self.batch_stats = {
            "total_batches": 0,
            "total_items": 0,
            "total_memory_mb": 0.0,
            "avg_batch_size": 0.0,
            "avg_memory_per_batch": 0.0
        }
        
        self._lock = threading.RLock()
    
    def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        分批处理项目
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            batch_size: 批大小
            
        Returns:
            处理结果列表
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        with self._lock:
            self.batch_stats["total_batches"] += 1
            self.batch_stats["total_items"] += len(items)
        
        # 分批处理
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # 检查内存使用
            if self._check_batch_memory(batch):
                # 处理批次
                try:
                    batch_result = process_func(batch)
                    if isinstance(batch_result, list):
                        results.extend(batch_result)
                    else:
                        results.append(batch_result)
                    
                    # 更新统计
                    with self._lock:
                        batch_memory = self._estimate_batch_memory(batch)
                        self.batch_stats["total_memory_mb"] += batch_memory
                        self.batch_stats["avg_batch_size"] = (
                            self.batch_stats["total_items"] / self.batch_stats["total_batches"]
                        )
                        self.batch_stats["avg_memory_per_batch"] = (
                            self.batch_stats["total_memory_mb"] / self.batch_stats["total_batches"]
                        )
                    
                    self.logger.debug(f"处理批次 {i//batch_size + 1}, 大小: {len(batch)}")
                    
                except Exception as e:
                    self.logger.error(f"批次处理失败: {e}")
                    # 继续处理下一批次
                    continue
            else:
                self.logger.warning(f"批次内存使用过高，跳过批次 {i//batch_size + 1}")
        
        return results
    
    async def process_in_batches_async(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], Any],
        batch_size: Optional[int] = None,
        max_concurrent: int = 3
    ) -> List[Any]:
        """
        异步分批处理项目
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数
            batch_size: 批大小
            max_concurrent: 最大并发数
            
        Returns:
            处理结果列表
        """
        batch_size = batch_size or self.config.batch_size
        
        # 创建批次
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # 使用信号量限制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                # 在线程池中执行处理函数
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_func, batch)
        
        # 并发处理批次
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"批次处理异常: {batch_result}")
                continue
            
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    def _check_batch_memory(self, batch: List[Any]) -> bool:
        """检查批次内存使用"""
        estimated_memory = self._estimate_batch_memory(batch)
        return estimated_memory <= self.config.max_batch_memory_mb
    
    def _estimate_batch_memory(self, batch: List[Any]) -> float:
        """估算批次内存使用（MB）"""
        try:
            import sys
            total_size = sum(sys.getsizeof(item) for item in batch)
            return total_size / 1024 / 1024
        except:
            # 如果无法计算，使用默认值
            return len(batch) * 0.001  # 假设每个项目1KB
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """获取批处理统计信息"""
        with self._lock:
            return self.batch_stats.copy()


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        初始化内存优化器
        
        Args:
            config: 内存配置
        """
        self.config = config or MemoryConfig()
        self.logger = StructuredLogger("memory_optimizer")
        
        # 组件
        self.memory_monitor = MemoryMonitor(self.config)
        self.resource_manager = ResourceManager(self.config)
        self.batch_processor = BatchProcessor(self.config)
        
        # 优化状态
        self._optimization_enabled = True
        self._lock = threading.RLock()
        
        # 启动监控
        self.memory_monitor.start_monitoring()
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        优化内存使用
        
        Returns:
            优化结果
        """
        with self._lock:
            if not self._optimization_enabled:
                return {"status": "disabled"}
            
            optimization_results = {
                "timestamp": datetime.now(),
                "before_stats": self.memory_monitor.get_memory_stats(),
                "actions_taken": []
            }
            
            # 1. 清理未使用的资源
            cleaned_resources = self.resource_manager.cleanup_unused_resources()
            if cleaned_resources > 0:
                optimization_results["actions_taken"].append(f"清理了 {cleaned_resources} 个未使用的资源")
            
            # 2. 触发垃圾回收
            if self.memory_monitor.is_memory_pressure():
                self.memory_monitor._trigger_gc()
                optimization_results["actions_taken"].append("触发垃圾回收")
            
            # 3. 记录优化后状态
            optimization_results["after_stats"] = self.memory_monitor.get_memory_stats()
            optimization_results["status"] = "completed"
            
            self.logger.info(f"内存优化完成，执行了 {len(optimization_results['actions_taken'])} 个操作")
            
            return optimization_results
    
    def process_large_query_batch(
        self,
        queries: List[str],
        process_func: Callable[[List[str]], List[FusedResult]]
    ) -> List[FusedResult]:
        """
        处理大批量查询
        
        Args:
            queries: 查询列表
            process_func: 处理函数
            
        Returns:
            处理结果列表
        """
        self.logger.info(f"开始处理大批量查询，数量: {len(queries)}")
        
        # 使用批处理器
        results = self.batch_processor.process_in_batches(
            queries,
            process_func,
            self.config.batch_size
        )
        
        # 优化内存使用
        self.optimize_memory_usage()
        
        self.logger.info(f"大批量查询处理完成，结果数量: {len(results)}")
        return results
    
    async def process_large_query_batch_async(
        self,
        queries: List[str],
        process_func: Callable[[List[str]], List[FusedResult]],
        max_concurrent: int = 3
    ) -> List[FusedResult]:
        """
        异步处理大批量查询
        
        Args:
            queries: 查询列表
            process_func: 处理函数
            max_concurrent: 最大并发数
            
        Returns:
            处理结果列表
        """
        self.logger.info(f"开始异步处理大批量查询，数量: {len(queries)}")
        
        # 使用异步批处理器
        results = await self.batch_processor.process_in_batches_async(
            queries,
            process_func,
            self.config.batch_size,
            max_concurrent
        )
        
        # 优化内存使用
        self.optimize_memory_usage()
        
        self.logger.info(f"异步大批量查询处理完成，结果数量: {len(results)}")
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            "memory_stats": self.memory_monitor.get_memory_stats(),
            "resource_stats": self.resource_manager.get_resource_stats(),
            "batch_stats": self.batch_processor.get_batch_stats(),
            "memory_trend": self.memory_monitor.get_memory_trend(),
            "optimization_enabled": self._optimization_enabled
        }
    
    def enable_optimization(self):
        """启用优化"""
        with self._lock:
            self._optimization_enabled = True
            self.logger.info("内存优化已启用")
    
    def disable_optimization(self):
        """禁用优化"""
        with self._lock:
            self._optimization_enabled = False
            self.logger.info("内存优化已禁用")
    
    def shutdown(self):
        """关闭优化器"""
        self.memory_monitor.stop_monitoring()
        self.logger.info("内存优化器已关闭")


# 全局实例
_global_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer(config: Optional[MemoryConfig] = None) -> MemoryOptimizer:
    """
    获取全局内存优化器实例
    
    Args:
        config: 内存配置
        
    Returns:
        内存优化器实例
    """
    global _global_memory_optimizer
    
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer(config)
    
    return _global_memory_optimizer


def cleanup_global_memory_optimizer():
    """清理全局内存优化器"""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is not None:
        _global_memory_optimizer.shutdown()
        _global_memory_optimizer = None
