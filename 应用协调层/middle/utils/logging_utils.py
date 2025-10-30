"""
日志工具函数
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict, Any
# 默认日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


def setup_logging(config: Optional[Dict[str, Any]] = None, log_file_path: Optional[str] = None):
    """
    设置日志配置
    
    Args:
        config: 日志配置字典，如果为None则使用默认配置
        log_file_path: 日志文件路径
    """
    if config is None:
        config = LOGGING_CONFIG.copy()
    
    # 如果指定了日志文件路径，更新配置
    if log_file_path:
        # 确保日志目录存在
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if 'handlers' in config and 'file' in config['handlers']:
            config['handlers']['file']['filename'] = log_file_path
        if 'handlers' in config and 'json_file' in config['handlers']:
            json_file_path = str(Path(log_file_path).with_suffix('.json'))
            config['handlers']['json_file']['filename'] = json_file_path
    
    # 确保日志目录存在
    for handler_name, handler_config in config.get('handlers', {}).items():
        if 'filename' in handler_config:
            Path(handler_config['filename']).parent.mkdir(parents=True, exist_ok=True)
    
    # 应用日志配置
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """记录错误日志"""
        extra = {"structured_data": kwargs} if kwargs else {}
        if error:
            extra["structured_data"] = extra.get("structured_data", {})
            extra["structured_data"]["error_type"] = error.__class__.__name__
            extra["structured_data"]["error_message"] = str(error)
        
        self.logger.error(message, extra=extra, exc_info=error is not None)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        extra = {"structured_data": kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"{name}.performance")
    
    def log_retrieval_performance(self, 
                                query: str,
                                response_time: float,
                                result_count: int,
                                modules_used: list,
                                fusion_method: str):
        """记录检索性能日志"""
        self.logger.info(
            f"检索性能 - 查询: {query[:50]}..., "
            f"响应时间: {response_time:.3f}s, "
            f"结果数量: {result_count}, "
            f"使用模块: {modules_used}, "
            f"融合方法: {fusion_method}",
            extra={
                "structured_data": {
                    "query_length": len(query),
                    "response_time": response_time,
                    "result_count": result_count,
                    "modules_used": modules_used,
                    "fusion_method": fusion_method
                }
            }
        )
    
    def log_module_performance(self,
                             module_name: str,
                             operation: str,
                             response_time: float,
                             success: bool,
                             error_message: Optional[str] = None):
        """记录模块性能日志"""
        status = "成功" if success else "失败"
        message = f"模块性能 - {module_name}.{operation}: {status}, 响应时间: {response_time:.3f}s"
        
        if error_message:
            message += f", 错误: {error_message}"
        
        extra = {
            "structured_data": {
                "module_name": module_name,
                "operation": operation,
                "response_time": response_time,
                "success": success,
                "error_message": error_message
            }
        }
        
        if success:
            self.logger.info(message, extra=extra)
        else:
            self.logger.error(message, extra=extra)


import time
import threading
import psutil
from datetime import datetime
from typing import List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, collection_interval: int = 60):
        """
        初始化指标收集器
        
        Args:
            collection_interval: 收集间隔（秒）
        """
        self.collection_interval = collection_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.custom_metrics: Dict[str, float] = {}
        self.callbacks: List[Callable[[PerformanceMetric], None]] = []
        
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        
        self.logger = get_logger(__name__)
    
    def start(self):
        """启动指标收集"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        self.logger.info("指标收集器已启动")
    
    def stop(self):
        """停止指标收集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.logger.info("指标收集器已停止")
    
    def _collect_loop(self):
        """指标收集循环"""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"指标收集失败: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存指标
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # 网络指标
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # 创建系统指标对象
            system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv
            )
            
            with self._lock:
                self.system_metrics_history.append(system_metrics)
            
            # 记录性能日志
            self.logger.debug(
                f"系统指标 - CPU: {cpu_percent:.1f}%, "
                f"内存: {memory_percent:.1f}%, "
                f"磁盘: {disk_usage_percent:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """记录自定义指标"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics_history[name].append(metric)
            self.custom_metrics[name] = value
        
        # 触发回调
        for callback in self.callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"指标回调执行失败: {e}")
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[PerformanceMetric]:
        """获取指标历史"""
        with self._lock:
            history = list(self.metrics_history[name])
            return history[-limit:] if limit else history
    
    def get_system_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """获取系统指标历史"""
        with self._lock:
            history = list(self.system_metrics_history)
            return history[-limit:] if limit else history
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        with self._lock:
            current_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            return {
                "system_metrics": {
                    "cpu_percent": current_system.cpu_percent if current_system else 0,
                    "memory_percent": current_system.memory_percent if current_system else 0,
                    "memory_used_gb": current_system.memory_used_gb if current_system else 0,
                    "memory_available_gb": current_system.memory_available_gb if current_system else 0,
                    "disk_usage_percent": current_system.disk_usage_percent if current_system else 0,
                    "timestamp": current_system.timestamp.isoformat() if current_system else None
                },
                "custom_metrics": self.custom_metrics.copy(),
                "collection_interval": self.collection_interval,
                "metrics_count": len(self.metrics_history),
                "system_metrics_count": len(self.system_metrics_history)
            }
    
    def add_callback(self, callback: Callable[[PerformanceMetric], None]):
        """添加指标回调"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[PerformanceMetric], None]):
        """移除指标回调"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, name: str):
        """
        初始化性能监控器
        
        Args:
            name: 监控器名称
        """
        self.name = name
        self.logger = StructuredLogger(f"{name}.performance")
        self.metrics_collector = None
        
        # 性能统计
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0
        })
        
        self._lock = threading.RLock()
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """设置指标收集器"""
        self.metrics_collector = collector
    
    def time_operation(self, operation_name: str):
        """操作计时装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error = None
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error = e
                    raise
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # 更新统计信息
                    self._update_operation_stats(operation_name, execution_time, success)
                    
                    # 记录性能日志
                    self.logger.info(
                        f"操作性能 - {operation_name}: "
                        f"{'成功' if success else '失败'}, "
                        f"耗时: {execution_time:.3f}s",
                        operation=operation_name,
                        execution_time=execution_time,
                        success=success,
                        error_type=error.__class__.__name__ if error else None
                    )
                    
                    # 记录到指标收集器
                    if self.metrics_collector:
                        self.metrics_collector.record_metric(
                            f"{self.name}.{operation_name}.execution_time",
                            execution_time,
                            {"success": str(success)}
                        )
            
            return wrapper
        return decorator
    
    def _update_operation_stats(self, operation_name: str, execution_time: float, success: bool):
        """更新操作统计信息"""
        with self._lock:
            stats = self.operation_stats[operation_name]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)
            
            if not success:
                stats["error_count"] += 1
    
    def get_operation_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """获取操作统计信息"""
        with self._lock:
            if operation_name:
                if operation_name in self.operation_stats:
                    stats = self.operation_stats[operation_name].copy()
                    if stats["count"] > 0:
                        stats["average_time"] = stats["total_time"] / stats["count"]
                        stats["success_rate"] = (stats["count"] - stats["error_count"]) / stats["count"]
                    return {operation_name: stats}
                else:
                    return {}
            else:
                result = {}
                for op_name, stats in self.operation_stats.items():
                    op_stats = stats.copy()
                    if op_stats["count"] > 0:
                        op_stats["average_time"] = op_stats["total_time"] / op_stats["count"]
                        op_stats["success_rate"] = (op_stats["count"] - op_stats["error_count"]) / op_stats["count"]
                        op_stats["min_time"] = op_stats["min_time"] if op_stats["min_time"] != float('inf') else 0
                    result[op_name] = op_stats
                return result
    
    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self.operation_stats.clear()
        self.logger.info("性能统计信息已重置")


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        """初始化告警管理器"""
        self.alert_rules = []
        self.alert_handlers = []
        self.logger = get_logger(__name__)
    
    def add_alert_rule(self, 
                      metric_name: str,
                      threshold: float,
                      comparison: str = "greater",
                      window_size: int = 5):
        """
        添加告警规则
        
        Args:
            metric_name: 指标名称
            threshold: 阈值
            comparison: 比较方式 (greater, less, equal)
            window_size: 窗口大小
        """
        rule = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "window_size": window_size,
            "triggered": False
        }
        self.alert_rules.append(rule)
        self.logger.info(f"添加告警规则: {metric_name} {comparison} {threshold}")
    
    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metric: PerformanceMetric):
        """检查告警"""
        for rule in self.alert_rules:
            if rule["metric_name"] == metric.name:
                should_trigger = False
                
                if rule["comparison"] == "greater" and metric.value > rule["threshold"]:
                    should_trigger = True
                elif rule["comparison"] == "less" and metric.value < rule["threshold"]:
                    should_trigger = True
                elif rule["comparison"] == "equal" and metric.value == rule["threshold"]:
                    should_trigger = True
                
                if should_trigger and not rule["triggered"]:
                    # 触发告警
                    alert_data = {
                        "rule": rule,
                        "metric": metric,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    for handler in self.alert_handlers:
                        try:
                            handler(f"告警触发: {metric.name}", alert_data)
                        except Exception as e:
                            self.logger.error(f"告警处理器执行失败: {e}")
                    
                    rule["triggered"] = True
                    self.logger.warning(f"告警触发: {metric.name} = {metric.value}, 阈值: {rule['threshold']}")
                
                elif not should_trigger and rule["triggered"]:
                    # 告警恢复
                    rule["triggered"] = False
                    self.logger.info(f"告警恢复: {metric.name} = {metric.value}")


# 全局实例
_metrics_collector = None
_alert_manager = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def setup_monitoring(collection_interval: int = 60, enable_alerts: bool = True):
    """
    设置监控系统
    
    Args:
        collection_interval: 指标收集间隔
        enable_alerts: 是否启用告警
    """
    # 启动指标收集器
    collector = get_metrics_collector()
    collector.collection_interval = collection_interval
    collector.start()
    
    # 设置告警
    if enable_alerts:
        alert_manager = get_alert_manager()
        
        # 添加默认告警规则
        alert_manager.add_alert_rule("cpu_percent", 80.0, "greater")
        alert_manager.add_alert_rule("memory_percent", 85.0, "greater")
        alert_manager.add_alert_rule("disk_usage_percent", 90.0, "greater")
        
        # 添加告警回调
        collector.add_callback(alert_manager.check_alerts)
        
        # 添加默认告警处理器
        def log_alert_handler(message: str, alert_data: Dict[str, Any]):
            logger = get_logger("alerts")
            logger.critical(f"{message}: {alert_data}")
        
        alert_manager.add_alert_handler(log_alert_handler)
    
    logger = get_logger(__name__)
    logger.info("监控系统已启动")


def shutdown_monitoring():
    """关闭监控系统"""
    global _metrics_collector
    if _metrics_collector:
        _metrics_collector.stop()
    
    logger = get_logger(__name__)
    logger.info("监控系统已关闭")