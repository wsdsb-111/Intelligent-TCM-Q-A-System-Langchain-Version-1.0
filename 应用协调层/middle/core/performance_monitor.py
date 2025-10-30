"""
性能监控工具
提供系统性能监控、分析和报告功能
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """性能告警"""
    level: str  # 'warning', 'critical'
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: int = 30):
        """
        初始化性能监控器
        
        Args:
            monitoring_interval: 监控间隔(秒)
        """
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        
        # 性能数据存储
        self.metrics_history = []
        self.alerts_history = []
        
        # 告警阈值 - 针对大型AI系统优化
        self.thresholds = {
            'cpu_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_percent': {'warning': 80.0, 'critical': 95.0},
            'memory_rss_mb': {'warning': 1500.0, 'critical': 2500.0},  # 提高阈值适应大型系统
            'disk_usage_percent': {'warning': 85.0, 'critical': 95.0}
        }
        
        # 自定义监控器
        self.custom_monitors = {}
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"性能监控已启动，间隔: {self.monitoring_interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("性能监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统性能指标
                self._collect_system_metrics()
                
                # 收集自定义指标
                self._collect_custom_metrics()
                
                # 检查告警
                self._check_alerts()
                
                # 清理历史数据
                self._cleanup_history()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """收集系统性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric('cpu_percent', cpu_percent, '%')
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            self._add_metric('memory_percent', memory.percent, '%')
            self._add_metric('memory_available_mb', memory.available / 1024 / 1024, 'MB')
            
            # 进程内存使用
            process = psutil.Process()
            memory_info = process.memory_info()
            self._add_metric('memory_rss_mb', memory_info.rss / 1024 / 1024, 'MB')
            self._add_metric('memory_vms_mb', memory_info.vms / 1024 / 1024, 'MB')
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            self._add_metric('disk_usage_percent', disk.percent, '%')
            self._add_metric('disk_free_gb', disk.free / 1024 / 1024 / 1024, 'GB')
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self._add_metric('network_bytes_sent', net_io.bytes_sent, 'bytes')
            self._add_metric('network_bytes_recv', net_io.bytes_recv, 'bytes')
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {e}")
    
    def _collect_custom_metrics(self):
        """收集自定义指标"""
        for name, monitor_func in self.custom_monitors.items():
            try:
                value = monitor_func()
                self._add_metric(name, value, '')
            except Exception as e:
                self.logger.error(f"收集自定义指标 {name} 失败: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """添加性能指标"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics_history.append(metric)
    
    def _check_alerts(self):
        """检查告警"""
        if not self.metrics_history:
            return
        
        # 获取最新的指标值
        latest_metrics = {}
        for metric in reversed(self.metrics_history):
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = metric
        
        # 检查每个指标的阈值
        for metric_name, thresholds in self.thresholds.items():
            if metric_name in latest_metrics:
                metric = latest_metrics[metric_name]
                
                # 检查严重告警
                if metric.value >= thresholds['critical']:
                    self._create_alert('critical', 
                                     f"{metric_name} 严重告警: {metric.value:.1f}{metric.unit} >= {thresholds['critical']}",
                                     metric_name, thresholds['critical'], metric.value)
                
                # 检查警告
                elif metric.value >= thresholds['warning']:
                    self._create_alert('warning',
                                     f"{metric_name} 警告: {metric.value:.1f}{metric.unit} >= {thresholds['warning']}",
                                     metric_name, thresholds['warning'], metric.value)
    
    def _create_alert(self, level: str, message: str, metric_name: str, threshold: float, current_value: float):
        """创建告警"""
        alert = PerformanceAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now()
        )
        
        self.alerts_history.append(alert)
        
        # 记录日志
        if level == 'critical':
            self.logger.critical(message)
        else:
            self.logger.warning(message)
    
    def _cleanup_history(self):
        """清理历史数据"""
        # 只保留最近24小时的数据
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        self.alerts_history = [a for a in self.alerts_history if a.timestamp > cutoff_time]
    
    def add_custom_monitor(self, name: str, monitor_func: Callable[[], float]):
        """添加自定义监控器"""
        self.custom_monitors[name] = monitor_func
        self.logger.info(f"添加自定义监控器: {name}")
    
    def remove_custom_monitor(self, name: str):
        """移除自定义监控器"""
        if name in self.custom_monitors:
            del self.custom_monitors[name]
            self.logger.info(f"移除自定义监控器: {name}")
    
    def get_latest_metrics(self) -> Dict[str, PerformanceMetric]:
        """获取最新指标"""
        latest_metrics = {}
        for metric in reversed(self.metrics_history):
            if metric.name not in latest_metrics:
                latest_metrics[metric.name] = metric
        return latest_metrics
    
    def get_metrics_history(self, metric_name: str, hours: int = 1) -> List[PerformanceMetric]:
        """获取指标历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history 
                if m.name == metric_name and m.timestamp > cutoff_time]
    
    def get_alerts(self, level: str = None, hours: int = 24) -> List[PerformanceAlert]:
        """获取告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts_history if a.timestamp > cutoff_time]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        latest_metrics = self.get_latest_metrics()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy',
            'latest_metrics': {},
            'recent_alerts': len(self.get_alerts(hours=1)),
            'critical_alerts': len(self.get_alerts(level='critical', hours=1))
        }
        
        # 添加最新指标
        for name, metric in latest_metrics.items():
            summary['latest_metrics'][name] = {
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat()
            }
        
        # 判断系统状态
        critical_alerts = self.get_alerts(level='critical', hours=1)
        if critical_alerts:
            summary['system_status'] = 'critical'
        elif self.get_alerts(level='warning', hours=1):
            summary['system_status'] = 'warning'
        
        return summary
    
    def export_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """导出指标数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        metrics_data = []
        for metric in self.metrics_history:
            if metric.timestamp > cutoff_time:
                metrics_data.append({
                    'name': metric.name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'tags': metric.tags
                })
        
        alerts_data = []
        for alert in self.alerts_history:
            if alert.timestamp > cutoff_time:
                alerts_data.append({
                    'level': alert.level,
                    'message': alert.message,
                    'metric_name': alert.metric_name,
                    'threshold': alert.threshold,
                    'current_value': alert.current_value,
                    'timestamp': alert.timestamp.isoformat()
                })
        
        return {
            'export_time': datetime.now().isoformat(),
            'time_range_hours': hours,
            'metrics': metrics_data,
            'alerts': alerts_data
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
    
    def profile_function(self, func_name: str):
        """函数性能分析装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_profile(func_name, execution_time, memory_delta)
            
            return wrapper
        return decorator
    
    def _record_profile(self, func_name: str, execution_time: float, memory_delta: int):
        """记录性能分析数据"""
        if func_name not in self.profiles:
            self.profiles[func_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'total_memory_delta': 0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'last_call': None
            }
        
        profile = self.profiles[func_name]
        profile['call_count'] += 1
        profile['total_time'] += execution_time
        profile['total_memory_delta'] += memory_delta
        profile['min_time'] = min(profile['min_time'], execution_time)
        profile['max_time'] = max(profile['max_time'], execution_time)
        profile['last_call'] = datetime.now()
        
        self.logger.debug(f"性能分析 - {func_name}: {execution_time:.3f}秒, 内存变化: {memory_delta/1024/1024:.1f}MB")
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """获取性能分析摘要"""
        summary = {}
        
        for func_name, profile in self.profiles.items():
            if profile['call_count'] > 0:
                summary[func_name] = {
                    'call_count': profile['call_count'],
                    'avg_time': profile['total_time'] / profile['call_count'],
                    'total_time': profile['total_time'],
                    'min_time': profile['min_time'],
                    'max_time': profile['max_time'],
                    'avg_memory_delta_mb': profile['total_memory_delta'] / profile['call_count'] / 1024 / 1024,
                    'last_call': profile['last_call'].isoformat() if profile['last_call'] else None
                }
        
        return summary
    
    def reset_profiles(self):
        """重置性能分析数据"""
        self.profiles.clear()
        self.logger.info("性能分析数据已重置")


# 全局实例
_global_monitor = None
_global_profiler = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def get_performance_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def start_performance_monitoring():
    """启动性能监控"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """停止性能监控"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


def profile_function(func_name: str):
    """性能分析装饰器"""
    profiler = get_performance_profiler()
    return profiler.profile_function(func_name)
