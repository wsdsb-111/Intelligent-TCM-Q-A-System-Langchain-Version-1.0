"""
监控API路由
提供系统监控、性能指标和健康检查的API端点
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import asdict

try:
    from fastapi import APIRouter, Depends, Query, HTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    class APIRouter:
        def __init__(self, **kwargs):
            self.routes = []
        
        def get(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append((path, func))
                return func
            return decorator
        
        def post(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append((path, func))
                return func
            return decorator
    
    def Depends(func):
        return func
    
    def Query(default=None, **kwargs):
        return default
    
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    
    FASTAPI_AVAILABLE = False

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from middle.utils.logging_utils import (
    get_metrics_collector, 
    get_alert_manager, 
    PerformanceMonitor,
    get_logger
)
from middle.core.retrieval_coordinator import HybridRetrievalCoordinator

# 创建路由器
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])
logger = get_logger(__name__)

# 全局监控器实例
_monitor = None
_coordinator = None


def get_monitor() -> PerformanceMonitor:
    """获取性能监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor("hybrid_retrieval_system")
        # 设置指标收集器
        collector = get_metrics_collector()
        _monitor.set_metrics_collector(collector)
    return _monitor


def get_coordinator() -> Optional[HybridRetrievalCoordinator]:
    """获取检索协调器实例"""
    global _coordinator
    if _coordinator is None:
        try:
            _coordinator = HybridRetrievalCoordinator()
        except Exception as e:
            logger.warning(f"无法创建检索协调器: {e}")
    return _coordinator


@router.get("/health")
async def health_check():
    """
    系统健康检查
    检查各个组件的健康状态
    """
    try:
        start_time = time.time()
        
        # 获取协调器
        coordinator = get_coordinator()
        
        health_status = {
            "overall_healthy": True,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # 检查检索协调器
        if coordinator:
            try:
                coord_health = await coordinator.health_check()
                health_status["components"]["retrieval_coordinator"] = {
                    "healthy": all(coord_health.values()),
                    "modules": coord_health,
                    "status": "healthy" if all(coord_health.values()) else "degraded"
                }
            except Exception as e:
                health_status["components"]["retrieval_coordinator"] = {
                    "healthy": False,
                    "error": str(e),
                    "status": "unhealthy"
                }
                health_status["overall_healthy"] = False
        else:
            health_status["components"]["retrieval_coordinator"] = {
                "healthy": False,
                "error": "协调器未初始化",
                "status": "unhealthy"
            }
            health_status["overall_healthy"] = False
        
        # 检查指标收集器
        try:
            collector = get_metrics_collector()
            current_metrics = collector.get_current_metrics()
            health_status["components"]["metrics_collector"] = {
                "healthy": True,
                "status": "healthy",
                "metrics_count": current_metrics.get("metrics_count", 0)
            }
        except Exception as e:
            health_status["components"]["metrics_collector"] = {
                "healthy": False,
                "error": str(e),
                "status": "unhealthy"
            }
            health_status["overall_healthy"] = False
        
        # 检查系统资源
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 定义健康阈值
            cpu_healthy = cpu_percent < 90
            memory_healthy = memory.percent < 90
            
            health_status["components"]["system_resources"] = {
                "healthy": cpu_healthy and memory_healthy,
                "status": "healthy" if (cpu_healthy and memory_healthy) else "degraded",
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3)
            }
            
            if not (cpu_healthy and memory_healthy):
                health_status["overall_healthy"] = False
                
        except Exception as e:
            health_status["components"]["system_resources"] = {
                "healthy": False,
                "error": str(e),
                "status": "unhealthy"
            }
            health_status["overall_healthy"] = False
        
        # 记录健康检查性能
        check_time = time.time() - start_time
        monitor = get_monitor()
        monitor.record_metric("health_check.duration", check_time)
        
        return {
            "success": True,
            "data": health_status,
            "response_time": check_time
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/metrics/current")
async def get_current_metrics():
    """
    获取当前系统指标
    """
    try:
        collector = get_metrics_collector()
        current_metrics = collector.get_current_metrics()
        
        return {
            "success": True,
            "data": current_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取当前指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")


@router.get("/metrics/history")
async def get_metrics_history(
    metric_name: Optional[str] = Query(None, description="指标名称"),
    limit: int = Query(100, description="返回数量限制"),
    hours: int = Query(24, description="时间范围（小时）")
):
    """
    获取指标历史数据
    """
    try:
        collector = get_metrics_collector()
        
        if metric_name:
            # 获取特定指标历史
            history = collector.get_metric_history(metric_name, limit)
            data = {
                "metric_name": metric_name,
                "history": [asdict(metric) for metric in history]
            }
        else:
            # 获取系统指标历史
            history = collector.get_system_metrics_history(limit)
            data = {
                "metric_type": "system",
                "history": [asdict(metric) for metric in history]
            }
        
        return {
            "success": True,
            "data": data,
            "limit": limit,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取指标历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标历史失败: {str(e)}")


@router.get("/performance/stats")
async def get_performance_stats(
    operation: Optional[str] = Query(None, description="操作名称")
):
    """
    获取性能统计信息
    """
    try:
        monitor = get_monitor()
        stats = monitor.get_operation_stats(operation)
        
        return {
            "success": True,
            "data": {
                "operation": operation,
                "stats": stats
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


@router.post("/performance/reset")
async def reset_performance_stats():
    """
    重置性能统计信息
    """
    try:
        monitor = get_monitor()
        monitor.reset_stats()
        
        return {
            "success": True,
            "message": "性能统计信息已重置",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"重置性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"重置性能统计失败: {str(e)}")


@router.get("/alerts/rules")
async def get_alert_rules():
    """
    获取告警规则
    """
    try:
        alert_manager = get_alert_manager()
        
        return {
            "success": True,
            "data": {
                "rules": alert_manager.alert_rules,
                "handlers_count": len(alert_manager.alert_handlers)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警规则失败: {str(e)}")


@router.post("/alerts/rules")
async def add_alert_rule(
    metric_name: str,
    threshold: float,
    comparison: str = "greater",
    window_size: int = 5
):
    """
    添加告警规则
    """
    try:
        alert_manager = get_alert_manager()
        alert_manager.add_alert_rule(metric_name, threshold, comparison, window_size)
        
        return {
            "success": True,
            "message": f"告警规则已添加: {metric_name} {comparison} {threshold}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"添加告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加告警规则失败: {str(e)}")


@router.get("/system/info")
async def get_system_info():
    """
    获取系统信息
    """
    try:
        import platform
        import psutil
        
        # 系统基本信息
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # 硬件信息
        hardware_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage('/').total / (1024**3) if hasattr(psutil, 'disk_usage') else 0
        }
        
        # 进程信息
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "memory_usage_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        return {
            "success": True,
            "data": {
                "system": system_info,
                "hardware": hardware_info,
                "process": process_info
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {str(e)}")


@router.get("/logs/recent")
async def get_recent_logs(
    level: Optional[str] = Query("INFO", description="日志级别"),
    limit: int = Query(100, description="返回数量限制"),
    hours: int = Query(24, description="时间范围（小时）")
):
    """
    获取最近的日志记录
    """
    try:
        # 这里应该从日志文件或日志存储中读取
        # 由于我们没有实现日志存储，这里返回模拟数据
        logs = []
        
        # 模拟日志数据
        for i in range(min(limit, 10)):
            logs.append({
                "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                "level": level,
                "message": f"模拟日志消息 {i+1}",
                "module": "test_module",
                "function": "test_function"
            })
        
        return {
            "success": True,
            "data": {
                "logs": logs,
                "total_count": len(logs),
                "level_filter": level,
                "time_range_hours": hours
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取日志记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取日志记录失败: {str(e)}")


@router.get("/status")
async def get_system_status():
    """
    获取系统整体状态
    """
    try:
        # 获取健康状态
        health_response = await health_check()
        health_data = health_response.get("data", {})
        
        # 获取当前指标
        metrics_response = await get_current_metrics()
        metrics_data = metrics_response.get("data", {})
        
        # 获取性能统计
        stats_response = await get_performance_stats()
        stats_data = stats_response.get("data", {})
        
        # 计算系统状态
        overall_status = "healthy"
        if not health_data.get("overall_healthy", False):
            overall_status = "unhealthy"
        elif any(comp.get("status") == "degraded" for comp in health_data.get("components", {}).values()):
            overall_status = "degraded"
        
        return {
            "success": True,
            "data": {
                "overall_status": overall_status,
                "health": health_data,
                "metrics": metrics_data,
                "performance": stats_data,
                "uptime": "N/A",  # 这里应该计算实际运行时间
                "version": "1.0.0"  # 这里应该从配置中获取版本信息
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


# 监控装饰器
def monitor_endpoint(operation_name: str):
    """监控API端点的装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = e
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 记录性能指标
                monitor.record_metric(f"api.{operation_name}.duration", execution_time)
                monitor.record_metric(f"api.{operation_name}.success", 1 if success else 0)
                
                # 记录日志
                logger.info(
                    f"API端点性能 - {operation_name}: "
                    f"{'成功' if success else '失败'}, "
                    f"耗时: {execution_time:.3f}s",
                    operation=operation_name,
                    execution_time=execution_time,
                    success=success,
                    error_type=error.__class__.__name__ if error else None
                )
        
        return wrapper
    return decorator


# 应用监控装饰器到所有端点
for route in router.routes:
    if hasattr(route, 'endpoint'):
        route.endpoint = monitor_endpoint(route.path)(route.endpoint)
