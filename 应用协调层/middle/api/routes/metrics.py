"""
指标和统计API路由
增强的性能监控和日志系统
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

try:
    from fastapi import APIRouter, Depends, Query
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
    
    def Query(**kwargs):
        return None
    
    FASTAPI_AVAILABLE = False

from ..models import MetricsResponse, StatisticsResponse, ErrorResponse
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from middle.integrations.hybrid_retriever import create_hybrid_retriever
from middle.utils.logging_utils import get_logger

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["metrics"])
logger = get_logger(__name__)

# 全局检索器实例
_retriever_instance = None
# 服务级别的指标存储
_service_metrics = {
    "start_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_response_time": 0.0,
    "endpoint_stats": {},
    "error_counts": {},
    "peak_memory_usage": 0,
    "peak_cpu_usage": 0
}


def get_retriever():
    """获取检索器实例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = create_hybrid_retriever()
    return _retriever_instance


def update_service_metrics(endpoint: str, response_time: float, success: bool, error_type: str = None):
    """更新服务级别指标"""
    global _service_metrics
    
    _service_metrics["total_requests"] += 1
    _service_metrics["total_response_time"] += response_time
    
    if success:
        _service_metrics["successful_requests"] += 1
    else:
        _service_metrics["failed_requests"] += 1
        if error_type:
            _service_metrics["error_counts"][error_type] = _service_metrics["error_counts"].get(error_type, 0) + 1
    
    # 端点统计
    if endpoint not in _service_metrics["endpoint_stats"]:
        _service_metrics["endpoint_stats"][endpoint] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0
        }
    
    endpoint_stats = _service_metrics["endpoint_stats"][endpoint]
    endpoint_stats["total_requests"] += 1
    endpoint_stats["total_response_time"] += response_time
    endpoint_stats["min_response_time"] = min(endpoint_stats["min_response_time"], response_time)
    endpoint_stats["max_response_time"] = max(endpoint_stats["max_response_time"], response_time)
    
    if success:
        endpoint_stats["successful_requests"] += 1
    else:
        endpoint_stats["failed_requests"] += 1


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(retriever=Depends(get_retriever)):
    """
    获取系统指标
    
    返回服务性能指标、检索指标和系统资源使用情况。
    """
    try:
        start_time = time.time()
        
        # 获取服务指标
        service_metrics = calculate_service_metrics()
        
        # 获取检索指标
        retrieval_metrics = get_retrieval_metrics(retriever)
        
        # 获取性能指标
        performance_metrics = get_current_performance_metrics()
        
        response_time = time.time() - start_time
        
        response = MetricsResponse(
            service_metrics=service_metrics,
            retrieval_metrics=retrieval_metrics,
            performance_metrics=performance_metrics
        )
        
        # 更新指标
        update_service_metrics("/metrics", response_time, True)
        
        return response
    
    except Exception as e:
        error_msg = f"获取指标失败: {str(e)}"
        logger.error(error_msg)
        
        update_service_metrics("/metrics", 0, False, "metrics_error")
        
        return MetricsResponse(
            service_metrics={"error": error_msg},
            retrieval_metrics={"error": error_msg},
            performance_metrics={"error": error_msg}
        )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    retriever=Depends(get_retriever),
    include_detailed: bool = Query(default=False, description="是否包含详细统计信息")
):
    """
    获取统计信息
    
    返回API使用统计、检索统计和融合统计信息。
    """
    try:
        start_time = time.time()
        
        # 获取API统计
        api_statistics = get_api_statistics(include_detailed)
        
        # 获取检索统计
        retrieval_statistics = retriever.get_statistics()
        
        # 获取融合统计
        fusion_statistics = retriever.coordinator.fusion_engine.get_fusion_statistics()
        
        response_time = time.time() - start_time
        
        response = StatisticsResponse(
            api_statistics=api_statistics,
            retrieval_statistics=retrieval_statistics,
            fusion_statistics=fusion_statistics
        )
        
        # 更新指标
        update_service_metrics("/statistics", response_time, True)
        
        return response
    
    except Exception as e:
        error_msg = f"获取统计信息失败: {str(e)}"
        logger.error(error_msg)
        
        update_service_metrics("/statistics", 0, False, "statistics_error")
        
        return StatisticsResponse(
            api_statistics={"error": error_msg},
            retrieval_statistics={"error": error_msg},
            fusion_statistics={"error": error_msg}
        )


@router.get("/metrics/performance")
async def get_performance_metrics():
    """
    获取实时性能指标
    
    返回当前的CPU、内存、磁盘等资源使用情况。
    """
    try:
        import psutil
        
        # CPU指标
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 内存指标
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 磁盘指标
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # 网络指标
        network_io = psutil.net_io_counters()
        
        # 进程指标
        process = psutil.Process()
        process_memory = process.memory_info()
        
        performance_data = {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency": cpu_freq._asdict() if cpu_freq else None
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            },
            "swap": {
                "total": swap.total,
                "used": swap.used,
                "free": swap.free,
                "percent": swap.percent
            },
            "disk": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent
            },
            "disk_io": disk_io._asdict() if disk_io else None,
            "network_io": network_io._asdict() if network_io else None,
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": process.create_time()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return performance_data
    
    except Exception as e:
        error_msg = f"获取性能指标失败: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "timestamp": datetime.now().isoformat()}


@router.get("/metrics/endpoints")
async def get_endpoint_metrics():
    """
    获取端点指标
    
    返回各API端点的调用统计和性能数据。
    """
    try:
        endpoint_metrics = {}
        
        for endpoint, stats in _service_metrics["endpoint_stats"].items():
            total_requests = stats["total_requests"]
            if total_requests > 0:
                avg_response_time = stats["total_response_time"] / total_requests
                success_rate = stats["successful_requests"] / total_requests
                
                endpoint_metrics[endpoint] = {
                    "total_requests": total_requests,
                    "successful_requests": stats["successful_requests"],
                    "failed_requests": stats["failed_requests"],
                    "success_rate": success_rate,
                    "average_response_time": avg_response_time,
                    "min_response_time": stats["min_response_time"] if stats["min_response_time"] != float('inf') else 0,
                    "max_response_time": stats["max_response_time"]
                }
        
        return {
            "endpoints": endpoint_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        error_msg = f"获取端点指标失败: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "timestamp": datetime.now().isoformat()}


@router.post("/metrics/reset")
async def reset_metrics():
    """
    重置指标
    
    清空所有统计数据，重新开始计数。
    """
    try:
        global _service_metrics
        
        # 重置服务指标
        _service_metrics = {
            "start_time": datetime.now(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "endpoint_stats": {},
            "error_counts": {},
            "peak_memory_usage": 0,
            "peak_cpu_usage": 0
        }
        
        # 重置检索器统计
        retriever = get_retriever()
        retriever.reset_statistics()
        
        logger.info("所有指标已重置")
        
        return {
            "success": True,
            "message": "所有指标已重置",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        error_msg = f"重置指标失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }


# 辅助函数

def calculate_service_metrics() -> Dict[str, Any]:
    """计算服务级别指标"""
    total_requests = _service_metrics["total_requests"]
    uptime = (datetime.now() - _service_metrics["start_time"]).total_seconds()
    
    metrics = {
        "uptime_seconds": uptime,
        "total_requests": total_requests,
        "successful_requests": _service_metrics["successful_requests"],
        "failed_requests": _service_metrics["failed_requests"],
        "success_rate": _service_metrics["successful_requests"] / total_requests if total_requests > 0 else 0,
        "requests_per_second": total_requests / uptime if uptime > 0 else 0,
        "average_response_time": _service_metrics["total_response_time"] / total_requests if total_requests > 0 else 0,
        "error_counts": _service_metrics["error_counts"].copy(),
        "peak_memory_usage": _service_metrics["peak_memory_usage"],
        "peak_cpu_usage": _service_metrics["peak_cpu_usage"]
    }
    
    return metrics


def get_retrieval_metrics(retriever) -> Dict[str, Any]:
    """获取检索相关指标"""
    try:
        stats = retriever.get_statistics()
        
        retrieval_stats = stats.get("retriever_stats", {})
        coordinator_stats = stats.get("coordinator_stats", {})
        
        metrics = {
            "retriever": {
                "total_queries": retrieval_stats.get("total_queries", 0),
                "successful_queries": retrieval_stats.get("successful_queries", 0),
                "failed_queries": retrieval_stats.get("failed_queries", 0),
                "average_response_time": retrieval_stats.get("average_response_time", 0),
                "document_conversions": retrieval_stats.get("document_conversion_count", 0)
            },
            "coordinator": {
                "total_queries": coordinator_stats.get("total_queries", 0),
                "successful_queries": coordinator_stats.get("successful_queries", 0),
                "module_usage": coordinator_stats.get("module_usage", {}),
                "fusion_method_usage": coordinator_stats.get("fusion_method_usage", {}),
                "error_counts": coordinator_stats.get("error_counts", {})
            }
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"获取检索指标失败: {str(e)}")
        return {"error": str(e)}


def get_current_performance_metrics() -> Dict[str, Any]:
    """获取当前性能指标"""
    try:
        import psutil
        
        # 更新峰值使用率
        current_memory = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent(interval=0.1)
        
        _service_metrics["peak_memory_usage"] = max(_service_metrics["peak_memory_usage"], current_memory)
        _service_metrics["peak_cpu_usage"] = max(_service_metrics["peak_cpu_usage"], current_cpu)
        
        return {
            "current_cpu_percent": current_cpu,
            "current_memory_percent": current_memory,
            "peak_cpu_percent": _service_metrics["peak_cpu_usage"],
            "peak_memory_percent": _service_metrics["peak_memory_usage"],
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }
    
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        return {"error": str(e)}


def get_api_statistics(include_detailed: bool = False) -> Dict[str, Any]:
    """获取API统计信息"""
    stats = calculate_service_metrics()
    
    if include_detailed:
        stats["endpoint_details"] = {}
        for endpoint, endpoint_stats in _service_metrics["endpoint_stats"].items():
            total = endpoint_stats["total_requests"]
            if total > 0:
                stats["endpoint_details"][endpoint] = {
                    "requests": total,
                    "success_rate": endpoint_stats["successful_requests"] / total,
                    "avg_response_time": endpoint_stats["total_response_time"] / total,
                    "min_response_time": endpoint_stats["min_response_time"] if endpoint_stats["min_response_time"] != float('inf') else 0,
                    "max_response_time": endpoint_stats["max_response_time"]
                }
    
    return stats