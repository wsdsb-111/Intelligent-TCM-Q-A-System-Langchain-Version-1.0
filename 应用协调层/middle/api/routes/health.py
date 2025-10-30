"""
健康检查和监控API路由
"""

import time
import psutil
import platform
from datetime import datetime
from typing import Dict, Any

try:
    from fastapi import APIRouter, Depends
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
    
    def Depends(func):
        return func
    
    FASTAPI_AVAILABLE = False

from ..models import HealthResponse, ErrorResponse, create_error_response
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from middle.integrations.hybrid_retriever import create_hybrid_retriever
from middle.utils.logging_utils import get_logger

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["health"])
logger = get_logger(__name__)

# 全局检索器实例
_retriever_instance = None


def get_retriever():
    """获取检索器实例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = create_hybrid_retriever()
    return _retriever_instance


@router.get("/health", response_model=HealthResponse)
async def health_check(retriever=Depends(get_retriever)):
    """
    系统健康检查
    
    检查所有检索模块和系统组件的健康状态。
    """
    try:
        start_time = time.time()
        
        # 执行检索器健康检查
        retriever_health = await retriever.health_check()
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 检查响应时间
        response_time = time.time() - start_time
        system_info["health_check_time"] = response_time
        
        # 构建响应
        overall_healthy = retriever_health.get("overall_healthy", False)
        status = "healthy" if overall_healthy else "unhealthy"
        
        # 提取模块状态
        modules = []
        if "coordinator" in retriever_health:
            coordinator_health = retriever_health["coordinator"]
            if "modules" in coordinator_health:
                for module in coordinator_health["modules"]:
                    modules.append({
                        "name": module.module_name,
                        "healthy": module.is_healthy,
                        "last_check": module.last_check.isoformat() if module.last_check else None,
                        "error_message": module.error_message,
                        "response_time": module.response_time
                    })
        
        response = HealthResponse(
            status=status,
            overall_healthy=overall_healthy,
            modules=modules,
            system_info=system_info
        )
        
        logger.info(f"健康检查完成: 状态={status}, 耗时={response_time:.3f}秒")
        return response
    
    except Exception as e:
        error_msg = f"健康检查失败: {str(e)}"
        logger.error(error_msg)
        
        # 返回不健康状态
        return HealthResponse(
            status="error",
            overall_healthy=False,
            modules=[],
            system_info={"error": error_msg}
        )


@router.get("/health/quick")
async def quick_health_check():
    """
    快速健康检查
    
    返回简单的服务状态，用于负载均衡器等快速检查。
    """
    try:
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "service": "hybrid-retrieval-api"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "service": "hybrid-retrieval-api"
        }


@router.get("/health/detailed")
async def detailed_health_check(retriever=Depends(get_retriever)):
    """
    详细健康检查
    
    返回详细的系统状态信息，包括性能指标和资源使用情况。
    """
    try:
        start_time = time.time()
        
        # 执行完整的健康检查
        retriever_health = await retriever.health_check()
        
        # 获取详细系统信息
        system_info = get_detailed_system_info()
        
        # 获取检索器统计信息
        retrieval_stats = retriever.get_statistics()
        
        # 计算健康检查时间
        health_check_time = time.time() - start_time
        
        # 构建详细响应
        detailed_info = {
            "service_info": {
                "name": "hybrid-retrieval-api",
                "version": "1.0.0",
                "uptime": get_service_uptime(),
                "health_check_time": health_check_time
            },
            "retriever_health": retriever_health,
            "system_info": system_info,
            "retrieval_statistics": retrieval_stats,
            "performance_metrics": get_performance_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        return detailed_info
    
    except Exception as e:
        error_msg = f"详细健康检查失败: {str(e)}"
        logger.error(error_msg)
        
        return {
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }


def get_system_info() -> Dict[str, Any]:
    """获取基础系统信息"""
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取系统信息失败: {str(e)}")
        return {"error": str(e)}


def get_detailed_system_info() -> Dict[str, Any]:
    """获取详细系统信息"""
    try:
        # 基础信息
        info = get_system_info()
        
        # 添加详细信息
        info.update({
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_details": psutil.virtual_memory()._asdict(),
            "swap_memory": psutil.swap_memory()._asdict(),
            "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None,
            "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "process_count": len(psutil.pids())
        })
        
        return info
    except Exception as e:
        logger.error(f"获取详细系统信息失败: {str(e)}")
        return get_system_info()


def get_service_uptime() -> float:
    """获取服务运行时间（秒）"""
    try:
        # 这里可以记录服务启动时间，暂时返回进程运行时间
        import os
        process = psutil.Process(os.getpid())
        return time.time() - process.create_time()
    except Exception:
        return 0.0


def get_performance_metrics() -> Dict[str, Any]:
    """获取性能指标"""
    try:
        return {
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if platform.system() != 'Windows' else psutil.disk_usage('C:').percent,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            "open_files": len(psutil.Process().open_files()) if hasattr(psutil.Process(), 'open_files') else None,
            "connections": len(psutil.net_connections()) if hasattr(psutil, 'net_connections') else None
        }
    except Exception as e:
        logger.error(f"获取性能指标失败: {str(e)}")
        return {"error": str(e)}