"""
API路由模块
"""

from .retrieval import router as retrieval_router
from .health import router as health_router
from .metrics import router as metrics_router

__all__ = [
    "retrieval_router",
    "health_router", 
    "metrics_router"
]