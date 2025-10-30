"""
RESTful API接口
提供基于FastAPI的混合检索服务接口
"""

from .app import create_app
from .models import *
from .routes import *

__all__ = [
    "create_app"
]