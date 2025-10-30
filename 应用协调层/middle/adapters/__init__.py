"""
适配器模块
包含检索模块的适配器（BM25已移除）
"""

# BM25适配器已从系统中移除
from .simple_vector_adapter import SimpleVectorAdapter  
from .graph_adapter import GraphRetrievalAdapter

__all__ = [
    "SimpleVectorAdapter",
    "GraphRetrievalAdapter"
]