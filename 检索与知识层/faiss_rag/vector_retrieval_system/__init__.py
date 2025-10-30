"""
向量检索子层 - 智能中医问答系统
支持CSV/JSON/Arrow数据集转换为Chroma向量数据库
"""

__version__ = "1.0.0"
__author__ = "中医问答系统开发团队"

from .data_loader import DataLoader
from .embedding_service import EmbeddingService
from .faiss_manager import FaissManager  # 使用 Faiss 替代 ChromaDB
from .vector_retrieval import VectorRetrieval
from .langchain_integration import TCMVectorRetriever, create_tcm_retriever

__all__ = [
    "DataLoader",
    "EmbeddingService", 
    "ChromaManager",
    "VectorRetrieval",
    "TCMVectorRetriever",
    "create_tcm_retriever"
]
