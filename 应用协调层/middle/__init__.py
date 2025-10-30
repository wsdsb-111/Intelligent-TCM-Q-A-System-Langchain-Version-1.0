"""
混合检索系统 - LangChain集成
智能中医问答系统的检索与知识层
"""

__version__ = "1.0.0"
__author__ = "智能中医问答系统开发团队"

# 核心组件
try:
    from .core.retrieval_coordinator import HybridRetrievalCoordinator
    from .core.result_fusion import ResultFusion
    from .models.data_models import RetrievalResult, FusedResult, RetrievalConfig
except ImportError as e:
    print(f"Warning: 核心组件导入失败: {e}")

# 集成组件
try:
    from .integrations.hybrid_retriever import HybridRetriever
    from .integrations.hybrid_tool import HybridRetrievalTool
except ImportError as e:
    print(f"Warning: 集成组件导入失败: {e}")

# API组件
try:
    from .api.app import create_app
except ImportError as e:
    print(f"Warning: API组件导入失败: {e}")

# 适配器组件（可选）（BM25已移除）
try:
    from .adapters.simple_vector_adapter import SimpleVectorAdapter  
    from .adapters.graph_adapter import GraphRetrievalAdapter
except ImportError as e:
    print(f"Warning: 适配器组件导入失败: {e}")

# 动态构建__all__列表
__all__ = []

# 添加可用的组件
if 'HybridRetrievalCoordinator' in locals():
    __all__.append('HybridRetrievalCoordinator')
if 'ResultFusion' in locals():
    __all__.append('ResultFusion')
if 'RetrievalResult' in locals():
    __all__.extend(['RetrievalResult', 'FusedResult', 'RetrievalConfig'])
if 'HybridRetriever' in locals():
    __all__.append('HybridRetriever')
if 'HybridRetrievalTool' in locals():
    __all__.append('HybridRetrievalTool')
if 'create_app' in locals():
    __all__.append('create_app')
if 'SimpleVectorAdapter' in locals():
    __all__.extend(['SimpleVectorAdapter', 'GraphRetrievalAdapter'])