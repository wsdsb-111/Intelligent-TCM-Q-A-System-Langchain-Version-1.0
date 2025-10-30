"""
LangChain兼容的混合检索器
实际实现在这里，不需要从其他地方导入
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # 如果LangChain不可用，创建基础类
    class BaseRetriever:
        def get_relevant_documents(self, query: str) -> List:
            raise NotImplementedError
        
        async def aget_relevant_documents(self, query: str) -> List:
            raise NotImplementedError
    
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    CallbackManagerForRetrieverRun = None
    AsyncCallbackManagerForRetrieverRun = None
    LANGCHAIN_AVAILABLE = False

import sys
import os
# 添加middle目录到路径，确保可以导入middle模块
middle_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(os.path.dirname(middle_dir))
sys.path.insert(0, middle_dir)
sys.path.insert(0, project_root)

from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.models.data_models import RetrievalConfig, FusedResult
from middle.utils.logging_utils import get_logger


class HybridRetriever(BaseRetriever):
    """
    LangChain兼容的混合检索器
    
    整合BM25、向量检索和知识图谱三个检索模块，
    提供符合LangChain标准的检索接口
    """
    
    def __init__(self,
                 coordinator: Optional[HybridRetrievalCoordinator] = None,
                 vector_adapter=None,
                 graph_adapter=None,
                 config: Optional[RetrievalConfig] = None,
                 **kwargs):
        """
        初始化混合检索器
        
        Args:
            coordinator: 混合检索协调器实例（如果提供，则忽略适配器参数）
            vector_adapter: 向量检索适配器（如果未提供coordinator）
            graph_adapter: 图检索适配器（如果未提供coordinator）
            config: 检索配置
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # 如果没有提供coordinator，则使用适配器创建一个
        if coordinator is None:
            coordinator = HybridRetrievalCoordinator(
                vector_adapter=vector_adapter,
                graph_adapter=graph_adapter,
                config=config
            )
        
        # 使用object.__setattr__绕过Pydantic的字段验证
        object.__setattr__(self, 'coordinator', coordinator)
        object.__setattr__(self, 'config', config or RetrievalConfig())
        object.__setattr__(self, 'logger', get_logger(__name__))
        
        # 统计信息
        object.__setattr__(self, 'stats', {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "document_conversion_count": 0
        })
        
        self.logger.info("LangChain混合检索器初始化完成")
    
    def _get_relevant_documents(self, 
                               query: str, 
                               *, 
                               run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        """
        LangChain 1.0+ 要求的抽象方法实现
        获取相关文档（同步）
        
        Args:
            query: 检索查询
            run_manager: LangChain回调管理器
            
        Returns:
            LangChain Document列表
        """
        try:
            if run_manager:
                run_manager.on_text(f"开始混合检索: {query}")
            
            # 使用asyncio运行异步检索
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                fused_results = loop.run_until_complete(
                    self.coordinator.retrieve(query, self.config)
                )
            finally:
                loop.close()
            
            # 转换为LangChain Document格式
            documents = self._convert_to_documents(fused_results)
            
            self.logger.info(f"同步检索完成: 查询='{query}', 文档数={len(documents)}")
            return documents
            
        except Exception as e:
            error_msg = f"同步检索失败: {str(e)}"
            self.logger.error(error_msg)
            
            if run_manager:
                run_manager.on_text(f"检索失败: {str(e)}")
            
            # 返回空列表而不是抛出异常
            return []
    
    # 注意：不需要重写 get_relevant_documents，BaseRetriever 会自动调用 _get_relevant_documents
    
    async def _aget_relevant_documents(self, 
                                      query: str, 
                                      *, 
                                      run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None) -> List[Document]:
        """
        LangChain 1.0+ 要求的异步抽象方法实现
        异步获取相关文档
        
        Args:
            query: 检索查询
            run_manager: LangChain异步回调管理器
            
        Returns:
            LangChain Document列表
        """
        try:
            if run_manager:
                await run_manager.on_text(f"开始异步混合检索: {query}")
            
            # 执行异步检索
            fused_results = await self.coordinator.retrieve(query, self.config)
            
            # 转换为LangChain Document格式
            documents = self._convert_to_documents(fused_results)
            
            if run_manager:
                await run_manager.on_text(f"异步检索完成，返回{len(documents)}个文档")
            
            self.logger.info(f"异步检索完成: 查询='{query}', 文档数={len(documents)}")
            return documents
            
        except Exception as e:
            error_msg = f"异步检索失败: {str(e)}"
            self.logger.error(error_msg)
            
            if run_manager:
                await run_manager.on_text(f"异步检索失败: {str(e)}")
            
            # 返回空列表而不是抛出异常
            return []
    
    # 注意：不需要重写 aget_relevant_documents，BaseRetriever 会自动调用 _aget_relevant_documents
    
    def _convert_to_documents(self, fused_results: List[FusedResult]) -> List[Document]:
        """
        将融合结果转换为LangChain Document格式
        
        Args:
            fused_results: 融合结果列表
            
        Returns:
            LangChain Document列表
        """
        documents = []
        
        for result in fused_results:
            # 构建元数据
            metadata = {
                "fused_score": result.fused_score,
                "source_scores": result.source_scores,
                "fusion_method": result.fusion_method.value if hasattr(result.fusion_method, 'value') else str(result.fusion_method),
                "contributing_sources": [source.value if hasattr(source, 'value') else str(source) for source in result.contributing_sources],
                "timestamp": result.timestamp.isoformat(),
                **result.metadata  # 包含原始元数据
            }
            
            # 添加实体和关系信息（如果存在）
            if result.entities:
                metadata["entities"] = result.entities
            if result.relationships:
                metadata["relationships"] = result.relationships
            
            # 创建Document
            document = Document(
                page_content=result.content,
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = getattr(self, 'stats', {})
        return {
            "retriever_stats": stats.copy(),
            "langchain_integration": {
                "available": LANGCHAIN_AVAILABLE
            }
        }


def create_hybrid_retriever(bm25_adapter=None,
                          vector_adapter=None,
                          graph_adapter=None,
                          config: Optional[RetrievalConfig] = None) -> HybridRetriever:
    """
    创建混合检索器的便捷函数
    
    注意：bm25_adapter参数已废弃，系统不再使用BM25检索
    
    Args:
        bm25_adapter: BM25检索适配器（已废弃）
        vector_adapter: 向量检索适配器
        graph_adapter: 图检索适配器
        config: 检索配置
        
    Returns:
        配置好的混合检索器实例
    """
    coordinator = HybridRetrievalCoordinator(
        vector_adapter=vector_adapter,
        graph_adapter=graph_adapter,
        config=config
    )
    
    return HybridRetriever(coordinator=coordinator, config=config)


async def create_hybrid_retriever_async(bm25_adapter=None,
                                      vector_adapter=None,
                                      graph_adapter=None,
                                      config: Optional[RetrievalConfig] = None) -> HybridRetriever:
    """
    异步创建混合检索器的便捷函数
    
    注意：bm25_adapter参数已废弃，系统不再使用BM25检索
    
    Args:
        bm25_adapter: BM25检索适配器（已废弃）
        vector_adapter: 向量检索适配器
        graph_adapter: 图检索适配器
        config: 检索配置
        
    Returns:
        配置好的混合检索器实例
    """
    coordinator = HybridRetrievalCoordinator(
        vector_adapter=vector_adapter,
        graph_adapter=graph_adapter,
        config=config
    )
    
    # 执行健康检查确保所有模块正常
    await coordinator.health_check()
    
    return HybridRetriever(coordinator=coordinator, config=config)


__all__ = [
    "HybridRetriever",
    "create_hybrid_retriever", 
    "create_hybrid_retriever_async"
]