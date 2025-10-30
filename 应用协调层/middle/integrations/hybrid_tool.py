"""
LangChain兼容的混合检索工具
提供混合检索的标准化工具接口，支持LangChain和MCP集成。
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..models.data_models import (
    RetrievalResult, FusedResult, RetrievalConfig, 
    RetrievalSource, FusionMethod
)
from ..core.retrieval_coordinator import HybridRetrievalCoordinator
from ..utils.logging_utils import get_logger, StructuredLogger

class HybridRetrievalTool:
    """混合检索工具"""
    
    def __init__(self, coordinator: Optional[HybridRetrievalCoordinator] = None):
        """
        初始化混合检索工具
        
        Args:
            coordinator: 检索协调器实例
        """
        self.coordinator = coordinator or HybridRetrievalCoordinator()
        self.logger = StructuredLogger("hybrid_tool")
        
    async def search(
        self,
        query: str,
        top_k: int = 5,
        config: Optional[RetrievalConfig] = None,
        fusion_method: FusionMethod = FusionMethod.RRF
    ) -> List[FusedResult]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            config: 检索配置
            fusion_method: 融合方法
            
        Returns:
            融合后的检索结果列表
        """
        try:
            self.logger.info(f"开始混合检索: {query[:50]}...")
            
            # 构建配置
            if config is None:
                config = RetrievalConfig(
                    top_k=top_k,
                    fusion_method=fusion_method
                )
            
            # 执行检索
            results = await self.coordinator.retrieve(query, config)
            
            self.logger.info(f"混合检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"混合检索失败: {e}")
            raise
    
    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        config: Optional[RetrievalConfig] = None,
        fusion_method: FusionMethod = FusionMethod.RRF
    ) -> Dict[str, List[FusedResult]]:
        """
        批量混合检索
        
        Args:
            queries: 查询字符串列表
            top_k: 每个查询返回结果数量
            config: 检索配置
            fusion_method: 融合方法
            
        Returns:
            查询到结果的映射字典
        """
        try:
            self.logger.info(f"开始批量混合检索，查询数量: {len(queries)}")
            
            # 构建配置
            if config is None:
                config = RetrievalConfig(
                    top_k=top_k,
                    fusion_method=fusion_method
                )
            
            # 并发执行检索
            tasks = [
                self.coordinator.retrieve(query, config)
                for query in queries
            ]
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            batch_results = {}
            for i, result in enumerate(results_list):
                query = queries[i]
                if isinstance(result, Exception):
                    self.logger.error(f"查询失败 '{query}': {result}")
                    batch_results[query] = []
                else:
                    batch_results[query] = result
            
            self.logger.info(f"批量混合检索完成，成功处理 {len(batch_results)} 个查询")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"批量混合检索失败: {e}")
            raise


def create_hybrid_tool() -> HybridRetrievalTool:
    """创建混合检索工具实例"""
    return HybridRetrievalTool()


async def create_hybrid_tool_async() -> HybridRetrievalTool:
    """异步创建混合检索工具实例"""
    return HybridRetrievalTool()


def mcp_hybrid_retrieval_tool() -> Dict[str, Any]:
    """创建MCP混合检索工具配置"""
    return {
        "name": "hybrid_retrieval",
        "description": "混合检索工具，支持BM25、向量和图检索的融合",
        "parameters": {
            "query": {"type": "string", "description": "查询字符串"},
            "top_k": {"type": "integer", "default": 5, "description": "返回结果数量"},
            "fusion_method": {"type": "string", "default": "rrf", "description": "融合方法"}
        }
    }


__all__ = [
    "HybridRetrievalTool",
    "create_hybrid_tool",
    "create_hybrid_tool_async", 
    "mcp_hybrid_retrieval_tool"
]