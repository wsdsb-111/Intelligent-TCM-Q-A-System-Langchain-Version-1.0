"""
MCP (Model Context Protocol) 工具集成
为混合检索系统提供符合MCP标准的工具接口
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import asdict, dataclass

import sys
import os
from pathlib import Path

import requests
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.models.data_models import RetrievalConfig, FusedResult, RetrievalSource, FusionMethod
from middle.utils.logging_utils import get_logger


class ExternalSearchUnavailable(Exception):
    """当外部网页搜索不可用时抛出的异常"""


@dataclass
class MCPSettings:
    api_key: str
    search_endpoint: str
    timeout: int
    default_summary: bool
    default_freshness: str


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_mcp_settings() -> MCPSettings:
    """从配置文件和环境变量加载MCP相关设置"""
    default = {
        "api_key": os.getenv("BOCHA_API_KEY", "sk-ee9c3bd6015549f9bed44cfa8fd6447c"),
        "search_endpoint": os.getenv("BOCHA_SEARCH_ENDPOINT", "https://api.bochaai.com/v1/web-search"),
        "timeout": int(os.getenv("BOCHA_SEARCH_TIMEOUT", "10")),
        "default_summary": _to_bool(os.getenv("MCP_WEB_SEARCH_DEFAULT_SUMMARY", "false"), False),
        "default_freshness": os.getenv("MCP_WEB_SEARCH_DEFAULT_FRESHNESS", "noLimit"),
    }

    config_path = Path(__file__).resolve().parent.parent / "config" / "mcp_tools_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            # 支持两种根节点: mcp_tools.web_search 或 web_search
            web_search_cfg = config_data.get("web_search")
            if web_search_cfg is None:
                web_search_cfg = config_data.get("mcp_tools", {}).get("web_search", {})

            if isinstance(web_search_cfg, dict):
                default.update({
                    "api_key": web_search_cfg.get("api_key", default["api_key"]),
                    "search_endpoint": web_search_cfg.get("search_endpoint", default["search_endpoint"]),
                    "timeout": web_search_cfg.get("timeout", default["timeout"]),
                    "default_summary": _to_bool(web_search_cfg.get("default_summary"), default["default_summary"]),
                    "default_freshness": web_search_cfg.get("default_freshness", default["default_freshness"]),
                })
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning("加载mcp_tools_config.yaml失败，将使用默认配置: %s", e)

    return MCPSettings(**default)


settings = _load_mcp_settings()


@dataclass
class SearchRequest:
    query: str
    limit: int
    include: Optional[str] = None
    exclude: Optional[str] = None
    freshness: Optional[str] = None
    summary: Optional[bool] = None
    count: Optional[int] = None


class SearchService:
    logger = get_logger(__name__)

    @classmethod
    def search_web(cls, request: SearchRequest) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": request.query,
            "count": request.count or request.limit,
            "include": request.include,
            "exclude": request.exclude,
            "freshness": request.freshness or settings.default_freshness,
            "summary": request.summary if request.summary is not None else settings.default_summary
        }
        # 清理 None 值，避免发送无效字段
        payload = {k: v for k, v in payload.items() if v not in [None, ""]}

        try:
            response = requests.post(
                settings.search_endpoint,
                json=payload,
                headers=headers,
                timeout=settings.timeout
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            normalized_results = []
            for item in results:
                normalized_results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "score": item.get("score"),
                        "source": item.get("source")
                    }
                )

            if normalized_results:
                return normalized_results

            raise ExternalSearchUnavailable("外部搜索服务未返回结果")

        except requests.exceptions.RequestException as req_err:
            raise ExternalSearchUnavailable(str(req_err)) from req_err
        except ExternalSearchUnavailable:
            raise
        except Exception as e:
            cls.logger.error("网页搜索失败", exc_info=True)
            raise ExternalSearchUnavailable(str(e)) from e


class MCPToolRegistry:
    """MCP工具注册器"""
    
    def __init__(self):
        self.tools = {}
        self.logger = get_logger(__name__)
        
    def register_tool(self, name: str, func, description: str, parameters: Dict[str, Any]):
        """注册MCP工具"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        self.logger.info(f"注册MCP工具: {name}")
        
    def get_tool(self, name: str):
        """获取工具"""
        return self.tools.get(name)
        
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in self.tools.items()
        ]


class HybridRetrievalMCPTool:
    """混合检索MCP工具"""
    
    def __init__(self, coordinator: Optional[HybridRetrievalCoordinator] = None):
        """
        初始化MCP工具
        
        Args:
            coordinator: 混合检索协调器实例
        """
        self.coordinator = coordinator or HybridRetrievalCoordinator()
        self.logger = get_logger(__name__)
        self.registry = MCPToolRegistry()
        
        # 注册工具
        self._register_tools()
        
    def _register_tools(self):
        """注册所有MCP工具"""
        
        # 混合检索工具
        self.registry.register_tool(
            name="hybrid_retrieval",
            func=self.hybrid_retrieval,
            description="智能中医混合检索工具，整合BM25、向量检索和知识图谱三种检索方式",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询内容"
                    },
                    "retrieval_type": {
                        "type": "string",
                        "enum": ["hybrid", "bm25", "vector", "graph"],
                        "default": "hybrid",
                        "description": "检索类型：hybrid(混合)、bm25(关键词)、vector(语义)、graph(知识图谱)"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                        "description": "返回结果数量"
                    },
                    "fusion_method": {
                        "type": "string",
                        "enum": ["rrf", "weighted"],
                        "default": "rrf",
                        "description": "融合方法：rrf(倒数排名融合)、weighted(加权融合)"
                    }
                },
                "required": ["query"]
            }
        )       
 
        # 批量检索工具
        self.registry.register_tool(
            name="batch_retrieval",
            func=self.batch_retrieval,
            description="批量混合检索工具，支持同时处理多个查询",
            parameters={
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "查询列表"
                    },
                    "retrieval_type": {
                        "type": "string",
                        "enum": ["hybrid", "bm25", "vector", "graph"],
                        "default": "hybrid",
                        "description": "检索类型"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 3,
                        "description": "每个查询返回的结果数量"
                    }
                },
                "required": ["queries"]
            }
        )
        
        # 上下文检索工具
        self.registry.register_tool(
            name="contextual_retrieval",
            func=self.contextual_retrieval,
            description="上下文感知检索工具，支持多轮对话场景",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "当前查询"
                    },
                    "context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "对话上下文"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "返回结果数量"
                    }
                },
                "required": ["query"]
            }
        )
        
        # 健康检查工具
        self.registry.register_tool(
            name="health_check",
            func=self.health_check,
            description="检查混合检索系统各模块的健康状态",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        
        # 统计信息工具
        self.registry.register_tool(
            name="get_statistics",
            func=self.get_statistics,
            description="获取混合检索系统的统计信息和性能指标",
            parameters={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "default": False,
                        "description": "是否包含详细统计信息"
                    }
                },
                "required": []
            }
        )

        # 网页搜索工具
        self.registry.register_tool(
            name="web_search",
            func=self.web_search,
            description="使用博查API进行网页搜索",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "limit": {"type": "integer", "description": "结果数量", "default": 5},
                    "include": {"type": "string", "description": "指定域名"}
                },
                "required": ["query"]
            }
        )
    
    async def hybrid_retrieval(self, 
                             query: str,
                             retrieval_type: str = "hybrid",
                             top_k: int = 5,
                             fusion_method: str = "rrf") -> Dict[str, Any]:
        """
        混合检索工具函数
        
        Args:
            query: 检索查询
            retrieval_type: 检索类型
            top_k: 返回结果数量
            fusion_method: 融合方法
            
        Returns:
            检索结果字典
        """
        try:
            self.logger.info(f"MCP混合检索: query='{query}', type={retrieval_type}, top_k={top_k}")
            
            # 构建检索配置
            fusion_enum = fusion_method
            if isinstance(fusion_method, str):
                try:
                    fusion_enum = FusionMethod(fusion_method.lower())
                except ValueError:
                    fusion_enum = FusionMethod.RRF if fusion_method.lower() == "rrf" else FusionMethod.WEIGHTED

            enable_vector = retrieval_type in ["hybrid", "vector", "bm25"]
            enable_graph = retrieval_type in ["hybrid", "graph"]

            config = RetrievalConfig(
                enable_vector=enable_vector,
                enable_graph=enable_graph,
                top_k=top_k,
                fusion_method=fusion_enum
            )
            
            # 执行检索
            results = await self.coordinator.retrieve(query, config)
            
            # 转换为MCP格式
            mcp_results = []
            for result in results:
                mcp_result = {
                    "content": result.content,
                    "score": result.fused_score,
                    "sources": [source.value if hasattr(source, 'value') else str(source) 
                              for source in result.contributing_sources],
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat()
                }
                
                # 添加实体和关系信息
                if result.entities:
                    mcp_result["entities"] = result.entities
                if result.relationships:
                    mcp_result["relationships"] = result.relationships
                    
                mcp_results.append(mcp_result)
            
            return {
                "success": True,
                "query": query,
                "retrieval_type": retrieval_type,
                "total_results": len(mcp_results),
                "results": mcp_results,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"MCP混合检索失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "retrieval_type": retrieval_type,
                "execution_time": datetime.now().isoformat()
            }  
  
    async def batch_retrieval(self,
                            queries: List[str],
                            retrieval_type: str = "hybrid",
                            top_k: int = 3) -> Dict[str, Any]:
        """
        批量检索工具函数
        
        Args:
            queries: 查询列表
            retrieval_type: 检索类型
            top_k: 每个查询返回的结果数量
            
        Returns:
            批量检索结果字典
        """
        try:
            self.logger.info(f"MCP批量检索: {len(queries)}个查询, type={retrieval_type}")
            
            # 构建检索配置
            config = RetrievalConfig(
                enable_bm25=(retrieval_type in ["hybrid", "bm25"]),
                enable_vector=(retrieval_type in ["hybrid", "vector"]),
                enable_graph=(retrieval_type in ["hybrid", "graph"]),
                top_k=top_k
            )
            
            # 执行批量检索
            batch_results = await self.coordinator.batch_retrieve(queries, config)
            
            # 转换为MCP格式
            mcp_batch_results = {}
            for query, results in batch_results.items():
                mcp_results = []
                for result in results:
                    mcp_result = {
                        "content": result.content,
                        "score": result.fused_score,
                        "sources": [source.value if hasattr(source, 'value') else str(source) 
                                  for source in result.contributing_sources],
                        "metadata": result.metadata
                    }
                    mcp_results.append(mcp_result)
                
                mcp_batch_results[query] = mcp_results
            
            return {
                "success": True,
                "total_queries": len(queries),
                "retrieval_type": retrieval_type,
                "results": mcp_batch_results,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"MCP批量检索失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_queries": len(queries),
                "execution_time": datetime.now().isoformat()
            }
    
    async def contextual_retrieval(self,
                                 query: str,
                                 context: Optional[List[Dict[str, str]]] = None,
                                 top_k: int = 5) -> Dict[str, Any]:
        """
        上下文检索工具函数
        
        Args:
            query: 当前查询
            context: 对话上下文
            top_k: 返回结果数量
            
        Returns:
            上下文检索结果字典
        """
        try:
            self.logger.info(f"MCP上下文检索: query='{query}', context_length={len(context) if context else 0}")
            
            # 构建增强查询（结合上下文）
            enhanced_query = query
            if context:
                # 提取上下文中的关键信息
                context_text = " ".join([msg["content"] for msg in context[-3:]])  # 使用最近3轮对话
                enhanced_query = f"{context_text} {query}"
            
            # 执行检索
            config = RetrievalConfig(top_k=top_k)
            results = await self.coordinator.retrieve(enhanced_query, config)
            
            # 转换为MCP格式
            mcp_results = []
            for result in results:
                mcp_result = {
                    "content": result.content,
                    "score": result.fused_score,
                    "sources": [source.value if hasattr(source, 'value') else str(source) 
                              for source in result.contributing_sources],
                    "metadata": result.metadata,
                    "context_enhanced": bool(context)
                }
                mcp_results.append(mcp_result)
            
            return {
                "success": True,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "context_used": bool(context),
                "total_results": len(mcp_results),
                "results": mcp_results,
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"MCP上下文检索失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_query": query,
                "execution_time": datetime.now().isoformat()
            }  
  
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查工具函数
        
        Returns:
            健康状态字典
        """
        try:
            self.logger.info("MCP健康检查")
            
            # 执行健康检查
            health_status = await self.coordinator.health_check()
            
            return {
                "success": True,
                "overall_healthy": health_status.overall_healthy,
                "modules": [module.to_dict() for module in health_status.modules],
                "check_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"MCP健康检查失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "overall_healthy": False,
                "check_time": datetime.now().isoformat()
            }
    
    async def get_statistics(self, include_details: bool = False) -> Dict[str, Any]:
        """
        获取统计信息工具函数
        
        Args:
            include_details: 是否包含详细信息
            
        Returns:
            统计信息字典
        """
        try:
            self.logger.info(f"MCP获取统计信息: include_details={include_details}")
            
            # 获取统计信息
            stats = self.coordinator.get_statistics()
            
            result = {
                "success": True,
                "statistics": stats,
                "collection_time": datetime.now().isoformat()
            }
            
            if not include_details:
                # 只返回基本统计信息
                basic_stats = {
                    "total_queries": stats.get("total_queries", 0),
                    "successful_queries": stats.get("successful_queries", 0),
                    "failed_queries": stats.get("failed_queries", 0),
                    "average_response_time": stats.get("average_response_time", 0.0)
                }
                result["statistics"] = basic_stats
            
            return result
            
        except Exception as e:
            self.logger.error(f"MCP获取统计信息失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "collection_time": datetime.now().isoformat()
            }

    async def web_search(
        self,
        query: str,
        limit: int = 5,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        freshness: Optional[str] = None,
        summary: Optional[bool] = None,
        count: Optional[int] = None
    ) -> Dict[str, Any]:
        request = SearchRequest(
            query=query,
            limit=limit,
            include=include,
            exclude=exclude,
            freshness=freshness,
            summary=summary,
            count=count
        )
        try:
            results = SearchService.search_web(request)
            return {
                "success": True,
                "results": results,
                "source": "external"
            }
        except ExternalSearchUnavailable as external_error:
            return {
                "success": True,
                "results": [
                    {
                        "title": "外部网页搜索暂不可用",
                        "url": settings.search_endpoint,
                        "snippet": f"原始查询: {query}。错误: {external_error}",
                        "source": "offline_fallback"
                    }
                ],
                "source": "offline_fallback",
                "external_error": str(external_error)
            }
        except Exception as e:
            self.logger.error("MCP网页搜索失败: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取所有工具定义（MCP格式）"""
        return self.registry.list_tools()
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用MCP工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            工具执行结果
        """
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"工具 '{tool_name}' 不存在",
                "available_tools": list(self.registry.tools.keys())
            }
        
        try:
            # 调用工具函数
            result = await tool["function"](**parameters)
            return result
            
        except Exception as e:
            self.logger.error(f"MCP工具调用失败: {tool_name}, 错误: {str(e)}")
            return {
                "success": False,
                "error": f"工具调用失败: {str(e)}",
                "tool_name": tool_name,
                "parameters": parameters
            }


# 全局MCP工具实例
_mcp_tool_instance = None


def get_mcp_tool(coordinator: Optional[HybridRetrievalCoordinator] = None) -> HybridRetrievalMCPTool:
    """
    获取MCP工具实例（单例模式）
    
    Args:
        coordinator: 混合检索协调器实例
        
    Returns:
        MCP工具实例
    """
    global _mcp_tool_instance
    
    if _mcp_tool_instance is None:
        _mcp_tool_instance = HybridRetrievalMCPTool(coordinator)
    
    return _mcp_tool_instance


# MCP工具装饰器
def mcp_tool(name: str, description: str, parameters: Dict[str, Any]):
    """
    MCP工具装饰器
    
    Args:
        name: 工具名称
        description: 工具描述
        parameters: 工具参数定义
    """
    def decorator(func):
        # 注册到全局工具注册器
        tool_instance = get_mcp_tool()
        tool_instance.registry.register_tool(name, func, description, parameters)
        return func
    return decorator


# 便捷函数
async def hybrid_retrieval_mcp(query: str, 
                             retrieval_type: str = "hybrid",
                             top_k: int = 5) -> Dict[str, Any]:
    """
    MCP兼容的混合检索便捷函数
    
    Args:
        query: 检索查询
        retrieval_type: 检索类型
        top_k: 返回结果数量
        
    Returns:
        检索结果字典
    """
    tool = get_mcp_tool()
    return await tool.hybrid_retrieval(query, retrieval_type, top_k)


__all__ = [
    "HybridRetrievalMCPTool",
    "MCPToolRegistry", 
    "get_mcp_tool",
    "mcp_tool",
    "hybrid_retrieval_mcp"
]