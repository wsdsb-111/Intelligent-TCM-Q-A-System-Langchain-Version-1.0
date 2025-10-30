"""
RAG链路
集成检索协调器和模型服务，实现完整的RAG问答流程

注意：系统使用二元路由（vector_only / hybrid），不再使用独立的纯知识图谱检索
- vector_only（ENTITY_DRIVEN）：纯向量检索
- hybrid（COMPLEX_REASONING）：混合检索（向量+图谱）
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from ..core.retrieval_coordinator import HybridRetrievalCoordinator
from ..models.data_models import RetrievalConfig, FusionMethod
from .model_service import get_model_service
from .prompt_templates import PromptTemplates
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class RAGChain:
    """
    RAG链路类
    协调检索和生成过程，实现端到端的问答
    """
    
    def __init__(self,
                 retrieval_coordinator: Optional[HybridRetrievalCoordinator] = None,
                 max_context_tokens: int = 1500,
                 max_retrieval_results: int = 5,
                 enable_context_truncation: bool = True,
                 enable_rerank: bool = True,
                 enable_query_expansion: bool = True):
        """
        初始化RAG链路
        
        Args:
            retrieval_coordinator: 检索协调器实例
            max_context_tokens: 最大上下文token数
            max_retrieval_results: 最大检索结果数
            enable_context_truncation: 是否启用上下文截断
            enable_rerank: 是否启用重排序
            enable_query_expansion: 是否启用查询扩展
        """
        self.retrieval_coordinator = retrieval_coordinator
        self.model_service = get_model_service()
        self.max_context_tokens = max_context_tokens
        self.max_retrieval_results = max_retrieval_results
        self.enable_context_truncation = enable_context_truncation
        self.enable_rerank = enable_rerank
        self.enable_query_expansion = enable_query_expansion
        self.logger = logger  # 添加logger属性
        
        # 初始化重排序器和查询扩展器（懒加载）
        self._reranker = None
        self._expander = None
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_retrieval_time": 0.0,
            "average_generation_time": 0.0,
            "average_total_time": 0.0
        }
        
        logger.info("RAG链路初始化完成")
    
    async def query(self,
                   question: str,
                   retrieval_config: Optional[RetrievalConfig] = None,
                   use_retrieval: bool = True,
                   system_prompt: str = None,
                   temperature: float = 0.5,  # 稍微调高默认温度
                   max_new_tokens: int = 512) -> Dict[str, Any]:
        """
        执行完整的RAG问答流程
        
        Args:
            question: 用户问题
            retrieval_config: 检索配置
            use_retrieval: 是否使用检索（False时直接生成）
            system_prompt: 自定义系统提示
            temperature: 生成温度
            max_new_tokens: 最大生成token数
            
        Returns:
            包含答案和元数据的字典
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            retrieval_results = []
            retrieval_time = 0.0
            
            # 步骤1: 检索阶段
            if use_retrieval and self.retrieval_coordinator:
                retrieval_start = time.time()
                retrieval_results = await self._retrieve(question, retrieval_config)
                retrieval_time = time.time() - retrieval_start
                logger.info(f"检索完成，耗时 {retrieval_time:.2f}s，获得 {len(retrieval_results)} 个结果")
            
            # 步骤2: 提取路由决策信息
            routing_decision = None
            routing_confidence = 0.0
            if retrieval_results and isinstance(retrieval_results[0], dict) and 'metadata' in retrieval_results[0]:
                routing_decision = retrieval_results[0].get('metadata', {}).get('routing_decision')
                routing_confidence = retrieval_results[0].get('metadata', {}).get('routing_confidence', 0.0)
            
            # 步骤3: 构建Prompt（根据路由决策选择模板）
            prompt_start = time.time()
            user_prompt, generation_mode = self._build_prompt_with_mode(question, retrieval_results, use_retrieval, routing_decision)
            prompt_time = time.time() - prompt_start
            
            # 步骤4: 生成答案（使用对应的生成模式）
            generation_start = time.time()
            system = system_prompt or PromptTemplates.SYSTEM_PROMPT
            generation_result = self.model_service.generate(
                query=user_prompt,
                system_prompt=system,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                mode=generation_mode  # 传递生成模式
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # 更新统计
            self._update_stats(True, retrieval_time, generation_time, total_time)
            
            # 构建返回结果
            result = {
                "success": True,
                "query": question,
                "answer": generation_result["answer"],
                "retrieval_results": self._format_retrieval_results(retrieval_results),
                "metadata": {
                    "retrieval_time": round(retrieval_time, 2),
                    "generation_time": round(generation_time, 2),
                    "prompt_time": round(prompt_time, 3),
                    "total_time": round(total_time, 2),
                    "num_retrieval_results": len(retrieval_results),
                    "use_retrieval": use_retrieval,
                    "routing_decision": routing_decision,  # 路由决策
                    "routing_confidence": round(routing_confidence, 2) if routing_confidence else None,  # 路由置信度
                    "generation_mode": generation_mode,  # 生成模式
                    **generation_result.get("metadata", {})
                }
            }
            
            logger.info(f"问答完成: {question[:50]}... | 总耗时: {total_time:.2f}s")
            return result
            
        except Exception as e:
            self.stats["failed_queries"] += 1
            logger.error(f"RAG问答失败: {e}", exc_info=True)
            
            return {
                "success": False,
                "query": question,
                "answer": None,
                "error": str(e),
                "metadata": {
                    "total_time": round(time.time() - start_time, 2)
                }
            }
    
    async def _retrieve(self,
                       query: str,
                       config: Optional[RetrievalConfig] = None) -> List[Dict[str, Any]]:
        """
        执行检索
        
        Args:
            query: 检索查询
            config: 检索配置
            
        Returns:
            检索结果列表
        """
        if not self.retrieval_coordinator:
            logger.warning("检索协调器未初始化，跳过检索")
            return []
        
        # 使用默认配置或自定义配置（BM25已移除）
        retrieval_config = config or RetrievalConfig(
            enable_vector=True,
            enable_graph=True,
            top_k=self.max_retrieval_results,
            fusion_method=FusionMethod.WEIGHTED  # 使用加权融合
        )
        
        try:
            # 调用检索协调器
            results = await self.retrieval_coordinator.retrieve(query, retrieval_config)
            
            # 转换为字典格式
            results_dicts = [r.to_dict() for r in results]
            
            # 记录路由信息（如果有）
            if results_dicts and 'metadata' in results_dicts[0]:
                routing_info = results_dicts[0].get('metadata', {})
                if 'routing_decision' in routing_info:
                    self.logger.debug(f"检索路由: {routing_info['routing_decision']}, "
                                    f"置信度: {routing_info.get('routing_confidence', 0):.2f}")
            
            # 根据token限制截断
            if self.enable_context_truncation:
                results_dicts = PromptTemplates.truncate_context_by_tokens(
                    results_dicts,
                    max_tokens=self.max_context_tokens
                )
            
            return results_dicts
            
        except Exception as e:
            logger.error(f"检索失败: {e}", exc_info=True)
            return []
    
    def _build_prompt(self,
                     query: str,
                     retrieval_results: List[Dict[str, Any]],
                     use_retrieval: bool) -> str:
        """
        构建用户提示词（传统方法，保持兼容性）
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果
            use_retrieval: 是否使用检索
            
        Returns:
            构建的提示词
        """
        if use_retrieval and retrieval_results:
            # 使用RAG模板
            prompt = PromptTemplates.build_rag_prompt(
                query=query,
                retrieval_results=retrieval_results,
                max_context_results=self.max_retrieval_results
            )
        else:
            # 直接问答模板
            prompt = PromptTemplates.build_direct_prompt(query)
        
        # 记录prompt长度
        estimated_tokens = PromptTemplates.estimate_tokens(prompt)
        logger.debug(f"构建Prompt完成，估计token数: {estimated_tokens}")
        
        return prompt
    
    def _build_prompt_with_mode(self,
                                query: str,
                                retrieval_results: List[Dict[str, Any]],
                                use_retrieval: bool,
                                routing_decision: str = None) -> Tuple[str, str]:
        """
        构建用户提示词（带模式选择）
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果
            use_retrieval: 是否使用检索
            routing_decision: 路由决策（二元路由："vector_only" 或 "hybrid"）
            
        Returns:
            (提示词, 生成模式)
        """
        if not use_retrieval or not retrieval_results:
            # 直接问答
            prompt = PromptTemplates.build_direct_prompt(query)
            return prompt, "default"
        
        # 根据路由决策选择不同的提示词模板（二元路由）
        if routing_decision == "vector_only":
            # 纯向量检索模式（ENTITY_DRIVEN）
            prompt = PromptTemplates.build_vector_prompt(
                query=query,
                retrieval_results=retrieval_results,
                max_context_results=self.max_retrieval_results
            )
            generation_mode = "vector"
            
        elif routing_decision == "hybrid":
            # 混合检索模式（COMPLEX_REASONING）
            # 需要分离向量和图谱结果
            vector_results = [r for r in retrieval_results if r.get('source') == 'vector']
            kg_results = [r for r in retrieval_results if r.get('source') == 'graph']
            
            # 如果没有分离结果，尝试从metadata中获取
            if not vector_results and not kg_results:
                # 降级：使用所有结果作为vector结果
                vector_results = retrieval_results[:3]
                kg_results = []
            
            prompt = PromptTemplates.build_hybrid_prompt(
                query=query,
                vector_results=vector_results,
                kg_results=kg_results,
                max_context_results=3
            )
            generation_mode = "hybrid"
            
        else:
            # 默认：使用传统RAG模板（用于兼容性）
            prompt = PromptTemplates.build_rag_prompt(
                query=query,
                retrieval_results=retrieval_results,
                max_context_results=self.max_retrieval_results
            )
            generation_mode = "default"
        
        # 记录prompt长度
        estimated_tokens = PromptTemplates.estimate_tokens(prompt)
        logger.debug(f"构建Prompt完成 (模式: {generation_mode}), 估计token数: {estimated_tokens}")
        
        return prompt, generation_mode
    
    def _format_retrieval_results(self, 
                                  results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        格式化检索结果用于返回
        
        Args:
            results: 原始检索结果
            
        Returns:
            格式化后的结果
        """
        formatted = []
        
        for result in results:
            formatted_result = {
                "content": result.get("content", ""),
                "fused_score": result.get("fused_score", 0.0),
                "source_scores": result.get("source_scores", {}),
                "contributing_sources": result.get("contributing_sources", []),
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", [])
            }
            formatted.append(formatted_result)
        
        return formatted
    
    def _update_stats(self, 
                     success: bool,
                     retrieval_time: float,
                     generation_time: float,
                     total_time: float):
        """更新统计信息"""
        if success:
            self.stats["successful_queries"] += 1
            
            # 更新平均时间
            count = self.stats["successful_queries"]
            
            self.stats["average_retrieval_time"] = (
                (self.stats["average_retrieval_time"] * (count - 1) + retrieval_time) / count
            )
            self.stats["average_generation_time"] = (
                (self.stats["average_generation_time"] * (count - 1) + generation_time) / count
            )
            self.stats["average_total_time"] = (
                (self.stats["average_total_time"] * (count - 1) + total_time) / count
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    async def batch_query(self,
                         questions: List[str],
                         retrieval_config: Optional[RetrievalConfig] = None,
                         **kwargs) -> List[Dict[str, Any]]:
        """
        批量问答
        
        Args:
            questions: 问题列表
            retrieval_config: 检索配置
            **kwargs: 其他参数传递给query方法
            
        Returns:
            答案列表
        """
        tasks = [
            self.query(question, retrieval_config, **kwargs)
            for question in questions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "query": questions[i],
                    "answer": None,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results


# 便捷函数
async def simple_query(question: str,
                      coordinator: HybridRetrievalCoordinator = None,
                      use_retrieval: bool = True) -> str:
    """
    简单问答函数，直接返回答案文本
    
    Args:
        question: 用户问题
        coordinator: 检索协调器
        use_retrieval: 是否使用检索
        
    Returns:
        答案文本
    """
    chain = RAGChain(retrieval_coordinator=coordinator)
    result = await chain.query(question, use_retrieval=use_retrieval)
    
    if result["success"]:
        return result["answer"]
    else:
        return f"抱歉，问答出现错误: {result.get('error', '未知错误')}"

