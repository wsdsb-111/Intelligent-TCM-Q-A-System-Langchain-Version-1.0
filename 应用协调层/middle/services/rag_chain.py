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
            # 步骤1: 检索阶段
            generation_contexts = []
            evaluation_contexts = []
            routing_decision = None  # 将在检索后设置
            routing_confidence = 0.0
            retrieval_time = 0.0
            all_retrieval_contexts = []  # 初始化
            
            if use_retrieval and self.retrieval_coordinator:
                retrieval_start = time.time()
                retrieve_result = await self.retrieval_coordinator.retrieve(question, retrieval_config)
                retrieval_time = time.time() - retrieval_start
                
                # 处理返回结果（可能是二元组或三元组）
                if len(retrieve_result) == 3:
                    # 混合检索：返回(generation_contexts, all_retrieval_contexts, evaluation_contexts)
                    generation_contexts, all_retrieval_contexts, evaluation_contexts = retrieve_result
                    routing_decision = "hybrid"  # 三元组表示混合检索
                    logger.debug(f"检测到混合检索(三元组): generation={len(generation_contexts)}, all_retrieval={len(all_retrieval_contexts)}")
                else:
                    # 纯向量检索：返回(generation_contexts, evaluation_contexts)
                    generation_contexts, evaluation_contexts = retrieve_result
                    all_retrieval_contexts = generation_contexts  # 纯向量时两者相同
                    routing_decision = "vector_only"  # 二元组表示纯向量检索
                    logger.debug(f"检测到纯向量检索(二元组): generation={len(generation_contexts)}")
                
                logger.info(f"检索完成，耗时 {retrieval_time:.2f}s，生成用 {len(generation_contexts)} 个，总检索 {len(all_retrieval_contexts)} 个结果，路由决策={routing_decision}")
                
                # 从协调器获取路由决策的详细信息（用于metadata中的confidence）
                routing_confidence = 0.0
                if hasattr(self.retrieval_coordinator, 'query_classifier') and self.retrieval_coordinator.query_classifier:
                    try:
                        query_type, confidence = self.retrieval_coordinator.query_classifier.classify_with_confidence(question)
                        if query_type:
                            routing_confidence = confidence
                            # 验证路由决策是否匹配（仅用于调试）
                            weights = self.retrieval_coordinator.query_classifier.get_fusion_weights(query_type)
                            expected_decision = "vector_only" if (weights['vector'] == 1.0 and weights['graph'] == 0.0) else "hybrid"
                            if expected_decision != routing_decision:
                                logger.warning(f"路由决策验证: 预期={expected_decision}, 实际={routing_decision}（使用实际值）")
                    except Exception as e:
                        logger.warning(f"获取路由详细信息失败: {e}")
            
            # 步骤2: 构建Prompt（根据路由决策选择模板）
            prompt_start = time.time()
            user_prompt, generation_mode = self._build_prompt_with_mode(question, generation_contexts, use_retrieval, routing_decision)
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
            gen_metadata = generation_result.get("metadata", {})
            
            # 确保routing_decision不为None
            if routing_decision is None:
                logger.warning(f"routing_decision为None，但all_retrieval_contexts有{len(all_retrieval_contexts)}个结果，推断为hybrid")
                routing_decision = "hybrid" if len(all_retrieval_contexts) == 10 else "vector_only"
            
            logger.info(f"准备格式化检索结果: routing_decision={routing_decision}, all_retrieval_contexts数量={len(all_retrieval_contexts)}")
            if all_retrieval_contexts:
                logger.info(f"all_retrieval_contexts类型: {type(all_retrieval_contexts)}, 第一个元素类型: {type(all_retrieval_contexts[0])}")
                logger.info(f"第一个元素内容示例: {str(all_retrieval_contexts[0])[:200] if all_retrieval_contexts else None}")
                if isinstance(all_retrieval_contexts[0], dict):
                    logger.info(f"第一个元素的keys: {list(all_retrieval_contexts[0].keys())}")
            
            # 格式化检索结果
            formatted_retrieval_results = self._format_retrieval_results(all_retrieval_contexts, routing_decision)
            
            # 格式化用于生成的文档（generation_contexts）
            # 注意：generation_contexts 在混合模式下是 3向量+5图谱（共8个）
            # 需要特殊处理，因为_format_retrieval_results默认认为前5个是向量
            if routing_decision == "hybrid" and len(generation_contexts) == 8:
                # 混合模式：前3个是向量，后5个是图谱
                formatted_generation_results = []
                for i, context in enumerate(generation_contexts):
                    if isinstance(context, str):
                        source = "vector" if i < 3 else "graph"
                        formatted_generation_results.append({
                            "content": context,
                            "fused_score": 1.0,
                            "source": source,
                            "source_scores": {},
                            "contributing_sources": [],
                            "entities": [],
                            "relationships": []
                        })
                    else:
                        # 字典格式，更新source字段
                        context_dict = context.copy() if isinstance(context, dict) else {"content": str(context)}
                        context_dict["source"] = "vector" if i < 3 else "graph"
                        formatted_generation_results.append(context_dict)
            else:
                # 纯向量或其他情况，使用标准格式化
                formatted_generation_results = self._format_retrieval_results(generation_contexts, routing_decision)
            
            # 验证格式化后的结果
            if formatted_retrieval_results:
                logger.info(f"格式化后第一个结果的keys: {list(formatted_retrieval_results[0].keys())}")
                logger.info(f"格式化后第一个结果的source: {formatted_retrieval_results[0].get('source')}")
            logger.info(f"格式化检索结果完成: results数量={len(formatted_retrieval_results)}")
            if formatted_retrieval_results:
                first_source = formatted_retrieval_results[0].get('source')
                all_sources = [r.get('source', 'MISSING') for r in formatted_retrieval_results]
                logger.info(f"第一个结果的source: {first_source}")
                logger.info(f"所有结果的source: {all_sources}")
                vector_count = sum(1 for s in all_sources if s == 'vector')
                graph_count = sum(1 for s in all_sources if s == 'graph')
                logger.info(f"source统计: vector={vector_count}, graph={graph_count}")
            
            # 统计用于生成的文档
            if formatted_generation_results:
                gen_vector_count = sum(1 for r in formatted_generation_results if r.get('source') == 'vector')
                gen_graph_count = sum(1 for r in formatted_generation_results if r.get('source') == 'graph')
                logger.info(f"用于生成的文档统计: vector={gen_vector_count}, graph={gen_graph_count}")
            
            result = {
                "success": True,
                "query": question,
                "answer": generation_result.get("answer", ""),
                "retrieval_results": formatted_retrieval_results,
                "metadata": {
                    "retrieval_time": round(retrieval_time, 2),
                    "generation_time": round(generation_time, 2),
                    "prompt_time": round(prompt_time, 3),
                    "total_time": round(total_time, 2),
                    "num_retrieval_results": len(all_retrieval_contexts),
                    "model": gen_metadata.get("model", "qwen-model"),
                    "temperature": temperature,
                    "use_retrieval": use_retrieval,
                    "query_expanded": self.enable_query_expansion,
                    "results_reranked": self.enable_rerank,
                    "routing_decision": routing_decision,
                    "routing_confidence": round(routing_confidence, 2) if routing_confidence else None,
                    "generation_mode": generation_mode,
                    "selected_for_generation": formatted_generation_results,  # 添加用于生成的文档
                    **gen_metadata
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
                                retrieval_results: Union[List[str], List[Dict[str, Any]]],
                                use_retrieval: bool,
                                routing_decision: str = None) -> Tuple[str, str]:
        """
        构建用户提示词（带模式选择）
        
        Args:
            query: 用户问题
            retrieval_results: 检索结果（List[str]或List[Dict]）
            use_retrieval: 是否使用检索
            routing_decision: 路由决策（"entity_driven" 或 "complex_reasoning"）
            
        Returns:
            (提示词, 生成模式)
        """
        if not use_retrieval or not retrieval_results:
            # 直接问答
            prompt = PromptTemplates.build_direct_prompt(query)
            return prompt, "default"
        
        # 如果是字符串列表，转换为字典格式（兼容PromptTemplates）
        if retrieval_results and isinstance(retrieval_results[0], str):
            # 转换为字典格式
            formatted_results = [{"content": text} for text in retrieval_results]
        else:
            formatted_results = retrieval_results
        
        # 根据路由决策选择不同的提示词模板
        if routing_decision == "vector_only":
            # 纯向量检索模式
            prompt = PromptTemplates.build_vector_prompt(
                query=query,
                retrieval_results=formatted_results,
                max_context_results=min(len(formatted_results), 3)
            )
            generation_mode = "vector"
            
        elif routing_decision == "hybrid":
            # 混合检索模式：前3个为向量，其余为图谱
            vector_results = formatted_results[:3]
            kg_results = formatted_results[3:] if len(formatted_results) > 3 else []
            
            # 使用混合模板
            if hasattr(PromptTemplates, 'build_hybrid_prompt'):
                prompt = PromptTemplates.build_hybrid_prompt(
                    query=query,
                    vector_results=vector_results,
                    kg_results=kg_results,
                    max_context_results=3
                )
            else:
                # 降级：使用RAG模板
                prompt = PromptTemplates.build_rag_prompt(
                    query=query,
                    retrieval_results=formatted_results,
                    max_context_results=len(formatted_results)
                )
            generation_mode = "hybrid"
            
        else:
            # 默认：使用传统RAG模板
            prompt = PromptTemplates.build_rag_prompt(
                query=query,
                retrieval_results=formatted_results,
                max_context_results=self.max_retrieval_results
            )
            generation_mode = "default"
        
        # 记录prompt长度
        estimated_tokens = PromptTemplates.estimate_tokens(prompt)
        logger.debug(f"构建Prompt完成 (模式: {generation_mode}), 估计token数: {estimated_tokens}")
        
        return prompt, generation_mode
    
    def _format_retrieval_results(self, 
                                  results: Union[List[str], List[Dict[str, Any]]],
                                  routing_decision: str = None) -> List[Dict[str, Any]]:
        """
        格式化检索结果用于返回
        
        Args:
            results: 原始检索结果（List[str]或List[Dict]）
            routing_decision: 路由决策（用于判断source）
            
        Returns:
            格式化后的结果
        """
        formatted = []
        
        if not results:
            return formatted
        
        logger.info(f"_format_retrieval_results: routing_decision={routing_decision}, results类型={type(results[0]) if results else None}, results数量={len(results)}")
        
        # 如果是字符串列表，转换为字典格式（参考hybrid_ragas_evaluator_v4的做法）
        if results and isinstance(results[0], str):
            logger.info(f"处理字符串列表，共{len(results)}个结果")
            for i, text in enumerate(results):
                # 根据路由决策和位置判断source
                source = "unknown"
                if routing_decision == "vector_only":
                    source = "vector"
                    logger.debug(f"位置{i}: routing_decision=vector_only -> source=vector")
                elif routing_decision == "hybrid":
                    # 混合模式：前5个是vector，后面5个是graph
                    if i < 5:
                        source = "vector"
                    elif i < 10:
                        source = "graph"
                    else:
                        source = "unknown"
                    logger.debug(f"位置{i}: routing_decision=hybrid -> source={source}")
                else:
                    # routing_decision为None或其他值时，根据结果数量推断
                    # 如果有10个结果，假设是混合模式（5向量+5图谱）
                    if len(results) == 10:
                        if i < 5:
                            source = "vector"
                        else:
                            source = "graph"
                        logger.debug(f"位置{i}: routing_decision=None，但结果数=10，推断source={source}")
                    elif len(results) == 3:
                        # 如果只有3个结果，假设是纯向量
                        source = "vector"
                        logger.debug(f"位置{i}: routing_decision=None，但结果数=3，推断source=vector")
                    else:
                        # 其他情况，根据位置推断（假设前一半是vector）
                        mid = len(results) // 2
                        if i < mid:
                            source = "vector"
                        else:
                            source = "graph"
                        logger.debug(f"位置{i}: routing_decision=None，根据位置推断source={source}")
                
                formatted_result = {
                    "content": text,
                    "fused_score": 1.0,
                    "source": source,  # 添加source字段（确保不为None）
                    "source_scores": {},
                    "contributing_sources": [],
                    "entities": [],
                    "relationships": []
                }
                
                # 再次确保source字段已设置
                if formatted_result.get("source") is None:
                    logger.error(f"位置{i}: 字符串列表格式化后source仍然是None！强制设置")
                    formatted_result["source"] = "vector" if i < 5 else "graph"
                
                formatted.append(formatted_result)
            
            logger.info(f"字符串列表处理完成，前5个source: {[r.get('source') for r in formatted[:5]]}, 后5个source: {[r.get('source') for r in formatted[5:]]}")
        else:
            # 字典格式
            logger.info(f"处理字典格式，共{len(results)}个结果，routing_decision={routing_decision}")
            for i, result in enumerate(results):
                # 获取原始source，可能是None、空字符串或不存在
                original_source = result.get("source")
                logger.debug(f"位置{i}: 原始source={original_source}, 字典keys={list(result.keys())}")
                
                # 确定最终的source值
                source = None  # 先设为None，后面一定会设置
                
                # 检查原字典是否有有效的source
                if original_source and original_source != "unknown" and original_source != "None":
                    # 如果原字典有有效的source，使用它
                    source = str(original_source)
                    logger.debug(f"位置{i}: 使用原字典的source={source}")
                else:
                    # 如果没有source或source无效，根据路由决策和位置推断
                    if routing_decision == "vector_only":
                        source = "vector"
                        logger.info(f"位置{i}: routing_decision=vector_only -> source=vector")
                    elif routing_decision == "hybrid":
                        # 混合模式：前5个是vector，后面5个是graph
                        if i < 5:
                            source = "vector"
                        elif i < 10:
                            source = "graph"
                        else:
                            source = "unknown"
                        logger.info(f"位置{i}: routing_decision=hybrid -> source={source}")
                    else:
                        # routing_decision为None时，根据结果数量推断
                        if len(results) == 10:
                            # 10个结果 = 混合模式（5向量+5图谱）
                            if i < 5:
                                source = "vector"
                            else:
                                source = "graph"
                            logger.info(f"位置{i}: routing_decision=None，但结果数=10，推断source={source}")
                        elif len(results) == 3:
                            # 3个结果 = 纯向量
                            source = "vector"
                            logger.info(f"位置{i}: routing_decision=None，但结果数=3，推断source=vector")
                        else:
                            # 其他情况，根据位置推断（假设前一半是vector）
                            mid = len(results) // 2
                            if i < mid:
                                source = "vector"
                            else:
                                source = "graph"
                            logger.info(f"位置{i}: routing_decision=None，根据位置推断source={source}")
                
                # 确保source不为None
                if source is None:
                    logger.error(f"位置{i}: source仍然是None，强制设置为unknown")
                    source = "unknown"
                
                # 构建格式化结果，确保所有字段都正确设置
                formatted_result = {
                    "content": result.get("content", ""),
                    "fused_score": result.get("fused_score", 0.0),
                    "source": source,  # 确保设置source字段
                    "source_scores": result.get("source_scores", {}),
                    "contributing_sources": result.get("contributing_sources", []),
                    "entities": result.get("entities", []),
                    "relationships": result.get("relationships", [])
                }
                
                # 最终验证：确保source字段一定不为None
                if formatted_result.get("source") is None or formatted_result.get("source") == "None":
                    logger.error(f"位置{i}: formatted_result的source为None或'None'！强制设置")
                    # 根据位置和结果数量推断
                    if len(results) == 10:
                        formatted_result["source"] = "vector" if i < 5 else "graph"
                    else:
                        formatted_result["source"] = "vector"  # 默认
                    logger.info(f"位置{i}: 强制设置source={formatted_result['source']}")
                
                formatted.append(formatted_result)
            
            # 验证所有结果的source字段
            sources_check = [r.get('source') for r in formatted]
            logger.info(f"字典格式处理完成，所有source: {sources_check}")
            logger.info(f"前5个source: {sources_check[:5]}, 后5个source: {sources_check[5:]}")
            
            # 统计
            none_count = sum(1 for s in sources_check if s is None)
            if none_count > 0:
                logger.error(f"警告：有{none_count}个结果的source字段为None！")
        
            logger.info(f"格式化完成，返回{len(formatted)}个结果")
            if formatted:
                sources = [r.get('source', 'MISSING') for r in formatted]
                logger.info(f"所有结果的source字段: {sources}")
                vector_count = sum(1 for s in sources if s == 'vector')
                graph_count = sum(1 for s in sources if s == 'graph')
                logger.info(f"source统计: vector={vector_count}, graph={graph_count}, unknown={len(sources) - vector_count - graph_count}")
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

