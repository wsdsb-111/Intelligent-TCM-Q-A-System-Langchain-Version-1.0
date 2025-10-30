"""
混合检索协调器
协调向量检索和知识图谱两个检索模块的并行执行和结果融合

注意：系统使用二元路由（vector_only / hybrid），不再使用独立的纯知识图谱检索
- vector_only（ENTITY_DRIVEN）：纯向量检索
- hybrid（COMPLEX_REASONING）：混合检索（向量+图谱）
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..models.data_models import (
    RetrievalResult, FusedResult, RetrievalConfig, ModuleConfig,
    RetrievalSource, FusionMethod, ModuleHealthStatus, SystemHealthStatus
)
from ..models.exceptions import (
    ModuleUnavailableError, TimeoutError as RetrievalTimeoutError,
    DataSourceError, handle_module_error, ErrorRecoveryStrategy
)
from ..core.result_fusion import ResultFusion
from ..utils.logging_utils import get_logger
from ..utils.query_classifier import get_query_classifier, QueryType


class HybridRetrievalCoordinator:
    """混合检索协调器类"""
    
    def __init__(self,
                 vector_adapter=None,
                 graph_adapter=None,
                 fusion_engine: Optional[ResultFusion] = None,
                 config: Optional[RetrievalConfig] = None,
                 module_config: Optional[ModuleConfig] = None,
                 use_intelligent_routing: bool = True,
                 intelligent_router_config: Optional[Dict] = None):
        """
        初始化混合检索协调器
        
        Args:
            vector_adapter: 向量检索适配器实例
            graph_adapter: 图检索适配器实例
            fusion_engine: 结果融合器实例
            config: 检索配置
            module_config: 模块配置
            use_intelligent_routing: 是否启用智能路由（默认True）
            intelligent_router_config: 智能路由器配置
        """
        
        self.vector_adapter = vector_adapter
        self.graph_adapter = graph_adapter
        
        # 初始化智能路由器
        self.use_intelligent_routing = use_intelligent_routing
        if use_intelligent_routing:
            # 从配置中提取智能路由器参数
            entity_csv_path = None
            qwen_api_config = None
            
            if intelligent_router_config:
                entity_csv_path = intelligent_router_config.get('entity_csv_path')
                qwen_api_config = intelligent_router_config.get('qwen_api_config')
            
            self.query_classifier = get_query_classifier(
                use_intelligent_routing=use_intelligent_routing,
                entity_csv_path=entity_csv_path,
                qwen_api_config=qwen_api_config
            )
        else:
            self.query_classifier = None
        
        # 初始化融合器
        self.fusion_engine = fusion_engine or ResultFusion()
        
        # 初始化配置
        self.config = config or RetrievalConfig()
        self.module_config = module_config or ModuleConfig()
        
        # 日志记录器
        self.logger = get_logger(__name__)
        
        # 模块状态跟踪
        self.module_status = {
            "vector": {"available": vector_adapter is not None, "last_error": None},
            "graph": {"available": graph_adapter is not None, "last_error": None}
        }
        
        # 性能统计
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "module_usage": {"vector": 0, "graph": 0},
            "average_response_time": 0.0,
            "fusion_method_usage": {},
            "error_counts": {"timeout": 0, "module_error": 0, "fusion_error": 0},
            "routing_decisions": {"vector_only": 0, "hybrid": 0}  # 路由统计（二元路由：vector_only 或 hybrid）
        }
        
        # 线程池用于并行执行
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="retrieval")
        
        self.logger.info("混合检索协调器初始化完成")
    
    async def retrieve(self, 
                      query: str, 
                      config: Optional[RetrievalConfig] = None) -> List[FusedResult]:
        """
        执行智能混合检索
        
        使用智能路由器对查询进行分类，根据分类结果选择最佳检索策略（二元路由）：
        - ENTITY_DRIVEN（实体主导型）：纯向量检索
        - COMPLEX_REASONING（复杂推理型）：混合检索（向量+图谱）
        
        Args:
            query: 检索查询
            config: 检索配置（可选，使用默认配置如果未提供）
            
        Returns:
            融合后的检索结果列表
        """
        start_time = time.time()
        retrieval_config = config or self.config
        
        try:
            self.stats["total_queries"] += 1
            self.logger.info(f"开始智能混合检索: '{query}'")
            
            # 智能路由分类
            routing_decision = None
            routing_confidence = 0.0
            
            if self.use_intelligent_routing and self.query_classifier:
                self.logger.info(f"使用智能路由: query_classifier={self.query_classifier is not None}")
                self.logger.info(f"智能路由器状态: {self.query_classifier.intelligent_router is not None}")

                try:
                    # 直接测试智能路由器
                    if self.query_classifier.intelligent_router:
                        route_result, confidence = self.query_classifier.intelligent_router.classify(query)
                        self.logger.info(f"智能路由器直接结果: route_result={route_result}, confidence={confidence}")

                        query_type, routing_confidence = self.query_classifier.classify_with_confidence(query)
                        self.logger.info(f"QueryClassifier结果: query_type={query_type}, confidence={routing_confidence}")

                        if query_type is None:
                            self.logger.warning("路由分类返回None，使用默认混合检索")
                            routing_decision = "hybrid"
                            results = await self._retrieve_hybrid(query, retrieval_config, None)
                        else:
                            weights = self.query_classifier.get_fusion_weights(query_type)
                            self.logger.info(f"路由分类: {query_type.value if query_type else 'None'}, 置信度: {routing_confidence:.2f}, "
                                           f"权重: vector={weights['vector']:.1f}, graph={weights['graph']:.1f}")

                            # 根据路由类型选择检索策略（二元路由：vector_only 或 hybrid）
                            if weights['vector'] == 1.0 and weights['graph'] == 0.0:
                                # 纯向量检索（ENTITY_DRIVEN）
                                routing_decision = "vector_only"
                                results = await self._retrieve_vector_only(query, retrieval_config)
                            else:
                                # 混合检索（COMPLEX_REASONING：向量50%图谱50%）
                                routing_decision = "hybrid"
                                results = await self._retrieve_hybrid(query, retrieval_config, weights)
                    else:
                        self.logger.warning("智能路由器未初始化，使用默认混合检索")
                        routing_decision = "hybrid"
                        results = await self._retrieve_hybrid(query, retrieval_config, None)
                except Exception as e:
                    self.logger.error(f"智能路由失败: {e}")
                    import traceback
                    self.logger.error(f"详细错误: {traceback.format_exc()}")
                    routing_decision = "hybrid"
                    results = await self._retrieve_hybrid(query, retrieval_config, None)
            else:
                # 不使用智能路由，执行传统混合检索
                self.logger.info("不使用智能路由，执行传统混合检索")
                routing_decision = "hybrid"
                results = await self._retrieve_hybrid(query, retrieval_config, None)
            
            # 更新路由统计
            if routing_decision:
                self.stats["routing_decisions"][routing_decision] = \
                    self.stats["routing_decisions"].get(routing_decision, 0) + 1
            
            # 检查是否有任何结果
            if not results:
                self.logger.warning(f"检索未返回结果: '{query}'")
                return []
            
            # 更新统计信息
            self._update_success_stats(start_time, retrieval_config.fusion_method)
            
            total_time = time.time() - start_time
            self.logger.info(f"智能混合检索完成: 查询='{query}', 路由={routing_decision}, "
                           f"结果数={len(results)}, 耗时={total_time:.3f}秒")
            
            # 为结果添加路由元数据
            for result in results:
                if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                    result.metadata['routing_decision'] = routing_decision
                    result.metadata['routing_confidence'] = routing_confidence
            
            return results
            
        except Exception as e:
            self._update_error_stats(e)
            error_msg = f"智能混合检索失败: {str(e)}"
            self.logger.error(error_msg)
            
            # 尝试降级策略
            fallback_results = await self._fallback_retrieve(query, retrieval_config)
            if fallback_results:
                self.logger.info(f"降级策略成功，返回{len(fallback_results)}个结果")
                return fallback_results
            
            raise handle_module_error("hybrid_coordinator", e)
    
    async def _retrieve_vector_only(self, 
                                   query: str, 
                                   config: RetrievalConfig,
                                   allow_fallback: bool = True) -> List[FusedResult]:
        """
        纯向量检索模式（不进行融合）
        
        Args:
            query: 检索查询
            config: 检索配置
            
        Returns:
            包装为FusedResult的向量检索结果
        """
        self.logger.debug(f"执行纯向量检索: '{query}'")
        
        try:
            vector_results = await self._safe_retrieve_vector(query, config)
            self.stats["module_usage"]["vector"] += 1
            
            # 将向量结果包装为FusedResult格式
            fused_results = [
                FusedResult(
                    content=r.content,
                    fused_score=r.score,
                    source_scores={"vector": r.score},
                    fusion_method=FusionMethod.WEIGHTED,
                    metadata=r.metadata.copy() if r.metadata else {},  # 确保metadata是可修改的副本
                    contributing_sources=[RetrievalSource.VECTOR],
                    entities=r.entities,
                    relationships=r.relationships,
                    timestamp=r.timestamp
                ) for r in vector_results
            ]
            
            self.logger.info(f"纯向量检索完成: 返回{len(fused_results)}个结果")
            
            # 零结果时返回空列表（不再降级到图谱检索）
            if len(fused_results) == 0:
                self.logger.warning(f"向量检索返回0结果")
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"纯向量检索失败: {e}")
            return []
    
    async def _retrieve_graph_only(self, 
                                  query: str, 
                                  config: RetrievalConfig,
                                  allow_fallback: bool = True) -> List[FusedResult]:
        """
        纯知识图谱检索模式（不进行融合）
        
        注意：此方法保留用于混合检索的降级逻辑，不再作为独立的路由类型使用
        
        Args:
            query: 检索查询
            config: 检索配置
            
        Returns:
            包装为FusedResult的图谱检索结果
        """
        self.logger.debug(f"执行纯知识图谱检索: '{query}'")
        
        try:
            graph_results = await self._safe_retrieve_graph(query, config)
            self.stats["module_usage"]["graph"] += 1
            
            # 将图谱结果包装为FusedResult格式
            fused_results = [
                FusedResult(
                    content=r.content,
                    fused_score=r.score,
                    source_scores={"graph": r.score},
                    fusion_method=FusionMethod.WEIGHTED,
                    metadata=r.metadata.copy() if r.metadata else {},  # 确保metadata是可修改的副本
                    contributing_sources=[RetrievalSource.GRAPH],
                    entities=r.entities,
                    relationships=r.relationships,
                    timestamp=r.timestamp
                ) for r in graph_results
            ]
            
            self.logger.info(f"纯知识图谱检索完成: 返回{len(fused_results)}个结果")
            
            # 零结果降级：如果图谱检索返回0结果，降级为向量检索
            if allow_fallback and len(fused_results) == 0 and self.vector_adapter and config.enable_vector:
                self.logger.warning(f"知识图谱检索返回0结果，降级为向量检索")
                return await self._retrieve_vector_only(query, config, allow_fallback=False)
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"纯知识图谱检索失败: {e}")
            # 异常降级：尝试向量检索
            if allow_fallback and self.vector_adapter and config.enable_vector:
                self.logger.warning(f"知识图谱检索异常，降级为向量检索")
                try:
                    return await self._retrieve_vector_only(query, config, allow_fallback=False)
                except:
                    return []
            return []
    
    async def _retrieve_hybrid(self, 
                              query: str, 
                              config: RetrievalConfig,
                              weights: Optional[Dict[str, float]] = None) -> List[FusedResult]:
        """
        混合检索模式（向量+图谱并行检索后融合）
        
        Args:
            query: 检索查询
            config: 检索配置
            weights: 自定义权重（可选）
            
        Returns:
            融合后的检索结果
        """
        self.logger.debug(f"执行混合检索（向量+图谱）: '{query}'")
        
        # 并行执行向量和图谱检索
        results_by_source = await self._parallel_retrieve_hybrid(query, config)
        
        self.logger.debug(f"并行检索返回的源: {list(results_by_source.keys())}")
        
        # 零结果降级处理
        vector_results = results_by_source.get("vector", [])
        graph_results = results_by_source.get("graph", [])
        
        self.logger.debug(f"向量结果数: {len(vector_results)}, 图谱结果数: {len(graph_results)}")
        
        # 如果向量检索返回0结果，但图谱有结果，则只使用图谱结果
        if len(vector_results) == 0 and len(graph_results) > 0:
            self.logger.warning(f"混合检索中向量返回0结果，仅使用知识图谱结果")
            # 直接将图谱结果包装为FusedResult，使用100%权重
            fused_results = [
                FusedResult(
                    content=r.content,
                    fused_score=r.score,
                    source_scores={"graph": r.score},
                    fusion_method=FusionMethod.WEIGHTED,
                    metadata=r.metadata,
                    contributing_sources=[RetrievalSource.GRAPH],
                    entities=r.entities,
                    relationships=r.relationships,
                    timestamp=r.timestamp
                ) for r in graph_results
            ]
            return fused_results
        
        # 如果图谱检索返回0结果，但向量有结果，则只使用向量结果
        if len(graph_results) == 0 and len(vector_results) > 0:
            self.logger.warning(f"混合检索中图谱返回0结果，仅使用向量结果")
            # 直接将向量结果包装为FusedResult，使用100%权重
            fused_results = [
                FusedResult(
                    content=r.content,
                    fused_score=r.score,
                    source_scores={"vector": r.score},
                    fusion_method=FusionMethod.WEIGHTED,
                    metadata=r.metadata,
                    contributing_sources=[RetrievalSource.VECTOR],
                    entities=r.entities,
                    relationships=r.relationships,
                    timestamp=r.timestamp
                ) for r in vector_results
            ]
            return fused_results
        
        # 如果两者都返回0结果
        if len(vector_results) == 0 and len(graph_results) == 0:
            self.logger.warning(f"混合检索未返回任何结果: '{query}'")
            return []
        
        # 如果提供了自定义权重，更新配置
        if weights:
            config_copy = RetrievalConfig(
                enable_vector=config.enable_vector,
                enable_graph=config.enable_graph,
                top_k=config.top_k,
                fusion_method=FusionMethod.WEIGHTED,  # 有自定义权重时强制使用WEIGHTED方法
                weights=weights,
                timeout=config.timeout
            )
        else:
            config_copy = config
        
        # 融合结果
        fused_results = self._fuse_results(results_by_source, config_copy)
        
        self.logger.info(f"混合检索完成: 返回{len(fused_results)}个结果")
        return fused_results
    
    async def _parallel_retrieve_hybrid(self, 
                                       query: str, 
                                       config: RetrievalConfig) -> Dict[str, List[RetrievalResult]]:
        """
        并行执行向量和图谱检索（混合模式）
        
        Args:
            query: 检索查询
            config: 检索配置
            
        Returns:
            各源检索结果字典
        """
        results_by_source = {}
        tasks = []
        
        # 创建检索任务（仅向量和图谱）
        if config.enable_vector and self.vector_adapter and self.module_status["vector"]["available"]:
            task = self._safe_retrieve_vector(query, config)
            tasks.append(("vector", task))
        
        if config.enable_graph and self.graph_adapter and self.module_status["graph"]["available"]:
            task = self._safe_retrieve_graph(query, config)
            tasks.append(("graph", task))
        
        # 并行执行任务
        if tasks:
            try:
                self.logger.debug(f"开始并行执行{len(tasks)}个检索任务: {[s for s, _ in tasks]}")
                
                # 使用asyncio.wait_for添加超时控制
                completed_tasks = await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                    timeout=config.timeout
                )
                
                self.logger.debug(f"并行任务完成，处理{len(completed_tasks)}个结果")
                
                # 处理结果
                for i, (source, _) in enumerate(tasks):
                    result = completed_tasks[i]
                    if isinstance(result, Exception):
                        self.logger.error(f"{source}检索失败: {str(result)}")
                        import traceback
                        self.logger.error(f"{source}详细错误: {traceback.format_exception(type(result), result, result.__traceback__)}")
                        self.module_status[source]["last_error"] = str(result)
                        # 不永久禁用模块，允许后续重试
                        # self.module_status[source]["available"] = False
                        results_by_source[source] = []
                    else:
                        result_count = len(result) if result else 0
                        self.logger.info(f"{source}检索成功: 返回{result_count}个结果")
                        if result_count > 0:
                            self.logger.debug(f"{source}第一个结果: {result[0]}")
                        results_by_source[source] = result or []
                        self.stats["module_usage"][source] += 1
                        self.module_status[source]["available"] = True
                        self.module_status[source]["last_error"] = None
                        
            except asyncio.TimeoutError:
                self.logger.error(f"检索超时: {config.timeout}秒")
                self.stats["error_counts"]["timeout"] += 1
                # 超时时返回已完成的结果
                for source, _ in tasks:
                    if source not in results_by_source:
                        results_by_source[source] = []
        
        return results_by_source
    
    async def _safe_retrieve_vector(self, query: str, config: RetrievalConfig) -> List[RetrievalResult]:
        """安全执行向量检索"""
        try:
            self.logger.debug(f"[Vector] 开始检索: query='{query}', top_k={config.top_k}")
            
            # 向量适配器的 search 方法是异步的
            results = await self.vector_adapter.search(query, top_k=config.top_k)
            
            self.logger.debug(f"[Vector] 原始结果数: {len(results) if results else 0}")
            if results:
                self.logger.debug(f"[Vector] 结果类型: {type(results)}, 第一个元素类型: {type(results[0])}")
            
            # 转换为标准格式
            converted = [self._convert_to_retrieval_result(r, RetrievalSource.VECTOR) for r in results]
            self.logger.debug(f"[Vector] 转换后结果数: {len(converted)}")
            
            return converted
            
        except Exception as e:
            self.logger.error(f"向量检索失败: {str(e)}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise
    
    async def _safe_retrieve_graph(self, query: str, config: RetrievalConfig) -> List[RetrievalResult]:
        """安全执行图检索"""
        try:
            self.logger.debug(f"[Graph] 开始检索: query='{query}', top_k={config.top_k}")
            
            # 图检索是异步的
            results = await self.graph_adapter.complex_query_search(query, top_k=config.top_k)
            
            self.logger.debug(f"[Graph] 检索结果数: {len(results) if results else 0}")
            
            # 结果已经是RetrievalResult格式
            return results
            
        except Exception as e:
            self.logger.error(f"图检索失败: {str(e)}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise
    
    def _convert_to_retrieval_result(self, result: Any, source: RetrievalSource) -> RetrievalResult:
        """将各模块的结果转换为标准RetrievalResult格式"""
        if isinstance(result, RetrievalResult):
            return result
        
        # 处理不同类型的结果
        if hasattr(result, 'to_retrieval_result'):
            return result.to_retrieval_result()
        
        # 基本转换
        if hasattr(result, 'content') and hasattr(result, 'score'):
            return RetrievalResult(
                content=result.content,
                score=result.score,
                source=source,
                metadata=getattr(result, 'metadata', {}),
                entities=getattr(result, 'entities', None),
                relationships=getattr(result, 'relationships', None),
                timestamp=datetime.now()
            )
        
        # 如果是字典格式
        if isinstance(result, dict):
            return RetrievalResult(
                content=result.get('content', str(result)),
                score=result.get('score', 1.0),
                source=source,
                metadata=result.get('metadata', {}),
                entities=result.get('entities', None),
                relationships=result.get('relationships', None),
                timestamp=datetime.now()
            )
        
        # 默认处理
        return RetrievalResult(
            content=str(result),
            score=1.0,
            source=source,
            metadata={},
            timestamp=datetime.now()
        )
    
    def _fuse_results(self, 
                     results_by_source: Dict[str, List[RetrievalResult]], 
                     config: RetrievalConfig) -> List[FusedResult]:
        """融合检索结果"""
        try:
            # 记录融合前各源的结果数量
            source_counts = {source: len(results) for source, results in results_by_source.items()}
            self.logger.debug(f"融合前各源结果数: {source_counts}")
            
            # 过滤空结果
            filtered_results = {
                source: results for source, results in results_by_source.items() 
                if results
            }
            
            if not filtered_results:
                self.logger.warning("融合前过滤后无有效结果")
                return []
            
            self.logger.debug(f"融合执行: 方法={config.fusion_method.value}, top_k={config.top_k}")
            
            # 执行融合
            fused_results = self.fusion_engine.fuse_results(
                filtered_results,
                method=config.fusion_method,
                weights=config.weights,
                top_k=config.top_k
            )
            
            # 更新融合方法使用统计
            method_name = config.fusion_method.value
            self.stats["fusion_method_usage"][method_name] = (
                self.stats["fusion_method_usage"].get(method_name, 0) + 1
            )
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"结果融合失败: {str(e)}")
            self.stats["error_counts"]["fusion_error"] += 1
            
            # 降级：返回第一个可用来源的结果
            for source, results in results_by_source.items():
                if results:
                    self.logger.info(f"融合失败，使用{source}的结果作为降级")
                    return [
                        FusedResult(
                            content=r.content,
                            fused_score=r.score,
                            source_scores={source: r.score},
                            fusion_method=FusionMethod.RRF,  # 默认方法
                            metadata=r.metadata,
                            contributing_sources=[r.source],
                            entities=r.entities,
                            relationships=r.relationships,
                            timestamp=datetime.now()
                        ) for r in results[:config.top_k]
                    ]
            
            return []   
 
    async def _fallback_retrieve(self, 
                               query: str, 
                               config: RetrievalConfig) -> List[FusedResult]:
        """降级检索策略（仅向量和图谱）"""
        try:
            self.logger.info("执行降级检索策略")
            
            # 尝试使用单个可用模块
            fallback_sources = []
            
            if config.enable_vector and self.vector_adapter and self.module_status["vector"]["available"]:
                fallback_sources.append("vector")
            if config.enable_graph and self.graph_adapter and self.module_status["graph"]["available"]:
                fallback_sources.append("graph")
            
            for source in fallback_sources:
                try:
                    if source == "vector":
                        results = await self._safe_retrieve_vector(query, config)
                    elif source == "graph":
                        results = await self._safe_retrieve_graph(query, config)
                    
                    if results:
                        # 转换为FusedResult格式
                        return [
                            FusedResult(
                                content=r.content,
                                fused_score=r.score,
                                source_scores={source: r.score},
                                fusion_method=FusionMethod.RRF,
                                metadata=r.metadata,
                                contributing_sources=[r.source],
                                entities=r.entities,
                                relationships=r.relationships,
                                timestamp=datetime.now()
                            ) for r in results[:config.top_k]
                        ]
                        
                except Exception as e:
                    self.logger.error(f"降级检索{source}失败: {str(e)}")
                    continue
            
            return []
            
        except Exception as e:
            self.logger.error(f"降级策略失败: {str(e)}")
            return []
    
    # 批量检索功能
    
    async def batch_retrieve(self, 
                           queries: List[str], 
                           config: Optional[RetrievalConfig] = None) -> Dict[str, List[FusedResult]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            config: 检索配置
            
        Returns:
            查询到结果的映射字典
        """
        retrieval_config = config or self.config
        results = {}
        
        self.logger.info(f"开始批量检索: {len(queries)}个查询")
        
        # 并发执行批量查询
        tasks = [self.retrieve(query, retrieval_config) for query in queries]
        
        try:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, query in enumerate(queries):
                result = completed_results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"查询'{query}'失败: {str(result)}")
                    results[query] = []
                else:
                    results[query] = result
            
            successful_count = sum(1 for r in results.values() if r)
            self.logger.info(f"批量检索完成: {successful_count}/{len(queries)}个查询成功")
            
        except Exception as e:
            self.logger.error(f"批量检索失败: {str(e)}")
            # 返回空结果
            results = {query: [] for query in queries}
        
        return results
    
    def batch_retrieve_sync(self, 
                          queries: List[str], 
                          config: Optional[RetrievalConfig] = None) -> Dict[str, List[FusedResult]]:
        """
        同步批量检索（为兼容性提供）
        """
        return asyncio.run(self.batch_retrieve(queries, config))
    
    # 健康检查和监控
    
    def get_health_status(self) -> SystemHealthStatus:
        """
        获取系统健康状态
        
        Returns:
            系统健康状态信息
        """
        module_statuses = []
        overall_healthy = True
        
        # 检查各模块状态
        for module_name, status in self.module_status.items():
            is_healthy = status["available"] and status["last_error"] is None
            
            module_status = ModuleHealthStatus(
                module_name=module_name,
                is_healthy=is_healthy,
                last_check=datetime.now(),
                error_message=status["last_error"],
                response_time=None  # 可以在实际检索时记录
            )
            
            module_statuses.append(module_status)
            
            if not is_healthy:
                overall_healthy = False
        
        return SystemHealthStatus(
            overall_healthy=overall_healthy,
            modules=module_statuses,
            timestamp=datetime.now()
        )
    
    async def health_check(self) -> SystemHealthStatus:
        """
        执行主动健康检查（仅检查向量和图谱模块）
        """
        self.logger.info("执行系统健康检查")
        
        # 测试查询
        test_query = "健康检查测试"
        
        # 检查各模块（仅向量和图谱）
        for module_name in ["vector", "graph"]:
            try:
                start_time = time.time()
                
                if module_name == "vector" and self.vector_adapter:
                    await self._safe_retrieve_vector(test_query, RetrievalConfig(top_k=1))
                elif module_name == "graph" and self.graph_adapter:
                    await self._safe_retrieve_graph(test_query, RetrievalConfig(top_k=1))
                
                response_time = time.time() - start_time
                
                # 更新状态
                self.module_status[module_name]["available"] = True
                self.module_status[module_name]["last_error"] = None
                
                self.logger.info(f"{module_name}模块健康检查通过，响应时间: {response_time:.3f}秒")
                
            except Exception as e:
                self.module_status[module_name]["available"] = False
                self.module_status[module_name]["last_error"] = str(e)
                self.logger.error(f"{module_name}模块健康检查失败: {str(e)}")
        
        return self.get_health_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            统计信息字典
        """
        total_queries = self.stats["total_queries"]
        success_rate = (
            self.stats["successful_queries"] / total_queries 
            if total_queries > 0 else 0.0
        )
        
        return {
            "total_queries": total_queries,
            "successful_queries": self.stats["successful_queries"],
            "failed_queries": self.stats["failed_queries"],
            "success_rate": success_rate,
            "average_response_time": self.stats["average_response_time"],
            "module_usage": self.stats["module_usage"].copy(),
            "fusion_method_usage": self.stats["fusion_method_usage"].copy(),
            "error_counts": self.stats["error_counts"].copy(),
            "routing_decisions": self.stats["routing_decisions"].copy(),  # 添加路由统计
            "module_status": {
                name: {
                    "available": status["available"],
                    "has_error": status["last_error"] is not None
                }
                for name, status in self.module_status.items()
            }
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "module_usage": {"vector": 0, "graph": 0},
            "average_response_time": 0.0,
            "fusion_method_usage": {},
            "error_counts": {"timeout": 0, "module_error": 0, "fusion_error": 0},
            "routing_decisions": {"vector_only": 0, "graph_only": 0, "hybrid": 0}
        }
        self.logger.info("统计信息已重置")
    
    # 配置管理
    
    def update_config(self, **kwargs):
        """
        更新检索配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"配置已更新: {key} = {value}")
        
        # 更新融合器配置
        if 'fusion_method' in kwargs or 'weights' in kwargs:
            self.logger.info("融合器配置已更新")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.to_dict()
    
    # 辅助方法
    
    def _update_success_stats(self, start_time: float, fusion_method: FusionMethod):
        """更新成功统计"""
        self.stats["successful_queries"] += 1
        
        # 更新平均响应时间
        response_time = time.time() - start_time
        total_queries = self.stats["successful_queries"]
        current_avg = self.stats["average_response_time"]
        
        self.stats["average_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def _update_error_stats(self, error: Exception):
        """更新错误统计"""
        self.stats["failed_queries"] += 1
        
        if isinstance(error, RetrievalTimeoutError):
            self.stats["error_counts"]["timeout"] += 1
        elif isinstance(error, (ModuleUnavailableError, DataSourceError)):
            self.stats["error_counts"]["module_error"] += 1
        else:
            self.stats["error_counts"]["fusion_error"] += 1
    
    # 上下文管理器支持
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
    
    def close(self):
        """关闭协调器，清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # 关闭各适配器（仅向量和图谱）
        if self.vector_adapter and hasattr(self.vector_adapter, 'close'):
            self.vector_adapter.close()
        if self.graph_adapter and hasattr(self.graph_adapter, 'close'):
            self.graph_adapter.close()
        
        self.logger.info("混合检索协调器已关闭")
    
    async def aclose(self):
        """异步关闭协调器"""
        self.close()
    
    def __repr__(self) -> str:
        """字符串表示"""
        available_modules = [
            name for name, status in self.module_status.items() 
            if status["available"]
        ]
        return f"HybridRetrievalCoordinator(modules={available_modules})"
# 降级和恢复策略
from enum import Enum
from dataclasses import dataclass
from typing import Callable
import threading


class FallbackStrategy(Enum):
    """降级策略枚举"""
    SKIP_MODULE = "skip_module"  # 跳过失败模块
    USE_CACHE = "use_cache"      # 使用缓存结果
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # 退避重试
    PARTIAL_RESULTS = "partial_results"  # 返回部分结果
    DEFAULT_RESPONSE = "default_response"  # 返回默认响应


class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    IMMEDIATE_RETRY = "immediate_retry"  # 立即重试
    SCHEDULED_RETRY = "scheduled_retry"  # 定时重试
    HEALTH_CHECK_BASED = "health_check_based"  # 基于健康检查
    MANUAL_RECOVERY = "manual_recovery"  # 手动恢复


@dataclass
class FallbackConfig:
    """降级配置"""
    strategy: FallbackStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 300  # 缓存生存时间（秒）
    partial_threshold: float = 0.5  # 部分结果阈值
    default_response: Any = None


@dataclass
class RecoveryConfig:
    """恢复配置"""
    strategy: RecoveryStrategy
    check_interval: float = 30.0  # 检查间隔（秒）
    max_recovery_attempts: int = 5
    recovery_timeout: float = 300.0  # 恢复超时（秒）


class ModuleHealthMonitor:
    """模块健康监控器"""
    
    def __init__(self):
        self.module_status = {}
        self.failure_counts = {}
        self.last_check_time = {}
        self.recovery_attempts = {}
        self.lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def update_module_status(self, module_name: str, is_healthy: bool, error: Exception = None):
        """更新模块状态"""
        with self.lock:
            self.module_status[module_name] = is_healthy
            self.last_check_time[module_name] = datetime.now()
            
            if not is_healthy:
                self.failure_counts[module_name] = self.failure_counts.get(module_name, 0) + 1
                self.logger.warning(f"模块 {module_name} 健康检查失败: {error}")
            else:
                self.failure_counts[module_name] = 0
                if module_name in self.recovery_attempts:
                    self.logger.info(f"模块 {module_name} 恢复正常")
                    del self.recovery_attempts[module_name]
    
    def is_module_healthy(self, module_name: str) -> bool:
        """检查模块是否健康"""
        with self.lock:
            return self.module_status.get(module_name, True)
    
    def get_failure_count(self, module_name: str) -> int:
        """获取模块失败次数"""
        with self.lock:
            return self.failure_counts.get(module_name, 0)
    
    def mark_recovery_attempt(self, module_name: str):
        """标记恢复尝试"""
        with self.lock:
            self.recovery_attempts[module_name] = datetime.now()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        with self.lock:
            return {
                "module_status": self.module_status.copy(),
                "failure_counts": self.failure_counts.copy(),
                "last_check_time": {k: v.isoformat() for k, v in self.last_check_time.items()},
                "recovery_attempts": {k: v.isoformat() for k, v in self.recovery_attempts.items()}
            }


class FallbackManager:
    """降级管理器"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.health_monitor = ModuleHealthMonitor()
        self.cache = {}  # 简单的内存缓存
        self.fallback_configs = {}
        self.recovery_configs = {}
        self.logger = get_logger(__name__)
        
        # 设置默认降级配置
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """设置默认配置"""
        default_fallback = FallbackConfig(
            strategy=FallbackStrategy.SKIP_MODULE,
            max_retries=2,
            retry_delay=1.0
        )
        
        default_recovery = RecoveryConfig(
            strategy=RecoveryStrategy.HEALTH_CHECK_BASED,
            check_interval=30.0,
            max_recovery_attempts=3
        )
        
        for module in ["vector", "graph"]:
            self.fallback_configs[module] = default_fallback
            self.recovery_configs[module] = default_recovery
    
    def set_fallback_config(self, module: str, config: FallbackConfig):
        """设置模块降级配置"""
        self.fallback_configs[module] = config
        self.logger.info(f"设置模块 {module} 降级配置: {config.strategy.value}")
    
    def set_recovery_config(self, module: str, config: RecoveryConfig):
        """设置模块恢复配置"""
        self.recovery_configs[module] = config
        self.logger.info(f"设置模块 {module} 恢复配置: {config.strategy.value}")
    
    async def execute_with_fallback(self, module_name: str, operation: Callable, *args, **kwargs):
        """执行带降级的操作"""
        config = self.fallback_configs.get(module_name, FallbackConfig(FallbackStrategy.SKIP_MODULE))
        
        # 检查模块健康状态
        if not self.health_monitor.is_module_healthy(module_name):
            return await self._handle_unhealthy_module(module_name, operation, *args, **kwargs)
        
        # 尝试执行操作
        for attempt in range(config.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                self.health_monitor.update_module_status(module_name, True)
                
                # 缓存成功结果
                if config.strategy == FallbackStrategy.USE_CACHE:
                    cache_key = self._generate_cache_key(module_name, args, kwargs)
                    self.cache[cache_key] = {
                        "result": result,
                        "timestamp": datetime.now(),
                        "ttl": config.cache_ttl
                    }
                
                return result
                
            except Exception as e:
                self.health_monitor.update_module_status(module_name, False, e)
                
                if attempt < config.max_retries:
                    if config.strategy == FallbackStrategy.RETRY_WITH_BACKOFF:
                        delay = config.retry_delay * (2 ** attempt)
                        self.logger.warning(f"模块 {module_name} 操作失败，{delay}秒后重试: {str(e)}")
                        await asyncio.sleep(delay)
                        continue
                
                # 最后一次尝试失败，执行降级策略
                return await self._execute_fallback_strategy(module_name, config, e, *args, **kwargs)
    
    async def _handle_unhealthy_module(self, module_name: str, operation: Callable, *args, **kwargs):
        """处理不健康的模块"""
        config = self.fallback_configs.get(module_name)
        
        if config.strategy == FallbackStrategy.USE_CACHE:
            cached_result = self._get_cached_result(module_name, args, kwargs)
            if cached_result:
                self.logger.info(f"模块 {module_name} 不健康，使用缓存结果")
                return cached_result
        
        # 尝试恢复模块
        await self._attempt_module_recovery(module_name)
        
        # 如果仍然不健康，执行降级策略
        if not self.health_monitor.is_module_healthy(module_name):
            error = Exception(f"模块 {module_name} 不可用")
            return await self._execute_fallback_strategy(module_name, config, error, *args, **kwargs)
        
        # 模块恢复，重新尝试操作
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            return await self._execute_fallback_strategy(module_name, config, e, *args, **kwargs)
    
    async def _execute_fallback_strategy(self, module_name: str, config: FallbackConfig, 
                                       error: Exception, *args, **kwargs):
        """执行降级策略"""
        self.logger.warning(f"执行模块 {module_name} 降级策略: {config.strategy.value}")
        
        if config.strategy == FallbackStrategy.SKIP_MODULE:
            return None  # 跳过该模块
        
        elif config.strategy == FallbackStrategy.USE_CACHE:
            cached_result = self._get_cached_result(module_name, args, kwargs)
            if cached_result:
                return cached_result
            else:
                return None  # 无缓存时跳过
        
        elif config.strategy == FallbackStrategy.PARTIAL_RESULTS:
            # 返回部分结果或空结果
            return []
        
        elif config.strategy == FallbackStrategy.DEFAULT_RESPONSE:
            return config.default_response
        
        else:
            # 默认跳过模块
            return None
    
    def _generate_cache_key(self, module_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        import hashlib
        key_data = f"{module_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, module_name: str, args: tuple, kwargs: dict):
        """获取缓存结果"""
        cache_key = self._generate_cache_key(module_name, args, kwargs)
        cached = self.cache.get(cache_key)
        
        if cached:
            # 检查缓存是否过期
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            if age < cached["ttl"]:
                self.logger.info(f"使用模块 {module_name} 的缓存结果")
                return cached["result"]
            else:
                # 缓存过期，删除
                del self.cache[cache_key]
        
        return None
    
    async def _attempt_module_recovery(self, module_name: str):
        """尝试模块恢复"""
        recovery_config = self.recovery_configs.get(module_name)
        if not recovery_config:
            return
        
        # 检查是否已经在恢复中
        if module_name in self.health_monitor.recovery_attempts:
            last_attempt = self.health_monitor.recovery_attempts[module_name]
            if (datetime.now() - last_attempt).total_seconds() < recovery_config.check_interval:
                return  # 还在恢复间隔内，不重复尝试
        
        self.health_monitor.mark_recovery_attempt(module_name)
        
        try:
            if recovery_config.strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                await self._immediate_recovery(module_name)
            elif recovery_config.strategy == RecoveryStrategy.HEALTH_CHECK_BASED:
                await self._health_check_recovery(module_name)
            elif recovery_config.strategy == RecoveryStrategy.SCHEDULED_RETRY:
                await self._scheduled_recovery(module_name, recovery_config)
        
        except Exception as e:
            self.logger.error(f"模块 {module_name} 恢复失败: {str(e)}")
    
    async def _immediate_recovery(self, module_name: str):
        """立即恢复"""
        adapter = getattr(self.coordinator, f"{module_name}_adapter", None)
        if adapter and hasattr(adapter, 'health_check'):
            try:
                is_healthy = await adapter.health_check()
                self.health_monitor.update_module_status(module_name, is_healthy)
            except Exception as e:
                self.health_monitor.update_module_status(module_name, False, e)
    
    async def _health_check_recovery(self, module_name: str):
        """基于健康检查的恢复"""
        await self._immediate_recovery(module_name)
    
    async def _scheduled_recovery(self, module_name: str, config: RecoveryConfig):
        """定时恢复"""
        # 这里可以实现更复杂的定时恢复逻辑
        await asyncio.sleep(config.check_interval)
        await self._immediate_recovery(module_name)
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """获取降级统计"""
        return {
            "health_summary": self.health_monitor.get_health_summary(),
            "cache_size": len(self.cache),
            "fallback_configs": {
                module: {
                    "strategy": config.strategy.value,
                    "max_retries": config.max_retries,
                    "retry_delay": config.retry_delay
                }
                for module, config in self.fallback_configs.items()
            },
            "recovery_configs": {
                module: {
                    "strategy": config.strategy.value,
                    "check_interval": config.check_interval,
                    "max_recovery_attempts": config.max_recovery_attempts
                }
                for module, config in self.recovery_configs.items()
            }
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.logger.info("降级缓存已清空")


# 在HybridRetrievalCoordinator类中添加降级管理器
def add_fallback_support_to_coordinator():
    """为协调器添加降级支持"""
    
    # 这个函数会在协调器初始化时调用
    def init_fallback_manager(self):
        """初始化降级管理器"""
        if not hasattr(self, 'fallback_manager'):
            self.fallback_manager = FallbackManager(self)
    
    # 修改检索方法以支持降级
    async def retrieve_with_fallback(self, query: str, config: RetrievalConfig = None) -> List[FusedResult]:
        """带降级的检索方法"""
        if not hasattr(self, 'fallback_manager'):
            self.init_fallback_manager()
        
        results = []
        
        
        # 向量检索
        if config.enable_vector and self.vector_adapter:
            vector_result = await self.fallback_manager.execute_with_fallback(
                "vector",
                self._safe_vector_search,
                query, config.top_k
            )
            if vector_result:
                results.extend(vector_result)
        
        # 图检索
        if config.enable_graph and self.graph_adapter:
            graph_result = await self.fallback_manager.execute_with_fallback(
                "graph",
                self._safe_graph_search,
                query, config.top_k
            )
            if graph_result:
                results.extend(graph_result)
        
        # 融合结果
        if results:
            return await self.fusion_engine.fuse_results(results, config.fusion_method)
        else:
            # 所有模块都失败，返回空结果或默认结果
            self.logger.warning("所有检索模块都不可用，返回空结果")
            return []
    
    # 安全的模块调用方法
    
    async def _safe_vector_search(self, query: str, top_k: int):
        """安全的向量搜索"""
        return await self.vector_adapter.search(query, top_k)
    
    async def _safe_graph_search(self, query: str, top_k: int):
        """安全的图搜索"""
        return await self.graph_adapter.search(query, top_k)
    
    # 将方法添加到HybridRetrievalCoordinator类
    HybridRetrievalCoordinator.init_fallback_manager = init_fallback_manager
    HybridRetrievalCoordinator.retrieve_with_fallback = retrieve_with_fallback
    HybridRetrievalCoordinator._safe_vector_search = _safe_vector_search
    HybridRetrievalCoordinator._safe_graph_search = _safe_graph_search


# 调用函数添加降级支持
add_fallback_support_to_coordinator()