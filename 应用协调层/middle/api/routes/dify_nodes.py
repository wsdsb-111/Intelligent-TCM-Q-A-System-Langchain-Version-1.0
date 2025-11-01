"""
Dify工作流节点路由
实现Dify工作流节点专用的API端点
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any, List, Tuple
import time
import asyncio

from ..schemas.dify_schemas import (
    DifyRetrievalRequest, DifyRetrievalResponse,
    DifyExpandRerankRequest, DifyExpandRerankResponse,
    DifyGenerateAnswerRequest, DifyGenerateAnswerResponse,
    DocumentSchema, RouterType
)
from ...models.data_models import RetrievalConfig
from ...utils.logging_utils import get_logger
from ...services.model_service import get_model_service
from ...services.rag_chain import RAGChain
from ...services.prompt_templates import PromptTemplates
from ...core.retrieval_coordinator import HybridRetrievalCoordinator
from ...utils.local_enhancer import LocalQueryExpander, LocalReranker, create_local_expander, create_local_reranker
import yaml
import os

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/dify", tags=["Dify节点"])

# 全局变量（将在应用启动时初始化）
_rag_chain: RAGChain = None
_retrieval_coordinator: HybridRetrievalCoordinator = None
_model_service = None
_prompt_templates = None
_expander: LocalQueryExpander = None
_reranker: LocalReranker = None


def get_rag_chain() -> RAGChain:
    """获取RAG链路依赖"""
    if _rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG链路未初始化"
        )
    return _rag_chain


def get_retrieval_coordinator() -> HybridRetrievalCoordinator:
    """获取检索协调器依赖"""
    if _retrieval_coordinator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="检索协调器未初始化"
        )
    return _retrieval_coordinator


def get_model_service_dependency():
    """获取模型服务依赖"""
    if _model_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型服务未初始化"
        )
    return _model_service


def _format_documents_for_response(documents: List[str], routing_decision: str, start_index: int = 0) -> List[DocumentSchema]:
    """
    格式化文档列表为DocumentSchema格式
    
    Args:
        documents: 文档字符串列表
        routing_decision: 路由决策（vector_only 或 hybrid）
        start_index: 起始索引（用于判断source）
    
    Returns:
        格式化后的文档列表
    """
    formatted = []
    for i, doc in enumerate(documents):
        # 根据路由决策和位置判断source
        if routing_decision == "vector_only":
            source = "vector"
        elif routing_decision == "hybrid":
            # 混合模式：前5个是vector，后5个是graph
            source = "vector" if (start_index + i) < 5 else "graph"
        else:
            source = "unknown"
        
        formatted.append(DocumentSchema(
            content=doc,
            source=source,
            fused_score=1.0,
            source_scores={},
            contributing_sources=[],
            entities=[],
            relationships=[]
        ))
    return formatted


@router.post("/retrieve_documents", 
            response_model=DifyRetrievalResponse,
            summary="检索与知识召回节点",
            description="执行文档检索（使用已全量加载的组件），根据路由类型执行精确召回规则")
async def retrieve_documents(
    request: DifyRetrievalRequest,
    coordinator: HybridRetrievalCoordinator = Depends(get_retrieval_coordinator)
) -> DifyRetrievalResponse:
    """
    Dify检索与知识召回节点
    
    - **query**: 用户查询
    - **router_type**: 路由类型（vector_only 或 hybrid）
    - **config**: 可选的检索配置
    
    精确召回规则：
    - vector_only: 召回3个向量文档，生成使用3个文档
    - hybrid: 召回5向量+5图谱（10个），生成用3向量+5图谱（8个）
    """
    try:
        logger.info(f"Dify检索节点: query='{request.query}', router_type={request.router_type.value}")
        start_time = time.time()
        
        # 根据路由类型执行精确召回规则
        if request.router_type == RouterType.VECTOR_ONLY:
            # 纯向量检索：召回3个，使用3个
            retrieval_config = RetrievalConfig(
                enable_vector=True,
                enable_graph=False,
                top_k=3  # 召回3个
            )
            
            # 执行检索（返回格式：Tuple[List[str], List[str]]）
            retrieve_result = await coordinator.retrieve(request.query, retrieval_config)
            generation_contexts, evaluation_contexts = retrieve_result
            all_retrieval_contexts = generation_contexts  # 总召回3个
            
            logger.info(f"纯向量检索完成: 召回{len(all_retrieval_contexts)}个，用于生成{len(generation_contexts)}个")
            
        else:  # hybrid
            # 混合检索：向量召回5个，图谱召回5个
            retrieval_config = RetrievalConfig(
                enable_vector=True,
                enable_graph=True,
                top_k=5  # 向量和图谱各召回5个
            )
            
            # 执行检索（返回格式：Tuple[List[str], List[str], List[str]]）
            retrieve_result = await coordinator.retrieve(request.query, retrieval_config)
            if len(retrieve_result) == 3:
                generation_contexts, all_retrieval_contexts, evaluation_contexts = retrieve_result
            else:
                # 兼容旧格式
                generation_contexts, evaluation_contexts = retrieve_result
                all_retrieval_contexts = generation_contexts
            
            logger.info(f"混合检索完成: 总召回{len(all_retrieval_contexts)}个（预期10个），用于生成{len(generation_contexts)}个（预期8个）")
        
        # 格式化文档（添加source字段）
        formatted_all_docs = _format_documents_for_response(all_retrieval_contexts, request.router_type.value)
        formatted_gen_docs = _format_documents_for_response(generation_contexts, request.router_type.value)
        
        # 统计信息
        vector_count = sum(1 for d in formatted_all_docs if d.source == "vector")
        graph_count = sum(1 for d in formatted_all_docs if d.source == "graph")
        
        retrieval_time = time.time() - start_time
        
        return DifyRetrievalResponse(
            success=True,
            documents=formatted_all_docs,
            generation_documents=formatted_gen_docs,
            routing_decision=request.router_type.value,
            retrieval_stats={
                "total_recalled": len(formatted_all_docs),
                "for_generation": len(formatted_gen_docs),
                "vector_count": vector_count,
                "graph_count": graph_count,
                "retrieval_time": round(retrieval_time, 2)
            }
        )
    
    except Exception as e:
        logger.error(f"Dify检索节点错误: {e}", exc_info=True)
        return DifyRetrievalResponse(
            success=False,
            error=str(e)
        )


@router.post("/expand_and_rerank",
            response_model=DifyExpandRerankResponse,
            summary="查询扩展与重排序节点",
            description="并行执行查询扩展和重排序（使用已全量加载的组件）")
async def expand_and_rerank(
    request: DifyExpandRerankRequest
) -> DifyExpandRerankResponse:
    """
    Dify查询扩展与重排序节点
    
    - **query**: 原始查询
    - **documents**: 待处理的文档列表
    - **parallel**: 是否并行执行
    
    使用已全量加载的查询扩展和重排序组件
    """
    try:
        logger.info(f"Dify扩展重排序节点: query='{request.query}', documents={len(request.documents)}个")
        start_time = time.time()
        
        # 检查组件是否已加载
        if _expander is None or _reranker is None:
            logger.warning("查询扩展或重排序组件未加载，返回原始文档")
            return DifyExpandRerankResponse(
                success=True,
                expanded_queries=[request.query],
                reranked_documents=request.documents
            )
        
        # 提取文档内容（用于重排序）
        document_contents = [doc.content for doc in request.documents]
        
        # 并行执行扩展和重排序
        if request.parallel:
            # 查询扩展（异步执行，但实际上expand是同步的）
            expanded_queries_task = asyncio.to_thread(_expander.expand, request.query, max_expansions=3)
            
            # 重排序（异步执行）
            rerank_task = asyncio.to_thread(_reranker.rerank, request.query, document_contents, top_k=len(document_contents))
            
            # 等待两个任务完成
            expanded_queries_result, rerank_result = await asyncio.gather(
                expanded_queries_task,
                rerank_task,
                return_exceptions=True
            )
            
            # 处理扩展结果
            if isinstance(expanded_queries_result, Exception):
                logger.error(f"查询扩展失败: {expanded_queries_result}")
                expanded_queries = [request.query]
            else:
                expanded_queries = expanded_queries_result if expanded_queries_result else [request.query]
            
            # 处理重排序结果
            if isinstance(rerank_result, Exception):
                logger.error(f"重排序失败: {rerank_result}")
                reranked_documents = request.documents  # 返回原始顺序
            else:
                # rerank_result是List[Tuple[int, float]]格式（索引，分数）
                # 需要根据索引重新排序文档
                if rerank_result and len(rerank_result) > 0:
                    # 按分数降序排序
                    sorted_indices = sorted(rerank_result, key=lambda x: x[1], reverse=True)
                    # 根据索引重新组织文档
                    reranked_documents = [request.documents[idx] for idx, _ in sorted_indices]
                else:
                    reranked_documents = request.documents
        else:
            # 串行执行
            expanded_queries = await asyncio.to_thread(_expander.expand, request.query, max_expansions=3)
            if not expanded_queries:
                expanded_queries = [request.query]
            
            rerank_result = await asyncio.to_thread(_reranker.rerank, request.query, document_contents, top_k=len(document_contents))
            if rerank_result and len(rerank_result) > 0:
                sorted_indices = sorted(rerank_result, key=lambda x: x[1], reverse=True)
                reranked_documents = [request.documents[idx] for idx, _ in sorted_indices]
            else:
                reranked_documents = request.documents
        
        processing_time = time.time() - start_time
        logger.info(f"Dify扩展重排序完成: 扩展查询{len(expanded_queries)}个, 重排序文档{len(reranked_documents)}个, 耗时{processing_time:.2f}秒")
        
        return DifyExpandRerankResponse(
            success=True,
            expanded_queries=expanded_queries,
            reranked_documents=reranked_documents
        )
    
    except Exception as e:
        logger.error(f"Dify扩展重排序节点错误: {e}", exc_info=True)
        return DifyExpandRerankResponse(
            success=False,
            error=str(e)
        )


@router.post("/generate_answer",
            response_model=DifyGenerateAnswerResponse,
            summary="回答生成节点",
            description="基于文档生成答案（使用已全量加载的组件）")
async def generate_answer(
    request: DifyGenerateAnswerRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
) -> DifyGenerateAnswerResponse:
    """
    Dify回答生成节点
    
    - **query**: 用户查询
    - **documents**: 已选择用于生成的文档（3或8个）
    - **routing_decision**: 路由决策（vector_only 或 hybrid）
    - **generation_params**: 生成参数（与评估系统一致）
    """
    try:
        logger.info(f"Dify生成节点: query='{request.query}', documents={len(request.documents)}个, routing={request.routing_decision.value}")
        start_time = time.time()
        
        # 将DocumentSchema转换为字典格式（用于提示词构建）
        document_dicts = [doc.dict() for doc in request.documents]
        
        # 使用RAGChain的提示词构建逻辑
        # 根据路由决策选择提示词模板
        prompt_templates = PromptTemplates()
        
        # 构建提示词（使用RAGChain的内部方法）
        if request.routing_decision == RouterType.VECTOR_ONLY:
            # 纯向量模式：所有文档都是向量文档
            prompt = prompt_templates.build_vector_prompt(
                query=request.query,
                retrieval_results=document_dicts,
                max_context_results=len(document_dicts)
            )
            generation_mode = "vector"
        else:  # hybrid
            # 混合模式：需要分开向量和图谱文档
            vector_results = [d for d in document_dicts if d.get('source') == 'vector']
            kg_results = [d for d in document_dicts if d.get('source') == 'graph']
            
            prompt = prompt_templates.build_hybrid_prompt(
                query=request.query,
                vector_results=vector_results,
                kg_results=kg_results,
                max_context_results=3  # 向量3个，图谱5个
            )
            generation_mode = "hybrid"
        
        # 使用模型服务生成答案
        model_service = get_model_service()
        
        generation_result = model_service.generate(
            query=prompt,
            system_prompt=None,
            max_new_tokens=request.generation_params.max_new_tokens,
            temperature=request.generation_params.temperature,
            top_p=request.generation_params.top_p,
            repetition_penalty=request.generation_params.repetition_penalty,
            num_beams=request.generation_params.num_beams,
            do_sample=request.generation_params.do_sample,
            length_penalty=request.generation_params.length_penalty,
            min_new_tokens=request.generation_params.min_new_tokens,
            no_repeat_ngram_size=request.generation_params.no_repeat_ngram_size,
            early_stopping=request.generation_params.early_stopping,
            use_cache=request.generation_params.use_cache
        )
        
        generation_time = time.time() - start_time
        
        answer = generation_result.get("answer", "")
        gen_metadata = generation_result.get("metadata", {})
        
        return DifyGenerateAnswerResponse(
            success=True,
            answer=answer,
            metadata={
                "routing_decision": request.routing_decision.value,
                "documents_used": len(request.documents),
                "generation_params": request.generation_params.dict(),
                "model": gen_metadata.get("model", "qwen3-1.7b-finetuned"),
                "generation_time": round(generation_time, 2),
                "generation_mode": generation_mode,
                "selected_for_generation": [doc.dict() for doc in request.documents]  # 包含source字段的文档
            }
        )
    
    except Exception as e:
        logger.error(f"Dify生成节点错误: {e}", exc_info=True)
        return DifyGenerateAnswerResponse(
            success=False,
            error=str(e)
        )


def init_dify_routes(
    rag_chain: RAGChain,
    retrieval_coordinator: HybridRetrievalCoordinator
):
    """
    初始化Dify节点路由的全局依赖
    
    Args:
        rag_chain: RAG链路实例
        retrieval_coordinator: 检索协调器实例
    """
    global _rag_chain, _retrieval_coordinator, _model_service, _prompt_templates, _expander, _reranker
    
    _rag_chain = rag_chain
    _retrieval_coordinator = retrieval_coordinator
    _model_service = get_model_service()
    _prompt_templates = PromptTemplates()
    
    # 加载查询扩展和重排序组件（全量加载）
    try:
        # 从配置文件读取模型路径
        # 路径解析：dify_nodes.py在 middle/api/routes/
        # 需要回到 middle/config/service_config.yaml
        # __file__: 应用协调层/middle/api/routes/dify_nodes.py
        # 向上两级: 应用协调层/middle/
        middle_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(middle_dir, "config", "service_config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 辅助函数：解析相对路径（参照main_app.py的方式）
            def resolve_path(relative_path: str) -> str:
                """解析相对路径为绝对路径"""
                if not relative_path:
                    return None
                if os.path.isabs(relative_path):
                    return relative_path
                # 从config目录解析（参照main_app.py的resolve_path逻辑）
                config_dir = os.path.dirname(config_path)
                # 相对路径通常是相对于项目根目录的，需要特殊处理
                if relative_path.startswith('../../../'):
                    # 向上三级到项目根目录，然后拼接剩余路径
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(config_dir)))
                    remaining_path = relative_path.replace('../../../', '')
                    resolved = os.path.abspath(os.path.join(project_root, remaining_path))
                else:
                    # 相对于config目录
                    resolved = os.path.abspath(os.path.join(config_dir, relative_path))
                return resolved
            
            # 加载查询扩展器
            query_expansion_config = config.get('retrieval', {}).get('enhancement', {}).get('query_expansion', {})
            if query_expansion_config.get('enabled', False):
                expander_path = query_expansion_config.get('model_path')
                if expander_path:
                    expander_path = resolve_path(expander_path)
                    if expander_path and os.path.exists(expander_path):
                        _expander = create_local_expander(expander_path)
                        logger.info(f"✅ 查询扩展器已加载: {expander_path}")
                    else:
                        logger.warning(f"查询扩展模型路径不存在: {expander_path}")
            
            # 加载重排序器
            reranking_config = config.get('retrieval', {}).get('enhancement', {}).get('reranking', {})
            if reranking_config.get('enabled', False):
                reranker_path = reranking_config.get('model_path')
                if reranker_path:
                    reranker_path = resolve_path(reranker_path)
                    if reranker_path and os.path.exists(reranker_path):
                        _reranker = create_local_reranker(reranker_path)
                        logger.info(f"✅ 重排序器已加载: {reranker_path}")
                    else:
                        logger.warning(f"重排序模型路径不存在: {reranker_path}")
        else:
            logger.warning(f"配置文件不存在: {config_path}，查询扩展和重排序组件将不可用")
    
    except Exception as e:
        logger.error(f"加载查询扩展和重排序组件失败: {e}", exc_info=True)
        # 不抛出异常，允许路由继续工作，只是扩展和重排序功能不可用
    
    logger.info("Dify节点路由初始化完成")

