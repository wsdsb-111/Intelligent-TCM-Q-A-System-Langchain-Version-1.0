"""
API v1 路由
实现主要的API端点
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any
import time
import asyncio

from .schemas import (
    QueryRequest, QueryResponse,
    RetrievalRequest, RetrievalResponse,
    HealthStatus, MultimodalRequest, MultimodalResponse,
    BatchQueryRequest, BatchQueryResponse,
    ErrorResponse
)
from ..services.model_service import get_model_service
from ..services.rag_chain import RAGChain
from ..core.retrieval_coordinator import HybridRetrievalCoordinator
from ..models.data_models import RetrievalConfig, FusionMethod
from ..utils.logging_utils import get_logger
from ..utils.query_classifier import get_query_classifier

logger = get_logger(__name__)
query_classifier = get_query_classifier()

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["v1"])

# 全局变量（将在应用启动时初始化）
_rag_chain: RAGChain = None
_retrieval_coordinator: HybridRetrievalCoordinator = None
_app_start_time = time.time()


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


@router.post("/query",
            response_model=QueryResponse,
            summary="问答接口",
            description="基于RAG的中医问答接口，返回生成的答案和检索结果")
async def query_endpoint(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
) -> QueryResponse:
    """
    智能路由问答接口（v4.0）
    
    - **query**: 用户问题（必填）
    - **top_k**: 返回的检索结果数量（默认5）
    - **enable_bm25**: 已废弃（BM25已完全移除）
    - **enable_vector**: 是否启用向量检索（默认true）
    - **enable_graph**: 是否启用知识图谱检索（默认true）
    - **fusion_method**: 融合方法（默认weighted）
    - **temperature**: 生成温度（默认0.7）
    - **max_new_tokens**: 最大生成token数（默认512）
    
    系统会自动使用BERT智能路由选择最佳检索策略
    """
    try:
        logger.info(f"收到问答请求: {request.query}")
        
        # 如果使用加权融合且未指定权重，使用查询分类器自动推荐
        weights = request.weights
        if request.fusion_method.value == "weighted" and weights is None:
            query_type, recommended_weights = query_classifier.classify_and_get_weights(request.query)
            weights = recommended_weights
            logger.info(f"查询类型: {query_type.value}, 自动推荐权重: {weights}")
        
        # 构建检索配置
        retrieval_config = RetrievalConfig(
            enable_vector=request.enable_vector,
            enable_graph=request.enable_graph,
            top_k=request.top_k,
            fusion_method=FusionMethod(request.fusion_method.value),
            weights=weights
        )
        
        # 执行RAG问答
        result = await rag_chain.query(
            question=request.query,
            retrieval_config=retrieval_config,
            use_retrieval=True,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens
        )
        
        # 转换为响应格式
        if result["success"]:
            return QueryResponse(
                success=True,
                query=result["query"],
                answer=result["answer"],
                retrieval_results=result["retrieval_results"],
                metadata=result["metadata"]
            )
        else:
            return QueryResponse(
                success=False,
                query=result["query"],
                answer=None,
                error=result.get("error", "未知错误"),
                metadata=result.get("metadata")
            )
    
    except Exception as e:
        logger.error(f"问答接口错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答处理失败: {str(e)}"
        )


@router.post("/retrieve",
            response_model=RetrievalResponse,
            summary="纯检索接口",
            description="仅执行检索，不生成答案")
async def retrieve_endpoint(
    request: RetrievalRequest,
    coordinator: HybridRetrievalCoordinator = Depends(get_retrieval_coordinator)
) -> RetrievalResponse:
    """
    纯检索接口
    
    - **query**: 检索查询（必填）
    - **top_k**: 返回结果数量（默认10）
    - **fusion_method**: 融合方法（默认rrf）
    """
    try:
        logger.info(f"收到检索请求: {request.query}")
        
        start_time = time.time()
        
        # 如果使用加权融合且未指定权重，使用查询分类器自动推荐
        weights = request.weights
        if request.fusion_method.value == "weighted" and weights is None:
            query_type, recommended_weights = query_classifier.classify_and_get_weights(request.query)
            weights = recommended_weights
            logger.info(f"查询类型: {query_type.value}, 自动推荐权重: {weights}")
        
        # 构建检索配置
        config = RetrievalConfig(
            enable_vector=request.enable_vector,
            enable_graph=request.enable_graph,
            top_k=request.top_k,
            fusion_method=FusionMethod(request.fusion_method.value),
            weights=weights
        )
        
        # 执行检索
        results = await coordinator.retrieve(request.query, config)
        
        retrieval_time = time.time() - start_time
        
        # 转换为响应格式
        results_dict = [r.to_dict() for r in results]
        
        # 确定使用的检索源（BM25已移除）
        sources_used = []
        if request.enable_vector:
            sources_used.append("vector")
        if request.enable_graph:
            sources_used.append("graph")
        
        return RetrievalResponse(
            success=True,
            query=request.query,
            results=results_dict,
            metadata={
                "retrieval_time": round(retrieval_time, 2),
                "num_results": len(results_dict),
                "fusion_method": request.fusion_method.value,
                "sources_used": sources_used
            }
        )
    
    except Exception as e:
        logger.error(f"检索接口错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索失败: {str(e)}"
        )


@router.get("/health",
           response_model=HealthStatus,
           summary="健康检查",
           description="检查服务健康状态和各模块可用性")
async def health_check() -> HealthStatus:
    """
    健康检查接口
    
    返回系统健康状态、模型加载情况、检索模块状态等
    """
    try:
        model_service = get_model_service()
        
        # 检查模型是否加载
        model_loaded = model_service._initialized
        
        # 检查检索模块状态
        retrieval_modules = {}
        if _retrieval_coordinator:
            module_status = _retrieval_coordinator.module_status
            for module, status_info in module_status.items():
                retrieval_modules[module] = "available" if status_info["available"] else "unavailable"
        else:
            retrieval_modules = {
                "bm25": "unknown",
                "vector": "unknown",
                "graph": "unknown"
            }
        
        # 计算运行时长
        uptime_seconds = int(time.time() - _app_start_time)
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        seconds = uptime_seconds % 60
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        
        # GPU信息
        gpu_memory = None
        if model_loaded and model_service.device == "cuda":
            import torch
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory = f"{allocated:.2f}GB / {total:.1f}GB"
        
        # 确定整体状态
        if model_loaded and all(v == "available" for v in retrieval_modules.values()):
            overall_status = "healthy"
        elif model_loaded:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # 获取统计信息
        stats = {}
        if _rag_chain:
            stats["rag"] = _rag_chain.get_stats()
        if model_loaded:
            stats["model"] = model_service.get_stats()
        
        return HealthStatus(
            status=overall_status,
            model_loaded=model_loaded,
            retrieval_modules=retrieval_modules,
            gpu_memory=gpu_memory,
            uptime=uptime_str,
            stats=stats if stats else None
        )
    
    except Exception as e:
        logger.error(f"健康检查错误: {e}", exc_info=True)
        return HealthStatus(
            status="unhealthy",
            model_loaded=False,
            retrieval_modules={},
            error=str(e)
        )


@router.post("/multimodal",
            response_model=MultimodalResponse,
            summary="多模态接口（预留）",
            description="多模态问答接口，支持图像输入（预留功能）")
async def multimodal_endpoint(request: MultimodalRequest) -> MultimodalResponse:
    """
    多模态接口（预留）
    
    - **query**: 文本查询
    - **image_base64**: 图像base64编码
    - **image_type**: 图像类型（tongue/pulse/other）
    
    注意：此接口为预留功能，将在后续版本实现
    """
    logger.info(f"收到多模态请求: {request.query}, 图像类型: {request.image_type}")
    
    return MultimodalResponse(
        success=True,
        query=request.query,
        answer="多模态功能暂未实现，敬请期待",
        image_analysis=None,
        note="多模态功能（舌诊、脉象图片分析）将在后续版本实现"
    )


@router.post("/batch_query",
            response_model=BatchQueryResponse,
            summary="批量问答（扩展功能）",
            description="批量处理多个问题")
async def batch_query_endpoint(
    request: BatchQueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
) -> BatchQueryResponse:
    """
    批量问答接口
    
    - **queries**: 问题列表（最多10个）
    - **top_k**: 每个问题返回的结果数量
    """
    try:
        logger.info(f"收到批量问答请求，共 {len(request.queries)} 个问题")
        
        start_time = time.time()
        
        # 构建检索配置
        retrieval_config = RetrievalConfig(
            enable_vector=request.enable_vector,
            enable_graph=request.enable_graph,
            top_k=request.top_k
        )
        
        # 批量处理
        results = await rag_chain.batch_query(
            questions=request.queries,
            retrieval_config=retrieval_config
        )
        
        total_time = time.time() - start_time
        
        # 转换为响应格式
        query_responses = []
        for result in results:
            if result.get("success"):
                query_responses.append(QueryResponse(
                    success=True,
                    query=result["query"],
                    answer=result["answer"],
                    retrieval_results=result.get("retrieval_results", []),
                    metadata=result.get("metadata")
                ))
            else:
                query_responses.append(QueryResponse(
                    success=False,
                    query=result["query"],
                    answer=None,
                    error=result.get("error", "未知错误")
                ))
        
        return BatchQueryResponse(
            success=True,
            results=query_responses,
            metadata={
                "total_queries": len(request.queries),
                "successful_queries": sum(1 for r in results if r.get("success")),
                "total_time": round(total_time, 2),
                "average_time_per_query": round(total_time / len(request.queries), 2)
            }
        )
    
    except Exception as e:
        logger.error(f"批量问答错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"批量问答失败: {str(e)}"
        )


# 初始化函数（由应用启动时调用）
def init_routes(rag_chain: RAGChain, retrieval_coordinator: HybridRetrievalCoordinator):
    """
    初始化路由的全局依赖
    
    Args:
        rag_chain: RAG链路实例
        retrieval_coordinator: 检索协调器实例
    """
    global _rag_chain, _retrieval_coordinator, _app_start_time
    _rag_chain = rag_chain
    _retrieval_coordinator = retrieval_coordinator
    _app_start_time = time.time()
    logger.info("API路由初始化完成")

