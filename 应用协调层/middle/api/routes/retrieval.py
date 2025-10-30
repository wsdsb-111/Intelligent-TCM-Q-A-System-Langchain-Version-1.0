"""
检索相关API路由
"""

import time
import uuid
from typing import Dict, Any
from datetime import datetime

try:
    from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    # 如果FastAPI不可用，创建基础类
    class APIRouter:
        def __init__(self, **kwargs):
            self.routes = []
        
        def post(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append((path, func))
                return func
            return decorator
        
        def get(self, path: str, **kwargs):
            def decorator(func):
                self.routes.append((path, func))
                return func
            return decorator
    
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
    
    def Depends(func):
        return func
    
    class BackgroundTasks:
        def add_task(self, func, *args, **kwargs):
            pass
    
    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code
    
    FASTAPI_AVAILABLE = False

from ..models import (
    RetrievalRequest, BatchRetrievalRequest, ContextualRetrievalRequest,
    RetrievalResponse, BatchRetrievalResponse, ErrorResponse,
    create_error_response, create_success_response
)
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from middle.integrations.hybrid_retriever import create_hybrid_retriever
from middle.models.data_models import RetrievalConfig, FusionMethod
from middle.utils.logging_utils import get_logger

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["retrieval"])
logger = get_logger(__name__)

# 全局检索器实例（可以考虑使用依赖注入）
_retriever_instance = None


def get_retriever():
    """获取检索器实例"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = create_hybrid_retriever()
    return _retriever_instance


def create_retrieval_config(request) -> RetrievalConfig:
    """根据请求创建检索配置"""
    # 映射融合方法
    fusion_method_map = {
        "rrf": FusionMethod.RRF,
        "weighted": FusionMethod.WEIGHTED,
        "rank_based": FusionMethod.RANK_BASED,
        "smart": FusionMethod.WEIGHTED  # 智能融合使用加权，由融合器自动选择
    }
    
    # 映射检索类型到模块启用状态（移除BM25支持）
    if request.retrieval_type == "bm25":
        # BM25已禁用，回退到向量检索
        enable_bm25, enable_vector, enable_graph = False, True, False
    elif request.retrieval_type == "vector":
        enable_bm25, enable_vector, enable_graph = False, True, False
    elif request.retrieval_type == "graph":
        enable_bm25, enable_vector, enable_graph = False, False, True
    else:  # hybrid
        enable_bm25, enable_vector, enable_graph = False, True, True
    
    return RetrievalConfig(
        enable_vector=enable_vector,
        enable_graph=enable_graph,
        top_k=request.top_k,
        fusion_method=fusion_method_map.get(request.fusion_method, FusionMethod.WEIGHTED),
        weights=request.weights or {"vector": 0.5, "graph": 0.5},
        timeout=request.timeout or 30
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
    background_tasks: BackgroundTasks,
    retriever=Depends(get_retriever)
):
    """
    基础检索接口
    
    执行单个查询的混合检索，支持多种检索类型和融合方法。
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] 开始检索: {request.query}")
        
        # 创建检索配置
        config = create_retrieval_config(request)
        
        # 更新检索器配置
        original_config = retriever.config
        retriever.config = config
        
        try:
            # 执行检索
            if request.fusion_method == "smart":
                # 使用智能融合
                fused_results = retriever.coordinator.fusion_engine.smart_fusion(
                    request.query,
                    await retriever.coordinator._parallel_retrieve(request.query, config),
                    top_k=request.top_k
                )
            else:
                # 使用标准检索
                fused_results = await retriever.coordinator.retrieve(request.query, config)
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 查询分析（如果是智能融合）
            query_analysis = None
            if request.fusion_method == "smart":
                query_type = retriever.coordinator.fusion_engine.detect_query_type(request.query)
                query_analysis = {
                    "query_type": query_type.value,
                    "detected_weights": retriever.coordinator.fusion_engine.get_scenario_weights(query_type)
                }
            
            # 创建成功响应
            response = create_success_response(
                query=request.query,
                retrieval_type=request.retrieval_type,
                fusion_method=request.fusion_method,
                results=fused_results,
                response_time=response_time,
                query_analysis=query_analysis
            )
            
            # 后台任务：记录统计信息
            background_tasks.add_task(
                log_retrieval_stats,
                request_id, request.query, len(fused_results), response_time, True
            )
            
            logger.info(f"[{request_id}] 检索完成: {len(fused_results)}个结果, 耗时{response_time:.3f}秒")
            return response
            
        finally:
            # 恢复原始配置
            retriever.config = original_config
    
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"检索失败: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        
        # 后台任务：记录错误统计
        background_tasks.add_task(
            log_retrieval_stats,
            request_id, request.query, 0, response_time, False, str(e)
        )
        
        # 返回错误响应
        error_response = create_error_response(
            error_code="RETRIEVAL_ERROR",
            error_message=error_msg,
            error_details={"query": request.query, "response_time": response_time},
            request_id=request_id
        )
        
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.post("/batch_retrieve", response_model=BatchRetrievalResponse)
async def batch_retrieve_documents(
    request: BatchRetrievalRequest,
    background_tasks: BackgroundTasks,
    retriever=Depends(get_retriever)
):
    """
    批量检索接口
    
    执行多个查询的批量检索，提高处理效率。
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] 开始批量检索: {len(request.queries)}个查询")
        
        # 创建检索配置
        config = create_retrieval_config(request)
        
        # 执行批量检索
        batch_results = await retriever.coordinator.batch_retrieve(request.queries, config)
        
        # 处理结果
        response_results = {}
        successful_count = 0
        failed_count = 0
        
        for query in request.queries:
            query_start_time = time.time()
            fused_results = batch_results.get(query, [])
            query_response_time = time.time() - query_start_time
            
            if fused_results:
                successful_count += 1
                # 创建单个查询的响应
                query_response = create_success_response(
                    query=query,
                    retrieval_type=request.retrieval_type,
                    fusion_method=request.fusion_method,
                    results=fused_results,
                    response_time=query_response_time
                )
            else:
                failed_count += 1
                # 创建失败响应
                query_response = RetrievalResponse(
                    success=False,
                    query=query,
                    retrieval_type=request.retrieval_type,
                    fusion_method=request.fusion_method,
                    total_results=0,
                    response_time=query_response_time,
                    results=[],
                    error_message="未找到相关结果"
                )
            
            response_results[query] = query_response
        
        # 计算总响应时间
        total_response_time = time.time() - start_time
        
        # 创建批量响应
        batch_response = BatchRetrievalResponse(
            success=True,
            total_queries=len(request.queries),
            successful_queries=successful_count,
            failed_queries=failed_count,
            total_response_time=total_response_time,
            results=response_results
        )
        
        # 后台任务：记录统计信息
        background_tasks.add_task(
            log_batch_retrieval_stats,
            request_id, len(request.queries), successful_count, total_response_time
        )
        
        logger.info(f"[{request_id}] 批量检索完成: {successful_count}/{len(request.queries)}成功, "
                   f"耗时{total_response_time:.3f}秒")
        
        return batch_response
    
    except Exception as e:
        total_response_time = time.time() - start_time
        error_msg = f"批量检索失败: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        
        # 返回错误响应
        error_response = create_error_response(
            error_code="BATCH_RETRIEVAL_ERROR",
            error_message=error_msg,
            error_details={"queries_count": len(request.queries), "response_time": total_response_time},
            request_id=request_id
        )
        
        raise HTTPException(status_code=500, detail=error_response.dict())


@router.post("/contextual_retrieve", response_model=RetrievalResponse)
async def contextual_retrieve_documents(
    request: ContextualRetrievalRequest,
    background_tasks: BackgroundTasks,
    retriever=Depends(get_retriever)
):
    """
    上下文检索接口
    
    基于对话上下文执行检索，支持多轮对话场景。
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] 开始上下文检索: {request.query}")
        
        # 处理上下文信息
        enhanced_query = request.query
        if request.use_context and request.context:
            # 简单的上下文处理：将最近的对话内容添加到查询中
            recent_context = []
            for msg in request.context[-3:]:  # 只使用最近3轮对话
                if msg['role'] == 'user':
                    recent_context.append(msg['content'])
            
            if recent_context:
                context_str = " ".join(recent_context)
                enhanced_query = f"{context_str} {request.query}"
        
        # 创建检索配置（移除BM25支持）
        config = RetrievalConfig(
            enable_bm25=False,  # BM25已禁用
            enable_vector=request.retrieval_type in ["vector", "hybrid"],
            enable_graph=request.retrieval_type in ["graph", "hybrid"],
            top_k=request.top_k,
            fusion_method=FusionMethod.WEIGHTED if request.fusion_method != "rrf" else FusionMethod.RRF,
            timeout=30
        )
        
        # 执行检索
        fused_results = await retriever.coordinator.retrieve(enhanced_query, config)
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 上下文分析
        query_analysis = {
            "original_query": request.query,
            "enhanced_query": enhanced_query,
            "context_used": request.use_context and len(request.context) > 0,
            "context_messages": len(request.context)
        }
        
        # 创建响应
        response = create_success_response(
            query=request.query,
            retrieval_type=request.retrieval_type,
            fusion_method=request.fusion_method,
            results=fused_results,
            response_time=response_time,
            query_analysis=query_analysis
        )
        
        # 后台任务：记录统计信息
        background_tasks.add_task(
            log_contextual_retrieval_stats,
            request_id, request.query, len(request.context), len(fused_results), response_time
        )
        
        logger.info(f"[{request_id}] 上下文检索完成: {len(fused_results)}个结果, 耗时{response_time:.3f}秒")
        return response
    
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"上下文检索失败: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        
        # 返回错误响应
        error_response = create_error_response(
            error_code="CONTEXTUAL_RETRIEVAL_ERROR",
            error_message=error_msg,
            error_details={"query": request.query, "context_length": len(request.context)},
            request_id=request_id
        )
        
        raise HTTPException(status_code=500, detail=error_response.dict())


# 后台任务函数

def log_retrieval_stats(request_id: str, query: str, result_count: int, 
                       response_time: float, success: bool, error: str = None):
    """记录检索统计信息"""
    try:
        stats = {
            "request_id": request_id,
            "query": query,
            "result_count": result_count,
            "response_time": response_time,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        # 这里可以将统计信息写入数据库或日志文件
        logger.info(f"检索统计: {stats}")
        
    except Exception as e:
        logger.error(f"记录检索统计失败: {str(e)}")


def log_batch_retrieval_stats(request_id: str, total_queries: int, 
                             successful_queries: int, response_time: float):
    """记录批量检索统计信息"""
    try:
        stats = {
            "request_id": request_id,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"批量检索统计: {stats}")
        
    except Exception as e:
        logger.error(f"记录批量检索统计失败: {str(e)}")


def log_contextual_retrieval_stats(request_id: str, query: str, context_length: int,
                                  result_count: int, response_time: float):
    """记录上下文检索统计信息"""
    try:
        stats = {
            "request_id": request_id,
            "query": query,
            "context_length": context_length,
            "result_count": result_count,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"上下文检索统计: {stats}")
        
    except Exception as e:
        logger.error(f"记录上下文检索统计失败: {str(e)}")