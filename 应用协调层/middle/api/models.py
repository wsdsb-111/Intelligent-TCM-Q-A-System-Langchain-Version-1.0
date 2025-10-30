"""
API数据模型
定义请求和响应的数据结构
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # 如果Pydantic不可用，创建基础类
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items()}
    
    def Field(**kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    PYDANTIC_AVAILABLE = False


class RetrievalTypeEnum(str, Enum):
    """检索类型枚举"""
    BM25 = "bm25"
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class FusionMethodEnum(str, Enum):
    """融合方法枚举"""
    RRF = "rrf"
    WEIGHTED = "weighted"
    RANK_BASED = "rank_based"
    SMART = "smart"


# 请求模型

class RetrievalRequest(BaseModel):
    """基础检索请求"""
    query: str = Field(..., description="检索查询内容", min_length=1, max_length=1000)
    retrieval_type: RetrievalTypeEnum = Field(
        default=RetrievalTypeEnum.HYBRID,
        description="检索类型"
    )
    top_k: int = Field(default=10, description="返回结果数量", ge=1, le=100)
    fusion_method: FusionMethodEnum = Field(
        default=FusionMethodEnum.SMART,
        description="融合方法"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="权重配置（用于加权融合）"
    )
    timeout: Optional[int] = Field(
        default=30,
        description="超时时间（秒）",
        ge=1,
        le=300
    )
    
    @validator('weights')
    def validate_weights(cls, v):
        if v is not None:
            # 检查权重值
            for source, weight in v.items():
                if weight < 0 or weight > 1:
                    raise ValueError(f"权重值必须在0-1之间: {source}={weight}")
            
            # 检查权重总和
            total = sum(v.values())
            if total <= 0:
                raise ValueError("权重总和必须大于0")
        
        return v


class BatchRetrievalRequest(BaseModel):
    """批量检索请求"""
    queries: List[str] = Field(..., description="查询列表", min_items=1, max_items=50)
    retrieval_type: RetrievalTypeEnum = Field(
        default=RetrievalTypeEnum.HYBRID,
        description="检索类型"
    )
    top_k: int = Field(default=10, description="每个查询返回结果数量", ge=1, le=50)
    fusion_method: FusionMethodEnum = Field(
        default=FusionMethodEnum.SMART,
        description="融合方法"
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="权重配置"
    )
    timeout: Optional[int] = Field(
        default=60,
        description="超时时间（秒）",
        ge=1,
        le=600
    )
    
    @validator('queries')
    def validate_queries(cls, v):
        # 检查查询内容
        for i, query in enumerate(v):
            if not query or len(query.strip()) == 0:
                raise ValueError(f"查询{i+1}不能为空")
            if len(query) > 1000:
                raise ValueError(f"查询{i+1}长度不能超过1000字符")
        return v


class ContextualRetrievalRequest(BaseModel):
    """上下文检索请求"""
    query: str = Field(..., description="当前查询内容", min_length=1, max_length=1000)
    context: List[Dict[str, str]] = Field(
        default=[],
        description="对话上下文",
        max_items=20
    )
    retrieval_type: RetrievalTypeEnum = Field(
        default=RetrievalTypeEnum.HYBRID,
        description="检索类型"
    )
    top_k: int = Field(default=10, description="返回结果数量", ge=1, le=100)
    fusion_method: FusionMethodEnum = Field(
        default=FusionMethodEnum.SMART,
        description="融合方法"
    )
    use_context: bool = Field(default=True, description="是否使用上下文信息")
    
    @validator('context')
    def validate_context(cls, v):
        for i, msg in enumerate(v):
            if not isinstance(msg, dict):
                raise ValueError(f"上下文消息{i+1}必须是字典格式")
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"上下文消息{i+1}必须包含role和content字段")
            if msg['role'] not in ['user', 'assistant', 'system']:
                raise ValueError(f"上下文消息{i+1}的role必须是user、assistant或system")
        return v


# 响应模型

class DocumentResult(BaseModel):
    """文档结果"""
    content: str = Field(..., description="文档内容")
    score: float = Field(..., description="相关性评分")
    source_scores: Dict[str, float] = Field(default={}, description="各来源评分")
    fusion_method: str = Field(..., description="融合方法")
    contributing_sources: List[str] = Field(default=[], description="贡献来源")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    entities: Optional[List[str]] = Field(default=None, description="识别的实体")
    relationships: Optional[List[str]] = Field(default=None, description="相关关系")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class RetrievalResponse(BaseModel):
    """检索响应"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="原始查询")
    retrieval_type: str = Field(..., description="检索类型")
    fusion_method: str = Field(..., description="融合方法")
    total_results: int = Field(..., description="结果总数")
    response_time: float = Field(..., description="响应时间（秒）")
    results: List[DocumentResult] = Field(default=[], description="检索结果")
    query_analysis: Optional[Dict[str, Any]] = Field(default=None, description="查询分析")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class BatchRetrievalResponse(BaseModel):
    """批量检索响应"""
    success: bool = Field(..., description="是否成功")
    total_queries: int = Field(..., description="查询总数")
    successful_queries: int = Field(..., description="成功查询数")
    failed_queries: int = Field(..., description="失败查询数")
    total_response_time: float = Field(..., description="总响应时间（秒）")
    results: Dict[str, RetrievalResponse] = Field(default={}, description="批量结果")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    overall_healthy: bool = Field(..., description="整体健康状态")
    modules: List[Dict[str, Any]] = Field(default=[], description="模块状态")
    system_info: Dict[str, Any] = Field(default={}, description="系统信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class MetricsResponse(BaseModel):
    """指标响应"""
    service_metrics: Dict[str, Any] = Field(default={}, description="服务指标")
    retrieval_metrics: Dict[str, Any] = Field(default={}, description="检索指标")
    performance_metrics: Dict[str, Any] = Field(default={}, description="性能指标")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class StatisticsResponse(BaseModel):
    """统计信息响应"""
    api_statistics: Dict[str, Any] = Field(default={}, description="API统计")
    retrieval_statistics: Dict[str, Any] = Field(default={}, description="检索统计")
    fusion_statistics: Dict[str, Any] = Field(default={}, description="融合统计")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


# 错误响应模型

class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误信息")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")


# 配置模型

class APIConfig(BaseModel):
    """API配置"""
    host: str = Field(default="0.0.0.0", description="服务主机")
    port: int = Field(default=8000, description="服务端口", ge=1, le=65535)
    workers: int = Field(default=1, description="工作进程数", ge=1, le=16)
    reload: bool = Field(default=False, description="是否自动重载")
    log_level: str = Field(default="INFO", description="日志级别")
    cors_origins: List[str] = Field(default=["*"], description="CORS允许的源")
    rate_limit: Dict[str, int] = Field(
        default={"requests": 100, "window": 60},
        description="速率限制配置"
    )
    enable_docs: bool = Field(default=True, description="是否启用API文档")
    enable_metrics: bool = Field(default=True, description="是否启用指标收集")


# 工具函数

def create_error_response(error_code: str, 
                         error_message: str, 
                         error_details: Optional[Dict[str, Any]] = None,
                         request_id: Optional[str] = None) -> ErrorResponse:
    """创建错误响应"""
    return ErrorResponse(
        error_code=error_code,
        error_message=error_message,
        error_details=error_details,
        request_id=request_id
    )


def create_success_response(query: str,
                           retrieval_type: str,
                           fusion_method: str,
                           results: List[Any],
                           response_time: float,
                           query_analysis: Optional[Dict[str, Any]] = None) -> RetrievalResponse:
    """创建成功响应"""
    # 转换结果格式
    document_results = []
    for result in results:
        if hasattr(result, 'content'):
            # FusedResult格式
            doc_result = DocumentResult(
                content=result.content,
                score=result.fused_score,
                source_scores=result.source_scores,
                fusion_method=result.fusion_method.value if hasattr(result.fusion_method, 'value') else str(result.fusion_method),
                contributing_sources=[s.value if hasattr(s, 'value') else str(s) for s in result.contributing_sources],
                metadata=result.metadata,
                entities=result.entities,
                relationships=result.relationships,
                timestamp=result.timestamp
            )
        else:
            # 其他格式
            doc_result = DocumentResult(
                content=str(result),
                score=1.0,
                fusion_method=fusion_method,
                contributing_sources=[retrieval_type]
            )
        document_results.append(doc_result)
    
    return RetrievalResponse(
        success=True,
        query=query,
        retrieval_type=retrieval_type,
        fusion_method=fusion_method,
        total_results=len(document_results),
        response_time=response_time,
        results=document_results,
        query_analysis=query_analysis
    )