"""
API数据模型
定义请求和响应的Pydantic模型
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum


class FusionMethodEnum(str, Enum):
    """融合方法枚举"""
    RRF = "rrf"
    WEIGHTED = "weighted"
    RANK_BASED = "rank_based"


class QueryRequest(BaseModel):
    """问答请求模型（v4.0：智能路由混合检索架构）"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: int = Field(5, description="返回结果数量", ge=1, le=20)
    enable_bm25: bool = Field(False, description="BM25已完全移除，此参数已废弃")
    enable_vector: bool = Field(True, description="是否启用向量检索")
    enable_graph: bool = Field(True, description="是否启用知识图谱检索")
    fusion_method: FusionMethodEnum = Field(FusionMethodEnum.WEIGHTED, description="融合方法")
    weights: Optional[Dict[str, float]] = Field(None, description="权重配置（智能路由会自动选择，手动指定会覆盖）")
    temperature: float = Field(0.7, description="生成温度", ge=0.0, le=2.0)
    max_new_tokens: int = Field(512, description="最大生成token数", ge=50, le=2048)
    stream: bool = Field(False, description="是否流式输出（预留）")
    image: Optional[str] = Field(None, description="图像base64编码（预留多模态）")
    session_id: Optional[str] = Field(None, description="会话ID（预留）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "人参和黄芪的配伍关系",
                "top_k": 5,
                "enable_bm25": False,
                "enable_vector": True,
                "enable_graph": True,
                "fusion_method": "weighted",
                "temperature": 0.7,
                "max_new_tokens": 512
            }
        }


class RetrievalRequest(BaseModel):
    """纯检索请求模型"""
    query: str = Field(..., description="检索查询", min_length=1, max_length=500)
    top_k: int = Field(10, description="返回结果数量", ge=1, le=50)
    enable_bm25: bool = Field(False, description="BM25已禁用")
    enable_vector: bool = Field(True, description="是否启用向量检索")
    enable_graph: bool = Field(True, description="是否启用图检索")
    fusion_method: FusionMethodEnum = Field(FusionMethodEnum.WEIGHTED, description="融合方法")
    weights: Optional[Dict[str, float]] = Field(None, description="权重配置（加权融合时使用，None则自动推荐）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "人参的功效",
                "top_k": 10,
                "fusion_method": "rrf"
            }
        }


class RetrievalResultSchema(BaseModel):
    """检索结果模型"""
    content: str = Field(..., description="文档内容")
    fused_score: float = Field(..., description="融合后评分")
    source_scores: Dict[str, float] = Field(default_factory=dict, description="各源评分")
    contributing_sources: List[str] = Field(default_factory=list, description="贡献的来源")
    entities: Optional[List[str]] = Field(None, description="实体列表")
    relationships: Optional[List[str]] = Field(None, description="关系列表")


class QueryMetadata(BaseModel):
    """问答元数据"""
    retrieval_time: float = Field(..., description="检索耗时（秒）")
    generation_time: float = Field(..., description="生成耗时（秒）")
    total_time: float = Field(..., description="总耗时（秒）")
    num_retrieval_results: int = Field(..., description="检索结果数量")
    model: str = Field(..., description="使用的模型")
    temperature: float = Field(..., description="生成温度")
    tokens_generated: Optional[int] = Field(None, description="生成的token数")
    tokens_per_second: Optional[float] = Field(None, description="生成速度")
    gpu_memory_used: Optional[str] = Field(None, description="GPU显存占用")


class QueryResponse(BaseModel):
    """问答响应模型"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="用户问题")
    answer: Optional[str] = Field(None, description="生成的答案")
    retrieval_results: List[RetrievalResultSchema] = Field(default_factory=list, description="检索结果")
    metadata: Optional[QueryMetadata] = Field(None, description="元数据")
    error: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "头痛怎么治疗",
                "answer": "根据中医理论，头痛的治疗需要辨证施治...",
                "retrieval_results": [
                    {
                        "content": "头痛分风寒、风热、血瘀等类型...",
                        "fused_score": 0.45,
                        "source_scores": {"bm25": 0.8, "vector": 0.7, "graph": 0.3},
                        "contributing_sources": ["bm25", "vector"],
                        "entities": ["头痛", "风寒", "风热"],
                        "relationships": ["治疗", "辨证"]
                    }
                ],
                "metadata": {
                    "retrieval_time": 0.15,
                    "generation_time": 1.2,
                    "total_time": 1.35,
                    "num_retrieval_results": 5,
                    "model": "qwen1.5-1.8b-tcm",
                    "temperature": 0.7,
                    "tokens_generated": 256,
                    "tokens_per_second": 213.3,
                    "gpu_memory_used": "5.2GB"
                }
            }
        }


class RetrievalMetadata(BaseModel):
    """检索元数据"""
    retrieval_time: float = Field(..., description="检索耗时（秒）")
    num_results: int = Field(..., description="结果数量")
    fusion_method: str = Field(..., description="融合方法")
    sources_used: List[str] = Field(..., description="使用的检索源")


class RetrievalResponse(BaseModel):
    """纯检索响应模型"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="检索查询")
    results: List[RetrievalResultSchema] = Field(default_factory=list, description="检索结果")
    metadata: Optional[RetrievalMetadata] = Field(None, description="元数据")
    error: Optional[str] = Field(None, description="错误信息")


class HealthStatus(BaseModel):
    """健康状态模型"""
    status: str = Field(..., description="健康状态", pattern="^(healthy|unhealthy|degraded)$")
    model_loaded: bool = Field(..., description="模型是否已加载")
    retrieval_modules: Dict[str, str] = Field(..., description="检索模块状态")
    gpu_memory: Optional[str] = Field(None, description="GPU显存使用情况")
    uptime: Optional[str] = Field(None, description="运行时长")
    stats: Optional[Dict[str, Any]] = Field(None, description="统计信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "retrieval_modules": {
                    "bm25": "available",
                    "vector": "available",
                    "graph": "available"
                },
                "gpu_memory": "5.2GB / 8.0GB",
                "uptime": "2h 30m 15s"
            }
        }


class MultimodalRequest(BaseModel):
    """多模态请求模型（预留）"""
    query: str = Field(..., description="文本查询")
    image_base64: Optional[str] = Field(None, description="图像base64编码")
    image_type: Optional[str] = Field(None, description="图像类型", pattern="^(tongue|pulse|other)$")
    
    @validator('image_base64')
    def validate_image(cls, v):
        if v and len(v) > 10_000_000:  # 限制10MB
            raise ValueError("图像大小不能超过10MB")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "这是什么舌象",
                "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
                "image_type": "tongue"
            }
        }


class MultimodalResponse(BaseModel):
    """多模态响应模型（预留）"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="查询")
    answer: Optional[str] = Field(None, description="答案")
    image_analysis: Optional[Dict[str, Any]] = Field(None, description="图像分析结果")
    note: str = Field("多模态功能将在后续版本实现", description="说明")
    error: Optional[str] = Field(None, description="错误信息")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = Field(False, description="是否成功")
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误")
    error_code: Optional[str] = Field(None, description="错误码")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "模型未初始化",
                "detail": "请先调用 /api/v1/health 检查系统状态",
                "error_code": "MODEL_NOT_LOADED"
            }
        }


class BatchQueryRequest(BaseModel):
    """批量问答请求（扩展功能）"""
    queries: List[str] = Field(..., description="问题列表", min_items=1, max_items=10)
    top_k: int = Field(5, description="每个问题返回的结果数量", ge=1, le=20)
    enable_bm25: bool = Field(False, description="BM25已禁用")
    enable_vector: bool = Field(True, description="是否启用向量检索")
    enable_graph: bool = Field(True, description="是否启用图检索")
    
    @validator('queries')
    def validate_queries(cls, v):
        if len(v) > 10:
            raise ValueError("批量查询最多支持10个问题")
        return v


class BatchQueryResponse(BaseModel):
    """批量问答响应"""
    success: bool = Field(..., description="是否成功")
    results: List[QueryResponse] = Field(..., description="答案列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="批量处理元数据")

