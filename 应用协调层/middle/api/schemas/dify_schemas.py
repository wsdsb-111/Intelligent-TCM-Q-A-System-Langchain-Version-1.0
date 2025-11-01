"""
Dify工作流节点API数据模型
定义Dify节点专用的请求和响应Pydantic模型
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RouterType(str, Enum):
    """路由类型枚举"""
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"


class RetrievalConfigModel(BaseModel):
    """检索配置模型（用于Dify节点）"""
    enable_vector: bool = Field(True, description="是否启用向量检索")
    enable_graph: bool = Field(True, description="是否启用知识图谱检索")
    vector_top_k: Optional[int] = Field(None, description="向量检索top_k（根据路由自动设置）")
    graph_top_k: Optional[int] = Field(None, description="图谱检索top_k（根据路由自动设置）")
    fusion_method: str = Field("weighted", description="融合方法")


class DifyRetrievalRequest(BaseModel):
    """Dify检索与知识召回节点请求模型"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=500)
    router_type: RouterType = Field(..., description="路由类型：vector_only 或 hybrid")
    config: Optional[RetrievalConfigModel] = Field(None, description="检索配置（可选）")


class DocumentSchema(BaseModel):
    """文档模型"""
    content: str = Field(..., description="文档内容")
    source: Optional[str] = Field(None, description="文档来源：vector 或 graph")
    fused_score: Optional[float] = Field(None, description="融合评分")
    source_scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="各源评分")
    contributing_sources: Optional[List[str]] = Field(default_factory=list, description="贡献的来源")
    entities: Optional[List[str]] = Field(default_factory=list, description="实体列表")
    relationships: Optional[List[str]] = Field(default_factory=list, description="关系列表")


class DifyRetrievalResponse(BaseModel):
    """Dify检索与知识召回节点响应模型"""
    success: bool = Field(..., description="是否成功")
    documents: Optional[List[DocumentSchema]] = Field(None, description="所有召回文档（含source字段）")
    generation_documents: Optional[List[DocumentSchema]] = Field(None, description="用于生成的文档")
    routing_decision: Optional[str] = Field(None, description="路由决策")
    retrieval_stats: Optional[Dict[str, Any]] = Field(None, description="检索统计信息")
    error: Optional[str] = Field(None, description="错误信息")


class DifyExpandRerankRequest(BaseModel):
    """Dify查询扩展与重排序节点请求模型"""
    query: str = Field(..., description="原始查询", min_length=1, max_length=500)
    documents: List[DocumentSchema] = Field(..., description="待处理的文档列表")
    parallel: bool = Field(True, description="是否并行执行扩展和重排序")


class DifyExpandRerankResponse(BaseModel):
    """Dify查询扩展与重排序节点响应模型"""
    success: bool = Field(..., description="是否成功")
    expanded_queries: Optional[List[str]] = Field(None, description="扩展后的查询列表")
    reranked_documents: Optional[List[DocumentSchema]] = Field(None, description="重排序后的文档列表")
    error: Optional[str] = Field(None, description="错误信息")


class GenerationParams(BaseModel):
    """生成参数模型"""
    max_new_tokens: int = Field(512, description="最大生成token数", ge=20, le=2048)
    temperature: float = Field(0.1, description="生成温度", ge=0.0, le=2.0)
    top_p: float = Field(0.4, description="nucleus sampling参数", ge=0.0, le=1.0)
    num_beams: int = Field(3, description="beam search束宽", ge=1, le=10)
    do_sample: bool = Field(False, description="是否采样")
    repetition_penalty: float = Field(1.3, description="重复惩罚", ge=1.0, le=2.0)
    length_penalty: float = Field(1.0, description="长度惩罚", ge=0.5, le=2.0)
    min_new_tokens: int = Field(20, description="最小生成token数", ge=1, le=100)
    no_repeat_ngram_size: int = Field(5, description="n-gram重复惩罚大小", ge=2, le=10)
    early_stopping: bool = Field(True, description="是否早停")
    use_cache: bool = Field(True, description="是否使用缓存")


class DifyGenerateAnswerRequest(BaseModel):
    """Dify回答生成节点请求模型"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=500)
    documents: List[DocumentSchema] = Field(..., description="已选择用于生成的文档（3或8个）")
    routing_decision: RouterType = Field(..., description="路由决策：vector_only 或 hybrid")
    generation_params: GenerationParams = Field(..., description="生成参数")


class DifyGenerateAnswerResponse(BaseModel):
    """Dify回答生成节点响应模型"""
    success: bool = Field(..., description="是否成功")
    answer: Optional[str] = Field(None, description="生成的答案")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据信息")
    error: Optional[str] = Field(None, description="错误信息")

