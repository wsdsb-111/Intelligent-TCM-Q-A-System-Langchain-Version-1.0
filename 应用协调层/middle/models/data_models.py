"""
数据模型定义
定义混合检索系统中使用的所有数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import json
from datetime import datetime


class RetrievalSource(Enum):
    """检索来源枚举"""
    VECTOR = "vector" 
    GRAPH = "graph"
    HYBRID = "hybrid"


class FusionMethod(Enum):
    """融合方法枚举"""
    RRF = "rrf"
    WEIGHTED = "weighted"
    RANK_BASED = "rank_based"


@dataclass
class RetrievalResult:
    """统一的检索结果格式"""
    content: str                           # 文档内容
    score: float                          # 相关性评分
    source: RetrievalSource               # 来源模块
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    entities: Optional[List[str]] = None   # 识别的实体
    relationships: Optional[List[str]] = None  # 相关关系
    timestamp: datetime = field(default_factory=datetime.now)  # 检索时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "score": self.score,
            "source": self.source.value,
            "metadata": self.metadata,
            "entities": self.entities,
            "relationships": self.relationships,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建实例"""
        return cls(
            content=data["content"],
            score=data["score"],
            source=RetrievalSource(data["source"]),
            metadata=data.get("metadata", {}),
            entities=data.get("entities"),
            relationships=data.get("relationships"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )


@dataclass
class FusedResult:
    """融合后的结果格式"""
    content: str                          # 文档内容
    fused_score: float                    # 融合后评分
    source_scores: Dict[str, float]       # 各模块的原始评分
    fusion_method: FusionMethod           # 融合方法
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    contributing_sources: List[RetrievalSource] = field(default_factory=list)  # 贡献的来源
    entities: Optional[List[str]] = None   # 合并的实体
    relationships: Optional[List[str]] = None  # 合并的关系
    timestamp: datetime = field(default_factory=datetime.now)  # 融合时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "fused_score": self.fused_score,
            "source_scores": self.source_scores,
            "fusion_method": self.fusion_method.value,
            "metadata": self.metadata,
            "contributing_sources": [s.value for s in self.contributing_sources],
            "entities": self.entities,
            "relationships": self.relationships,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RetrievalConfig:
    """检索配置（v4.0：智能路由混合检索）"""
    enable_vector: bool = True            # 启用向量检索
    enable_graph: bool = True             # 启用知识图谱检索
    top_k: int = 10                       # 返回结果数量
    fusion_method: FusionMethod = FusionMethod.WEIGHTED  # 融合方法（智能路由推荐使用WEIGHTED）
    weights: Optional[Dict[str, float]] = None  # 加权融合的权重
    timeout: int = 120                    # 超时时间（秒）- 增加以支持混合检索并行
    score_threshold: float = 0.0          # 最低评分阈值
    enable_caching: bool = True           # 启用缓存
    cache_ttl: int = 3600                 # 缓存生存时间（秒）
    
    def __post_init__(self):
        """初始化后处理"""
        if self.weights is None:
            self.weights = {
                "vector": 0.5, 
                "graph": 0.5
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enable_vector": self.enable_vector,
            "enable_graph": self.enable_graph,
            "top_k": self.top_k,
            "fusion_method": self.fusion_method.value,
            "weights": self.weights,
            "timeout": self.timeout,
            "score_threshold": self.score_threshold,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl
        }


@dataclass
class ModuleConfig:
    """模块配置"""
    vector_config: Dict[str, Any] = field(default_factory=dict)
    graph_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """设置默认配置"""
        
        if not self.vector_config:
            self.vector_config = {
                "persist_directory": "faiss_rag/向量数据库_简单查询",
                "model_path": "BAAI/bge-base-zh",
                "timeout": 15,
                "collection_name": "tcm_qa_collection"
            }
        
        if not self.graph_config:
            self.graph_config = {
                "neo4j_uri": "bolt://localhost:7687",
                "dump_path": "Graphrag/Knowledge_Graph/neo4j.dump",
                "timeout": 20,
                "username": "neo4j",
                "password": "password"
            }


# 特定检索模块的结果类型



@dataclass
class VectorResult:
    """向量检索结果"""
    content: str
    score: float
    similarity_score: float
    embedding_model: str = "BAAI/bge-base-zh"
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_retrieval_result(self) -> RetrievalResult:
        """转换为统一的检索结果格式"""
        base_metadata = {
            "similarity_score": self.similarity_score,
            "embedding_model": self.embedding_model,
            "document_id": self.document_id
        }
        
        # 如果有额外的metadata，合并进去
        if self.metadata:
            base_metadata.update(self.metadata)
        
        return RetrievalResult(
            content=self.content,
            score=self.score,
            source=RetrievalSource.VECTOR,
            metadata=base_metadata
        )


@dataclass
class EntityResult:
    """实体检索结果"""
    entity_name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 1.0


@dataclass
class RelationshipResult:
    """关系检索结果"""
    source_entity: str
    target_entity: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    score: float = 1.0


@dataclass
class PathResult:
    """路径检索结果"""
    start_entity: str
    end_entity: str
    path: List[Dict[str, Any]] = field(default_factory=list)
    path_length: int = 0
    score: float = 1.0
    
    def to_retrieval_result(self) -> RetrievalResult:
        """转换为统一的检索结果格式"""
        # 构建路径描述
        path_description = f"从 {self.start_entity} 到 {self.end_entity} 的关系路径："
        for i, step in enumerate(self.path):
            if i < len(self.path) - 1:
                path_description += f"\n{i+1}. {step.get('source', '')} --{step.get('relationship', '')}--> {step.get('target', '')}"
        
        return RetrievalResult(
            content=path_description,
            score=self.score,
            source=RetrievalSource.GRAPH,
            metadata={
                "start_entity": self.start_entity,
                "end_entity": self.end_entity,
                "path_length": self.path_length,
                "path_details": self.path
            },
            entities=[self.start_entity, self.end_entity],
            relationships=[step.get('relationship', '') for step in self.path]
        )


# 批量操作相关的数据类型

@dataclass
class BatchRetrievalRequest:
    """批量检索请求"""
    queries: List[str]
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "queries": self.queries,
            "config": self.config.to_dict()
        }


@dataclass
class BatchRetrievalResponse:
    """批量检索响应"""
    results: Dict[str, List[FusedResult]]
    total_queries: int
    successful_queries: int
    failed_queries: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "results": {
                query: [result.to_dict() for result in results] 
                for query, results in self.results.items()
            },
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "processing_time": self.processing_time
        }


# 健康检查和统计相关的数据类型

@dataclass
class ModuleHealthStatus:
    """模块健康状态"""
    module_name: str
    is_healthy: bool
    last_check: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "module_name": self.module_name,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat(),
            "error_message": self.error_message,
            "response_time": self.response_time
        }


@dataclass
class SystemHealthStatus:
    """系统健康状态"""
    overall_healthy: bool
    modules: List[ModuleHealthStatus] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "overall_healthy": self.overall_healthy,
            "modules": [module.to_dict() for module in self.modules],
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RetrievalStatistics:
    """检索统计信息"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    module_usage: Dict[str, int] = field(default_factory=dict)
    fusion_method_usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / max(self.total_queries, 1),
            "average_response_time": self.average_response_time,
            "module_usage": self.module_usage,
            "fusion_method_usage": self.fusion_method_usage
        }