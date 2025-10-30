"""
中医知识图谱自定义异常类
定义中医知识图谱系统中使用的各种异常类型
"""

from typing import Optional, Any


class GraphKnowledgePipelineError(Exception):
    """中医知识图谱基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details


class ConfigurationError(GraphKnowledgePipelineError):
    """配置错误"""
    pass


class FileProcessingError(GraphKnowledgePipelineError):
    """文件处理错误"""
    pass


class DocumentProcessingError(GraphKnowledgePipelineError):
    """文档处理错误"""
    pass


class GraphRAGProcessingError(GraphKnowledgePipelineError):
    """GraphRAG处理错误"""
    pass


class DataValidationError(GraphKnowledgePipelineError):
    """数据验证错误"""
    pass


class DataConversionError(GraphKnowledgePipelineError):
    """数据转换错误"""
    pass


class Neo4jConnectionError(GraphKnowledgePipelineError):
    """Neo4j连接错误"""
    pass


class Neo4jImportError(GraphKnowledgePipelineError):
    """Neo4j导入错误"""
    pass


class EntityExtractionError(GraphKnowledgePipelineError):
    """实体提取错误"""
    pass


class RelationshipExtractionError(GraphKnowledgePipelineError):
    """关系提取错误"""
    pass


class FileWatcherError(GraphKnowledgePipelineError):
    """文件监控错误"""
    pass


class APIError(GraphKnowledgePipelineError):
    """API调用错误"""
    pass


class RetryableError(GraphKnowledgePipelineError):
    """可重试的错误"""
    
    def __init__(self, message: str, retry_count: int = 0, max_retries: int = 3, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_count = retry_count
        self.max_retries = max_retries
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """增加重试次数"""
        self.retry_count += 1