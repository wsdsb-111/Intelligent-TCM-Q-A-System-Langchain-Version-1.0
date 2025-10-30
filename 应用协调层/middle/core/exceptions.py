""
"
混合检索系统异常定义
定义检索相关的异常类和错误处理机制
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    CONFIGURATION = "configuration"
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    PROCESSING = "processing"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"


class HybridRetrievalError(Exception):
    """混合检索系统基础异常类"""
    
    def __init__(self, 
                 message: str,
                 error_code: str = None,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict[str, Any] = None,
                 cause: Exception = None,
                 recoverable: bool = True):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            category: 错误类别
            severity: 错误严重程度
            context: 错误上下文信息
            cause: 原始异常
            recoverable: 是否可恢复
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        
        # 如果有原始异常，保存其信息
        if cause:
            self.context.update({
                "original_error_type": cause.__class__.__name__,
                "original_error_message": str(cause)
            })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# 配置相关异常
class ConfigurationError(HybridRetrievalError):
    """配置错误"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if config_key:
            self.context["config_key"] = config_key


class InvalidConfigurationError(ConfigurationError):
    """无效配置错误"""
    
    def __init__(self, message: str, config_key: str = None, expected_type: str = None, **kwargs):
        super().__init__(message, config_key, **kwargs)
        if expected_type:
            self.context["expected_type"] = expected_type


class MissingConfigurationError(ConfigurationError):
    """缺失配置错误"""
    
    def __init__(self, config_key: str, **kwargs):
        message = f"缺少必需的配置项: {config_key}"
        super().__init__(message, config_key, **kwargs)


# 连接相关异常
class ConnectionError(HybridRetrievalError):
    """连接错误"""
    
    def __init__(self, message: str, service: str = None, endpoint: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if service:
            self.context["service"] = service
        if endpoint:
            self.context["endpoint"] = endpoint


class DatabaseConnectionError(ConnectionError):
    """数据库连接错误"""
    
    def __init__(self, message: str, database_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if database_type:
            self.context["database_type"] = database_type


class ExternalServiceError(ConnectionError):
    """外部服务错误"""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None, **kwargs):
        super().__init__(message, service_name, **kwargs)
        self.category = ErrorCategory.EXTERNAL_SERVICE
        if status_code:
            self.context["status_code"] = status_code


# 认证相关异常
class AuthenticationError(HybridRetrievalError):
    """认证错误"""
    
    def __init__(self, message: str, auth_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        if auth_type:
            self.context["auth_type"] = auth_type


class AuthorizationError(AuthenticationError):
    """授权错误"""
    
    def __init__(self, message: str, required_permission: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if required_permission:
            self.context["required_permission"] = required_permission


# 验证相关异常
class ValidationError(HybridRetrievalError):
    """验证错误"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if field:
            self.context["field"] = field
        if value is not None:
            self.context["value"] = str(value)


class QueryValidationError(ValidationError):
    """查询验证错误"""
    
    def __init__(self, message: str, query: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if query:
            self.context["query"] = query


class ParameterValidationError(ValidationError):
    """参数验证错误"""
    
    def __init__(self, message: str, parameter_name: str = None, **kwargs):
        super().__init__(message, parameter_name, **kwargs)


# 处理相关异常
class ProcessingError(HybridRetrievalError):
    """处理错误"""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if operation:
            self.context["operation"] = operation


class RetrievalError(ProcessingError):
    """检索错误"""
    
    def __init__(self, message: str, module: str = None, query: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if module:
            self.context["module"] = module
        if query:
            self.context["query"] = query


class FusionError(ProcessingError):
    """融合错误"""
    
    def __init__(self, message: str, fusion_method: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if fusion_method:
            self.context["fusion_method"] = fusion_method


class IndexError(ProcessingError):
    """索引错误"""
    
    def __init__(self, message: str, index_type: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if index_type:
            self.context["index_type"] = index_type


# 超时相关异常
class TimeoutError(HybridRetrievalError):
    """超时错误"""
    
    def __init__(self, message: str, timeout_duration: float = None, operation: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        if timeout_duration:
            self.context["timeout_duration"] = timeout_duration
        if operation:
            self.context["operation"] = operation


class RetrievalTimeoutError(TimeoutError):
    """检索超时错误"""
    
    def __init__(self, message: str, module: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if module:
            self.context["module"] = module


class ConnectionTimeoutError(TimeoutError):
    """连接超时错误"""
    
    def __init__(self, message: str, service: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if service:
            self.context["service"] = service


# 资源相关异常
class ResourceError(HybridRetrievalError):
    """资源错误"""
    
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if resource_type:
            self.context["resource_type"] = resource_type


class ResourceNotFoundError(ResourceError):
    """资源未找到错误"""
    
    def __init__(self, message: str, resource_id: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if resource_id:
            self.context["resource_id"] = resource_id


class ResourceExhaustedError(ResourceError):
    """资源耗尽错误"""
    
    def __init__(self, message: str, resource_limit: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if resource_limit:
            self.context["resource_limit"] = resource_limit


class MemoryError(ResourceExhaustedError):
    """内存不足错误"""
    
    def __init__(self, message: str, memory_usage: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if memory_usage:
            self.context["memory_usage"] = memory_usage


# 数据相关异常
class DataError(HybridRetrievalError):
    """数据错误"""
    
    def __init__(self, message: str, data_source: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_CORRUPTION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        if data_source:
            self.context["data_source"] = data_source


class DataCorruptionError(DataError):
    """数据损坏错误"""
    
    def __init__(self, message: str, file_path: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if file_path:
            self.context["file_path"] = file_path


class DataFormatError(DataError):
    """数据格式错误"""
    
    def __init__(self, message: str, expected_format: str = None, actual_format: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if expected_format:
            self.context["expected_format"] = expected_format
        if actual_format:
            self.context["actual_format"] = actual_format


# 模块特定异常


class VectorRetrievalError(RetrievalError):
    """向量检索错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, module="vector", **kwargs)


class GraphRetrievalError(RetrievalError):
    """图检索错误"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, module="graph", **kwargs)


class ChromaDBError(VectorRetrievalError):
    """ChromaDB错误"""
    
    def __init__(self, message: str, collection_name: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if collection_name:
            self.context["collection_name"] = collection_name


class Neo4jError(GraphRetrievalError):
    """Neo4j错误"""
    
    def __init__(self, message: str, query: str = None, **kwargs):
        super().__init__(message, **kwargs)
        if query:
            self.context["neo4j_query"] = query


# 异常工厂类
class ExceptionFactory:
    """异常工厂类"""
    
    @staticmethod
    def create_from_category(category: ErrorCategory, 
                           message: str, 
                           **kwargs) -> HybridRetrievalError:
        """根据类别创建异常"""
        exception_map = {
            ErrorCategory.CONFIGURATION: ConfigurationError,
            ErrorCategory.CONNECTION: ConnectionError,
            ErrorCategory.AUTHENTICATION: AuthenticationError,
            ErrorCategory.VALIDATION: ValidationError,
            ErrorCategory.PROCESSING: ProcessingError,
            ErrorCategory.TIMEOUT: TimeoutError,
            ErrorCategory.RESOURCE: ResourceError,
            ErrorCategory.EXTERNAL_SERVICE: ExternalServiceError,
            ErrorCategory.DATA_CORRUPTION: DataError,
            ErrorCategory.UNKNOWN: HybridRetrievalError
        }
        
        exception_class = exception_map.get(category, HybridRetrievalError)
        return exception_class(message, category=category, **kwargs)
    
    @staticmethod
    def wrap_exception(original_exception: Exception, 
                      message: str = None,
                      category: ErrorCategory = None) -> HybridRetrievalError:
        """包装原始异常"""
        if isinstance(original_exception, HybridRetrievalError):
            return original_exception
        
        # 根据原始异常类型推断类别
        if category is None:
            category = ExceptionFactory._infer_category(original_exception)
        
        # 使用原始异常消息或提供的消息
        error_message = message or str(original_exception)
        
        return ExceptionFactory.create_from_category(
            category=category,
            message=error_message,
            cause=original_exception
        )
    
    @staticmethod
    def _infer_category(exception: Exception) -> ErrorCategory:
        """推断异常类别"""
        exception_name = exception.__class__.__name__.lower()
        
        if "timeout" in exception_name:
            return ErrorCategory.TIMEOUT
        elif "connection" in exception_name or "network" in exception_name:
            return ErrorCategory.CONNECTION
        elif "auth" in exception_name or "permission" in exception_name:
            return ErrorCategory.AUTHENTICATION
        elif "validation" in exception_name or "value" in exception_name:
            return ErrorCategory.VALIDATION
        elif "memory" in exception_name or "resource" in exception_name:
            return ErrorCategory.RESOURCE
        elif "file" in exception_name or "io" in exception_name:
            return ErrorCategory.DATA_CORRUPTION
        else:
            return ErrorCategory.UNKNOWN


# 错误收集器
class ErrorCollector:
    """错误收集器"""
    
    def __init__(self, max_errors: int = 1000):
        """
        初始化错误收集器
        
        Args:
            max_errors: 最大错误数量
        """
        self.max_errors = max_errors
        self.errors: List[HybridRetrievalError] = []
        self.error_counts: Dict[str, int] = {}
        self.category_counts: Dict[ErrorCategory, int] = {}
        self.severity_counts: Dict[ErrorSeverity, int] = {}
    
    def collect_error(self, error: HybridRetrievalError):
        """收集错误"""
        # 添加到错误列表
        self.errors.append(error)
        
        # 限制错误数量
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
        
        # 更新统计
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        self.category_counts[error.category] = self.category_counts.get(error.category, 0) + 1
        self.severity_counts[error.severity] = self.severity_counts.get(error.severity, 0) + 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "total_errors": len(self.errors),
            "error_counts": self.error_counts.copy(),
            "category_counts": {cat.value: count for cat, count in self.category_counts.items()},
            "severity_counts": {sev.value: count for sev, count in self.severity_counts.items()},
            "recent_errors": [error.to_dict() for error in self.errors[-10:]]  # 最近10个错误
        }
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[HybridRetrievalError]:
        """根据类别获取错误"""
        return [error for error in self.errors if error.category == category]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[HybridRetrievalError]:
        """根据严重程度获取错误"""
        return [error for error in self.errors if error.severity == severity]
    
    def clear_errors(self):
        """清空错误"""
        self.errors.clear()
        self.error_counts.clear()
        self.category_counts.clear()
        self.severity_counts.clear()


# 全局错误收集器实例
_global_error_collector = None


def get_error_collector() -> ErrorCollector:
    """获取全局错误收集器"""
    global _global_error_collector
    if _global_error_collector is None:
        _global_error_collector = ErrorCollector()
    return _global_error_collector


def collect_error(error: Exception, 
                 message: str = None,
                 category: ErrorCategory = None) -> HybridRetrievalError:
    """
    收集错误的便捷函数
    
    Args:
        error: 原始异常
        message: 自定义错误消息
        category: 错误类别
        
    Returns:
        包装后的异常
    """
    if isinstance(error, HybridRetrievalError):
        wrapped_error = error
    else:
        wrapped_error = ExceptionFactory.wrap_exception(error, message, category)
    
    # 收集到全局收集器
    collector = get_error_collector()
    collector.collect_error(wrapped_error)
    
    return wrapped_error


__all__ = [
    # 枚举
    "ErrorSeverity",
    "ErrorCategory",
    
    # 基础异常
    "HybridRetrievalError",
    
    # 配置异常
    "ConfigurationError",
    "InvalidConfigurationError", 
    "MissingConfigurationError",
    
    # 连接异常
    "ConnectionError",
    "DatabaseConnectionError",
    "ExternalServiceError",
    
    # 认证异常
    "AuthenticationError",
    "AuthorizationError",
    
    # 验证异常
    "ValidationError",
    "QueryValidationError",
    "ParameterValidationError",
    
    # 处理异常
    "ProcessingError",
    "RetrievalError",
    "FusionError",
    "IndexError",
    
    # 超时异常
    "TimeoutError",
    "RetrievalTimeoutError",
    "ConnectionTimeoutError",
    
    # 资源异常
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceExhaustedError",
    "MemoryError",
    
    # 数据异常
    "DataError",
    "DataCorruptionError",
    "DataFormatError",
    
    # 模块异常
    "VectorRetrievalError",
    "GraphRetrievalError",
    "ChromaDBError",
    "Neo4jError",
    
    # 工具类
    "ExceptionFactory",
    "ErrorCollector",
    "get_error_collector",
    "collect_error"
]