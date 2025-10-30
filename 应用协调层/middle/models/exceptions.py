"""
异常处理模块
定义混合检索系统中使用的所有异常类
"""

from typing import Optional, Dict, Any


class RetrievalError(Exception):
    """检索基础异常"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ModuleUnavailableError(RetrievalError):
    """模块不可用异常"""
    
    def __init__(self, module_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"检索模块 '{module_name}' 不可用: {reason}"
        super().__init__(message, "MODULE_UNAVAILABLE", details)
        self.module_name = module_name
        self.reason = reason


class TimeoutError(RetrievalError):
    """超时异常"""
    
    def __init__(self, operation: str, timeout_seconds: int, details: Optional[Dict[str, Any]] = None):
        message = f"操作 '{operation}' 超时 ({timeout_seconds}秒)"
        super().__init__(message, "TIMEOUT", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class DataSourceError(RetrievalError):
    """数据源异常"""
    
    def __init__(self, data_source: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"数据源 '{data_source}' 错误: {reason}"
        super().__init__(message, "DATA_SOURCE_ERROR", details)
        self.data_source = data_source
        self.reason = reason


class ConfigurationError(RetrievalError):
    """配置错误异常"""
    
    def __init__(self, config_key: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"配置项 '{config_key}' 错误: {reason}"
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key
        self.reason = reason


class FusionError(RetrievalError):
    """结果融合异常"""
    
    def __init__(self, fusion_method: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"结果融合 '{fusion_method}' 失败: {reason}"
        super().__init__(message, "FUSION_ERROR", details)
        self.fusion_method = fusion_method
        self.reason = reason


class ValidationError(RetrievalError):
    """数据验证异常"""
    
    def __init__(self, field_name: str, value: Any, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"字段 '{field_name}' 验证失败 (值: {value}): {reason}"
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name
        self.value = value
        self.reason = reason


class ScoringError(RetrievalError):
    """评分计算异常"""
    
    def __init__(self, scoring_method: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"评分计算 '{scoring_method}' 失败: {reason}"
        super().__init__(message, "SCORING_ERROR", details)
        self.scoring_method = scoring_method
        self.reason = reason


class AuthenticationError(RetrievalError):
    """认证异常"""
    
    def __init__(self, service: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"服务 '{service}' 认证失败: {reason}"
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.service = service
        self.reason = reason


class RateLimitError(RetrievalError):
    """频率限制异常"""
    
    def __init__(self, limit: int, window_seconds: int, details: Optional[Dict[str, Any]] = None):
        message = f"请求频率超限: {limit}次/{window_seconds}秒"
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.limit = limit
        self.window_seconds = window_seconds


class ResourceExhaustedError(RetrievalError):
    """资源耗尽异常"""
    
    def __init__(self, resource_type: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"资源 '{resource_type}' 耗尽: {reason}"
        super().__init__(message, "RESOURCE_EXHAUSTED", details)
        self.resource_type = resource_type
        self.reason = reason


# 异常处理工具函数

def handle_module_error(module_name: str, error: Exception) -> ModuleUnavailableError:
    """处理模块错误，转换为标准异常"""
    if isinstance(error, RetrievalError):
        return error
    
    return ModuleUnavailableError(
        module_name=module_name,
        reason=str(error),
        details={"original_error": error.__class__.__name__}
    )


def handle_timeout_error(operation: str, timeout_seconds: int, error: Exception) -> TimeoutError:
    """处理超时错误"""
    return TimeoutError(
        operation=operation,
        timeout_seconds=timeout_seconds,
        details={"original_error": str(error)}
    )


def handle_data_source_error(data_source: str, error: Exception) -> DataSourceError:
    """处理数据源错误"""
    return DataSourceError(
        data_source=data_source,
        reason=str(error),
        details={"original_error": error.__class__.__name__}
    )


# 异常恢复策略

class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int = 3) -> bool:
        """判断是否应该重试"""
        if attempt >= max_attempts:
            return False
        
        # 对于某些类型的错误，不进行重试
        if isinstance(error, (ConfigurationError, ValidationError, AuthenticationError)):
            return False
        
        # 对于临时性错误，可以重试
        if isinstance(error, (TimeoutError, ResourceExhaustedError, DataSourceError)):
            return True
        
        return False
    
    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
        """获取重试延迟时间（指数退避）"""
        return base_delay * (2 ** attempt)
    
    @staticmethod
    def should_fallback(error: Exception) -> bool:
        """判断是否应该使用降级策略"""
        # 对于模块不可用错误，使用降级策略
        if isinstance(error, ModuleUnavailableError):
            return True
        
        # 对于超时错误，也可以考虑降级
        if isinstance(error, TimeoutError):
            return True
        
        return False