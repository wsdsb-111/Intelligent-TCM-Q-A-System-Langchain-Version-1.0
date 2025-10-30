"""

错误处理器
统一的错误处理机制和错误报告
"""

import traceback
import functools
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from middle.core.exceptions import (
    HybridRetrievalError,
    ErrorSeverity,
    ErrorCategory,
    ExceptionFactory,
    collect_error
)
from middle.utils.logging_utils import get_logger, StructuredLogger


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, name: str):
        """
        初始化错误处理器
        
        Args:
            name: 处理器名称
        """
        self.name = name
        self.logger = StructuredLogger(f"{name}.error_handler")
        self.error_callbacks: List[Callable[[HybridRetrievalError], None]] = []
        
        # 错误统计
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": defaultdict(int),
            "errors_by_severity": defaultdict(int),
            "recent_errors": deque(maxlen=100)
        }
    
    def handle_error(self, 
                    error: Exception,
                    context: Dict[str, Any] = None,
                    reraise: bool = True) -> Optional[HybridRetrievalError]:
        """
        处理错误
        
        Args:
            error: 原始异常
            context: 错误上下文
            reraise: 是否重新抛出异常
            
        Returns:
            包装后的异常（如果不重新抛出）
        """
        # 包装异常
        if isinstance(error, HybridRetrievalError):
            wrapped_error = error
        else:
            wrapped_error = ExceptionFactory.wrap_exception(error)
        
        # 添加上下文信息
        if context:
            wrapped_error.context.update(context)
        
        # 记录错误
        self._log_error(wrapped_error)
        
        # 更新统计
        self._update_statistics(wrapped_error)
        
        # 触发回调
        self._trigger_callbacks(wrapped_error)
        
        # 收集错误
        collect_error(wrapped_error)
        
        if reraise:
            raise wrapped_error
        else:
            return wrapped_error
    
    def _log_error(self, error: HybridRetrievalError):
        """记录错误日志"""
        log_data = {
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "context": error.context
        }
        
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(
                f"严重错误: {error.message}",
                **log_data
            )
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"警告错误: {error.message}",
                **log_data
            )
        else:
            self.logger.info(
                f"轻微错误: {error.message}",
                **log_data
            )
        
        # 记录堆栈跟踪（对于严重错误）
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(
                f"错误堆栈跟踪: {error.error_code}",
                stack_trace=traceback.format_exc()
            )
    
    def _update_statistics(self, error: HybridRetrievalError):
        """更新错误统计"""
        self.error_stats["total_errors"] += 1
        self.error_stats["errors_by_category"][error.category.value] += 1
        self.error_stats["errors_by_severity"][error.severity.value] += 1
        self.error_stats["recent_errors"].append({
            "timestamp": error.timestamp.isoformat(),
            "error_code": error.error_code,
            "message": error.message,
            "category": error.category.value,
            "severity": error.severity.value
        })
    
    def _trigger_callbacks(self, error: HybridRetrievalError):
        """触发错误回调"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(
                    f"错误回调执行失败: {str(e)}",
                    callback_error=str(e),
                    original_error=error.error_code
                )
    
    def add_error_callback(self, callback: Callable[[HybridRetrievalError], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable[[HybridRetrievalError], None]):
        """移除错误回调"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        return {
            "handler_name": self.name,
            "total_errors": self.error_stats["total_errors"],
            "errors_by_category": dict(self.error_stats["errors_by_category"]),
            "errors_by_severity": dict(self.error_stats["errors_by_severity"]),
            "recent_errors": list(self.error_stats["recent_errors"])
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": defaultdict(int),
            "errors_by_severity": defaultdict(int),
            "recent_errors": deque(maxlen=100)
        }


class RetryHandler:
    """重试处理器"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 retryable_exceptions: List[type] = None):
        """
        初始化重试处理器
        
        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            backoff_factor: 退避因子
            retryable_exceptions: 可重试的异常类型
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            Exception  # 默认所有异常都可重试
        ]
        
        self.logger = get_logger(__name__)
        
        # 重试统计
        self.retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "retry_counts": defaultdict(int)
        }
    
    def is_retryable(self, error: Exception) -> bool:
        """判断异常是否可重试"""
        # 检查是否为HybridRetrievalError且标记为可恢复
        if isinstance(error, HybridRetrievalError):
            return error.recoverable
        
        # 检查是否为可重试的异常类型
        for exc_type in self.retryable_exceptions:
            if isinstance(error, exc_type):
                return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行带重试的函数调用
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.retry_stats["total_attempts"] += 1
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                    self.logger.info(
                        f"重试成功: 函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功"
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries and self.is_retryable(e):
                    delay = self.calculate_delay(attempt)
                    self.retry_stats["retry_counts"][attempt] += 1
                    
                    self.logger.warning(
                        f"重试: 函数 {func.__name__} 第 {attempt + 1} 次尝试失败，"
                        f"{delay:.2f}秒后重试。错误: {str(e)}"
                    )
                    
                    import time
                    time.sleep(delay)
                else:
                    self.retry_stats["failed_retries"] += 1
                    break
        
        # 所有重试都失败了
        self.logger.error(
            f"重试失败: 函数 {func.__name__} 在 {self.max_retries + 1} 次尝试后仍然失败"
        )
        raise last_error
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """获取重试统计"""
        return {
            "max_retries": self.max_retries,
            "total_attempts": self.retry_stats["total_attempts"],
            "successful_retries": self.retry_stats["successful_retries"],
            "failed_retries": self.retry_stats["failed_retries"],
            "retry_counts": dict(self.retry_stats["retry_counts"]),
            "success_rate": (
                self.retry_stats["successful_retries"] / 
                max(1, self.retry_stats["total_attempts"])
            )
        }


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        初始化熔断器
        
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间（秒）
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # 熔断器状态
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = get_logger(__name__)
        
        # 熔断器统计
        self.circuit_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opened_count": 0,
            "state_changes": []
        }
    
    def _change_state(self, new_state: str):
        """改变熔断器状态"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            self.circuit_stats["state_changes"].append({
                "timestamp": datetime.now().isoformat(),
                "from_state": old_state,
                "to_state": new_state,
                "failure_count": self.failure_count
            })
            
            self.logger.info(f"熔断器状态变更: {old_state} -> {new_state}")
            
            if new_state == "OPEN":
                self.circuit_stats["circuit_opened_count"] += 1
    
    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        self.circuit_stats["total_calls"] += 1
        
        # 检查熔断器状态
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self._change_state("HALF_OPEN")
            else:
                raise HybridRetrievalError(
                    "熔断器开启，拒绝调用",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.HIGH,
                    context={
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
                    }
                )
        
        try:
            result = func(*args, **kwargs)
            
            # 调用成功
            self.circuit_stats["successful_calls"] += 1
            
            if self.state == "HALF_OPEN":
                # 半开状态下成功，重置熔断器
                self.failure_count = 0
                self._change_state("CLOSED")
                self.logger.info("熔断器重置成功")
            
            return result
            
        except self.expected_exception as e:
            # 调用失败
            self.circuit_stats["failed_calls"] += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self._change_state("OPEN")
                self.logger.warning(
                    f"熔断器开启: 失败次数 {self.failure_count} 达到阈值 {self.failure_threshold}"
                )
            
            raise e
    
    def get_circuit_statistics(self) -> Dict[str, Any]:
        """获取熔断器统计"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "statistics": self.circuit_stats.copy()
        }
    
    def reset(self):
        """手动重置熔断器"""
        self.failure_count = 0
        self.last_failure_time = None
        self._change_state("CLOSED")
        self.logger.info("熔断器手动重置")


# 装饰器
def handle_errors(error_handler: ErrorHandler = None, 
                 context: Dict[str, Any] = None,
                 reraise: bool = True):
    """
    错误处理装饰器
    
    Args:
        error_handler: 错误处理器
        context: 错误上下文
        reraise: 是否重新抛出异常
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or get_default_error_handler()
            func_context = context or {}
            func_context.update({
                "function": func.__name__,
                "module": func.__module__
            })
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler.handle_error(e, func_context, reraise)
        
        return wrapper
    return decorator


def retry_on_error(retry_handler: RetryHandler = None):
    """
    重试装饰器
    
    Args:
        retry_handler: 重试处理器
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = retry_handler or get_default_retry_handler()
            return handler.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


def circuit_breaker(breaker: CircuitBreaker = None):
    """
    熔断器装饰器
    
    Args:
        breaker: 熔断器实例
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = breaker or get_default_circuit_breaker()
            return cb.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# 全局实例
_default_error_handler = None
_default_retry_handler = None
_default_circuit_breaker = None


def get_default_error_handler() -> ErrorHandler:
    """获取默认错误处理器"""
    global _default_error_handler
    if _default_error_handler is None:
        _default_error_handler = ErrorHandler("default")
    return _default_error_handler


def get_default_retry_handler() -> RetryHandler:
    """获取默认重试处理器"""
    global _default_retry_handler
    if _default_retry_handler is None:
        _default_retry_handler = RetryHandler()
    return _default_retry_handler


def get_default_circuit_breaker() -> CircuitBreaker:
    """获取默认熔断器"""
    global _default_circuit_breaker
    if _default_circuit_breaker is None:
        _default_circuit_breaker = CircuitBreaker()
    return _default_circuit_breaker


__all__ = [
    "ErrorHandler",
    "RetryHandler", 
    "CircuitBreaker",
    "handle_errors",
    "retry_on_error",
    "circuit_breaker",
    "get_default_error_handler",
    "get_default_retry_handler",
    "get_default_circuit_breaker"
]