"""
错误处理工具
提供统一的错误处理和异常管理功能
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime
from enum import Enum


class ErrorLevel(Enum):
    """错误级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化错误处理器
        
        Args:
            logger: 日志记录器，如果为None则使用默认logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_error": None
        }
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    level: ErrorLevel = ErrorLevel.ERROR,
                    reraise: bool = False) -> Dict[str, Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文信息
            level: 错误级别
            reraise: 是否重新抛出异常
            
        Returns:
            错误信息字典
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "level": level.value,
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        # 更新统计信息
        self.error_stats["total_errors"] += 1
        error_type = type(error).__name__
        self.error_stats["error_types"][error_type] = self.error_stats["error_types"].get(error_type, 0) + 1
        self.error_stats["last_error"] = error_info
        
        # 记录日志
        log_message = f"错误处理: {error_type} - {str(error)}"
        if context:
            log_message += f" | 上下文: {context}"
        
        if level == ErrorLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == ErrorLevel.INFO:
            self.logger.info(log_message)
        elif level == ErrorLevel.WARNING:
            self.logger.warning(log_message)
        elif level == ErrorLevel.ERROR:
            self.logger.error(log_message)
        elif level == ErrorLevel.CRITICAL:
            self.logger.critical(log_message)
        
        # 是否重新抛出异常
        if reraise:
            raise error
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return self.error_stats.copy()
    
    def clear_stats(self):
        """清空错误统计"""
        self.error_stats = {
            "total_errors": 0,
            "error_types": {},
            "last_error": None
        }


def safe_execute(func, *args, error_handler: Optional[ErrorHandler] = None, **kwargs) -> Any:
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        error_handler: 错误处理器
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或None（如果出错）
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, {"function": func.__name__, "args": args, "kwargs": kwargs})
        else:
            logging.error(f"函数 {func.__name__} 执行失败: {str(e)}")
        return None


def safe_execute_async(async_func, *args, error_handler: Optional[ErrorHandler] = None, **kwargs):
    """
    安全执行异步函数
    
    Args:
        async_func: 要执行的异步函数
        *args: 函数参数
        error_handler: 错误处理器
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或None（如果出错）
    """
    import asyncio
    
    async def _safe_wrapper():
        try:
            return await async_func(*args, **kwargs)
        except Exception as e:
            if error_handler:
                error_handler.handle_error(e, {"function": async_func.__name__, "args": args, "kwargs": kwargs})
            else:
                logging.error(f"异步函数 {async_func.__name__} 执行失败: {str(e)}")
            return None
    
    return _safe_wrapper()


# 全局错误处理器实例
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

def set_global_error_handler(handler: ErrorHandler):
    """设置全局错误处理器"""
    global _global_error_handler
    _global_error_handler = handler
