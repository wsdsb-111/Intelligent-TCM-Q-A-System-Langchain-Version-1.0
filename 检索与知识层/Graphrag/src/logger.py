"""
日志系统配置
提供统一的日志记录功能
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from .config import LoggingConfig


class Logger:
    """日志管理器"""
    
    def __init__(self, name: str = "GraphKnowledgePipeline"):
        self.name = name
        self._logger: Optional[logging.Logger] = None
    
    def setup_logger(self, config: LoggingConfig) -> logging.Logger:
        """设置日志器"""
        if self._logger is not None:
            return self._logger
        
        # 创建日志器
        self._logger = logging.getLogger(self.name)
        self._logger.setLevel(getattr(logging, config.level.upper()))
        
        # 清除现有的处理器
        self._logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(config.format)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # 文件处理器
        if config.file_path:
            # 确保日志目录存在
            log_file = Path(config.file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用RotatingFileHandler支持日志轮转
            file_handler = logging.handlers.RotatingFileHandler(
                config.file_path,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        return self._logger
    
    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        if self._logger is None:
            # 使用默认配置
            default_config = LoggingConfig()
            return self.setup_logger(default_config)
        return self._logger


# 全局日志器实例
logger_manager = Logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器的便捷函数"""
    if name:
        return logging.getLogger(f"GraphKnowledgePipeline.{name}")
    return logger_manager.get_logger()


def setup_logging(config: LoggingConfig):
    """设置全局日志配置"""
    logger_manager.setup_logger(config)