"""
实体提取配置管理器
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EntityExtractionConfig:
    """实体提取配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认路径
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config" / "entity_extraction.yaml"
        return str(config_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"实体提取配置加载成功: {self.config_file}")
                return config.get('entity_extraction', {})
            else:
                logger.warning(f"配置文件不存在: {self.config_file}，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'default_mode': 'knowledge_graph',
            'knowledge_graph': {
                'csv_file_path': '测试与质量保障层/testdataset/merged_datasets_classified.csv',
                'enabled': True,
                'cache_enabled': True,
                'cache_size': 1000
            },
            'llm': {
                'enabled': False,
                'model_service': None,
                'max_tokens': 100,
                'temperature': 0.1
            },
            'rule': {
                'enabled': True,
                'predefined_entities': {
                    'herbs': True,
                    'symptoms': True,
                    'formulas': True
                }
            },
            'logging': {
                'level': 'INFO',
                'enable_debug': False
            }
        }
    
    def get_default_mode(self) -> str:
        """获取默认模式"""
        return self.config.get('default_mode', 'knowledge_graph')
    
    def is_kg_enabled(self) -> bool:
        """是否启用知识图谱模式"""
        return self.config.get('knowledge_graph', {}).get('enabled', True)
    
    def get_kg_csv_path(self) -> str:
        """获取知识图谱CSV文件路径"""
        return self.config.get('knowledge_graph', {}).get('csv_file_path', 
            '测试与质量保障层/testdataset/merged_datasets_classified.csv')
    
    def is_llm_enabled(self) -> bool:
        """是否启用LLM模式"""
        return self.config.get('llm', {}).get('enabled', False)
    
    def is_rule_enabled(self) -> bool:
        """是否启用规则模式"""
        return self.config.get('rule', {}).get('enabled', True)
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self.config.get('llm', {})
    
    def get_rule_config(self) -> Dict[str, Any]:
        """获取规则配置"""
        return self.config.get('rule', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self.config

# 全局配置实例
_config_instance = None

def get_entity_config() -> EntityExtractionConfig:
    """获取实体提取配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = EntityExtractionConfig()
    return _config_instance

def reload_config():
    """重新加载配置"""
    global _config_instance
    _config_instance = None
    return get_entity_config()
