"""
配置管理器
支持YAML配置文件和环境变量
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from ..models.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config = {}
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """获取默认配置文件路径"""
        # 首先检查环境变量
        env_config_path = os.getenv("HYBRID_RETRIEVAL_CONFIG_PATH")
        if env_config_path:
            return Path(env_config_path)
        
        # 然后检查当前目录和上级目录
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "config" / "hybrid_retrieval.yaml",
            current_dir / "hybrid_retrieval.yaml",
            current_dir.parent / "config" / "hybrid_retrieval.yaml",
            Path(__file__).parent / "hybrid_retrieval.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 如果都不存在，返回默认路径
        return current_dir / "config" / "hybrid_retrieval.yaml"
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"已加载配置文件: {self.config_path}")
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                self._config = {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise ConfigurationError("config_file", f"无法加载配置文件: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 "section.subsection.key" 格式
            default: 默认值
        
        Returns:
            配置值
        """
        # 首先检查环境变量
        env_value = self._get_env_value(key)
        if env_value is not None:
            return env_value
        
        # 然后从配置文件获取
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def _get_env_value(self, key: str) -> Optional[Any]:
        """从环境变量获取配置值"""
        # 将点号分隔的键转换为环境变量格式
        env_key = f"HYBRID_RETRIEVAL_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        
        if env_value is None:
            return None
        
        # 尝试转换数据类型
        return self._convert_env_value(env_value)
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值的数据类型"""
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON格式
        if value.startswith('{') or value.startswith('['):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # 字符串
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self._config
        
        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """保存配置到文件"""
        save_path = path or self.config_path
        
        try:
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise ConfigurationError("config_save", f"无法保存配置文件: {e}")
    
    def reload(self):
        """重新加载配置文件"""
        self._load_config()
        logger.info("配置文件已重新加载")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()
    
    def update(self, config_dict: Dict[str, Any]):
        """更新配置"""
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """加载配置文件"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any):
    """设置配置值的便捷函数"""
    get_config_manager().set(key, value)


# 配置验证器

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_retrieval_config(config: Dict[str, Any]) -> bool:
        """验证检索配置"""
        required_sections = ['retrieval', 'api']
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(section, f"缺少必需的配置节: {section}")
        
        # 验证检索模块配置
        retrieval_config = config['retrieval']
        if 'modules' not in retrieval_config:
            raise ConfigurationError('retrieval.modules', "缺少模块配置")
        
        modules = retrieval_config['modules']
        required_modules = ['vector', 'graph']
        
        for module in required_modules:
            if module not in modules:
                raise ConfigurationError(f'retrieval.modules.{module}', f"缺少模块配置: {module}")
        
        return True
    
    @staticmethod
    def validate_module_config(module_name: str, config: Dict[str, Any]) -> bool:
        """验证单个模块配置"""
        if module_name == 'vector':
            required_keys = ['persist_directory', 'model_path']
        elif module_name == 'graph':
            required_keys = ['neo4j_uri']
        else:
            return True
        
        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f'{module_name}.{key}', f"缺少必需的配置项: {key}")
        
        return True