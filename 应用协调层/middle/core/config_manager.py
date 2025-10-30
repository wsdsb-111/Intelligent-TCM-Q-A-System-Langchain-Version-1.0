"""
配置管
理系统
支持YAML配置文件解析、环境变量支持和配置热重载功能
"""

import os
import yaml
import json
import threading
import time
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from middle.utils.logging_utils import get_logger


@dataclass
class ConfigSource:
    """配置源定义"""
    path: str
    type: str  # yaml, json, env
    priority: int = 0  # 优先级，数字越大优先级越高
    required: bool = True
    last_modified: Optional[datetime] = None


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    source: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更监听器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = get_logger(__name__)
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self.config_manager.is_watched_file(file_path):
            self.logger.info(f"配置文件变更: {file_path}")
            self.config_manager.reload_config(file_path)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 enable_hot_reload: bool = True,
                 enable_env_override: bool = True):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
            enable_hot_reload: 是否启用热重载
            enable_env_override: 是否启用环境变量覆盖
        """
        self.logger = get_logger(__name__)
        
        # 配置目录
        if config_dir is None:
            config_dir = os.path.join(project_root, "langchain", "config")
        self.config_dir = Path(config_dir)
        
        # 配置选项
        self.enable_hot_reload = enable_hot_reload
        self.enable_env_override = enable_env_override
        
        # 配置数据
        self.config_data: Dict[str, Any] = {}
        self.config_sources: List[ConfigSource] = []
        self.change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # 热重载相关
        self.observer = None
        self.watched_files: Dict[str, ConfigSource] = {}
        self._lock = threading.RLock()
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化配置管理器"""
        self.logger.info("初始化配置管理器")
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载默认配置源
        self._register_default_sources()
        
        # 加载所有配置
        self._load_all_configs()
        
        # 启动热重载监听
        if self.enable_hot_reload:
            self._start_hot_reload()
        
        self.logger.info("配置管理器初始化完成")
    
    def _register_default_sources(self):
        """注册默认配置源"""
        default_configs = [
            ("hybrid_retrieval.yaml", "yaml", 100, True),
            ("mcp_tools_config.yaml", "yaml", 90, False),
            ("logging.yaml", "yaml", 80, False),
            ("database.yaml", "yaml", 70, False)
        ]
        
        for filename, config_type, priority, required in default_configs:
            config_path = self.config_dir / filename
            if config_path.exists() or required:
                self.register_config_source(
                    str(config_path), 
                    config_type, 
                    priority, 
                    required
                )
    
    def register_config_source(self, 
                             path: str, 
                             config_type: str, 
                             priority: int = 0, 
                             required: bool = True):
        """
        注册配置源
        
        Args:
            path: 配置文件路径
            config_type: 配置类型 (yaml, json, env)
            priority: 优先级
            required: 是否必需
        """
        source = ConfigSource(
            path=path,
            type=config_type,
            priority=priority,
            required=required
        )
        
        self.config_sources.append(source)
        self.config_sources.sort(key=lambda x: x.priority, reverse=True)
        
        # 如果启用热重载，添加到监听列表
        if self.enable_hot_reload and config_type in ['yaml', 'json']:
            self.watched_files[path] = source
        
        self.logger.info(f"注册配置源: {path} (类型: {config_type}, 优先级: {priority})")
    
    def _load_all_configs(self):
        """加载所有配置"""
        with self._lock:
            self.config_data.clear()
            
            # 按优先级加载配置
            for source in self.config_sources:
                try:
                    config = self._load_config_file(source)
                    if config:
                        self._merge_config(config)
                        source.last_modified = datetime.now()
                        self.logger.debug(f"加载配置: {source.path}")
                except Exception as e:
                    if source.required:
                        self.logger.error(f"加载必需配置失败: {source.path}, 错误: {e}")
                        raise
                    else:
                        self.logger.warning(f"加载可选配置失败: {source.path}, 错误: {e}")
            
            # 应用环境变量覆盖
            if self.enable_env_override:
                self._apply_env_overrides()
    
    def _load_config_file(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """加载单个配置文件"""
        if not os.path.exists(source.path):
            if source.required:
                raise FileNotFoundError(f"必需的配置文件不存在: {source.path}")
            return None
        
        try:
            with open(source.path, 'r', encoding='utf-8') as f:
                if source.type == 'yaml':
                    return yaml.safe_load(f)
                elif source.type == 'json':
                    return json.load(f)
                else:
                    self.logger.warning(f"不支持的配置类型: {source.type}")
                    return None
        except Exception as e:
            self.logger.error(f"解析配置文件失败: {source.path}, 错误: {e}")
            raise
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """合并配置数据"""
        def deep_merge(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self.config_data, new_config)
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_prefix = "HYBRID_RETRIEVAL_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # 尝试转换数据类型
                converted_value = self._convert_env_value(value)
                
                # 设置配置值
                self._set_nested_value(self.config_data, config_key, converted_value)
                self.logger.debug(f"环境变量覆盖: {config_key} = {converted_value}")
    
    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值的数据类型"""
        # 尝试转换为布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 尝试转换为数字
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 尝试转换为JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # 返回字符串
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any):
        """设置嵌套配置值"""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _start_hot_reload(self):
        """启动热重载监听"""
        if not self.watched_files:
            return
        
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            
            # 监听配置目录
            self.observer.schedule(handler, str(self.config_dir), recursive=True)
            
            # 监听其他配置文件的目录
            watched_dirs = set()
            for file_path in self.watched_files.keys():
                dir_path = os.path.dirname(file_path)
                if dir_path not in watched_dirs:
                    self.observer.schedule(handler, dir_path, recursive=False)
                    watched_dirs.add(dir_path)
            
            self.observer.start()
            self.logger.info("配置热重载监听已启动")
            
        except Exception as e:
            self.logger.error(f"启动热重载失败: {e}")
    
    def is_watched_file(self, file_path: str) -> bool:
        """检查是否为监听的配置文件"""
        return file_path in self.watched_files
    
    def reload_config(self, file_path: Optional[str] = None):
        """重新加载配置"""
        with self._lock:
            old_config = self.config_data.copy()
            
            if file_path:
                # 重新加载特定文件
                source = self.watched_files.get(file_path)
                if source:
                    try:
                        config = self._load_config_file(source)
                        if config:
                            # 重新加载所有配置以保持优先级
                            self._load_all_configs()
                            self.logger.info(f"重新加载配置文件: {file_path}")
                    except Exception as e:
                        self.logger.error(f"重新加载配置失败: {file_path}, 错误: {e}")
                        return
            else:
                # 重新加载所有配置
                self._load_all_configs()
                self.logger.info("重新加载所有配置")
            
            # 触发变更回调
            self._trigger_change_callbacks(old_config, self.config_data)
    
    def _trigger_change_callbacks(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """触发配置变更回调"""
        changes = self._find_config_changes(old_config, new_config)
        
        for change in changes:
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    self.logger.error(f"配置变更回调执行失败: {e}")
    
    def _find_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChangeEvent]:
        """查找配置变更"""
        changes = []
        
        def compare_dict(old_dict: Dict[str, Any], new_dict: Dict[str, Any], prefix: str = ""):
            # 检查新增和修改的键
            for key, new_value in new_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if key not in old_dict:
                    # 新增的配置
                    changes.append(ConfigChangeEvent(
                        source=full_key,
                        old_value=None,
                        new_value=new_value
                    ))
                elif old_dict[key] != new_value:
                    if isinstance(old_dict[key], dict) and isinstance(new_value, dict):
                        # 递归比较嵌套字典
                        compare_dict(old_dict[key], new_value, full_key)
                    else:
                        # 修改的配置
                        changes.append(ConfigChangeEvent(
                            source=full_key,
                            old_value=old_dict[key],
                            new_value=new_value
                        ))
            
            # 检查删除的键
            for key, old_value in old_dict.items():
                if key not in new_dict:
                    full_key = f"{prefix}.{key}" if prefix else key
                    changes.append(ConfigChangeEvent(
                        source=full_key,
                        old_value=old_value,
                        new_value=None
                    ))
        
        compare_dict(old_config, new_config)
        return changes
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        current = self.config_data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, persist: bool = False):
        """设置配置值"""
        with self._lock:
            old_value = self.get(key)
            self._set_nested_value(self.config_data, key, value)
            
            # 触发变更回调
            change = ConfigChangeEvent(
                source=key,
                old_value=old_value,
                new_value=value
            )
            
            for callback in self.change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    self.logger.error(f"配置变更回调执行失败: {e}")
            
            # 持久化到文件
            if persist:
                self._persist_config()
    
    def _persist_config(self):
        """持久化配置到文件"""
        # 找到主配置文件（优先级最高的YAML文件）
        main_config_file = None
        for source in self.config_sources:
            if source.type == 'yaml' and os.path.exists(source.path):
                main_config_file = source.path
                break
        
        if main_config_file:
            try:
                with open(main_config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                self.logger.info(f"配置已持久化到: {main_config_file}")
            except Exception as e:
                self.logger.error(f"持久化配置失败: {e}")
    
    def add_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """添加配置变更回调"""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """移除配置变更回调"""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config_data.copy()
    
    def get_config_sources(self) -> List[ConfigSource]:
        """获取配置源列表"""
        return self.config_sources.copy()
    
    def validate_config(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """验证配置"""
        # 这里可以添加配置验证逻辑
        # 例如使用jsonschema进行验证
        return True
    
    def export_config(self, format: str = 'yaml') -> str:
        """导出配置"""
        if format == 'yaml':
            return yaml.dump(self.config_data, default_flow_style=False, allow_unicode=True)
        elif format == 'json':
            return json.dumps(self.config_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def stop(self):
        """停止配置管理器"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("配置热重载监听已停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 全局配置管理器实例
_config_manager_instance = None


def get_config_manager(**kwargs) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager_instance
    
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager(**kwargs)
    
    return _config_manager_instance


def get_config(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any, persist: bool = False):
    """设置配置值的便捷函数"""
    get_config_manager().set(key, value, persist)


__all__ = [
    "ConfigManager",
    "ConfigSource", 
    "ConfigChangeEvent",
    "get_config_manager",
    "get_config",
    "set_config"
]