"""
中医知识图谱简化的配置管理系统
专门用于中医CSV导出功能，移除了Neo4j相关配置
"""

import os
import yaml
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path
import logging


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


@dataclass
class GraphRAGConfig:
    """中医GraphRAG配置"""
    # OpenAI配置
    api_key: str = ""
    api_base: str = ""
    model: str = "gpt-4"
    embedding_model: str = "text-embedding-ada-002"
    
    # Ollama本地模型配置
    use_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_model_path: str = "D:\\RAG_ZS\\yi-6b"
    
    # 通用配置
    max_tokens: int = 2000
    temperature: float = 0.0
    chunk_size: int = 1200
    chunk_overlap: int = 100
    entity_types: list = field(default_factory=lambda: ["organization", "person", "geo", "event"])
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.use_ollama:
            # Ollama配置验证
            if not self.ollama_model:
                errors.append("Ollama模型名称不能为空")
                
            if not self.ollama_base_url:
                errors.append("Ollama基础URL不能为空")
                
            if not self.ollama_model_path:
                errors.append("Ollama模型路径不能为空")
        else:
            # OpenAI配置验证
            if not self.model:
                errors.append("GraphRAG模型不能为空")
                
            if not self.embedding_model:
                errors.append("嵌入模型不能为空")
        
        # 通用配置验证
        if self.max_tokens <= 0:
            errors.append("最大token数必须大于0")
            
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("温度值必须在0.0到2.0之间")
            
        if self.chunk_size <= 0:
            errors.append("分块大小必须大于0")
            
        if self.chunk_overlap < 0:
            errors.append("分块重叠不能为负数")
            
        if self.chunk_overlap >= self.chunk_size:
            errors.append("分块重叠不能大于等于分块大小")
            
        if not self.entity_types:
            errors.append("实体类型列表不能为空")
            
        return errors


@dataclass
class CSVExportConfig:
    """中医CSV导出配置"""
    output_directory: str = "output"
    encoding: str = "utf-8"
    delimiter: str = ","
    quote_char: str = '"'
    include_headers: bool = True
    generate_import_guide: bool = True
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if not self.output_directory:
            errors.append("输出目录不能为空")
            
        if not self.encoding:
            errors.append("编码格式不能为空")
            
        if len(self.delimiter) != 1:
            errors.append("分隔符必须是单个字符")
            
        if len(self.quote_char) != 1:
            errors.append("引号字符必须是单个字符")
            
        return errors


@dataclass
class FileWatcherConfig:
    """文件监控配置"""
    input_directory: str = "input"
    processed_directory: str = "processed"
    supported_formats: list = field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx"])
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if not self.input_directory:
            errors.append("输入目录不能为空")
            
        if not self.processed_directory:
            errors.append("已处理目录不能为空")
            
        if not self.supported_formats:
            errors.append("支持的文件格式列表不能为空")
            
        if self.max_file_size <= 0:
            errors.append("最大文件大小必须大于0")
            
        return errors


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            errors.append(f"日志级别必须是以下之一: {', '.join(valid_levels)}")
            
        if not self.format:
            errors.append("日志格式不能为空")
            
        if not self.file_path:
            errors.append("日志文件路径不能为空")
            
        if self.max_file_size <= 0:
            errors.append("最大文件大小必须大于0")
            
        if self.backup_count < 0:
            errors.append("备份数量不能为负数")
            
        return errors


@dataclass
class SystemConfig:
    """中医知识图谱系统总配置"""
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    csv_export: CSVExportConfig = field(default_factory=CSVExportConfig)
    file_watcher: FileWatcherConfig = field(default_factory=FileWatcherConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SystemConfig':
        """从配置文件加载配置"""
        config_file = Path(config_path)
        if not config_file.exists():
            # 如果配置文件不存在，创建默认配置文件
            default_config = cls()
            default_config.save_to_file(config_path)
            return default_config
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls._from_dict(config_data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置对象"""
        config = cls()
        
        if 'graphrag' in data:
            config.graphrag = GraphRAGConfig(**data['graphrag'])
        
        if 'csv_export' in data:
            config.csv_export = CSVExportConfig(**data['csv_export'])
        
        if 'file_watcher' in data:
            config.file_watcher = FileWatcherConfig(**data['file_watcher'])
        
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        return config
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_data = {
            'graphrag': {
                'api_key': self.graphrag.api_key,
                'api_base': self.graphrag.api_base,
                'model': self.graphrag.model,
                'embedding_model': self.graphrag.embedding_model,
                'use_ollama': self.graphrag.use_ollama,
                'ollama_base_url': self.graphrag.ollama_base_url,
                'ollama_model': self.graphrag.ollama_model,
                'ollama_model_path': self.graphrag.ollama_model_path,
                'max_tokens': self.graphrag.max_tokens,
                'temperature': self.graphrag.temperature,
                'chunk_size': self.graphrag.chunk_size,
                'chunk_overlap': self.graphrag.chunk_overlap,
                'entity_types': self.graphrag.entity_types
            },
            'csv_export': {
                'output_directory': self.csv_export.output_directory,
                'encoding': self.csv_export.encoding,
                'delimiter': self.csv_export.delimiter,
                'quote_char': self.csv_export.quote_char,
                'include_headers': self.csv_export.include_headers,
                'generate_import_guide': self.csv_export.generate_import_guide
            },
            'file_watcher': {
                'input_directory': self.file_watcher.input_directory,
                'processed_directory': self.file_watcher.processed_directory,
                'supported_formats': self.file_watcher.supported_formats,
                'max_file_size': self.file_watcher.max_file_size
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count
            }
        }
        
        # 确保配置目录存在
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def load_from_env(self):
        """从环境变量加载配置（覆盖文件配置）"""
        # 尝试加载.env文件
        self._load_env_file()
        
        # GraphRAG配置
        if os.getenv('GRAPHRAG_API_KEY'):
            self.graphrag.api_key = os.getenv('GRAPHRAG_API_KEY')
        if os.getenv('GRAPHRAG_API_BASE'):
            self.graphrag.api_base = os.getenv('GRAPHRAG_API_BASE')
        if os.getenv('GRAPHRAG_CHAT_MODEL'):
            self.graphrag.model = os.getenv('GRAPHRAG_CHAT_MODEL')
        elif os.getenv('GRAPHRAG_MODEL'):
            self.graphrag.model = os.getenv('GRAPHRAG_MODEL')
        if os.getenv('GRAPHRAG_EMBEDDING_MODEL'):
            self.graphrag.embedding_model = os.getenv('GRAPHRAG_EMBEDDING_MODEL')
        
        # Ollama配置
        if os.getenv('GRAPHRAG_USE_OLLAMA'):
            self.graphrag.use_ollama = os.getenv('GRAPHRAG_USE_OLLAMA').lower() in ['true', '1', 'yes']
        if os.getenv('OLLAMA_BASE_URL'):
            self.graphrag.ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        if os.getenv('OLLAMA_MODEL'):
            self.graphrag.ollama_model = os.getenv('OLLAMA_MODEL')
        if os.getenv('OLLAMA_MODEL_PATH'):
            self.graphrag.ollama_model_path = os.getenv('OLLAMA_MODEL_PATH')
        
        # 通用配置
        if os.getenv('GRAPHRAG_MAX_TOKENS'):
            try:
                self.graphrag.max_tokens = int(os.getenv('GRAPHRAG_MAX_TOKENS'))
            except ValueError:
                pass
        if os.getenv('GRAPHRAG_TEMPERATURE'):
            try:
                self.graphrag.temperature = float(os.getenv('GRAPHRAG_TEMPERATURE'))
            except ValueError:
                pass
        
        # CSV导出配置
        if os.getenv('CSV_OUTPUT_DIRECTORY'):
            self.csv_export.output_directory = os.getenv('CSV_OUTPUT_DIRECTORY')
        if os.getenv('CSV_ENCODING'):
            self.csv_export.encoding = os.getenv('CSV_ENCODING')
        
        # 文件监控配置
        if os.getenv('INPUT_DIRECTORY'):
            self.file_watcher.input_directory = os.getenv('INPUT_DIRECTORY')
        if os.getenv('PROCESSED_DIRECTORY'):
            self.file_watcher.processed_directory = os.getenv('PROCESSED_DIRECTORY')
    
    def validate(self) -> List[str]:
        """验证整个系统配置"""
        all_errors = []
        
        # 验证各个子配置
        all_errors.extend([f"GraphRAG配置: {error}" for error in self.graphrag.validate()])
        all_errors.extend([f"CSV导出配置: {error}" for error in self.csv_export.validate()])
        all_errors.extend([f"文件监控配置: {error}" for error in self.file_watcher.validate()])
        all_errors.extend([f"日志配置: {error}" for error in self.logging.validate()])
        
        return all_errors
    
    def _load_env_file(self):
        """加载.env文件"""
        env_file = Path(".env")
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
            except Exception as e:
                pass  # 静默处理.env文件加载错误


class SimpleConfigManager:
    """中医知识图谱简化的配置管理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> SystemConfig:
        """加载配置"""
        if self._config is None:
            self._config = self._load_and_validate_config()
        return self._config
    
    def detect_autodl_environment(self) -> bool:
        """检测是否在 AutoDL 环境中"""
        try:
            # 检查是否在 AutoDL 环境中
            # AutoDL 通常有特定的环境变量或文件
            autodl_indicators = [
                os.environ.get('AUTODL_CONTAINER'),
                os.environ.get('AUTODL_TASK_ID'),
                os.path.exists('/autodl-tmp'),
                os.path.exists('/root/autodl-tmp'),
                'autodl' in platform.node().lower(),
                'autodl' in os.environ.get('HOSTNAME', '').lower()
            ]
            
            # 检查是否有 NVIDIA GPU
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                has_nvidia_gpu = result.returncode == 0
            except:
                has_nvidia_gpu = False
            
            # 检查是否有 RTX 4090
            is_rtx_4090 = False
            if has_nvidia_gpu:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                         capture_output=True, text=True, timeout=5)
                    gpu_names = result.stdout.strip().split('\n')
                    is_rtx_4090 = any('4090' in name for name in gpu_names)
                except:
                    pass
            
            return any(autodl_indicators) and has_nvidia_gpu and is_rtx_4090
            
        except Exception as e:
            self.logger.warning(f"检测 AutoDL 环境失败: {e}")
            return False
    
    def get_optimal_config_path(self) -> str:
        """获取最优配置文件路径"""
        if self.detect_autodl_environment():
            autodl_config = Path(self.config_path).parent / "config_autodl.yaml"
            if autodl_config.exists():
                self.logger.info("检测到 AutoDL 4090 环境，使用优化配置")
                return str(autodl_config)
        
        return self.config_path
    
    def _load_and_validate_config(self) -> SystemConfig:
        """加载并验证配置"""
        try:
            # 使用最优配置文件路径
            optimal_config_path = self.get_optimal_config_path()
            config = SystemConfig.load_from_file(optimal_config_path)
            config.load_from_env()
            
            # 验证配置
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "配置验证失败:\n" + "\n".join(validation_errors)
                self.logger.error(error_msg)
                raise ConfigValidationError(error_msg)
            
            self.logger.info("配置加载和验证成功")
            return config
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            raise
    
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> SystemConfig:
        """重新加载配置"""
        self._config = None
        return self.load_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要（不包含敏感信息）"""
        config = self.get_config()
        return {
            'graphrag': {
                'use_ollama': config.graphrag.use_ollama,
                'model': config.graphrag.ollama_model if config.graphrag.use_ollama else config.graphrag.model,
                'embedding_model': config.graphrag.embedding_model,
                'max_tokens': config.graphrag.max_tokens,
                'temperature': config.graphrag.temperature,
                'chunk_size': config.graphrag.chunk_size,
                'api_key': '***' if config.graphrag.api_key else '',
                'ollama_base_url': config.graphrag.ollama_base_url if config.graphrag.use_ollama else None
            },
            'csv_export': {
                'output_directory': config.csv_export.output_directory,
                'encoding': config.csv_export.encoding,
                'delimiter': config.csv_export.delimiter,
                'include_headers': config.csv_export.include_headers
            },
            'file_watcher': {
                'input_directory': config.file_watcher.input_directory,
                'processed_directory': config.file_watcher.processed_directory,
                'max_file_size': config.file_watcher.max_file_size
            }
        }


# 全局配置管理器实例
config_manager = SimpleConfigManager()