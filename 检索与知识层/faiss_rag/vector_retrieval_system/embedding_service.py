"""
嵌入服务模块 - 使用 GTE 文本向量模型生成文本向量
支持SentenceTransformer和Ollama模型
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any
import logging
import numpy as np
from pathlib import Path
from .config import MODEL_CONFIG

# 尝试导入Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logger = logging.getLogger(__name__)

# 全局模型缓存，避免重复加载
_model_cache = {}
_last_model_config = None

class EmbeddingService:
    """文本嵌入服务，支持SentenceTransformer和Ollama模型"""
    
    def __init__(self, model_path: str = None, cache_dir: str = None, lazy_load: bool = True):
        """
        初始化嵌入服务
        
        Args:
            model_path: 模型路径，默认使用配置中的模型
            cache_dir: 模型缓存目录
            lazy_load: 是否延迟加载模型（默认True，提高启动速度）
        """
        self.model_path = model_path or MODEL_CONFIG["embedding_model"]
        self.cache_dir = cache_dir or MODEL_CONFIG["model_cache_dir"]
        self.lazy_load = lazy_load
        
        # 调试信息
        logger.info(f"🔍 EmbeddingService初始化:")
        logger.info(f"  传入model_path: {model_path}")
        logger.info(f"  传入cache_dir: {cache_dir}")
        logger.info(f"  最终model_path: {self.model_path}")
        logger.info(f"  最终cache_dir: {self.cache_dir}")
        logger.info(f"  延迟加载: {lazy_load}")
        
        self.max_length = MODEL_CONFIG["max_length"]
        self.batch_size = MODEL_CONFIG["batch_size"]
        self.device = MODEL_CONFIG.get("device", "cpu")
        
        # 判断是否为Ollama模型
        self.is_ollama_model = self._is_ollama_model()
        
        # 模型缓存键
        self._cache_key = f"{self.model_path}_{self.cache_dir}_{self.device}"
        
        # 初始化模型
        self.model = None
        self.ollama_client = None
        
        if not lazy_load:
            self._load_model()
        else:
            logger.info("🚀 使用延迟加载模式，模型将在首次使用时加载")
    
    def _is_ollama_model(self) -> bool:
        """判断是否为Ollama模型"""
        return ("ollama" in self.model_path.lower() or 
                "quentinz" in self.model_path.lower())
    
    def _load_model(self):
        """加载嵌入模型"""
        global _model_cache, _last_model_config
        
        try:
            # 检查缓存
            if self._cache_key in _model_cache:
                logger.info("🔄 使用缓存的模型，跳过加载")
                self.model = _model_cache[self._cache_key]
                return
            
            logger.info(f"正在加载嵌入模型: {self.model_path}")
            logger.info(f"🔍 模型检测结果: is_ollama_model = {self.is_ollama_model}")
            logger.info(f"🔍 缓存目录: {self.cache_dir}")
            
            if self.is_ollama_model:
                logger.info("🚀 使用Ollama模型加载")
                self._load_ollama_model()
            else:
                logger.info("🚀 使用SentenceTransformer模型加载")
                self._load_sentence_transformer_model()
            
            # 缓存模型
            _model_cache[self._cache_key] = self.model
            _last_model_config = self._cache_key
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e
    
    def _ensure_model_loaded(self):
        """确保模型已加载（用于延迟加载）"""
        if self.model is None:
            logger.info("🔧 模型未加载，开始加载...")
            self._load_model()
    
    def _load_ollama_model(self):
        """加载Ollama模型"""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama未安装，请先安装: pip install ollama")
        
        try:
            self.ollama_client = ollama.Client()
            logger.info(f"Ollama客户端初始化成功，使用模型: {self.model_path}")
            
            # 检查模型是否存在
            models = self.ollama_client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_path not in model_names:
                logger.error(f"Ollama模型 {self.model_path} 不存在")
                logger.info(f"可用模型: {model_names}")
                raise ValueError(f"模型 {self.model_path} 未找到")
            
            logger.info(f"Ollama模型加载成功: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Ollama模型加载失败: {e}")
            raise e
    
    def _load_sentence_transformer_model(self):
        """加载SentenceTransformer模型"""
        # 检查 model_path 是否为本地路径
        model_path_obj = Path(self.model_path)
        is_local_path = model_path_obj.is_absolute() or (
            "\\" in self.model_path or "/" in self.model_path
        )
        
        logger.info(f"🔍 模型路径分析:")
        logger.info(f"  原始路径: {self.model_path}")
        logger.info(f"  是否本地路径: {is_local_path}")
        logger.info(f"  路径是否存在: {model_path_obj.exists()}")
        
        # 如果是本地路径，检查是否存在
        if is_local_path:
            if not model_path_obj.exists():
                logger.error(f"❌ 本地模型路径不存在: {self.model_path}")
                logger.info("回退到使用 HuggingFace GTE 模型")
                self.model = SentenceTransformer("Alibaba-NLP/gte-base-zh")
            else:
                # 检查关键文件
                config_file = model_path_obj / "config.json"
                if not config_file.exists():
                    logger.warning(f"⚠️  模型目录缺少 config.json: {self.model_path}")
                
                resolved_path = model_path_obj.resolve()
                logger.info(f"✅ 从本地路径加载模型: {resolved_path}")
                logger.info(f"   config.json存在: {(resolved_path / 'config.json').exists()}")
                # 使用绝对路径加载，并强制仅使用本地文件
                self.model = SentenceTransformer(
                    model_name_or_path=str(resolved_path),
                    cache_folder=str(resolved_path)
                )
        else:
            # HuggingFace 模型名称，直接加载
            logger.info(f"📥 从 HuggingFace 加载模型: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
        
        # 设置设备
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA不可用，回退到CPU")
            self.device = "cpu"
        
        self.model = self.model.to(self.device)
        
        logger.info(f"SentenceTransformer模型加载成功，使用设备: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU信息: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        对单个文本进行向量化
        
        Args:
            text: 输入文本
            
        Returns:
            文本向量
        """
        try:
            # 确保模型已加载
            self._ensure_model_loaded()
            
            if not text or not text.strip():
                # 返回零向量
                return np.zeros(self.get_embedding_dimension())
            
            # 截断过长的文本
            if len(text) > self.max_length:
                text = text[:self.max_length]
            
            if self.is_ollama_model:
                return self._encode_with_ollama([text])[0]
            else:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return embedding
            
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            return np.zeros(self.get_embedding_dimension())
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量对文本进行向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        try:
            # 确保模型已加载
            self._ensure_model_loaded()
            
            if not texts:
                return []
            
            # 预处理文本
            processed_texts = []
            for text in texts:
                if not text or not text.strip():
                    processed_texts.append("")
                else:
                    # 截断过长的文本
                    if len(text) > self.max_length:
                        text = text[:self.max_length]
                    processed_texts.append(text)
            
            if self.is_ollama_model:
                return self._encode_with_ollama(processed_texts)
            else:
                # 批量编码 - 完全禁用进度条
                embeddings = self.model.encode(
                    processed_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=self.device
                )
                return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"批量向量化失败: {e}")
            return [np.zeros(self.get_embedding_dimension()) for _ in texts]
    
    def _encode_with_ollama(self, texts: List[str]) -> List[np.ndarray]:
        """
        使用Ollama模型进行向量化
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        embeddings = []
        
        for text in texts:
            try:
                if not text or not text.strip():
                    embeddings.append(np.zeros(512))  # GTE 512维
                    continue
                
                # 使用Ollama生成嵌入向量
                response = self.ollama_client.embeddings(
                    model=self.model_path,
                    prompt=text
                )
                
                # 提取向量
                embedding = np.array(response['embedding'], dtype=np.float32)
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Ollama向量化失败: {e}")
                embeddings.append(np.zeros(512))  # GTE 512维
        
        return embeddings
    
    def encode_data_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对数据批次进行向量化处理
        
        Args:
            data: 包含文本的数据列表
            
        Returns:
            包含向量的数据列表
        """
        try:
            texts = [item["text"] for item in data]
            embeddings = self.encode_batch(texts)
            
            # 合并向量到数据中
            for i, (item, embedding) in enumerate(zip(data, embeddings)):
                item["embedding"] = embedding
                item["embedding_dim"] = len(embedding)
            
            logger.info(f"成功处理{len(data)}条数据的向量化")
            return data
            
        except Exception as e:
            logger.error(f"数据向量化失败: {e}")
            return data
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        if self.is_ollama_model:
            return 512  # GTE 模型默认维度
        else:
            # 如果模型未加载，返回默认维度
            if self.model is None:
                return 512  # GTE 模型默认维度
            return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            相似度分数
        """
        try:
            # 确保向量是numpy数组
            if not isinstance(embedding1, np.ndarray):
                embedding1 = np.array(embedding1)
            if not isinstance(embedding2, np.ndarray):
                embedding2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "model_path": self.model_path,
            "cache_dir": self.cache_dir,
            "is_ollama_model": self.is_ollama_model,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "embedding_dimension": self.get_embedding_dimension()
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "cuda_version": torch.version.cuda
            })
        
        return info