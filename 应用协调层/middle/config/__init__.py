"""
配置管理模块
"""

from .config_manager import ConfigManager, load_config

# 默认配置
DEFAULT_CONFIG = {
    "retrieval": {
        "top_k": 10,
        "fusion_method": "weighted",
        "weights": {
            "vector": 0.5,
            "graph": 0.5
        }
    }
}

HYBRID_RETRIEVAL_CONFIG = {
    "vector": {
        "enabled": True,
        "persist_directory": "../../../检索与知识层/faiss_rag/向量数据库_简单查询",
        "model_path": "../../../Model Layer/model/iic/nlp_gte_sentence-embedding_chinese-small"
    },
    "graph": {
        "enabled": True,
        "neo4j_uri": "neo4j://127.0.0.1:7687",
        "username": "neo4j",
        "password": "hx1230047"
    }
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1
}

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

__all__ = [
    "ConfigManager",
    "load_config", 
    "DEFAULT_CONFIG",
    "HYBRID_RETRIEVAL_CONFIG",
    "API_CONFIG",
    "LOGGING_CONFIG"
]