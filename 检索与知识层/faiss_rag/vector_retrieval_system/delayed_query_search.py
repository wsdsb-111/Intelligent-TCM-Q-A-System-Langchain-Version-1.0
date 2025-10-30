"""
延迟查询的向量检索
先编码，卸载模型，再查询 ChromaDB
"""

import torch
import gc
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DelayedQuerySearch:
    """延迟查询的向量检索"""
    
    def __init__(self, model_path: str, chroma_path: str, collection_name: str):
        self.model_path = model_path
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.model = None
        self.client = None
        self.collection = None
    
    def _load_model(self):
        """加载模型"""
        if self.model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"加载模型: {self.model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_path, device=device)
        logger.info(f"模型加载完成，设备: {device}")
    
    def _unload_model(self):
        """卸载模型，释放资源"""
        if self.model is None:
            return
        
        logger.info("卸载模型...")
        del self.model
        self.model = None
        
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        logger.info("模型已卸载")
    
    def _load_chroma(self):
        """加载 ChromaDB"""
        if self.collection is not None:
            return
        
        import chromadb
        
        logger.info(f"连接 ChromaDB: {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_collection(self.collection_name)
        logger.info(f"ChromaDB 连接成功")
    
    def encode_text(self, text: str) -> List[float]:
        """编码文本"""
        self._load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def search(self, query_text: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """搜索（先编码，卸载模型，再查询）"""
        # 步骤 1: 加载模型并编码
        logger.info(f"步骤 1: 编码查询文本")
        query_embedding = self.encode_text(query_text)
        logger.info(f"编码完成，维度: {len(query_embedding)}")
        
        # 步骤 2: 卸载模型
        logger.info(f"步骤 2: 卸载模型")
        self._unload_model()
        
        # 步骤 3: 查询 ChromaDB
        logger.info(f"步骤 3: 查询 ChromaDB")
        self._load_chroma()
        
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        logger.info(f"查询完成")
        
        # 步骤 4: 格式化结果
        results = []
        if raw_results and raw_results.get('ids') and raw_results['ids'][0]:
            for i in range(len(raw_results['ids'][0])):
                result = {
                    'id': raw_results['ids'][0][i],
                    'content': raw_results['documents'][0][i],
                    'metadata': raw_results['metadatas'][0][i],
                    'distance': raw_results['distances'][0][i],
                    'score': 1 - raw_results['distances'][0][i]
                }
                results.append(result)
        
        return results
