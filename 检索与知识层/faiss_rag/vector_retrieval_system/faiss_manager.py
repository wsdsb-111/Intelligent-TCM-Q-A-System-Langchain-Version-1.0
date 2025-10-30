"""
Faiss 向量数据库管理器
替代 ChromaDB，解决 Windows 上的死锁问题
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FaissManager:
    """Faiss 向量数据库管理器"""
    
    def __init__(self, persist_directory: str, dimension: int = 512, use_gpu: bool = True):
        """
        初始化 Faiss 管理器
        
        Args:
            persist_directory: 持久化目录
            dimension: 向量维度
            use_gpu: 是否使用 GPU
        """
        # 转换为绝对路径
        self.persist_directory = Path(persist_directory).resolve()
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # 确保目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.index_file = self.persist_directory / "faiss.index"
        self.metadata_file = self.persist_directory / "metadata.pkl"
        self.documents_file = self.persist_directory / "documents.json"
        
        # 初始化索引
        self.index = None
        self.metadata = []  # 存储文档元数据
        self.documents = []  # 存储文档内容
        
        # 加载或创建索引
        self._load_or_create_index()
        
        logger.info(f"Faiss 管理器初始化完成")
        logger.info(f"  维度: {self.dimension}")
        logger.info(f"  GPU: {self.use_gpu}")
        logger.info(f"  文档数: {len(self.documents)}")
    
    def _load_or_create_index(self):
        """加载或创建索引"""
        if self.index_file.exists():
            logger.info(f"加载现有索引: {self.index_file}")
            self._load_index()
        else:
            logger.info("创建新索引")
            self._create_index()
    
    def _create_index(self):
        """创建新的 Faiss 索引"""
        # 使用 IndexFlatIP (内积，适合归一化向量的余弦相似度)
        cpu_index = faiss.IndexFlatIP(self.dimension)
        
        if self.use_gpu:
            # 转移到 GPU
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("索引已转移到 GPU")
        else:
            self.index = cpu_index
            logger.info("使用 CPU 索引")
        
        self.metadata = []
        self.documents = []
    
    def _load_index(self):
        """加载索引和元数据"""
        try:
            # 使用临时文件避免路径空格问题（Windows）
            import tempfile
            import shutil
            
            # 复制到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp:
                tmp_path = tmp.name
            
            try:
                shutil.copy2(str(self.index_file), tmp_path)
                # 从临时文件加载
                cpu_index = faiss.read_index(tmp_path)
            finally:
                # 清理临时文件
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("索引已加载到 GPU")
            else:
                self.index = cpu_index
                logger.info("索引已加载到 CPU")
            
            # 加载元数据
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"加载元数据: {len(self.metadata)} 条")
            
            # 加载文档
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"加载文档: {len(self.documents)} 条")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            logger.info("创建新索引")
            self._create_index()
    
    def save_index(self):
        """保存索引和元数据"""
        try:
            # 确保目录存在
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # 保存索引（需要先转回 CPU）
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            
            # 使用短路径名避免空格问题（Windows）
            import tempfile
            import shutil
            
            # 先保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as tmp:
                tmp_path = tmp.name
            
            try:
                faiss.write_index(cpu_index, tmp_path)
                # 移动到目标位置
                shutil.move(tmp_path, str(self.index_file))
                logger.info(f"索引已保存: {self.index_file}")
            finally:
                # 清理临时文件
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            
            # 保存元数据
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"元数据已保存: {len(self.metadata)} 条")
            
            # 保存文档
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"文档已保存: {len(self.documents)} 条")
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档到索引
        
        Args:
            documents: 文档列表，每个文档包含 'embedding', 'text', 'metadata'
        
        Returns:
            是否成功
        """
        try:
            if not documents:
                return False
            
            # 提取向量
            embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
            
            # 归一化向量（用于余弦相似度）
            faiss.normalize_L2(embeddings)
            
            # 添加到索引
            self.index.add(embeddings)
            
            # 保存文档和元数据
            for doc in documents:
                self.documents.append(doc.get('text', ''))
                self.metadata.append(doc.get('metadata', {}))
            
            logger.info(f"添加 {len(documents)} 个文档到索引")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("索引为空")
                return []
            
            # 确保是 numpy 数组
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # 归一化查询向量
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # 搜索
            n_results = min(n_results, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, n_results)
            
            # 格式化结果
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue
                
                result = {
                    'id': f"doc_{idx}",
                    'content': self.documents[idx],
                    'text': self.documents[idx],  # 兼容性
                    'metadata': self.metadata[idx],
                    'distance': float(distance),
                    'score': float(distance)  # Faiss 返回的是相似度分数（内积）
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def count(self) -> int:
        """获取文档数量"""
        return self.index.ntotal if self.index else 0
    
    def reset(self):
        """重置索引"""
        logger.info("重置索引")
        self._create_index()
        
        # 删除文件
        for file in [self.index_file, self.metadata_file, self.documents_file]:
            if file.exists():
                file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents": self.count(),
            "dimension": self.dimension,
            "use_gpu": self.use_gpu,
            "index_type": "IndexFlatIP",
            "persist_directory": str(self.persist_directory)
        }
