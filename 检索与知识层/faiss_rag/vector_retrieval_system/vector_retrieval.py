"""
向量检索主模块 - 整合数据加载、嵌入生成和向量存储
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from .data_loader import DataLoader
from .embedding_service import EmbeddingService
from .faiss_manager import FaissManager  # 使用 Faiss 替代 ChromaDB
from .config import RETRIEVAL_CONFIG

logger = logging.getLogger(__name__)

class VectorRetrieval:
    """向量检索系统主类"""
    
    def __init__(self, 
                 persist_directory: Union[str, Path] = None,
                 collection_name: str = None,
                 model_path: str = None,
                 use_simple_store: bool = False,
                 use_gpu: bool = None):
        """
        初始化向量检索系统
        
        Args:
            persist_directory: Chroma数据库持久化目录
            collection_name: 集合名称
            model_path: 嵌入模型路径
            use_simple_store: 是否使用简单向量存储（绕过SQLite问题）
        """
        # 初始化各个组件（使用延迟加载优化启动速度）
        self.data_loader = DataLoader()
        self.embedding_service = EmbeddingService(model_path=model_path, lazy_load=True)  # 延迟加载模型
        
        # 使用 Faiss 替代 ChromaDB（解决 Windows 死锁问题）
        import faiss
        from pathlib import Path as PathLib
        
        # 确定持久化目录
        if persist_directory:
            faiss_dir = PathLib(persist_directory)  # 直接使用传入的路径
        else:
            faiss_dir = PathLib(__file__).parent.parent / "向量数据库_faiss"
        
        # 允许外部控制是否使用GPU，默认保持与现有逻辑一致
        use_gpu_flag = (faiss.get_num_gpus() > 0) if (use_gpu is None) else bool(use_gpu)
        self.faiss_manager = FaissManager(
            persist_directory=str(faiss_dir),
            dimension=768,  # GTE Base 模型维度
            use_gpu=use_gpu_flag
        )
        
        # 保留 chroma_manager 引用以兼容旧代码
        self.chroma_manager = self.faiss_manager
        
        # 检索配置
        self.top_k = RETRIEVAL_CONFIG["top_k"]
        self.score_threshold = RETRIEVAL_CONFIG["score_threshold"]
        self.rerank = RETRIEVAL_CONFIG["rerank"]
        
        logger.info("向量检索系统初始化完成")
        try:
            stats = self.faiss_manager.get_stats()
            logger.info(f"Faiss 初始化统计: total_documents={stats.get('total_documents')}, dimension={stats.get('dimension')}, use_gpu={stats.get('use_gpu')}, dir={stats.get('persist_directory')}")
        except Exception as _e:
            logger.warning(f"无法获取Faiss统计: {_e}")
    
    def build_vector_database(self, data_sources: List[str] = None) -> bool:
        """
        构建向量数据库
        
        Args:
            data_sources: 数据源列表，可选['csv', 'json', 'arrow']，默认全部
            
        Returns:
            是否构建成功
        """
        try:
            logger.info("开始构建向量数据库...")
            start_time = time.time()
            
            # 加载数据
            if data_sources is None:
                data_sources = ['csv', 'json', 'arrow']
            
            all_data = []
            
            if 'csv' in data_sources:
                logger.info("加载CSV数据...")
                csv_data = self.data_loader.load_csv_data()
                all_data.extend(csv_data)
            
            if 'json' in data_sources:
                logger.info("加载JSON数据...")
                json_data = self.data_loader.load_json_data()
                all_data.extend(json_data)
            
            if 'arrow' in data_sources:
                logger.info("加载Arrow数据...")
                arrow_data = self.data_loader.load_arrow_data()
                all_data.extend(arrow_data)
            
            if not all_data:
                logger.warning("没有加载到任何数据")
                return False
            
            logger.info(f"总共加载了{len(all_data)}条原始数据")
            
            # 预处理数据
            logger.info("预处理数据...")
            processed_data = self.data_loader.prepare_for_embedding(all_data)
            
            if not processed_data:
                logger.warning("预处理后没有有效数据")
                return False
            
            # 生成向量嵌入
            logger.info("生成向量嵌入...")
            vectorized_data = self.embedding_service.encode_data_batch(processed_data)
            
            # 存储到Chroma数据库
            logger.info("存储到向量数据库...")
            success = self.chroma_manager.add_documents(vectorized_data)
            
            if success:
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"向量数据库构建完成，耗时: {duration:.2f}秒")
                
                # 输出统计信息
                collection_info = self.chroma_manager.get_collection_info()
                logger.info(f"数据库统计: {collection_info}")
                
                return True
            else:
                logger.error("向量数据库构建失败")
                return False
                
        except Exception as e:
            logger.error(f"构建向量数据库时发生错误: {e}")
            return False
    
    def search(self, query: str, top_k: int = None, 
               score_threshold: float = None,
               source_filter: str = None) -> List[Dict[str, Any]]:
        """
        向量检索搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            source_filter: 数据源过滤
            
        Returns:
            搜索结果列表
        """
        try:
            if not query or not query.strip():
                logger.warning("查询文本为空")
                return []
            
            # 使用默认参数
            top_k = top_k or self.top_k
            score_threshold = score_threshold or self.score_threshold
            
            # 生成查询向量
            logger.info(f"[DEBUG] 开始生成查询向量...")
            query_embedding = self.embedding_service.encode_text(query)
            logger.info(f"[DEBUG] 查询向量生成完成，维度: {len(query_embedding)}")
            
            # 使用 Faiss 搜索（不支持过滤，忽略 source_filter）
            logger.info(f"[DEBUG] 开始 Faiss 查询，top_k={top_k}...")
            results = self.faiss_manager.search(
                query_embedding=query_embedding,
                n_results=top_k
            )
            logger.info(f"[DEBUG] Faiss 查询完成，返回 {len(results)} 个结果")
            
            # 过滤低分结果
            filtered_results = []
            for result in results:
                if result["score"] >= score_threshold:
                    filtered_results.append(result)
            
            logger.info(f"搜索完成，返回{len(filtered_results)}个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """
        批量搜索
        
        Args:
            queries: 查询文本列表
            top_k: 每个查询返回结果数量
            
        Returns:
            每个查询的搜索结果列表
        """
        try:
            results = []
            for query in queries:
                query_results = self.search(query, top_k=top_k)
                results.append(query_results)
            
            logger.info(f"批量搜索完成，处理了{len(queries)}个查询")
            return results
            
        except Exception as e:
            logger.error(f"批量搜索失败: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            # 使用 Faiss 的统计信息
            stats = self.faiss_manager.get_stats()
            embedding_dim = self.embedding_service.get_embedding_dimension()
            
            stats.update({
                "embedding_dimension": embedding_dim,
                "model_name": self.embedding_service.model_path,
                "backend": "Faiss"  # 标识使用 Faiss
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            # 返回基本信息
            return {
                "total_documents": 0,
                "embedding_dimension": 512,  # GTE 默认维度
                "backend": "Faiss",
                "error": str(e)
            }
    
    def reset_database(self) -> bool:
        """重置向量数据库"""
        try:
            logger.info("重置向量数据库...")
            success = self.chroma_manager.reset_collection()
            
            if success:
                logger.info("向量数据库重置成功")
            else:
                logger.error("向量数据库重置失败")
            
            return success
            
        except Exception as e:
            logger.error(f"重置数据库时发生错误: {e}")
            return False
    
    def export_database(self, export_path: Union[str, Path]) -> bool:
        """导出数据库"""
        try:
            success = self.chroma_manager.export_collection(export_path)
            
            if success:
                logger.info(f"数据库导出成功: {export_path}")
            else:
                logger.error("数据库导出失败")
            
            return success
            
        except Exception as e:
            logger.error(f"导出数据库时发生错误: {e}")
            return False
    
    def similarity_search(self, query: str, candidate_texts: List[str]) -> List[Dict[str, Any]]:
        """
        相似度搜索（不依赖Chroma数据库）
        
        Args:
            query: 查询文本
            candidate_texts: 候选文本列表
            
        Returns:
            相似度结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.encode_text(query)
            
            # 生成候选向量
            candidate_embeddings = self.embedding_service.encode_batch(candidate_texts)
            
            # 计算相似度
            similarities = []
            for i, (text, embedding) in enumerate(zip(candidate_texts, candidate_embeddings)):
                similarity = self.embedding_service.similarity(query_embedding, embedding)
                similarities.append({
                    "index": i,
                    "text": text,
                    "similarity": similarity
                })
            
            # 按相似度排序
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"相似度搜索完成，处理了{len(candidate_texts)}个候选文本")
            return similarities
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
