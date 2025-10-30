"""
LangChain集成准备模块 - 为未来集成LangChain框架做准备
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from .vector_retrieval import VectorRetrieval
from .config import CHROMA_CONFIG

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    """基础检索器抽象类，兼容LangChain接口"""
    
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """获取相关文档"""
        pass
    
    @abstractmethod
    def aget_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """异步获取相关文档"""
        pass

class TCMVectorRetriever(BaseRetriever):
    """中医向量检索器，兼容LangChain接口"""
    
    def __init__(self, vector_retrieval: VectorRetrieval = None):
        """
        初始化中医向量检索器
        
        Args:
            vector_retrieval: 向量检索系统实例
        """
        if vector_retrieval is None:
            self.vector_retrieval = VectorRetrieval()
        else:
            self.vector_retrieval = vector_retrieval
        
        logger.info("中医向量检索器初始化完成")
    
    def get_relevant_documents(self, query: str, 
                              top_k: int = 5,
                              score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        获取相关文档（同步）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相关文档列表
        """
        try:
            results = self.vector_retrieval.search(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
            
            # 转换为LangChain兼容格式
            documents = []
            for result in results:
                documents.append({
                    "page_content": result["text"],
                    "metadata": result["metadata"],
                    "score": result.get("score", 0.0)
                })
            
            logger.info(f"检索到{len(documents)}个相关文档")
            return documents
            
        except Exception as e:
            logger.error(f"获取相关文档失败: {e}")
            return []
    
    async def aget_relevant_documents(self, query: str,
                                    top_k: int = 5,
                                    score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        获取相关文档（异步）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 相似度阈值
            
        Returns:
            相关文档列表
        """
        # 目前使用同步实现，未来可以改为真正的异步
        return self.get_relevant_documents(query, top_k, score_threshold)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        return self.vector_retrieval.get_database_stats()

class LangChainCompatibleVectorStore:
    """LangChain兼容的向量存储类"""
    
    def __init__(self, vector_retrieval: VectorRetrieval = None):
        """
        初始化向量存储
        
        Args:
            vector_retrieval: 向量检索系统实例
        """
        if vector_retrieval is None:
            self.vector_retrieval = VectorRetrieval()
        else:
            self.vector_retrieval = vector_retrieval
    
    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        """
        相似度搜索，返回文本列表
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            文本列表
        """
        try:
            results = self.vector_retrieval.search(query=query, top_k=k)
            return [result["text"] for result in results]
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文本, 分数)的元组列表
        """
        try:
            results = self.vector_retrieval.search(query=query, top_k=k)
            return [(result["text"], result.get("score", 0.0)) for result in results]
        except Exception as e:
            logger.error(f"带分数相似度搜索失败: {e}")
            return []
    
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """
        添加文本到向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            
        Returns:
            文档ID列表
        """
        try:
            # 准备数据
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append({
                    "text": text,
                    "metadata": metadata,
                    "source": "langchain_add",
                    "original_id": f"langchain_{i}"
                })
            
            # 生成向量
            vectorized_data = self.vector_retrieval.embedding_service.encode_data_batch(documents)
            
            # 添加到数据库
            success = self.vector_retrieval.chroma_manager.add_documents(vectorized_data)
            
            if success:
                return [doc["original_id"] for doc in vectorized_data]
            else:
                return []
                
        except Exception as e:
            logger.error(f"添加文本失败: {e}")
            return []
    
    @classmethod
    def from_texts(cls, texts: List[str], 
                   metadatas: List[Dict[str, Any]] = None,
                   persist_directory: Union[str, Path] = None,
                   collection_name: str = None) -> 'LangChainCompatibleVectorStore':
        """
        从文本列表创建向量存储
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            persist_directory: 持久化目录
            collection_name: 集合名称
            
        Returns:
            向量存储实例
        """
        # 创建向量检索系统
        vector_retrieval = VectorRetrieval(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # 创建向量存储实例
        store = cls(vector_retrieval)
        
        # 添加文本
        store.add_texts(texts, metadatas)
        
        return store

class TCMDocumentProcessor:
    """中医文档处理器，用于LangChain集成"""
    
    @staticmethod
    def process_tcm_document(document: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理中医文档，提取关键信息
        
        Args:
            document: 原始文档
            
        Returns:
            处理后的文档
        """
        try:
            # 提取关键信息
            text = document.get("text", "")
            metadata = document.get("metadata", {})
            
            # 提取症状、诊断、方剂等信息
            processed_metadata = {
                "source": metadata.get("source", "unknown"),
                "output": metadata.get("output", ""),
                "instruction": metadata.get("instruction", ""),
                "input": metadata.get("input", ""),
                "document_type": "tcm_case"
            }
            
            # 识别中医术语
            tcm_terms = TCMDocumentProcessor._extract_tcm_terms(text)
            processed_metadata["tcm_terms"] = tcm_terms
            
            return {
                "text": text,
                "metadata": processed_metadata
            }
            
        except Exception as e:
            logger.error(f"处理中医文档失败: {e}")
            return document
    
    @staticmethod
    def _extract_tcm_terms(text: str) -> List[str]:
        """
        提取中医术语
        
        Args:
            text: 输入文本
            
        Returns:
            中医术语列表
        """
        # 简单的中医术语识别（可以后续优化）
        tcm_keywords = [
            "症状", "诊断", "方剂", "药材", "脉象", "舌象", 
            "辨证", "治法", "处方", "复诊", "按语",
            "湿热", "血虚", "风燥", "阴虚", "阳虚", "气虚",
            "当归", "川芎", "生地黄", "白芍", "甘草"
        ]
        
        found_terms = []
        for term in tcm_keywords:
            if term in text:
                found_terms.append(term)
        
        return found_terms

# LangChain集成工厂函数
def create_tcm_retriever(persist_directory: Union[str, Path] = None,
                        collection_name: str = None) -> TCMVectorRetriever:
    """
    创建中医向量检索器
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
        
    Returns:
        中医向量检索器实例
    """
    vector_retrieval = VectorRetrieval(
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return TCMVectorRetriever(vector_retrieval)

def create_tcm_vector_store(persist_directory: Union[str, Path] = None,
                          collection_name: str = None) -> LangChainCompatibleVectorStore:
    """
    创建中医向量存储
    
    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称
        
    Returns:
        中医向量存储实例
    """
    vector_retrieval = VectorRetrieval(
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return LangChainCompatibleVectorStore(vector_retrieval)
