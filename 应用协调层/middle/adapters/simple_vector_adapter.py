#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的向量检索适配器
直接基于Faiss向量数据库，避免复杂的依赖关系
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "检索与知识层" / "faiss_rag"))

# 直接导入向量检索系统
try:
    from vector_retrieval_system.vector_retrieval import VectorRetrieval
    VECTOR_RETRIEVAL_AVAILABLE = True
except ImportError as e:
    logging.error(f"无法导入向量检索系统: {e}")
    VECTOR_RETRIEVAL_AVAILABLE = False
    VectorRetrieval = None

# 导入数据模型
# 添加应用协调层到路径，确保绝对导入可用
sys.path.insert(0, str(project_root / "应用协调层"))

try:
    from middle.models.data_models import VectorResult, RetrievalResult, RetrievalSource
except ImportError as e:
    logging.error(f"无法导入数据模型: {e}")
    # 不再使用相对导入，避免"attempted relative import beyond top-level package"错误
    raise

class SimpleVectorAdapter:
    """
    简化的向量检索适配器
    直接基于VectorRetrieval，提供标准化的异步接口
    支持关键词提取和文档增强功能
    """
    
    def __init__(self, 
                 persist_directory: str = None,
                 model_path: str = None,
                 timeout: int = 15,
                 score_threshold: float = 0.0,
                 enable_keyword_enhancement: bool = True,
                 keyword_csv_path: str = None):
        """
        初始化简化向量检索适配器
        
        Args:
            persist_directory: Faiss数据库持久化目录
            model_path: 嵌入模型路径
            timeout: 搜索超时时间（秒）
            score_threshold: 相似度阈值
            enable_keyword_enhancement: 是否启用关键词增强
            keyword_csv_path: 关键词库CSV文件路径
        """
        # 设置默认路径
        if persist_directory is None:
            persist_directory = str(project_root / "检索与知识层" / "faiss_rag" / "向量数据库_768维")
        
        if model_path is None:
            model_path = r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base"
        
        if keyword_csv_path is None:
            keyword_csv_path = str(project_root / "检索与知识层" / "keyword" / "knowledge_graph_entities_only.csv")
        
        self.persist_directory = persist_directory
        self.model_path = model_path
        self.timeout = timeout
        self.score_threshold = score_threshold
        self.enable_keyword_enhancement = enable_keyword_enhancement
        
        self._retrieval_system = None
        self._is_initialized = False
        self._initialization_error = None
        
        # 统计信息
        self._search_count = 0
        self._total_search_time = 0.0
        self._error_count = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"简化向量检索适配器初始化 - 数据库路径: {persist_directory}")
        
        # 初始化关键词库
        self.keyword_library = []
        if self.enable_keyword_enhancement:
            self.keyword_library = self._load_keyword_library(keyword_csv_path)
            self.logger.info(f"关键词增强功能已启用，加载了 {len(self.keyword_library)} 个关键词")
    
    def _load_keyword_library(self, keyword_csv_path: str) -> List[str]:
        """加载关键词库"""
        try:
            if not os.path.exists(keyword_csv_path):
                self.logger.warning(f"关键词库文件不存在: {keyword_csv_path}")
                return []
            
            # 读取文件，这是一个每行一个实体的CSV文件
            with open(keyword_csv_path, 'r', encoding='utf-8') as f:
                keywords = [line.strip() for line in f if line.strip()]
            
            # 去重并过滤（保留1个字符及以上的实体，支持单字实体如"姜"、"枣"）
            unique_keywords = list(set(keywords))
            filtered_keywords = [kw for kw in unique_keywords if len(kw) >= 1]
            
            # 定义停用词（过滤非实体词）
            stopwords = {
                '请', '如果', '就', '怎么', '什么', '哪些', '为什么', '如何', '哪个',
                '的', '了', '着', '过', '得', '地', '在', '是', '有', '和', '与', '或',
                '但', '而', '却', '都', '也', '还', '又', '再', '才', '只', '仅',
                '不', '没', '无', '非', '未', '勿', '别', '不要', '不用', '不必'
            }
            
            # 过滤停用词
            if stopwords:
                filtered_keywords = [kw for kw in filtered_keywords if kw not in stopwords]
            
            self.logger.info(f"✅ 加载了 {len(filtered_keywords)} 个关键词（原始: {len(keywords)}, 去重后: {len(unique_keywords)}）")
            return filtered_keywords
        except Exception as e:
            self.logger.error(f"❌ 加载关键词库失败: {e}")
            return []
    
    def _extract_keywords_from_document(self, document: str, top_k: int = 10) -> List[str]:
        """从文档中提取关键词"""
        if not self.keyword_library:
            return []
        
        # 使用高效的字符串匹配，按关键词长度排序（长关键词优先）
        found_keywords = []
        
        # 按关键词长度降序排序，避免短关键词覆盖长关键词
        sorted_keywords = sorted(self.keyword_library, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            if keyword in document:
                # 计算关键词在文档中的出现次数
                freq = document.count(keyword)
                if freq > 0:
                    # 计算关键词重要性分数（频率 * 长度权重）
                    importance_score = freq * len(keyword)
                    found_keywords.append((keyword, importance_score))
        
        # 按重要性分数排序并返回前k个
        found_keywords.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, score in found_keywords[:top_k]]
    
    def _enhance_document_with_keywords(self, document: str, metadata: dict = None, top_k: int = 8) -> str:
        """用关键词信息增强文档，优先使用metadata中的完整答案"""
        # 优先使用metadata中的完整答案内容
        if metadata and 'answer' in metadata and metadata['answer']:
            # 使用完整的答案内容
            answer_content = metadata['answer']
            question = metadata.get('question', '')
            
            # 提取关键词（从答案内容中提取）
            keywords = self._extract_keywords_from_document(answer_content, top_k)
            
            if keywords and self.enable_keyword_enhancement:
                # 构建增强文档：关键词 + 完整答案（避免重复）
                keyword_info = f"关键词: {', '.join(keywords)}"
                enhanced_doc = f"{keyword_info}\n\n文档内容:\n{answer_content}"
            else:
                # 直接返回完整答案
                enhanced_doc = answer_content
        else:
            # 降级：使用原始文档内容
            if self.enable_keyword_enhancement and self.keyword_library:
                keywords = self._extract_keywords_from_document(document, top_k)
                if keywords:
                    keyword_info = f"关键词: {', '.join(keywords)}"
                    enhanced_doc = f"{keyword_info}\n\n文档内容:\n{document}"
                else:
                    enhanced_doc = f"文档内容:\n{document}"
            else:
                enhanced_doc = document
        
        return enhanced_doc
    
    def _initialize_system(self):
        """初始化向量检索系统"""
        if self._is_initialized:
            return
        
        if not VECTOR_RETRIEVAL_AVAILABLE:
            error_msg = "向量检索系统不可用"
            self._initialization_error = error_msg
            raise RuntimeError(error_msg)
        
        try:
            self.logger.info("正在初始化向量检索系统...")
            start_time = time.time()
            
            # 检查数据库目录是否存在
            if not Path(self.persist_directory).exists():
                raise FileNotFoundError(f"向量数据库目录不存在: {self.persist_directory}")
            
            # 初始化向量检索系统
            # 为避免GPU Faiss在服务进程下异常，强制在服务端使用CPU索引加载
            self._retrieval_system = VectorRetrieval(
                persist_directory=self.persist_directory,
                model_path=self.model_path,
                use_gpu=False
            )
            
            self._is_initialized = True
            init_time = time.time() - start_time
            self.logger.info(f"✅ 向量检索系统初始化成功，耗时: {init_time:.2f}秒")
            try:
                stats = self._retrieval_system.get_database_stats()
                self.logger.info(f"向量库统计: total_documents={stats.get('total_documents')}, dimension={stats.get('embedding_dimension')}, backend={stats.get('backend')}, dir={stats.get('persist_directory', 'N/A')}")
            except Exception as _e:
                self.logger.warning(f"无法获取向量库统计信息: {_e}")
            
        except Exception as e:
            error_msg = f"向量检索系统初始化失败: {e}"
            self._initialization_error = error_msg
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def search(self, query: str, top_k: int = 5) -> List[VectorResult]:
        """
        异步搜索向量数据库
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self._is_initialized:
            self._initialize_system()
        
        if self._retrieval_system is None:
            raise RuntimeError("向量检索系统未初始化")
        
        try:
            start_time = time.time()
            
            # 执行搜索
            results = self._retrieval_system.search(query, top_k=top_k)
            
            # 转换为标准格式并应用关键词增强
            vector_results = []
            for result in results:
                content = result.get('content', result.get('text', ''))
                metadata = result.get('metadata', {})
                score = float(result.get('score', result.get('distance', 0.0)))
                document_id = result.get('id', result.get('doc_id', ''))
                
                # 如果分数低于阈值，跳过
                if score < self.score_threshold:
                    continue
                
                # 检查是否有完整的对话格式
                full_conversation = metadata.get('full_conversation')
                if full_conversation:
                    # 如果有完整对话格式，使用答案作为内容
                    answer = metadata.get('answer', '')
                    enhanced_content = self._enhance_document_with_keywords(answer, metadata, top_k=8)
                else:
                    # 否则使用原始内容
                    enhanced_content = self._enhance_document_with_keywords(content, metadata, top_k=8)
                
                vector_result = VectorResult(
                    content=enhanced_content,
                    score=score,
                    similarity_score=score,
                    embedding_model=self.model_path,
                    document_id=document_id,
                    metadata=metadata  # 传递完整metadata，包含对话格式
                )
                vector_results.append(vector_result)
            
            # 更新统计信息
            search_time = time.time() - start_time
            self._search_count += 1
            self._total_search_time += search_time
            
            self.logger.debug(f"向量搜索完成: {len(vector_results)}个结果，耗时: {search_time:.3f}秒")
            if len(vector_results) == 0:
                try:
                    stats = self._retrieval_system.get_database_stats()
                    self.logger.warning(f"向量搜索返回0结果，库统计: total_documents={stats.get('total_documents')}, dimension={stats.get('embedding_dimension')}, dir={stats.get('persist_directory', 'N/A')}")
                except Exception as _e:
                    self.logger.warning(f"无法获取库统计进行诊断: {_e}")
            return vector_results
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"向量搜索失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self._total_search_time / max(self._search_count, 1)
        return {
            "search_count": self._search_count,
            "error_count": self._error_count,
            "average_search_time": avg_time,
            "total_search_time": self._total_search_time,
            "is_initialized": self._is_initialized,
            "initialization_error": self._initialization_error
        }
    
    def is_available(self) -> bool:
        """检查适配器是否可用"""
        return self._is_initialized and self._retrieval_system is not None
