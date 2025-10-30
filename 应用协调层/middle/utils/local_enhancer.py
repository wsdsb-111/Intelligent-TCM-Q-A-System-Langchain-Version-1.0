"""
基于本地模型的查询扩展和重排序
使用 text2vec-paraphrase 和 bge-reranker
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from ..utils.logging_utils import get_logger


class LocalQueryExpander:
    """基于text2vec-paraphrase的查询扩展器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化查询扩展器
        
        Args:
            model_path: text2vec-paraphrase模型路径
            device: 设备 (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = get_logger(__name__)
        
        self.model = None
        self._load_model()
        
        # 中医术语改写库
        self.paraphrase_templates = {
            "治疗": ["调理", "医治", "疗法"],
            "感冒": ["风寒", "风热", "外感"],
            "头痛": ["头疼", "偏头痛"],
            "咳嗽": ["咳", "久咳"],
            "方法": ["方式", "疗法"],
            "推荐": ["建议", "介绍"],
        }
    
    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"加载text2vec-paraphrase模型: {self.model_path}")
            
            self.model = SentenceTransformer(
                self.model_path,
                device=self.device
            )
            
            self.logger.info(f"✅ 查询扩展模型加载成功")
            self.logger.info(f"   设备: {self.device}")
            self.logger.info(f"   维度: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    async def expand_query(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        扩展查询
        
        Args:
            query: 原始查询
            max_expansions: 最大扩展数量
            
        Returns:
            扩展后的查询列表
        """
        self.logger.info(f"扩展查询: '{query}'")
        
        expanded_queries = [query]  # 始终包含原始查询
        
        try:
            # 生成术语替换的候选查询
            candidates = self._generate_candidates(query)
            
            if not candidates:
                self.logger.info("未生成候选查询，返回原始查询")
                return expanded_queries
            
            # 计算相似度
            query_emb = self.model.encode(query, convert_to_tensor=True)
            candidate_embs = self.model.encode(candidates, convert_to_tensor=True)
            
            # 计算余弦相似度
            from torch.nn.functional import cosine_similarity
            similarities = cosine_similarity(query_emb.unsqueeze(0), candidate_embs)
            
            # 选择top-k最相似的（相似度 > 0.7）
            for i, sim in enumerate(similarities):
                if sim > 0.7 and len(expanded_queries) < max_expansions + 1:
                    expanded_queries.append(candidates[i])
                    self.logger.debug(f"  扩展: {candidates[i]} (相似度={sim:.4f})")
            
            self.logger.info(f"查询扩展完成: {len(expanded_queries)}个查询")
            
        except Exception as e:
            self.logger.error(f"查询扩展失败: {e}")
        
        return expanded_queries
    
    def _generate_candidates(self, query: str) -> List[str]:
        """生成候选改写查询"""
        candidates = []
        
        # 基于术语替换生成候选
        for term, paraphrases in self.paraphrase_templates.items():
            if term in query:
                for para in paraphrases:
                    candidate = query.replace(term, para)
                    if candidate != query and candidate not in candidates:
                        candidates.append(candidate)
        
        return candidates[:10]  # 最多10个候选


class LocalReranker:
    """基于bge-reranker的重排序器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化重排序器
        
        Args:
            model_path: bge-reranker模型路径
            device: 设备 (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = get_logger(__name__)
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            self.logger.info(f"加载bge-reranker模型: {self.model_path}")
            
            self.model = CrossEncoder(
                self.model_path,
                max_length=512,
                device=self.device
            )
            
            self.logger.info(f"✅ 重排序模型加载成功")
            self.logger.info(f"   设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    async def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k结果
            
        Returns:
            [(原始索引, 分数), ...] 按分数降序排列
        """
        if not documents:
            return []
        
        self.logger.info(f"重排序: {len(documents)}个文档")
        
        try:
            # 构建query-document对
            pairs = [[query, doc] for doc in documents]
            
            # 批量预测
            raw_scores = self.model.predict(pairs, show_progress_bar=False)
            
            # bge-reranker返回的是logits（范围约-10到10），需要归一化到0-1
            # 使用sigmoid函数归一化
            normalized_scores = 1 / (1 + np.exp(-np.array(raw_scores)))
            
            # 创建索引-分数对并排序
            indexed_scores = list(enumerate(normalized_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            avg_raw_score = np.mean(raw_scores)
            avg_norm_score = np.mean(normalized_scores)
            self.logger.info(f"重排序完成: 原始平均分数={avg_raw_score:.4f}, 归一化平均分数={avg_norm_score:.4f}")
            
            return indexed_scores[:top_k]
            
        except Exception as e:
            self.logger.error(f"重排序失败: {e}")
            # 失败时返回原始顺序
            return [(i, 1.0) for i in range(min(top_k, len(documents)))]


# 便捷函数
def create_local_expander(model_path: str) -> LocalQueryExpander:
    """创建本地查询扩展器"""
    return LocalQueryExpander(model_path)


def create_local_reranker(model_path: str) -> LocalReranker:
    """创建本地重排序器"""
    return LocalReranker(model_path)

