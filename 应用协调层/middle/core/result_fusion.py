"""
结果融合器
实现RRF和加权融合算法，融合多个检索模块的结果
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
from datetime import datetime

from ..models.data_models import (
    RetrievalResult, FusedResult, RetrievalSource, FusionMethod
)
from ..models.exceptions import FusionError, ValidationError
from ..utils.logging_utils import get_logger


class ResultFusion:
    """结果融合器类"""
    
    def __init__(self, 
                 default_method: FusionMethod = FusionMethod.RRF,
                 rrf_k: int = 60,  # 标准RRF k值，降低排名差异的影响
                 default_weights: Optional[Dict[str, float]] = None):
        """
        初始化结果融合器
        
        Args:
            default_method: 默认融合方法
            rrf_k: RRF算法的k参数
            default_weights: 默认权重配置
        """
        self.default_method = default_method
        self.rrf_k = rrf_k
        self.default_weights = default_weights or {
            "vector": 0.5,
            "graph": 0.5
        }
        self.logger = get_logger(__name__)
        
        # 融合统计信息
        self.fusion_stats = {
            "total_fusions": 0,
            "method_usage": defaultdict(int),
            "average_input_size": 0.0,
            "average_output_size": 0.0
        }
    
    def fuse_results(self, 
                    results_by_source: Dict[str, List[RetrievalResult]],
                    method: Optional[FusionMethod] = None,
                    weights: Optional[Dict[str, float]] = None,
                    top_k: int = 10) -> List[FusedResult]:
        """
        融合多个来源的检索结果
        
        Args:
            results_by_source: 按来源分组的检索结果
            method: 融合方法
            weights: 权重配置（用于加权融合）
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        try:
            # 参数验证
            if not results_by_source:
                return []
            
            method = method or self.default_method
            weights = weights or self.default_weights
            
            # 确保 method 是 FusionMethod 枚举类型
            if isinstance(method, str):
                try:
                    method = FusionMethod(method)
                except ValueError:
                    self.logger.error(f"无效的融合方法字符串: {method}")
                    method = self.default_method
            elif method is not None and not isinstance(method, FusionMethod):
                self.logger.warning(f"method 参数类型错误: {type(method)}, 使用默认方法")
                method = self.default_method
            
            # 更新统计信息
            self._update_fusion_stats(results_by_source, method)
            
            # 根据方法选择融合算法
            if method == FusionMethod.RRF:
                fused_results = self._rrf_fusion(results_by_source, top_k)
            elif method == FusionMethod.WEIGHTED:
                fused_results = self._weighted_fusion(results_by_source, weights, top_k)
            elif method == FusionMethod.RANK_BASED:
                fused_results = self._rank_based_fusion(results_by_source, weights, top_k)
            else:
                raise FusionError(str(method), f"不支持的融合方法: {method}")
            
            self.logger.info(f"融合完成: 方法={method.value}, 输入源数={len(results_by_source)}, 输出数={len(fused_results)}")
            return fused_results
            
        except Exception as e:
            error_msg = f"结果融合失败: {str(e)}"
            self.logger.error(error_msg)
            raise FusionError(str(method) if method else "unknown", error_msg) from e
    
    def _rrf_fusion(self, 
                   results_by_source: Dict[str, List[RetrievalResult]], 
                   top_k: int) -> List[FusedResult]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        
        算法公式: score = Σ(1/(k + rank_i))
        其中 k 是常数（通常为60），rank_i 是结果在第i个列表中的排名
        """
        try:
            # 收集所有唯一的内容
            content_to_results = {}
            content_to_ranks = defaultdict(dict)
            content_to_original_scores = defaultdict(dict)
            
            # 为每个来源的结果分配排名和存储原始评分
            for source, results in results_by_source.items():
                for rank, result in enumerate(results):
                    content = result.content
                    
                    # 存储结果信息
                    if content not in content_to_results:
                        content_to_results[content] = result
                    
                    # 存储排名信息
                    content_to_ranks[content][source] = rank + 1  # 排名从1开始
                    # 存储原始评分
                    content_to_original_scores[content][source] = result.score
            
            # 计算RRF评分
            fused_results = []
            for content, result in content_to_results.items():
                rrf_score = 0.0
                source_scores = {}
                contributing_sources = []
                rrf_contributions = {}  # 记录每个来源的RRF贡献
                
                # 计算每个来源的RRF贡献
                for source, results in results_by_source.items():
                    if source in content_to_ranks[content]:
                        rank = content_to_ranks[content][source]
                        rrf_contribution = 1.0 / (self.rrf_k + rank)
                        rrf_score += rrf_contribution
                        rrf_contributions[source] = rrf_contribution
                        
                        # 存储原始评分，而不是RRF贡献值
                        source_scores[source] = content_to_original_scores[content][source]
                        
                        # 尝试转换为RetrievalSource，如果失败则使用字符串
                        try:
                            contributing_sources.append(RetrievalSource(source))
                        except ValueError:
                            # 对于测试中的自定义来源名称，我们跳过或使用默认值
                            pass
                    else:
                        source_scores[source] = 0.0
                        rrf_contributions[source] = 0.0
                
                # 记录详细的RRF计算过程
                self.logger.debug(f"RRF计算 - 内容: {content[:50]}...")
                self.logger.debug(f"  排名信息: {content_to_ranks[content]}")
                self.logger.debug(f"  原始评分: {content_to_original_scores[content]}")
                self.logger.debug(f"  RRF贡献: {rrf_contributions}")
                self.logger.debug(f"  最终RRF评分: {rrf_score:.6f}")
                
                # 创建融合结果
                fused_result = FusedResult(
                    content=content,
                    fused_score=rrf_score,
                    source_scores=source_scores,
                    fusion_method=FusionMethod.RRF,
                    metadata=result.metadata.copy(),
                    contributing_sources=contributing_sources,
                    entities=result.entities,
                    relationships=result.relationships,
                    timestamp=datetime.now()
                )
                
                fused_results.append(fused_result)
            
            # 按RRF评分排序并返回top_k
            fused_results.sort(key=lambda x: x.fused_score, reverse=True)
            
            # 记录融合统计
            self.logger.info(f"RRF融合统计: 总计{len(fused_results)}个唯一文档, 返回top {top_k}")
            source_counts = {}
            for result in fused_results[:top_k]:
                for source in result.contributing_sources:
                    source_name = source.value if hasattr(source, 'value') else str(source)
                    source_counts[source_name] = source_counts.get(source_name, 0) + 1
            self.logger.info(f"Top {top_k} 来源分布: {source_counts}")
            
            # 记录被排除的向量结果
            if len(fused_results) > top_k:
                excluded = fused_results[top_k:]
                vector_excluded = [r for r in excluded if any(
                    (s.value if hasattr(s, 'value') else str(s)) == 'vector' 
                    for s in r.contributing_sources
                )]
                if vector_excluded:
                    self.logger.warning(f"有{len(vector_excluded)}个向量结果被排除在top {top_k}之外")
                    self.logger.debug(f"首个被排除的向量结果: RRF={vector_excluded[0].fused_score:.6f}")
            
            return fused_results[:top_k]
            
        except Exception as e:
            raise FusionError("rrf", f"RRF融合失败: {str(e)}") from e
    
    def _weighted_fusion(self, 
                        results_by_source: Dict[str, List[RetrievalResult]],
                        weights: Dict[str, float],
                        top_k: int) -> List[FusedResult]:
        """
        加权融合算法
        
        算法公式: score = Σ(weight_i * normalized_score_i)
        """
        try:
            # 验证权重
            self._validate_weights(weights, list(results_by_source.keys()))
            
            # 标准化各来源的评分
            normalized_results = self._normalize_scores_by_source(results_by_source)
            
            # 收集所有唯一内容
            content_to_results = {}
            content_to_scores = defaultdict(dict)
            
            # 收集每个内容在各来源中的标准化评分
            for source, results in normalized_results.items():
                weight = weights.get(source, 0.0)
                
                for result in results:
                    content = result.content
                    
                    if content not in content_to_results:
                        content_to_results[content] = result
                    
                    # 加权评分
                    weighted_score = result.score * weight
                    content_to_scores[content][source] = weighted_score
            
            # 计算最终融合评分
            fused_results = []
            for content, result in content_to_results.items():
                total_score = sum(content_to_scores[content].values())
                source_scores = content_to_scores[content].copy()
                contributing_sources = []
                for source in source_scores.keys():
                    try:
                        contributing_sources.append(RetrievalSource(source))
                    except ValueError:
                        # 对于测试中的自定义来源名称，我们跳过
                        pass
                
                # 创建融合结果
                fused_result = FusedResult(
                    content=content,
                    fused_score=total_score,
                    source_scores=source_scores,
                    fusion_method=FusionMethod.WEIGHTED,
                    metadata=result.metadata.copy(),
                    contributing_sources=contributing_sources,
                    entities=result.entities,
                    relationships=result.relationships,
                    timestamp=datetime.now()
                )
                
                fused_results.append(fused_result)
            
            # 按加权评分排序并返回top_k
            fused_results.sort(key=lambda x: x.fused_score, reverse=True)
            return fused_results[:top_k]
            
        except Exception as e:
            raise FusionError("weighted", f"加权融合失败: {str(e)}") from e
    
    def _rank_based_fusion(self,
                          results_by_source: Dict[str, List[RetrievalResult]],
                          weights: Dict[str, float],
                          top_k: int) -> List[FusedResult]:
        """
        基于排名的融合算法
        结合排名和权重进行融合
        """
        try:
            # 验证权重
            self._validate_weights(weights, list(results_by_source.keys()))
            
            content_to_results = {}
            content_to_rank_scores = defaultdict(dict)
            
            # 计算基于排名的评分
            for source, results in results_by_source.items():
                weight = weights.get(source, 0.0)
                
                for rank, result in enumerate(results):
                    content = result.content
                    
                    if content not in content_to_results:
                        content_to_results[content] = result
                    
                    # 排名评分：排名越靠前评分越高
                    rank_score = 1.0 / (rank + 1)  # 排名从1开始
                    weighted_rank_score = rank_score * weight
                    content_to_rank_scores[content][source] = weighted_rank_score
            
            # 创建融合结果
            fused_results = []
            for content, result in content_to_results.items():
                total_score = sum(content_to_rank_scores[content].values())
                source_scores = content_to_rank_scores[content].copy()
                contributing_sources = []
                for source in source_scores.keys():
                    try:
                        contributing_sources.append(RetrievalSource(source))
                    except ValueError:
                        # 对于测试中的自定义来源名称，我们跳过
                        pass
                
                fused_result = FusedResult(
                    content=content,
                    fused_score=total_score,
                    source_scores=source_scores,
                    fusion_method=FusionMethod.RANK_BASED,
                    metadata=result.metadata.copy(),
                    contributing_sources=contributing_sources,
                    entities=result.entities,
                    relationships=result.relationships,
                    timestamp=datetime.now()
                )
                
                fused_results.append(fused_result)
            
            # 按评分排序并返回top_k
            fused_results.sort(key=lambda x: x.fused_score, reverse=True)
            return fused_results[:top_k]
            
        except Exception as e:
            raise FusionError("rank_based", f"排名融合失败: {str(e)}") from e
    
    def _normalize_scores_by_source(self, 
                                   results_by_source: Dict[str, List[RetrievalResult]]
                                   ) -> Dict[str, List[RetrievalResult]]:
        """
        按来源标准化评分
        将每个来源的评分标准化到[0,1]范围
        """
        normalized_results = {}
        
        for source, results in results_by_source.items():
            if not results:
                normalized_results[source] = []
                continue
            
            # 获取最大和最小评分
            scores = [result.score for result in results]
            max_score = max(scores)
            min_score = min(scores)
            
            # 避免除零错误
            score_range = max_score - min_score
            if score_range == 0:
                # 所有评分相同，设为1.0
                normalized_results[source] = [
                    RetrievalResult(
                        content=result.content,
                        score=1.0,
                        source=result.source,
                        metadata=result.metadata,
                        entities=result.entities,
                        relationships=result.relationships,
                        timestamp=result.timestamp
                    ) for result in results
                ]
            else:
                # 线性标准化到[0,1]
                normalized_results[source] = [
                    RetrievalResult(
                        content=result.content,
                        score=(result.score - min_score) / score_range,
                        source=result.source,
                        metadata=result.metadata,
                        entities=result.entities,
                        relationships=result.relationships,
                        timestamp=result.timestamp
                    ) for result in results
                ]
        
        return normalized_results
    
    def _validate_weights(self, weights: Dict[str, float], sources: List[str]):
        """验证权重配置"""
        # 检查权重是否为负数
        for source, weight in weights.items():
            if weight < 0:
                raise ValidationError("weights", weight, f"权重不能为负数: {source}={weight}")
        
        # 检查是否所有来源都有权重
        missing_sources = set(sources) - set(weights.keys())
        if missing_sources:
            self.logger.warning(f"以下来源缺少权重配置，将使用0.0: {missing_sources}")
        
        # 检查权重总和
        total_weight = sum(weights.get(source, 0.0) for source in sources)
        if total_weight == 0:
            raise ValidationError("weights", weights, "权重总和不能为0")
        
        # 标准化权重（可选）
        if abs(total_weight - 1.0) > 0.01:
            self.logger.info(f"权重总和为{total_weight:.3f}，建议标准化为1.0")
    
    def _update_fusion_stats(self, 
                           results_by_source: Dict[str, List[RetrievalResult]], 
                           method: FusionMethod):
        """更新融合统计信息"""
        self.fusion_stats["total_fusions"] += 1
        self.fusion_stats["method_usage"][method.value] += 1
        
        # 计算平均输入大小
        total_input_size = sum(len(results) for results in results_by_source.values())
        current_avg = self.fusion_stats["average_input_size"]
        total_count = self.fusion_stats["total_fusions"]
        self.fusion_stats["average_input_size"] = (
            (current_avg * (total_count - 1) + total_input_size) / total_count
        )
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        return {
            "total_fusions": self.fusion_stats["total_fusions"],
            "method_usage": dict(self.fusion_stats["method_usage"]),
            "average_input_size": self.fusion_stats["average_input_size"],
            "average_output_size": self.fusion_stats["average_output_size"],
            "rrf_k_parameter": self.rrf_k,
            "default_weights": self.default_weights.copy()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.fusion_stats = {
            "total_fusions": 0,
            "method_usage": defaultdict(int),
            "average_input_size": 0.0,
            "average_output_size": 0.0
        }
        self.logger.info("融合统计信息已重置")
    
    # 高级融合功能
    
    def adaptive_fusion(self, 
                       results_by_source: Dict[str, List[RetrievalResult]],
                       query_context: Optional[Dict[str, Any]] = None,
                       top_k: int = 10) -> List[FusedResult]:
        """
        自适应融合
        根据查询上下文和结果质量自动选择最佳融合方法
        """
        try:
            # 分析结果质量
            quality_analysis = self._analyze_result_quality(results_by_source)
            
            # 根据分析结果选择融合方法
            if quality_analysis["score_variance"] > 0.3:
                # 评分差异较大，使用RRF
                method = FusionMethod.RRF
                weights = None
            elif quality_analysis["source_balance"] < 0.5:
                # 来源不平衡，使用加权融合
                method = FusionMethod.WEIGHTED
                weights = self._calculate_adaptive_weights(results_by_source, quality_analysis)
            else:
                # 使用默认方法
                method = self.default_method
                weights = self.default_weights
            
            self.logger.info(f"自适应融合选择方法: {method.value}")
            return self.fuse_results(results_by_source, method, weights, top_k)
            
        except Exception as e:
            self.logger.error(f"自适应融合失败: {str(e)}")
            # 降级到默认方法
            return self.fuse_results(results_by_source, self.default_method, self.default_weights, top_k)
    
    def _analyze_result_quality(self, 
                               results_by_source: Dict[str, List[RetrievalResult]]
                               ) -> Dict[str, float]:
        """分析结果质量"""
        analysis = {
            "score_variance": 0.0,
            "source_balance": 0.0,
            "average_scores": {},
            "result_counts": {}
        }
        
        all_scores = []
        total_results = 0
        
        # 收集统计信息
        for source, results in results_by_source.items():
            if results:
                scores = [result.score for result in results]
                analysis["average_scores"][source] = sum(scores) / len(scores)
                analysis["result_counts"][source] = len(results)
                all_scores.extend(scores)
                total_results += len(results)
        
        # 计算评分方差
        if all_scores:
            mean_score = sum(all_scores) / len(all_scores)
            variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
            analysis["score_variance"] = math.sqrt(variance)
        
        # 计算来源平衡度
        if len(results_by_source) > 1 and total_results > 0:
            counts = list(analysis["result_counts"].values())
            max_count = max(counts)
            min_count = min(counts)
            analysis["source_balance"] = min_count / max_count if max_count > 0 else 0.0
        
        return analysis
    
    def _calculate_adaptive_weights(self, 
                                   results_by_source: Dict[str, List[RetrievalResult]],
                                   quality_analysis: Dict[str, float]) -> Dict[str, float]:
        """计算自适应权重"""
        weights = {}
        
        # 基于结果数量和平均评分计算权重
        total_score = 0.0
        source_scores = {}
        
        for source, results in results_by_source.items():
            if results:
                avg_score = quality_analysis["average_scores"].get(source, 0.0)
                result_count = len(results)
                
                # 综合评分：平均分数 * log(结果数量 + 1)
                combined_score = avg_score * math.log(result_count + 1)
                source_scores[source] = combined_score
                total_score += combined_score
        
        # 标准化权重
        if total_score > 0:
            for source in results_by_source.keys():
                weights[source] = source_scores.get(source, 0.0) / total_score
        else:
            # 均匀分配权重
            num_sources = len(results_by_source)
            for source in results_by_source.keys():
                weights[source] = 1.0 / num_sources if num_sources > 0 else 0.0
        
        return weights
    
    def diversity_fusion(self, 
                        results_by_source: Dict[str, List[RetrievalResult]],
                        diversity_threshold: float = 0.7,
                        top_k: int = 10) -> List[FusedResult]:
        """
        多样性融合
        在保证相关性的同时增加结果多样性
        """
        try:
            # 首先进行常规融合
            initial_results = self.fuse_results(results_by_source, top_k=top_k * 2)
            
            # 多样性选择
            diverse_results = []
            selected_contents = set()
            
            for result in initial_results:
                # 计算与已选结果的相似度
                similarity = self._calculate_content_similarity(
                    result.content, [r.content for r in diverse_results]
                )
                
                # 如果相似度低于阈值或者是第一个结果，则选择
                if similarity < diversity_threshold or len(diverse_results) == 0:
                    diverse_results.append(result)
                    selected_contents.add(result.content)
                    
                    if len(diverse_results) >= top_k:
                        break
            
            self.logger.info(f"多样性融合完成: 原始{len(initial_results)}个，多样性{len(diverse_results)}个")
            return diverse_results
            
        except Exception as e:
            self.logger.error(f"多样性融合失败: {str(e)}")
            # 降级到常规融合
            return self.fuse_results(results_by_source, top_k=top_k)
    
    def _calculate_content_similarity(self, content: str, existing_contents: List[str]) -> float:
        """计算内容相似度（简单的基于词汇重叠）"""
        if not existing_contents:
            return 0.0
        
        content_words = set(content.split())
        max_similarity = 0.0
        
        for existing_content in existing_contents:
            existing_words = set(existing_content.split())
            
            if len(content_words) == 0 and len(existing_words) == 0:
                similarity = 1.0
            elif len(content_words) == 0 or len(existing_words) == 0:
                similarity = 0.0
            else:
                # Jaccard相似度
                intersection = len(content_words & existing_words)
                union = len(content_words | existing_words)
                similarity = intersection / union if union > 0 else 0.0
            
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def explain_fusion(self, 
                      fused_result: FusedResult,
                      results_by_source: Dict[str, List[RetrievalResult]]) -> Dict[str, Any]:
        """
        解释融合结果
        提供融合过程的详细说明
        """
        explanation = {
            "fusion_method": fused_result.fusion_method.value,
            "final_score": fused_result.fused_score,
            "source_contributions": {},
            "ranking_details": {},
            "fusion_rationale": ""
        }
        
        # 分析各来源贡献
        for source, score in fused_result.source_scores.items():
            contribution_pct = (score / fused_result.fused_score * 100) if fused_result.fused_score > 0 else 0
            explanation["source_contributions"][source] = {
                "score": score,
                "contribution_percentage": contribution_pct
            }
        
        # 分析排名详情
        for source, results in results_by_source.items():
            for rank, result in enumerate(results):
                if result.content == fused_result.content:
                    explanation["ranking_details"][source] = {
                        "rank": rank + 1,
                        "original_score": result.score,
                        "total_results": len(results)
                    }
                    break
        
        # 生成融合理由
        if fused_result.fusion_method == FusionMethod.RRF:
            explanation["fusion_rationale"] = f"使用RRF算法融合，k={self.rrf_k}，综合考虑各来源的排名位置"
        elif fused_result.fusion_method == FusionMethod.WEIGHTED:
            explanation["fusion_rationale"] = "使用加权融合，根据预设权重组合各来源评分"
        elif fused_result.fusion_method == FusionMethod.RANK_BASED:
            explanation["fusion_rationale"] = "使用排名融合，基于排名位置和权重计算最终评分"
        
        return explanation