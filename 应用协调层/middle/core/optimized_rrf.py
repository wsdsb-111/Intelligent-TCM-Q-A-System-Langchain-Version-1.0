"""
优化的RRF算法实现
提供更高效的RRF计算和批量处理功能
"""

import time
import math
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

from middle.models.data_models import RetrievalResult, FusedResult, RetrievalSource, FusionMethod
from middle.models.exceptions import FusionError


@dataclass
class RRFConfig:
    """RRF配置"""
    k: int = 60  # RRF常数
    max_results: int = 1000  # 最大结果数
    batch_size: int = 100  # 批处理大小
    enable_preprocessing: bool = True  # 启用预处理
    enable_parallel_processing: bool = False  # 启用并行处理


class OptimizedRRF:
    """优化的RRF算法实现"""
    
    def __init__(self, config: RRFConfig = None):
        """
        初始化优化的RRF算法
        
        Args:
            config: RRF配置
        """
        self.config = config or RRFConfig()
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self.stats = {
            'total_fusions': 0,
            'total_time': 0.0,
            'avg_time_per_fusion': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def fuse_results(self, results_by_source: Dict[str, List[RetrievalResult]], 
                    top_k: int) -> List[FusedResult]:
        """
        优化的RRF融合算法
        
        Args:
            results_by_source: 按来源分组的结果
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        start_time = time.time()
        
        try:
            # 预处理阶段
            if self.config.enable_preprocessing:
                results_by_source = self._preprocess_results(results_by_source)
            
            # 执行RRF计算
            fused_results = self._calculate_rrf(results_by_source, top_k)
            
            # 后处理阶段
            fused_results = self._postprocess_results(fused_results)
            
            # 更新统计信息
            execution_time = time.time() - start_time
            self._update_stats(execution_time)
            
            self.logger.debug(f"RRF融合完成: {len(fused_results)}个结果, 耗时{execution_time:.3f}秒")
            
            return fused_results
            
        except Exception as e:
            raise FusionError("optimized_rrf", f"优化RRF融合失败: {str(e)}") from e
    
    def _preprocess_results(self, results_by_source: Dict[str, List[RetrievalResult]]) -> Dict[str, List[RetrievalResult]]:
        """预处理结果"""
        processed_results = {}
        
        for source, results in results_by_source.items():
            if not results:
                continue
            
            # 限制结果数量
            if len(results) > self.config.max_results:
                results = results[:self.config.max_results]
                self.logger.debug(f"限制{source}源结果数量: {len(results)}")
            
            # 过滤无效结果
            valid_results = []
            for result in results:
                if (isinstance(result, RetrievalResult) and 
                    result.content and 
                    isinstance(result.score, (int, float)) and
                    not math.isnan(result.score) and
                    not math.isinf(result.score)):
                    valid_results.append(result)
            
            if valid_results:
                processed_results[source] = valid_results
        
        return processed_results
    
    def _calculate_rrf(self, results_by_source: Dict[str, List[RetrievalResult]], 
                      top_k: int) -> List[FusedResult]:
        """计算RRF评分"""
        # 收集所有唯一内容
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
        
        # 批量计算RRF评分
        fused_results = []
        for content, result in content_to_results.items():
            rrf_score = self._calculate_single_rrf_score(
                content, content_to_ranks, content_to_original_scores, results_by_source
            )
            
            if rrf_score > 0:  # 只保留有效评分的结果
                fused_result = self._create_fused_result(
                    result, rrf_score, content_to_original_scores[content], results_by_source
                )
                fused_results.append(fused_result)
        
        # 按RRF评分排序并返回top_k
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)
        return fused_results[:top_k]
    
    def _calculate_single_rrf_score(self, content: str, content_to_ranks: Dict[str, Dict[str, int]], 
                                   content_to_original_scores: Dict[str, Dict[str, float]], 
                                   results_by_source: Dict[str, List[RetrievalResult]]) -> float:
        """计算单个内容的RRF评分"""
        rrf_score = 0.0
        
        for source in results_by_source.keys():
            if source in content_to_ranks[content]:
                rank = content_to_ranks[content][source]
                rrf_contribution = 1.0 / (self.config.k + rank)
                rrf_score += rrf_contribution
        
        return rrf_score
    
    def _create_fused_result(self, result: RetrievalResult, rrf_score: float, 
                           source_scores: Dict[str, float], 
                           results_by_source: Dict[str, List[RetrievalResult]]) -> FusedResult:
        """创建融合结果"""
        # 确定贡献来源
        contributing_sources = []
        for source in results_by_source.keys():
            if source in source_scores:
                try:
                    contributing_sources.append(RetrievalSource(source))
                except ValueError:
                    # 对于测试中的自定义来源名称，跳过
                    pass
        
        return FusedResult(
            content=result.content,
            fused_score=rrf_score,
            source_scores=source_scores,
            fusion_method=FusionMethod.RRF,
            metadata=result.metadata.copy() if result.metadata else {},
            contributing_sources=contributing_sources,
            entities=result.entities,
            relationships=result.relationships,
            timestamp=time.time()
        )
    
    def _postprocess_results(self, fused_results: List[FusedResult]) -> List[FusedResult]:
        """后处理结果"""
        if not fused_results:
            return fused_results
        
        # 归一化评分
        max_score = max(result.fused_score for result in fused_results)
        if max_score > 0:
            for result in fused_results:
                result.fused_score = result.fused_score / max_score
        
        return fused_results
    
    def _update_stats(self, execution_time: float):
        """更新统计信息"""
        self.stats['total_fusions'] += 1
        self.stats['total_time'] += execution_time
        self.stats['avg_time_per_fusion'] = self.stats['total_time'] / self.stats['total_fusions']
    
    def batch_fuse_results(self, batch_results: List[Dict[str, List[RetrievalResult]]], 
                          top_k: int) -> List[List[FusedResult]]:
        """批量融合结果"""
        start_time = time.time()
        batch_fused_results = []
        
        for i, results_by_source in enumerate(batch_results):
            try:
                fused_results = self.fuse_results(results_by_source, top_k)
                batch_fused_results.append(fused_results)
                
                # 每处理一定数量的批次后记录进度
                if (i + 1) % 10 == 0:
                    self.logger.info(f"批量处理进度: {i + 1}/{len(batch_results)}")
                    
            except Exception as e:
                self.logger.error(f"批量处理第{i+1}个批次失败: {e}")
                batch_fused_results.append([])
        
        total_time = time.time() - start_time
        self.logger.info(f"批量融合完成: {len(batch_results)}个批次, 总耗时{total_time:.3f}秒")
        
        return batch_fused_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.stats.copy()
        
        if stats['total_fusions'] > 0:
            stats['throughput'] = stats['total_fusions'] / max(stats['total_time'], 0.001)
        else:
            stats['throughput'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_fusions': 0,
            'total_time': 0.0,
            'avg_time_per_fusion': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("RRF性能统计已重置")


class RRFPerformanceAnalyzer:
    """RRF性能分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_history = []
    
    def analyze_rrf_performance(self, results_by_source: Dict[str, List[RetrievalResult]], 
                               fused_results: List[FusedResult], 
                               execution_time: float) -> Dict[str, Any]:
        """分析RRF性能"""
        analysis = {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'input_stats': self._analyze_input(results_by_source),
            'output_stats': self._analyze_output(fused_results),
            'efficiency_metrics': self._calculate_efficiency_metrics(results_by_source, fused_results, execution_time)
        }
        
        self.analysis_history.append(analysis)
        
        # 只保留最近100条分析记录
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]
        
        return analysis
    
    def _analyze_input(self, results_by_source: Dict[str, List[RetrievalResult]]) -> Dict[str, Any]:
        """分析输入数据"""
        total_results = sum(len(results) for results in results_by_source.values())
        unique_contents = set()
        
        for results in results_by_source.values():
            for result in results:
                unique_contents.add(result.content)
        
        return {
            'sources_count': len(results_by_source),
            'total_results': total_results,
            'unique_contents': len(unique_contents),
            'avg_results_per_source': total_results / len(results_by_source) if results_by_source else 0,
            'duplication_rate': (total_results - len(unique_contents)) / total_results if total_results > 0 else 0
        }
    
    def _analyze_output(self, fused_results: List[FusedResult]) -> Dict[str, Any]:
        """分析输出数据"""
        if not fused_results:
            return {
                'output_count': 0,
                'avg_score': 0.0,
                'score_range': (0.0, 0.0),
                'score_variance': 0.0
            }
        
        scores = [result.fused_score for result in fused_results]
        
        return {
            'output_count': len(fused_results),
            'avg_score': sum(scores) / len(scores),
            'score_range': (min(scores), max(scores)),
            'score_variance': self._calculate_variance(scores)
        }
    
    def _calculate_efficiency_metrics(self, results_by_source: Dict[str, List[RetrievalResult]], 
                                    fused_results: List[FusedResult], 
                                    execution_time: float) -> Dict[str, Any]:
        """计算效率指标"""
        input_count = sum(len(results) for results in results_by_source.values())
        output_count = len(fused_results)
        
        return {
            'compression_ratio': output_count / input_count if input_count > 0 else 0,
            'processing_rate': input_count / execution_time if execution_time > 0 else 0,
            'output_rate': output_count / execution_time if execution_time > 0 else 0
        }
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """计算方差"""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        return variance
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """获取性能趋势"""
        if len(self.analysis_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_analyses = self.analysis_history[-10:]  # 最近10次分析
        
        execution_times = [analysis['execution_time'] for analysis in recent_analyses]
        processing_rates = [analysis['efficiency_metrics']['processing_rate'] for analysis in recent_analyses]
        
        # 计算趋势
        time_trend = self._calculate_trend(execution_times)
        rate_trend = self._calculate_trend(processing_rates)
        
        return {
            'execution_time_trend': time_trend,
            'processing_rate_trend': rate_trend,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'avg_processing_rate': sum(processing_rates) / len(processing_rates)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'
        
        # 简单的线性趋势计算
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'degrading'
        else:
            return 'stable'
