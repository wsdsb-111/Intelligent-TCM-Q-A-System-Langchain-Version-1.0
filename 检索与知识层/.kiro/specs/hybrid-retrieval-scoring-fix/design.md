# 混合检索系统评分计算修复设计文档

## 概述

本设计文档详细说明如何修复混合检索系统中的评分计算问题。当前系统存在的主要问题是RRF融合算法计算错误，导致所有检索源返回相同评分，无法体现不同检索方法的特点。

## 架构

### 核心组件修改

```
混合检索协调器 (HybridRetrievalCoordinator)
├── BM25适配器 (MockBM25Adapter) - 修复评分计算
├── 向量适配器 (MockVectorAdapter) - 修复评分计算  
├── 图适配器 (MockGraphAdapter) - 修复评分计算
└── 结果融合器 (ResultFusion) - 修复RRF算法

评分计算流程:
1. 各适配器生成差异化评分
2. RRF算法正确计算排名贡献
3. 融合器输出准确的最终评分
4. 显示层展示详细的评分信息
```

### 数据流设计

```mermaid
graph TD
    A[查询输入] --> B[并行检索]
    B --> C[BM25检索<br/>评分范围: 0.1-1.0]
    B --> D[向量检索<br/>评分范围: 0.1-0.95]
    B --> E[图检索<br/>评分范围: 0.1-0.98]
    
    C --> F[RRF融合算法]
    D --> F
    E --> F
    
    F --> G[计算排名贡献<br/>1/(k+rank)]
    G --> H[累加各源贡献]
    H --> I[生成融合评分]
    I --> J[排序输出结果]
    
    J --> K[显示详细评分信息]
```

## 组件和接口

### 1. 评分计算接口改进

```python
class ScoringStrategy:
    """评分策略基类"""
    
    def calculate_score(self, 
                       query: str, 
                       content: str, 
                       match_info: Dict[str, Any]) -> float:
        """计算评分"""
        pass
    
    def get_score_range(self) -> Tuple[float, float]:
        """获取评分范围"""
        pass

class BM25ScoringStrategy(ScoringStrategy):
    """BM25评分策略 - 关键词匹配"""
    
    def calculate_score(self, query: str, content: str, match_info: Dict) -> float:
        # 基础匹配分数
        base_score = match_info.get('match_weight', 0.5)
        
        # 词频加权
        tf_boost = match_info.get('tf_score', 0.0)
        
        # 长度惩罚
        length_penalty = 1.0 / (1.0 + len(content.split()) * 0.01)
        
        # 随机扰动（模拟BM25的复杂性）
        random_factor = random.uniform(0.85, 1.15)
        
        score = (base_score + tf_boost) * length_penalty * random_factor
        return max(0.1, min(1.0, score))

class VectorScoringStrategy(ScoringStrategy):
    """向量评分策略 - 语义相似度"""
    
    def calculate_score(self, query: str, content: str, match_info: Dict) -> float:
        # 语义匹配分数
        semantic_score = match_info.get('semantic_score', 0.5)
        
        # 内容相关性
        content_relevance = match_info.get('content_relevance', 0.0)
        
        # 概念扩展奖励
        concept_bonus = match_info.get('concept_bonus', 0.0)
        
        # 语义随机性
        semantic_noise = random.uniform(0.9, 1.1)
        
        score = (semantic_score * 0.7 + content_relevance * 0.2 + concept_bonus * 0.1) * semantic_noise
        return max(0.1, min(0.95, score))

class GraphScoringStrategy(ScoringStrategy):
    """图评分策略 - 实体关系"""
    
    def calculate_score(self, query: str, content: str, match_info: Dict) -> float:
        # 实体匹配分数
        entity_score = match_info.get('entity_score', 0.5)
        
        # 关系复杂度奖励
        relation_bonus = match_info.get('relation_complexity', 0.0)
        
        # 结构化知识奖励
        structure_bonus = 0.05 if any(word in content for word in ["治疗", "功效", "主治"]) else 0.0
        
        # 图随机性
        graph_noise = random.uniform(0.88, 1.12)
        
        score = (entity_score + relation_bonus + structure_bonus) * graph_noise
        return max(0.1, min(0.98, score))
```

### 2. RRF融合算法修复

```python
class ImprovedRRFFusion:
    """改进的RRF融合算法"""
    
    def __init__(self, k: int = 60):
        self.k = k
        self.logger = get_logger(__name__)
    
    def fuse_results(self, results_by_source: Dict[str, List[RetrievalResult]], 
                    top_k: int = 10) -> List[FusedResult]:
        """执行RRF融合"""
        
        # 1. 构建内容到结果的映射
        content_map = {}
        content_rankings = defaultdict(dict)
        
        # 2. 记录每个内容在各源中的排名
        for source, results in results_by_source.items():
            for rank, result in enumerate(results):
                content = result.content
                
                # 存储原始结果
                if content not in content_map:
                    content_map[content] = result
                
                # 记录排名（从1开始）
                content_rankings[content][source] = rank + 1
        
        # 3. 计算RRF评分
        fused_results = []
        for content, original_result in content_map.items():
            rrf_score = 0.0
            source_contributions = {}
            contributing_sources = []
            
            # 计算每个源的RRF贡献
            for source, results in results_by_source.items():
                if source in content_rankings[content]:
                    rank = content_rankings[content][source]
                    rrf_contribution = 1.0 / (self.k + rank)
                    rrf_score += rrf_contribution
                    source_contributions[source] = rrf_contribution
                    
                    # 添加贡献源
                    try:
                        contributing_sources.append(RetrievalSource(source))
                    except ValueError:
                        # 处理测试中的自定义源名称
                        pass
                else:
                    source_contributions[source] = 0.0
            
            # 创建融合结果
            fused_result = FusedResult(
                content=content,
                fused_score=rrf_score,
                source_scores=source_contributions,
                fusion_method=FusionMethod.RRF,
                metadata=original_result.metadata.copy(),
                contributing_sources=contributing_sources,
                entities=original_result.entities,
                relationships=original_result.relationships,
                timestamp=datetime.now()
            )
            
            fused_results.append(fused_result)
            
            # 调试日志
            self.logger.debug(f"RRF计算: 内容='{content[:50]}...', 评分={rrf_score:.6f}, 贡献={source_contributions}")
        
        # 4. 按RRF评分排序
        fused_results.sort(key=lambda x: x.fused_score, reverse=True)
        
        return fused_results[:top_k]
```

### 3. 检索适配器改进

```python
class ImprovedMockBM25Adapter:
    """改进的BM25适配器"""
    
    def __init__(self):
        self.knowledge_base = ChineseMedicalKnowledgeBase()
        self.scoring_strategy = BM25ScoringStrategy()
        self.logger = get_logger(__name__)
    
    async def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25搜索 - 关键词精确匹配"""
        await asyncio.sleep(0.08 + random.uniform(0, 0.04))
        
        results = []
        query_lower = query.lower()
        
        # 查找匹配的知识条目
        for key, contents in self.knowledge_base.knowledge_base.items():
            match_weight = self._calculate_match_weight(query_lower, key)
            
            if match_weight > 0:
                for i, content in enumerate(contents):
                    # 计算详细的匹配信息
                    match_info = {
                        'match_weight': match_weight,
                        'tf_score': query_lower.count(key.lower()) * 0.1,
                        'position_penalty': i * 0.1,
                        'content_length': len(content.split())
                    }
                    
                    # 使用评分策略计算分数
                    score = self.scoring_strategy.calculate_score(query, content, match_info)
                    
                    # 位置惩罚
                    score *= (1.0 - match_info['position_penalty'])
                    
                    result = RetrievalResult(
                        content=content,
                        score=max(0.1, score),
                        source=RetrievalSource.BM25,
                        metadata={
                            "source": "bm25",
                            "keyword": key,
                            "match_weight": match_weight,
                            "tf_score": match_info['tf_score'],
                            "doc_id": f"bm25_{key}_{i}",
                            "calculation_details": match_info
                        },
                        entities=[key] if key in content else [],
                        relationships=["治疗", "功效"] if any(word in content for word in ["治疗", "功效"]) else []
                    )
                    results.append(result)
        
        # 按评分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _calculate_match_weight(self, query: str, key: str) -> float:
        """计算匹配权重"""
        if key in query:
            return 1.0  # 完全匹配
        elif any(word in query for word in key.split()):
            return 0.7  # 部分匹配
        else:
            # 检查同义词
            for synonym, standard_term in self.knowledge_base.keyword_mapping.items():
                if synonym in query and standard_term == key:
                    return 0.8
            return 0.0
```

## 数据模型

### 评分详情模型

```python
@dataclass
class ScoringDetails:
    """评分计算详情"""
    original_score: float
    rrf_contribution: float
    rank_position: int
    calculation_method: str
    debug_info: Dict[str, Any]

@dataclass
class EnhancedFusedResult(FusedResult):
    """增强的融合结果"""
    scoring_details: Dict[str, ScoringDetails]
    fusion_explanation: str
    quality_metrics: Dict[str, float]
```

## 错误处理

### 评分计算异常处理

```python
class ScoringErrorHandler:
    """评分错误处理器"""
    
    def handle_empty_results(self, source: str) -> List[RetrievalResult]:
        """处理空结果"""
        self.logger.warning(f"检索源 {source} 返回空结果")
        return []
    
    def handle_invalid_score(self, score: float, source: str) -> float:
        """处理无效评分"""
        if score <= 0 or math.isnan(score) or math.isinf(score):
            self.logger.warning(f"检索源 {source} 评分无效: {score}, 使用默认值0.001")
            return 0.001
        return max(0.001, min(1.0, score))
    
    def handle_fusion_failure(self, results_by_source: Dict) -> List[FusedResult]:
        """处理融合失败"""
        self.logger.error("RRF融合失败，使用降级策略")
        
        # 返回第一个可用源的结果
        for source, results in results_by_source.items():
            if results:
                return [self._convert_to_fused_result(r, source) for r in results[:5]]
        
        return []
```

## 测试策略

### 单元测试覆盖

1. **RRF算法测试**
   - 验证排名计算公式 1/(k+rank)
   - 测试多源结果融合
   - 验证评分累加逻辑

2. **评分策略测试**
   - BM25评分范围验证
   - 向量评分差异性测试
   - 图评分结构化奖励测试

3. **异常处理测试**
   - 空结果处理
   - 无效评分修正
   - 融合失败降级

### 集成测试场景

1. **端到端评分流程**
   - 完整查询执行
   - 评分差异验证
   - 融合结果正确性

2. **性能测试**
   - 评分计算耗时
   - 内存使用优化
   - 并发处理能力

## 性能优化

### 评分计算优化

1. **缓存机制**
   - 评分结果缓存
   - 匹配权重缓存
   - RRF计算缓存

2. **并行计算**
   - 多源评分并行
   - 批量RRF计算
   - 异步结果处理

3. **内存优化**
   - 结果对象复用
   - 大对象及时释放
   - 内存池管理