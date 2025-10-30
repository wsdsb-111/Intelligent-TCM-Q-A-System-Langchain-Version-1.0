"""
查询分类器 - 根据查询特征选择最佳融合权重
集成智能路由器，支持基于BERT的查询分类
"""

import re
from typing import Dict, Tuple, Optional
from enum import Enum

# 导入智能路由器
try:
    from .intelligent_router import IntelligentRouter, RouteType, get_intelligent_router
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    RouteType = None


class QueryType(Enum):
    """查询类型枚举（扩展版）"""
    # 原有类型（兼容性保留）
    ENTITY = "entity"           # 实体查询：人参、黄芪
    SYMPTOM = "symptom"         # 症状查询：头痛、失眠
    RELATIONSHIP = "relationship"  # 关系查询：配伍、组成
    TREATMENT = "treatment"     # 治疗查询：如何治疗、怎么调理
    GENERAL = "general"         # 通用查询
    
    # 新增路由类型（智能路由 - 二元路由）
    ENTITY_DRIVEN = "entity_driven"          # 实体主导型查询（纯向量检索）
    COMPLEX_REASONING = "complex_reasoning"  # 复杂推理查询（混合检索，向量50%图谱50%）


class QueryClassifier:
    """
    查询分类器（增强版）
    
    支持两种工作模式：
    1. 传统模式：基于关键词的细粒度分类（ENTITY, SYMPTOM, RELATIONSHIP等）
    2. 智能路由模式：基于BERT的粗粒度路由（VECTOR, GRAPH, HYBRID）
    """
    
    def __init__(self, 
                 use_intelligent_routing: bool = True,
                 entity_csv_path: Optional[str] = None,
                 qwen_api_config: Optional[Dict] = None):
        """
        初始化分类器
        
        Args:
            use_intelligent_routing: 是否使用智能路由模式
            entity_csv_path: 实体关键词库CSV文件路径
            qwen_api_config: Qwen-Flash API配置
        """
        self.use_intelligent_routing = use_intelligent_routing
        
        # 初始化智能路由器
        self.intelligent_router = None
        if use_intelligent_routing and ROUTER_AVAILABLE:
            self.intelligent_router = get_intelligent_router(
                entity_csv_path=entity_csv_path,
                qwen_api_config=qwen_api_config
            )
        
        # 实体关键词
        self.entity_keywords = [
            '人参', '黄芪', '当归', '白术', '茯苓', '甘草', '附子', '干姜',
            '功效', '性味', '归经', '用法', '用量', '禁忌', '配伍'
        ]
        
        # 症状关键词
        self.symptom_keywords = [
            '头痛', '失眠', '咳嗽', '发热', '腹痛', '便秘', '腹泻', '乏力',
            '盗汗', '自汗', '眩晕', '耳鸣', '心悸', '胸闷', '气短'
        ]
        
        # 关系关键词
        self.relationship_keywords = [
            '配伍', '组成', '成分', '包含', '搭配', '一起', '结合',
            '相互', '关系', '区别', '比较'
        ]
        
        # 治疗关键词
        self.treatment_keywords = [
            '治疗', '调理', '缓解', '改善', '怎么', '如何', '什么方法',
            '用什么', '吃什么', '喝什么', '方剂', '处方'
        ]
        
        # 疑问词
        self.question_words = ['什么', '怎么', '如何', '为何', '哪些', '多少']
        
    def classify(self, query: str) -> QueryType:
        """
        分类查询类型
        
        根据配置选择使用智能路由模式或传统模式
        
        Args:
            query: 用户查询
            
        Returns:
            QueryType: 查询类型
        """
        if self.use_intelligent_routing and self.intelligent_router:
            # 使用智能路由模式
            route_type, _ = self.intelligent_router.classify(query)
            return self._route_type_to_query_type(route_type)
        else:
            # 使用传统分类模式
            return self._classify_traditional(query)
    
    def _classify_traditional(self, query: str) -> QueryType:
        """
        传统分类方法（基于关键词）
        
        Args:
            query: 用户查询
            
        Returns:
            QueryType: 查询类型
        """
        query = query.strip()
        
        # 检查关系查询（优先级最高）
        if any(keyword in query for keyword in self.relationship_keywords):
            return QueryType.RELATIONSHIP
        
        # 检查治疗查询
        if any(keyword in query for keyword in self.treatment_keywords):
            return QueryType.TREATMENT
        
        # 检查症状查询
        if any(keyword in query for keyword in self.symptom_keywords):
            return QueryType.SYMPTOM
        
        # 检查实体查询
        if any(keyword in query for keyword in self.entity_keywords):
            # 如果有疑问词，可能是询问实体属性
            if any(word in query for word in self.question_words):
                return QueryType.ENTITY
            return QueryType.ENTITY
        
        # 默认通用查询
        return QueryType.GENERAL
    
    def classify_with_confidence(self, query: str) -> Tuple[QueryType, float]:
        """
        分类查询并返回置信度
        
        Args:
            query: 用户查询
            
        Returns:
            (查询类型, 置信度分数)
        """
        if self.use_intelligent_routing and self.intelligent_router:
            route_type, confidence = self.intelligent_router.classify(query)
            query_type = self._route_type_to_query_type(route_type)
            return query_type, confidence
        else:
            query_type = self._classify_traditional(query)
            # 传统模式没有置信度，返回固定值
            return query_type, 0.8
    
    def _route_type_to_query_type(self, route_type) -> QueryType:
        """
        将RouteType转换为QueryType
        
        Args:
            route_type: RouteType枚举值
            
        Returns:
            QueryType枚举值
        """
        if not ROUTER_AVAILABLE or route_type is None:
            return QueryType.GENERAL
        
        # 导入RouteType（如果可用）
        from .intelligent_router import RouteType
        
        if route_type == RouteType.ENTITY_DRIVEN:
            return QueryType.ENTITY_DRIVEN
        elif route_type == RouteType.COMPLEX_REASONING:
            return QueryType.COMPLEX_REASONING
        else:
            return QueryType.GENERAL
    
    def get_fusion_weights(self, query_type: QueryType) -> Dict[str, float]:
        """
        根据查询类型获取融合权重
        
        Args:
            query_type: 查询类型
            
        Returns:
            权重字典
        """
        weight_configs = {
            # 智能路由权重配置（优先）- 二元路由
            QueryType.ENTITY_DRIVEN: {
                "vector": 1.0,  # 实体主导型查询（纯向量检索）
                "graph": 0.0
            },
            QueryType.COMPLEX_REASONING: {
                "vector": 0.5,  # 复杂推理查询（混合检索1:1，向量50%图谱50%）
                "graph": 0.5
            },
            
            # 传统分类权重配置（兼容）
            QueryType.ENTITY: {
                "vector": 0.4,  # 实体语义理解
                "graph": 0.6    # 图谱提供实体属性和关系
            },
            QueryType.SYMPTOM: {
                "vector": 0.6,  # 症状描述需要语义理解
                "graph": 0.4
            },
            QueryType.RELATIONSHIP: {
                "vector": 0.3,
                "graph": 0.7    # 关系推理依赖图谱
            },
            QueryType.TREATMENT: {
                "vector": 0.6,  # 治疗方法语义多样
                "graph": 0.4
            },
            QueryType.GENERAL: {
                "vector": 0.5,  # 默认均衡
                "graph": 0.5
            }
        }
        
        return weight_configs.get(query_type, weight_configs[QueryType.GENERAL])
    
    def classify_and_get_weights(self, query: str) -> Tuple[QueryType, Dict[str, float]]:
        """
        分类查询并返回推荐权重
        
        Args:
            query: 用户查询
            
        Returns:
            (查询类型, 权重字典)
        """
        query_type = self.classify(query)
        weights = self.get_fusion_weights(query_type)
        return query_type, weights
    
    def explain_classification(self, query: str) -> str:
        """
        解释分类结果
        
        Args:
            query: 用户查询
            
        Returns:
            解释文本
        """
        query_type, confidence = self.classify_with_confidence(query)
        weights = self.get_fusion_weights(query_type)
        
        type_descriptions = {
            # 智能路由类型 - 二元路由
            QueryType.ENTITY_DRIVEN: "实体主导型查询（纯向量检索）",
            QueryType.COMPLEX_REASONING: "复杂推理查询（混合检索，向量50%图谱50%）",
            
            # 传统分类类型
            QueryType.ENTITY: "实体查询（药材/方剂名称）",
            QueryType.SYMPTOM: "症状查询",
            QueryType.RELATIONSHIP: "关系查询（配伍/组成）",
            QueryType.TREATMENT: "治疗查询",
            QueryType.GENERAL: "通用查询"
        }
        
        explanation = f"查询类型: {type_descriptions.get(query_type, '未知类型')}\n"
        explanation += f"置信度: {confidence:.2f}\n"
        explanation += f"推荐权重: 向量={weights['vector']:.1f}, "
        explanation += f"图谱={weights['graph']:.1f}"
        
        return explanation
    
    def get_retrieval_strategy(self, query: str) -> Dict[str, any]:
        """
        获取检索策略配置
        
        Args:
            query: 用户查询
            
        Returns:
            检索策略字典
        """
        query_type, confidence = self.classify_with_confidence(query)
        weights = self.get_fusion_weights(query_type)
        
        # 判断是否启用单一检索模式
        enable_vector = weights['vector'] > 0
        enable_graph = weights['graph'] > 0
        is_single_source = (weights['vector'] == 1.0 or weights['graph'] == 1.0)
        
        return {
            "query_type": query_type.value,
            "confidence": confidence,
            "enable_vector": enable_vector,
            "enable_graph": enable_graph,
            "weights": weights,
            "is_single_source": is_single_source
        }


# 创建全局分类器实例
_classifier = None

def get_query_classifier(use_intelligent_routing: bool = True,
                        entity_csv_path: Optional[str] = None,
                        qwen_api_config: Optional[Dict] = None) -> QueryClassifier:
    """
    获取查询分类器单例
    
    Args:
        use_intelligent_routing: 是否使用智能路由模式
        entity_csv_path: 实体关键词库CSV文件路径
        qwen_api_config: Qwen-Flash API配置
        
    Returns:
        QueryClassifier实例
    """
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier(
            use_intelligent_routing=use_intelligent_routing,
            entity_csv_path=entity_csv_path,
            qwen_api_config=qwen_api_config
        )
    return _classifier

