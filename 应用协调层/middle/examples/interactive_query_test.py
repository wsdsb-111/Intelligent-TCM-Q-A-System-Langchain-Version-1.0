"""
交互式混合检索测试程序
支持自由输入查询，如"肚子疼怎么办"，测试完整的混合检索系统
"""

import asyncio
import sys
import os
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入核心模块
from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.core.result_fusion import ResultFusion
from middle.core.scoring_strategies import ScoringErrorHandler
from middle.core.performance_optimization import get_performance_optimizer, get_performance_report
from middle.core.optimized_rrf import OptimizedRRF, RRFConfig
from middle.core.performance_monitor import get_performance_monitor, start_performance_monitoring, stop_performance_monitoring
from middle.models.data_models import (
    RetrievalResult, RetrievalConfig, RetrievalSource, FusionMethod
)
from middle.integrations.hybrid_retriever import create_hybrid_retriever
from middle.integrations.mcp_tools import get_mcp_tool
from middle.utils.logging_utils import setup_monitoring, get_logger
from middle.core.config_manager import get_config_manager


class ChineseMedicalKnowledgeBase:
    """中医知识库模拟器"""
    
    def __init__(self):
        # 扩展的知识库，包含更多内容和同义词映射
        self.knowledge_base = {
            # 症状相关
            "肚子疼": [
                "腹痛可能由脾胃虚寒、食积、肝气郁结等引起",
                "急性腹痛可用藿香正气散，慢性腹痛可用理中汤",
                "腹痛伴腹泻多为脾胃虚弱，可用参苓白术散调理"
            ],
            "头痛": [
                "头痛分风寒、风热、血瘀、肝阳上亢等类型",
                "风寒头痛用川芎茶调散，风热头痛用桑菊饮",
                "血瘀头痛可用血府逐瘀汤，肝阳上亢用天麻钩藤饮"
            ],
            "感冒": [
                "感冒分风寒感冒和风热感冒两大类",
                "风寒感冒用麻黄汤或桂枝汤，症见恶寒重、发热轻",
                "风热感冒用银翘散或桑菊饮，症见发热重、恶寒轻"
            ],
            "咳嗽": [
                "咳嗽有外感咳嗽和内伤咳嗽之分",
                "风寒咳嗽用三拗汤，风热咳嗽用桑菊饮",
                "燥咳用桑杏汤，痰湿咳嗽用二陈汤"
            ],
            "失眠": [
                "失眠多因心肾不交、肝火扰心、痰热内扰等引起",
                "心肾不交型失眠用交泰丸，肝火扰心用龙胆泻肝汤",
                "痰热内扰型失眠用温胆汤，心脾两虚用归脾汤",
                "睡前可用酸枣仁汤安神，或按摩神门、三阴交等穴位"
            ],
            "胃痛": [
                "胃痛常见寒邪犯胃、肝气犯胃、胃阴不足等证型",
                "寒邪犯胃用良附丸，肝气犯胃用柴胡疏肝散",
                "胃阴不足用一贯煎，脾胃虚寒用理中汤"
            ],
            
            # 季节养生
            "春季养生": [
                "春季宜养肝，多食绿色蔬菜如菠菜、韭菜、芹菜",
                "春季易肝火旺，可用菊花茶、决明子茶清肝明目",
                "春季适合户外运动，但要注意防风保暖"
            ],
            "夏季养生": [
                "夏季宜养心，多食红色食物如西红柿、红豆、红枣",
                "夏季炎热，可用绿豆汤、冬瓜汤清热解暑",
                "夏季出汗多，要注意补充水分和电解质"
            ],
            "秋季养生": [
                "秋季宜养肺，多食白色食物如梨、百合、银耳、莲藕",
                "秋季干燥，可用蜂蜜润燥，多喝温开水",
                "秋季适合食用核桃、芝麻等坚果类食物补肾",
                "秋季可多食山药、南瓜、红薯等健脾胃的食物"
            ],
            "冬季养生": [
                "冬季宜养肾，多食黑色食物如黑豆、黑芝麻、黑木耳",
                "冬季寒冷，可用当归生姜羊肉汤温阳散寒",
                "冬季适合进补，可用人参、鹿茸等温补之品"
            ],
            
            # 中药相关
            "人参": [
                "人参大补元气，复脉固脱，补脾益肺，生津养血",
                "主治大病、久病、失血、脱汗等导致的元气欲脱",
                "人参分红参、白参、野山参，功效略有不同"
            ],
            "黄芪": [
                "黄芪补气固表，利尿托毒，排脓，敛疮生肌",
                "主治气虚乏力、中气下陷、久泻脱肛、便血崩漏",
                "黄芪配当归为当归补血汤，补气生血"
            ],
            "当归": [
                "当归补血活血，调经止痛，润燥滑肠",
                "主治血虚萎黄、眩晕心悸、月经不调、经闭痛经",
                "当归头补血，当归身养血，当归尾破血"
            ],
            
            # 方剂相关
            "四君子汤": [
                "四君子汤由人参、白术、茯苓、甘草组成",
                "功效益气健脾，主治脾胃气虚证",
                "为补气方剂之祖方，后世补气方多从此方化裁"
            ],
            "六味地黄丸": [
                "六味地黄丸由熟地黄、山茱萸、山药、泽泻、茯苓、牡丹皮组成",
                "功效滋阴补肾，主治肾阴虚证",
                "为补阴方剂之祖方，临床应用极为广泛"
            ],
            
            # 食疗相关
            "食疗": [
                "药食同源，食物也有寒热温凉之性",
                "脾胃虚寒者宜食温热性食物，如生姜、桂圆、红枣",
                "阴虚火旺者宜食滋阴食物，如银耳、百合、梨"
            ]
        }
        
        # 同义词和关键词映射
        self.keyword_mapping = {
            # 症状同义词
            "睡不着": "失眠",
            "睡不好": "失眠", 
            "入睡困难": "失眠",
            "多梦": "失眠",
            "腹痛": "肚子疼",
            "腹部疼痛": "肚子疼",
            "肚疼": "肚子疼",
            "偏头痛": "头痛",
            "头疼": "头痛",
            "发烧": "感冒",
            "发热": "感冒",
            "鼻塞": "感冒",
            "流鼻涕": "感冒",
            "嗓子疼": "感冒",
            "咽痛": "感冒",
            
            # 季节相关
            "春天": "春季养生",
            "夏天": "夏季养生", 
            "秋天": "秋季养生",
            "冬天": "冬季养生",
            "春季": "春季养生",
            "夏季": "夏季养生",
            "秋季": "秋季养生", 
            "冬季": "冬季养生",
            
            # 养生相关
            "吃什么": "食疗",
            "食物": "食疗",
            "饮食": "食疗",
            "营养": "食疗",
            "进补": "食疗"
        }
    
    def search_knowledge(self, query: str, source: RetrievalSource) -> List[RetrievalResult]:
        """搜索知识库"""
        results = []
        query_lower = query.lower()
        
        # 第一步：通过同义词映射扩展查询
        mapped_keywords = []
        for synonym, standard_term in self.keyword_mapping.items():
            if synonym in query_lower:
                mapped_keywords.append(standard_term)
        
        # 第二步：直接关键词匹配
        matched_keys = []
        for key in self.knowledge_base.keys():
            # 直接匹配
            if key in query_lower:
                matched_keys.append((key, 1.0))  # 完全匹配，权重1.0
            # 部分匹配
            elif any(word in query_lower for word in key.split()):
                matched_keys.append((key, 0.8))  # 部分匹配，权重0.8
        
        # 第三步：通过映射的关键词匹配
        for mapped_key in mapped_keywords:
            if mapped_key in self.knowledge_base:
                matched_keys.append((mapped_key, 0.9))  # 同义词匹配，权重0.9
        
        # 第四步：智能语义匹配
        semantic_matches = self._semantic_match(query_lower)
        matched_keys.extend(semantic_matches)
        
        # 去重并按权重排序
        unique_matches = {}
        for key, weight in matched_keys:
            if key not in unique_matches or unique_matches[key] < weight:
                unique_matches[key] = weight
        
        sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)
        
        # 生成结果
        for key, match_weight in sorted_matches[:3]:  # 最多取3个匹配的主题
            contents = self.knowledge_base[key]
            for i, content in enumerate(contents):
                base_score = (0.9 - i * 0.1) * match_weight  # 基础评分乘以匹配权重
                
                # 根据检索源调整评分
                if source == RetrievalSource.BM25:
                    score = base_score * 0.95
                elif source == RetrievalSource.VECTOR:
                    score = base_score * 1.0
                elif source == RetrievalSource.GRAPH:
                    score = base_score * 1.05
                else:
                    score = base_score
                
                result = RetrievalResult(
                    content=content,
                    score=max(0.1, score),
                    source=source,
                    metadata={
                        "source": source.value,
                        "keyword": key,
                        "match_weight": match_weight,
                        "doc_id": f"{source.value}_{key}_{i}",
                        "timestamp": datetime.now().isoformat()
                    },
                    entities=[key] if key in content else [],
                    relationships=["治疗", "功效", "主治"] if any(word in content for word in ["治疗", "功效", "主治"]) else []
                )
                results.append(result)
        
        # 如果仍然没有匹配结果，返回相关的通用建议
        if not results:
            general_advice = self._get_contextual_advice(query_lower)
            
            for i, advice in enumerate(general_advice):
                result = RetrievalResult(
                    content=advice,
                    score=0.3 - i * 0.05,  # 降低通用建议的评分
                    source=source,
                    metadata={
                        "source": source.value,
                        "type": "contextual_advice",
                        "doc_id": f"{source.value}_advice_{i}"
                    }
                )
                results.append(result)
        
        return results[:5]  # 返回前5个结果
    
    def _semantic_match(self, query: str) -> List[tuple]:
        """语义匹配，基于查询内容推断可能的主题"""
        matches = []
        
        # 症状相关的语义匹配
        if any(word in query for word in ["疼", "痛", "不舒服", "难受"]):
            if "肚" in query or "腹" in query:
                matches.append(("肚子疼", 0.85))
            elif "头" in query:
                matches.append(("头痛", 0.85))
            elif "胃" in query:
                matches.append(("胃痛", 0.85))
        
        # 睡眠相关
        if any(word in query for word in ["睡", "眠", "觉"]):
            matches.append(("失眠", 0.9))
        
        # 感冒相关
        if any(word in query for word in ["感冒", "发烧", "发热", "咳嗽", "鼻塞"]):
            matches.append(("感冒", 0.9))
            if "咳" in query:
                matches.append(("咳嗽", 0.85))
        
        # 季节养生相关
        season_keywords = {
            "春": "春季养生",
            "夏": "夏季养生", 
            "秋": "秋季养生",
            "冬": "冬季养生"
        }
        
        for season_char, season_topic in season_keywords.items():
            if season_char in query:
                matches.append((season_topic, 0.9))
        
        # 饮食相关
        if any(word in query for word in ["吃", "食", "喝", "营养", "补"]):
            matches.append(("食疗", 0.8))
            # 如果提到季节，优先匹配季节养生
            for season_char, season_topic in season_keywords.items():
                if season_char in query:
                    matches.append((season_topic, 0.95))
                    break
        
        return matches
    
    def _get_contextual_advice(self, query: str) -> List[str]:
        """根据查询内容返回相关的建议"""
        if any(word in query for word in ["睡", "眠"]):
            return [
                "失眠可能与心神不宁、肝火旺盛有关，建议规律作息",
                "可尝试睡前用温水泡脚，按摩涌泉穴助眠",
                "避免睡前饮用咖啡、茶等刺激性饮品"
            ]
        
        elif any(word in query for word in ["吃", "食", "营养"]):
            return [
                "中医讲究药食同源，食物有寒热温凉之性",
                "建议根据个人体质选择合适的食物",
                "脾胃虚弱者宜食温和易消化的食物"
            ]
        
        elif any(word in query for word in ["疼", "痛"]):
            return [
                "疼痛需要辨别寒热虚实，对症治疗",
                "急性疼痛建议及时就医，慢性疼痛可考虑中医调理",
                "可配合针灸、推拿等中医外治法"
            ]
        
        else:
            return [
                "建议咨询专业中医师进行辨证论治",
                "中医治疗需要根据个人体质和具体症状来制定方案",
                "可以考虑中医的望闻问切四诊合参来确定治疗方向"
            ]
    
    def search_knowledge_bm25(self, query: str) -> List[RetrievalResult]:
        """BM25风格搜索 - 使用改进的BM25评分策略"""
        results = []
        query_lower = query.lower()
        
        # 导入BM25评分策略
        from middle.core.scoring_strategies import BM25ScoringStrategy, ScoringConfig, ScoringMethod
        
        # 创建BM25评分策略
        bm25_config = ScoringConfig(
            method=ScoringMethod.BM25,
            k1=1.2,
            b=0.75,
            enable_position_penalty=True,
            enable_length_normalization=True
        )
        bm25_strategy = BM25ScoringStrategy(bm25_config)
        
        # 计算词频统计信息
        term_frequency = self._calculate_term_frequency()
        total_documents = len(self.knowledge_base)
        
        # BM25特点：对关键词精确匹配给高分，对词频敏感
        for key, contents in self.knowledge_base.items():
            match_type = self._determine_match_type(query_lower, key)
            
            if match_type != "no_match":
                for i, content in enumerate(contents):
                    # 构建元数据
                    metadata = {
                        "source": "bm25",
                        "keyword": key,
                        "match_type": match_type,
                        "term_frequency": term_frequency,
                        "total_documents": total_documents,
                        "avg_doc_length": self._calculate_avg_doc_length(),
                        "doc_id": f"bm25_{key}_{i}",
                        "position": i,
                        "content_length": len(content),
                        "query_terms": self._tokenize(query_lower),
                        "content_terms": self._tokenize(content.lower())
                    }
                    
                    # 使用BM25策略计算评分
                    try:
                        score = bm25_strategy.calculate_score(
                            query=query,
                            content=content,
                            metadata=metadata,
                            position=i,
                            total_docs=total_documents
                        )
                        
                        # 应用关键词匹配度加权
                        match_weight = self._get_match_weight(match_type)
                        final_score = score * match_weight
                        
                        # 添加词频奖励
                        tf_boost = self._calculate_tf_boost(query_lower, key, content)
                        final_score += tf_boost
                        
                        # 添加匹配度奖励
                        if match_type == "exact_match":
                            final_score += 0.1  # 精确匹配额外奖励
                        elif match_type == "partial_match":
                            final_score += 0.05  # 部分匹配奖励
                        
                        # BM25评分范围：0.4-0.8
                        final_score = max(0.4, min(0.8, final_score))
                        
                    except Exception as e:
                        # 如果评分计算失败，使用默认评分
                        final_score = 0.3
                    
                    result = RetrievalResult(
                        content=content,
                        score=final_score,
                        source=RetrievalSource.BM25,
                        metadata=metadata,
                        entities=[key] if key in content else [],
                        relationships=["治疗", "功效"] if any(word in content for word in ["治疗", "功效"]) else []
                    )
                    results.append(result)
        
        # 按评分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]
    
    def search_knowledge_vector(self, query: str) -> List[RetrievalResult]:
        """向量风格搜索 - 使用改进的向量评分策略"""
        results = []
        query_lower = query.lower()
        
        # 导入向量评分策略
        from middle.core.scoring_strategies import VectorScoringStrategy, ScoringConfig, ScoringMethod
        
        # 创建向量评分策略
        vector_config = ScoringConfig(
            method=ScoringMethod.VECTOR,
            alpha=0.6,  # 语义相似度权重
            enable_position_penalty=True,
            enable_length_normalization=True
        )
        vector_strategy = VectorScoringStrategy(vector_config)
        
        # 向量检索特点：语义相关性，能找到相关但不完全匹配的内容
        semantic_matches = self._semantic_match(query_lower)
        
        # 扩展语义匹配
        all_candidates = []
        
        # 直接匹配
        for key, contents in self.knowledge_base.items():
            if key in query_lower:
                all_candidates.append((key, 0.9))
        
        # 语义匹配
        all_candidates.extend(semantic_matches)
        
        # 同义词匹配
        for synonym, standard_term in self.keyword_mapping.items():
            if synonym in query_lower and standard_term in self.knowledge_base:
                all_candidates.append((standard_term, 0.85))
        
        # 相关概念匹配（向量检索的优势）
        related_concepts = self._get_related_concepts(query_lower)
        all_candidates.extend(related_concepts)
        
        # 去重并生成结果
        unique_matches = {}
        for key, weight in all_candidates:
            if key not in unique_matches or unique_matches[key] < weight:
                unique_matches[key] = weight
        
        for key, match_weight in unique_matches.items():
            if key in self.knowledge_base:
                contents = self.knowledge_base[key]
                for i, content in enumerate(contents):
                    # 构建元数据
                    metadata = {
                        "source": "vector",
                        "keyword": key,
                        "similarity": match_weight,
                        "doc_id": f"vector_{key}_{i}",
                        "position": i,
                        "content_length": len(content),
                        "entities": self._extract_entities(content),
                        "relationships": self._extract_relationships(content),
                        "query_entities": self._extract_query_entities(query_lower),
                        "query_relationships": self._extract_query_relationships(query_lower),
                        "semantic_noise": self._simulate_semantic_noise(query_lower, content)
                    }
                    
                    # 使用向量策略计算评分
                    try:
                        score = vector_strategy.calculate_score(
                            query=query,
                            content=content,
                            metadata=metadata,
                            position=i,
                            total_docs=len(self.knowledge_base)
                        )
                        
                        # 应用概念扩展奖励
                        concept_bonus = self._calculate_concept_expansion_bonus(query_lower, key, content)
                        final_score = score + concept_bonus
                        
                        # 添加高相似度奖励
                        if match_weight > 0.8:
                            final_score += 0.1  # 高相似度额外奖励
                        elif match_weight > 0.6:
                            final_score += 0.05  # 中等相似度奖励
                        
                        # 应用语义噪声模拟
                        noise_factor = self._apply_semantic_noise(final_score, metadata.get('semantic_noise', 0.0))
                        final_score = final_score * noise_factor
                        
                        # 向量检索评分范围：0.5-0.9
                        final_score = max(0.5, min(0.9, final_score))
                        
                    except Exception as e:
                        # 如果向量计算失败，使用简化评分
                        simplified_score = self._calculate_simplified_vector_score(query_lower, key, content, i)
                        final_score = simplified_score
                    
                    result = RetrievalResult(
                        content=content,
                        score=final_score,
                        source=RetrievalSource.VECTOR,
                        metadata=metadata,
                        entities=metadata.get("entities", []),
                        relationships=metadata.get("relationships", [])
                    )
                    results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]
    
    def search_knowledge_graph(self, query: str) -> List[RetrievalResult]:
        """图风格搜索 - 使用改进的图评分策略"""
        results = []
        query_lower = query.lower()
        
        # 导入图评分策略
        from middle.core.scoring_strategies import GraphScoringStrategy, ScoringConfig, ScoringMethod
        
        # 创建图评分策略
        graph_config = ScoringConfig(
            method=ScoringMethod.GRAPH,
            enable_position_penalty=True,
            enable_length_normalization=True
        )
        graph_strategy = GraphScoringStrategy(graph_config)
        
        # 图检索特点：关注实体关系，对结构化知识给高分
        entity_matches = []
        
        # 识别查询中的实体
        for key in self.knowledge_base.keys():
            if key in query_lower:
                entity_matches.append((key, 1.0))
        
        # 通过同义词识别实体
        for synonym, standard_term in self.keyword_mapping.items():
            if synonym in query_lower and standard_term in self.knowledge_base:
                entity_matches.append((standard_term, 0.9))
        
        # 图检索的优势：找到相关实体
        for key, contents in self.knowledge_base.items():
            for content in contents:
                # 如果内容中包含查询相关的实体，也算匹配
                if any(word in content.lower() for word in query_lower.split() if len(word) > 1):
                    entity_matches.append((key, 0.7))
                    break
        
        # 去重
        unique_entities = {}
        for entity, weight in entity_matches:
            if entity not in unique_entities or unique_entities[entity] < weight:
                unique_entities[entity] = weight
        
        for entity, match_weight in unique_entities.items():
            contents = self.knowledge_base[entity]
            for i, content in enumerate(contents):
                # 构建元数据
                entities = self._extract_entities(content)
                relationships = self._extract_relationships(content)
                query_entities = self._extract_query_entities(query_lower)
                
                metadata = {
                    "source": "graph",
                    "entity": entity,
                    "doc_id": f"graph_{entity}_{i}",
                    "position": i,
                    "content_length": len(content),
                    "entities": entities,
                    "relationships": relationships,
                    "query_entities": query_entities,
                    "structured_data": self._has_structured_data(content),
                    "entity_relationships": self._extract_entity_relationships(content),
                    "graph_complexity": self._calculate_graph_complexity(content)
                }
                
                # 使用图策略计算评分
                try:
                    score = graph_strategy.calculate_score(
                        query=query,
                        content=content,
                        metadata=metadata,
                        position=i,
                        total_docs=len(self.knowledge_base)
                    )
                    
                    # 应用结构化知识奖励
                    structure_bonus = self._calculate_structured_knowledge_bonus(content, metadata)
                    final_score = score + structure_bonus
                    
                    # 应用实体关系奖励
                    entity_relation_bonus = self._calculate_entity_relation_bonus(query_lower, entity, content)
                    final_score = final_score + entity_relation_bonus
                    
                    # 添加实体匹配奖励
                    if match_weight > 0.9:
                        final_score += 0.15  # 高实体匹配奖励
                    elif match_weight > 0.7:
                        final_score += 0.1   # 中等实体匹配奖励
                    
                    # 添加结构化数据奖励
                    if metadata.get('structured_data', False):
                        final_score += 0.1  # 结构化数据奖励
                    
                    # 图检索评分范围：0.6-0.9
                    final_score = max(0.6, min(0.9, final_score))
                    
                except Exception as e:
                    # 如果图计算失败，使用简化评分
                    simplified_score = self._calculate_simplified_graph_score(query_lower, entity, content, i)
                    final_score = simplified_score
                
                result = RetrievalResult(
                    content=content,
                    score=final_score,
                    source=RetrievalSource.GRAPH,
                    metadata=metadata,
                    entities=entities,
                    relationships=relationships
                )
                results.append(result)
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]
    
    def _get_related_concepts(self, query: str) -> List[tuple]:
        """获取相关概念（向量检索的优势）"""
        related = []
        
        # 症状相关的概念扩展
        if any(word in query for word in ["疼", "痛"]):
            related.extend([("人参", 0.6), ("当归", 0.6), ("黄芪", 0.5)])
        
        if any(word in query for word in ["睡", "眠"]):
            related.extend([("四君子汤", 0.5), ("六味地黄丸", 0.6)])
        
        # 季节相关的概念扩展
        if "秋" in query:
            related.extend([("人参", 0.7), ("当归", 0.6), ("食疗", 0.8)])
        
        if any(word in query for word in ["吃", "食"]):
            related.extend([("春季养生", 0.4), ("夏季养生", 0.4), ("冬季养生", 0.4)])
        
        return related
    
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """计算内容相关性"""
        query_words = set(query.split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # 计算词汇重叠度
        overlap = len(query_words.intersection(content_words))
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)
    
    def _calculate_term_frequency(self) -> Dict[str, int]:
        """计算词频统计"""
        term_freq = {}
        for key, contents in self.knowledge_base.items():
            for content in contents:
                words = self._tokenize(content.lower())
                for word in words:
                    term_freq[word] = term_freq.get(word, 0) + 1
        return term_freq
    
    def _calculate_avg_doc_length(self) -> float:
        """计算平均文档长度"""
        total_length = 0
        total_docs = 0
        for key, contents in self.knowledge_base.items():
            for content in contents:
                total_length += len(self._tokenize(content.lower()))
                total_docs += 1
        return total_length / total_docs if total_docs > 0 else 100
    
    def _determine_match_type(self, query: str, key: str) -> str:
        """确定匹配类型"""
        # 精确匹配
        if key in query:
            return "exact"
        
        # 部分匹配
        if any(word in query for word in key.split()):
            return "partial"
        
        # 同义词匹配
        for synonym, standard_term in self.keyword_mapping.items():
            if synonym in query and standard_term == key:
                return "synonym"
        
        return "no_match"
    
    def _get_match_weight(self, match_type: str) -> float:
        """获取匹配类型权重"""
        weights = {
            "exact": 1.0,
            "partial": 0.7,
            "synonym": 0.8,
            "no_match": 0.0
        }
        return weights.get(match_type, 0.0)
    
    def _calculate_tf_boost(self, query: str, key: str, content: str) -> float:
        """计算词频奖励"""
        # 查询中关键词出现次数
        query_tf = query.count(key.lower())
        
        # 内容中关键词出现次数
        content_tf = content.lower().count(key.lower())
        
        # 词频奖励：查询词频 * 内容词频 * 权重
        tf_boost = (query_tf * content_tf * 0.05)
        
        return min(0.2, tf_boost)  # 最大奖励0.2
    
    def _calculate_simplified_bm25_score(self, query: str, key: str, content: str, position: int) -> float:
        """计算简化的BM25评分（备用方案）"""
        # 基础匹配分数
        if key in query:
            base_score = 0.8
        elif any(word in query for word in key.split()):
            base_score = 0.6
        else:
            base_score = 0.4
        
        # 位置惩罚
        position_penalty = 1.0 / (1.0 + position * 0.1)
        
        # 词频奖励
        tf_boost = self._calculate_tf_boost(query, key, content)
        
        # 综合评分
        score = (base_score * position_penalty) + tf_boost
        
        return max(0.1, min(0.9, score))
    
    def _has_structured_data(self, content: str) -> bool:
        """检查内容是否包含结构化数据"""
        # 检查是否包含结构化信息
        structured_indicators = [
            "组成", "配伍", "功效", "主治", "用法", "用量", 
            "方剂", "汤剂", "丸剂", "散剂", "膏剂"
        ]
        
        content_lower = content.lower()
        for indicator in structured_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _extract_entity_relationships(self, content: str) -> List[Dict[str, str]]:
        """提取实体关系"""
        relationships = []
        content_lower = content.lower()
        
        # 定义关系模式
        relation_patterns = [
            ("治疗", "症状", "药物"),
            ("功效", "药物", "作用"),
            ("主治", "药物", "疾病"),
            ("组成", "方剂", "药物"),
            ("配伍", "药物", "药物")
        ]
        
        # 简化的关系提取
        for pattern in relation_patterns:
            if pattern[0] in content_lower:
                relationships.append({
                    "relation": pattern[0],
                    "subject": "entity",
                    "object": "target"
                })
        
        return relationships
    
    def _calculate_graph_complexity(self, content: str) -> float:
        """计算图复杂度"""
        complexity = 0.0
        content_lower = content.lower()
        
        # 实体数量
        entities = self._extract_entities(content)
        entity_complexity = min(0.3, len(entities) * 0.05)
        complexity += entity_complexity
        
        # 关系数量
        relationships = self._extract_relationships(content)
        relation_complexity = min(0.3, len(relationships) * 0.05)
        complexity += relation_complexity
        
        # 结构化程度
        if self._has_structured_data(content):
            complexity += 0.2
        
        # 关系复杂度
        entity_relationships = self._extract_entity_relationships(content)
        relationship_complexity = min(0.2, len(entity_relationships) * 0.05)
        complexity += relationship_complexity
        
        return min(1.0, complexity)
    
    def _calculate_structured_knowledge_bonus(self, content: str, metadata: Dict[str, Any]) -> float:
        """计算结构化知识奖励"""
        bonus = 0.0
        
        # 结构化数据奖励
        if metadata.get('structured_data', False):
            bonus += 0.1
        
        # 图复杂度奖励
        graph_complexity = metadata.get('graph_complexity', 0.0)
        bonus += graph_complexity * 0.1
        
        # 实体关系奖励
        entity_relationships = metadata.get('entity_relationships', [])
        bonus += len(entity_relationships) * 0.02
        
        # 关系多样性奖励
        relationships = metadata.get('relationships', [])
        unique_relations = len(set(relationships))
        bonus += unique_relations * 0.03
        
        return min(0.3, bonus)  # 最大奖励0.3
    
    def _calculate_entity_relation_bonus(self, query: str, entity: str, content: str) -> float:
        """计算实体关系奖励"""
        bonus = 0.0
        
        # 查询实体匹配奖励
        query_entities = self._extract_query_entities(query)
        if entity in query_entities:
            bonus += 0.1
        
        # 实体关系匹配奖励
        content_entities = self._extract_entities(content)
        entity_overlap = len(set(query_entities).intersection(set(content_entities)))
        bonus += entity_overlap * 0.05
        
        # 关系匹配奖励
        query_relationships = self._extract_query_relationships(query)
        content_relationships = self._extract_relationships(content)
        relationship_overlap = len(set(query_relationships).intersection(set(content_relationships)))
        bonus += relationship_overlap * 0.03
        
        # 实体-关系三元组奖励
        if self._has_entity_relation_triple(query, entity, content):
            bonus += 0.05
        
        return min(0.2, bonus)  # 最大奖励0.2
    
    def _has_entity_relation_triple(self, query: str, entity: str, content: str) -> bool:
        """检查是否存在实体-关系三元组"""
        # 简化的三元组检查
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 检查查询中的实体是否在内容中，并且有明确的关系
        if entity in query_lower and entity in content_lower:
            # 检查是否有关系词连接
            relation_words = ["治疗", "功效", "主治", "组成", "配伍"]
            for rel_word in relation_words:
                if rel_word in query_lower and rel_word in content_lower:
                    return True
        
        return False
    
    def _calculate_simplified_graph_score(self, query: str, entity: str, content: str, position: int) -> float:
        """计算简化的图评分（备用方案）"""
        # 基础实体匹配分数
        if entity in query.lower():
            base_score = 0.8
        elif any(word in query.lower() for word in entity.split()):
            base_score = 0.6
        else:
            base_score = 0.4
        
        # 位置惩罚
        position_penalty = 1.0 / (1.0 + position * 0.1)
        
        # 关系复杂度奖励
        relationships = self._extract_relationships(content)
        relation_bonus = len(relationships) * 0.05
        
        # 结构化知识奖励
        structure_bonus = 0.1 if self._has_structured_data(content) else 0.0
        
        # 综合评分
        score = (base_score * position_penalty) + relation_bonus + structure_bonus
        
        return max(0.1, min(0.9, score))
    
    def _tokenize(self, text: str) -> List[str]:
        """分词方法"""
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if len(word) > 1]
    
    def _extract_entities(self, content: str) -> List[str]:
        """提取内容中的实体"""
        entities = []
        content_lower = content.lower()
        
        # 提取中医相关实体
        medical_entities = ["人参", "黄芪", "当归", "四君子汤", "六味地黄丸", "感冒", "头痛", "失眠", "胃痛"]
        for entity in medical_entities:
            if entity in content_lower:
                entities.append(entity)
        
        # 提取症状相关实体
        symptom_entities = ["肚子疼", "腹痛", "头痛", "失眠", "咳嗽", "感冒", "发烧"]
        for entity in symptom_entities:
            if entity in content_lower:
                entities.append(entity)
        
        return list(set(entities))  # 去重
    
    def _extract_relationships(self, content: str) -> List[str]:
        """提取内容中的关系"""
        relationships = []
        content_lower = content.lower()
        
        # 提取中医关系
        medical_relationships = ["治疗", "功效", "主治", "组成", "配伍", "调理", "补气", "补血", "清热"]
        for rel in medical_relationships:
            if rel in content_lower:
                relationships.append(rel)
        
        return list(set(relationships))  # 去重
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """提取查询中的实体"""
        entities = []
        
        # 从知识库中查找匹配的实体
        for key in self.knowledge_base.keys():
            if key in query:
                entities.append(key)
        
        # 从同义词映射中查找
        for synonym, standard_term in self.keyword_mapping.items():
            if synonym in query and standard_term not in entities:
                entities.append(standard_term)
        
        return entities
    
    def _extract_query_relationships(self, query: str) -> List[str]:
        """提取查询中的关系"""
        relationships = []
        
        # 常见的关系词汇
        relationship_words = ["治疗", "怎么", "如何", "方法", "功效", "作用", "怎么办", "怎么治"]
        for word in relationship_words:
            if word in query:
                relationships.append(word)
        
        return relationships
    
    def _simulate_semantic_noise(self, query: str, content: str) -> float:
        """模拟语义噪声"""
        # 基于查询和内容的相似度计算噪声水平
        query_words = set(self._tokenize(query))
        content_words = set(self._tokenize(content.lower()))
        
        if not query_words:
            return 0.0
        
        # 计算词汇重叠度
        overlap = len(query_words.intersection(content_words))
        similarity = overlap / len(query_words)
        
        # 相似度越高，噪声越小
        noise_level = max(0.0, 0.2 - similarity * 0.2)
        
        return noise_level
    
    def _calculate_concept_expansion_bonus(self, query: str, key: str, content: str) -> float:
        """计算概念扩展奖励"""
        bonus = 0.0
        
        # 相关概念奖励
        related_concepts = self._get_related_concepts(query)
        for concept, weight in related_concepts:
            if concept == key:
                bonus += weight * 0.1
        
        # 语义相似度奖励
        semantic_similarity = self._calculate_semantic_similarity(query, content)
        bonus += semantic_similarity * 0.05
        
        # 概念层次奖励
        concept_hierarchy_bonus = self._calculate_concept_hierarchy_bonus(query, key)
        bonus += concept_hierarchy_bonus
        
        return min(0.3, bonus)  # 最大奖励0.3
    
    def _calculate_semantic_similarity(self, query: str, content: str) -> float:
        """计算语义相似度"""
        query_words = set(self._tokenize(query))
        content_words = set(self._tokenize(content.lower()))
        
        if not query_words:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_concept_hierarchy_bonus(self, query: str, key: str) -> float:
        """计算概念层次奖励"""
        bonus = 0.0
        
        # 如果查询和关键词在同一个概念层次，给予奖励
        concept_hierarchies = {
            "症状": ["肚子疼", "头痛", "失眠", "咳嗽", "感冒"],
            "中药": ["人参", "黄芪", "当归"],
            "方剂": ["四君子汤", "六味地黄丸"],
            "养生": ["春季养生", "夏季养生", "秋季养生", "冬季养生"]
        }
        
        for hierarchy, concepts in concept_hierarchies.items():
            if key in concepts:
                # 检查查询中是否包含同层次的概念
                for concept in concepts:
                    if concept in query and concept != key:
                        bonus += 0.05
                        break
        
        return min(0.1, bonus)  # 最大奖励0.1
    
    def _apply_semantic_noise(self, score: float, noise_level: float) -> float:
        """应用语义噪声"""
        if noise_level <= 0:
            return 1.0
        
        # 噪声会降低评分
        noise_factor = 1.0 - noise_level
        return max(0.5, noise_factor)  # 最小保持50%的评分
    
    def _calculate_simplified_vector_score(self, query: str, key: str, content: str, position: int) -> float:
        """计算简化的向量评分（备用方案）"""
        # 基础语义相似度
        semantic_similarity = self._calculate_semantic_similarity(query, content)
        
        # 位置惩罚
        position_penalty = 1.0 / (1.0 + position * 0.1)
        
        # 概念扩展奖励
        concept_bonus = self._calculate_concept_expansion_bonus(query, key, content)
        
        # 综合评分
        score = (semantic_similarity * 0.7 + concept_bonus * 0.3) * position_penalty
        
        return max(0.1, min(0.9, score))
    
    def _has_structured_data(self, content: str) -> bool:
        """检查内容是否包含结构化数据"""
        # 检查是否包含结构化信息
        structured_indicators = [
            "组成", "配伍", "功效", "主治", "用法", "用量", 
            "方剂", "汤剂", "丸剂", "散剂", "膏剂"
        ]
        
        content_lower = content.lower()
        for indicator in structured_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _extract_entity_relationships(self, content: str) -> List[Dict[str, str]]:
        """提取实体关系"""
        relationships = []
        content_lower = content.lower()
        
        # 定义关系模式
        relation_patterns = [
            ("治疗", "症状", "药物"),
            ("功效", "药物", "作用"),
            ("主治", "药物", "疾病"),
            ("组成", "方剂", "药物"),
            ("配伍", "药物", "药物")
        ]
        
        # 简化的关系提取
        for pattern in relation_patterns:
            if pattern[0] in content_lower:
                relationships.append({
                    "relation": pattern[0],
                    "subject": "entity",
                    "object": "target"
                })
        
        return relationships
    
    def _calculate_graph_complexity(self, content: str) -> float:
        """计算图复杂度"""
        complexity = 0.0
        content_lower = content.lower()
        
        # 实体数量
        entities = self._extract_entities(content)
        entity_complexity = min(0.3, len(entities) * 0.05)
        complexity += entity_complexity
        
        # 关系数量
        relationships = self._extract_relationships(content)
        relation_complexity = min(0.3, len(relationships) * 0.05)
        complexity += relation_complexity
        
        # 结构化程度
        if self._has_structured_data(content):
            complexity += 0.2
        
        # 关系复杂度
        entity_relationships = self._extract_entity_relationships(content)
        relationship_complexity = min(0.2, len(entity_relationships) * 0.05)
        complexity += relationship_complexity
        
        return min(1.0, complexity)
    
    def _calculate_structured_knowledge_bonus(self, content: str, metadata: Dict[str, Any]) -> float:
        """计算结构化知识奖励"""
        bonus = 0.0
        
        # 结构化数据奖励
        if metadata.get('structured_data', False):
            bonus += 0.1
        
        # 图复杂度奖励
        graph_complexity = metadata.get('graph_complexity', 0.0)
        bonus += graph_complexity * 0.1
        
        # 实体关系奖励
        entity_relationships = metadata.get('entity_relationships', [])
        bonus += len(entity_relationships) * 0.02
        
        # 关系多样性奖励
        relationships = metadata.get('relationships', [])
        unique_relations = len(set(relationships))
        bonus += unique_relations * 0.03
        
        return min(0.3, bonus)  # 最大奖励0.3
    
    def _calculate_entity_relation_bonus(self, query: str, entity: str, content: str) -> float:
        """计算实体关系奖励"""
        bonus = 0.0
        
        # 查询实体匹配奖励
        query_entities = self._extract_query_entities(query)
        if entity in query_entities:
            bonus += 0.1
        
        # 实体关系匹配奖励
        content_entities = self._extract_entities(content)
        entity_overlap = len(set(query_entities).intersection(set(content_entities)))
        bonus += entity_overlap * 0.05
        
        # 关系匹配奖励
        query_relationships = self._extract_query_relationships(query)
        content_relationships = self._extract_relationships(content)
        relationship_overlap = len(set(query_relationships).intersection(set(content_relationships)))
        bonus += relationship_overlap * 0.03
        
        # 实体-关系三元组奖励
        if self._has_entity_relation_triple(query, entity, content):
            bonus += 0.05
        
        return min(0.2, bonus)  # 最大奖励0.2
    
    def _has_entity_relation_triple(self, query: str, entity: str, content: str) -> bool:
        """检查是否存在实体-关系三元组"""
        # 简化的三元组检查
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 检查查询中的实体是否在内容中，并且有明确的关系
        if entity in query_lower and entity in content_lower:
            # 检查是否有关系词连接
            relation_words = ["治疗", "功效", "主治", "组成", "配伍"]
            for rel_word in relation_words:
                if rel_word in query_lower and rel_word in content_lower:
                    return True
        
        return False
    
    def _calculate_simplified_graph_score(self, query: str, entity: str, content: str, position: int) -> float:
        """计算简化的图评分（备用方案）"""
        # 基础实体匹配分数
        if entity in query.lower():
            base_score = 0.8
        elif any(word in query.lower() for word in entity.split()):
            base_score = 0.6
        else:
            base_score = 0.4
        
        # 位置惩罚
        position_penalty = 1.0 / (1.0 + position * 0.1)
        
        # 关系复杂度奖励
        relationships = self._extract_relationships(content)
        relation_bonus = len(relationships) * 0.05
        
        # 结构化知识奖励
        structure_bonus = 0.1 if self._has_structured_data(content) else 0.0
        
        # 综合评分
        score = (base_score * position_penalty) + relation_bonus + structure_bonus
        
        return max(0.1, min(0.9, score))


class MockBM25Adapter:
    """模拟BM25适配器 - 专注关键词精确匹配"""
    
    def __init__(self):
        self.knowledge_base = ChineseMedicalKnowledgeBase()
        self.logger = get_logger(__name__)
    
    async def search(self, query: str, top_k: int = 10):
        """BM25搜索 - 关键词匹配，对精确匹配给高分"""
        await asyncio.sleep(0.08 + random.uniform(0, 0.04))  # 模拟搜索延迟
        self.logger.info(f"BM25搜索: {query}")
        
        results = self.knowledge_base.search_knowledge_bm25(query)
        return results[:top_k]
    
    async def health_check(self):
        """健康检查"""
        return True


class MockVectorAdapter:
    """模拟向量检索适配器 - 专注语义相似度"""
    
    def __init__(self):
        self.knowledge_base = ChineseMedicalKnowledgeBase()
        self.logger = get_logger(__name__)
    
    async def search(self, query: str, top_k: int = 10):
        """向量搜索 - 语义相似度，对相关概念给高分"""
        await asyncio.sleep(0.12 + random.uniform(0, 0.06))  # 模拟搜索延迟
        self.logger.info(f"向量搜索: {query}")
        
        results = self.knowledge_base.search_knowledge_vector(query)
        return results[:top_k]
    
    async def health_check(self):
        """健康检查"""
        return True


class MockGraphAdapter:
    """模拟图检索适配器 - 专注实体关系"""
    
    def __init__(self):
        self.knowledge_base = ChineseMedicalKnowledgeBase()
        self.logger = get_logger(__name__)
    
    async def search(self, query: str, top_k: int = 10):
        """图搜索 - 实体关系，对结构化知识给高分"""
        await asyncio.sleep(0.15 + random.uniform(0, 0.1))  # 模拟搜索延迟
        self.logger.info(f"图检索搜索: {query}")
        
        results = self.knowledge_base.search_knowledge_graph(query)
        return results[:top_k]
    
    async def complex_query_search(self, query: str, top_k: int = 10):
        """复杂图查询"""
        return await self.search(query, top_k)
    
    async def health_check(self):
        """健康检查"""
        return True


class InteractiveQueryTester:
    """交互式查询测试器"""
    
    def __init__(self):
        self.coordinator = None
        self.mcp_tool = None
        self.query_history = []
        self.logger = get_logger(__name__)
        self.error_handler = ScoringErrorHandler(max_errors=50, enable_fallback=True)
        
        # 性能优化组件
        self.performance_optimizer = get_performance_optimizer()
        self.optimized_rrf = OptimizedRRF(RRFConfig(k=60, max_results=100, batch_size=50))
        self.performance_monitor = get_performance_monitor()
        
        # 评分统计和分析
        self.scoring_stats = {
            'total_queries': 0,
            'bm25_scores': [],
            'vector_scores': [],
            'graph_scores': [],
            'fusion_scores': [],
            'query_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def initialize(self):
        """初始化系统"""
        print("🚀 正在初始化混合检索系统...")
        
        # 设置监控
        setup_monitoring(collection_interval=30, enable_alerts=False)
        
        # 启动性能监控
        start_performance_monitoring()
        print("📊 性能监控已启动")
        
        # 创建适配器
        bm25_adapter = MockBM25Adapter()
        vector_adapter = MockVectorAdapter()
        graph_adapter = MockGraphAdapter()
        
        # 创建融合器
        fusion_engine = ResultFusion()
        
        # 创建协调器
        self.coordinator = HybridRetrievalCoordinator(
            bm25_adapter=bm25_adapter,
            vector_adapter=vector_adapter,
            graph_adapter=graph_adapter,
            fusion_engine=fusion_engine
        )
        
        # 初始化MCP工具
        self.mcp_tool = get_mcp_tool(self.coordinator)
        
        # 执行健康检查
        health_status = await self.coordinator.health_check()
        
        print("✅ 系统初始化完成!")
        print(f"   系统健康状态: {'正常' if health_status.overall_healthy else '异常'}")
        
        for module in health_status.modules:
            status = "✅" if module.is_healthy else "❌"
            print(f"   {module.module_name}: {status}")
        
        print()
    
    def display_welcome(self):
        """显示欢迎信息"""
        print("=" * 60)
        print("🏥 智能中医混合检索系统 - 交互式测试")
        print("=" * 60)
        print()
        print("💡 使用说明:")
        print("   • 输入中医相关查询，如：'肚子疼怎么办'、'人参的功效'")
        print("   • 输入 'help' 查看更多命令")
        print("   • 输入 'quit' 或 'exit' 退出程序")
        print("   • 输入 'stats' 查看系统统计")
        print("   • 输入 'health' 查看系统健康状态")
        print()
        print("🔍 系统特性:")
        print("   • BM25关键词检索 + 向量语义检索 + 知识图谱检索")
        print("   • 智能结果融合 (RRF算法)")
        print("   • 实时性能监控")
        print("   • 容错和降级策略")
        print()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """处理查询"""
        start_time = time.time()
        
        # 创建检索配置
        config = RetrievalConfig(
            enable_bm25=True,
            enable_vector=True,
            enable_graph=True,
            top_k=5,
            fusion_method=FusionMethod.RRF,
            timeout=30
        )
        
        try:
            # 执行检索
            results = await self.coordinator.retrieve(query, config)
            
            # 检查结果是否为空
            if not results:
                self.logger.warning(f"查询 '{query}' 返回空结果")
                fallback_results = self.error_handler.handle_empty_results(query, "coordinator")
                results = fallback_results
            
            # 验证评分有效性
            if results:
                scores = [result.fused_score for result in results]
                valid_scores = self.error_handler.handle_invalid_scores(scores, {"query": query})
                
                # 如果评分被修正，更新结果
                for i, (result, valid_score) in enumerate(zip(results, valid_scores)):
                    if result.fused_score != valid_score:
                        result.metadata = result.metadata or {}
                        result.metadata['score_corrected'] = True
                        result.metadata['original_score'] = result.fused_score
                        result.fused_score = valid_score
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 收集评分统计
            self._collect_scoring_stats(results, execution_time)
            
            # 记录查询历史
            query_record = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "result_count": len(results),
                "success": True,
                "error_handler_stats": self.error_handler.get_error_statistics()
            }
            self.query_history.append(query_record)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "execution_time": execution_time,
                "result_count": len(results),
                "error_handler_stats": self.error_handler.get_error_statistics()
            }
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 使用错误处理器处理异常
            context = {
                "query": query,
                "execution_time": execution_time,
                "coordinator_available": self.coordinator is not None
            }
            
            # 尝试降级策略
            try:
                fallback_results = self.error_handler.handle_fusion_failure(e, {})
                self.logger.info("使用降级策略处理检索失败")
                
                query_record = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "result_count": len(fallback_results),
                    "success": True,
                    "fallback": True,
                    "error": str(e)
                }
                self.query_history.append(query_record)
                
                return {
                    "success": True,
                    "query": query,
                    "results": fallback_results,
                    "execution_time": execution_time,
                    "result_count": len(fallback_results),
                    "fallback": True,
                    "error": str(e)
                }
                
            except Exception as fallback_error:
                self.logger.error(f"降级策略也失败了: {str(fallback_error)}")
            
            # 记录失败的查询
            query_record = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "error": str(e),
                    "fallback_error": str(fallback_error),
                "success": False
            }
            self.query_history.append(query_record)
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                    "fallback_error": str(fallback_error),
                "execution_time": execution_time
            }
    
    def display_results(self, result_data: Dict[str, Any]):
        """显示检索结果 - 增强版本，包含详细评分计算过程"""
        if not result_data["success"]:
            print(f"❌ 查询失败: {result_data['error']}")
            return
        
        query = result_data["query"]
        results = result_data["results"]
        execution_time = result_data["execution_time"]
        
        print(f"🔍 查询: '{query}'")
        print(f"⏱️  执行时间: {execution_time:.3f}秒")
        print(f"📊 结果数量: {len(results)}")
        print()
        
        if not results:
            print("😔 未找到相关结果")
            return
        
        # 显示融合统计信息
        self._display_fusion_statistics(results)
        
        # 显示错误处理信息（如果有）
        if "error_handler_stats" in result_data:
            self._display_error_handler_info(result_data["error_handler_stats"])
        
        print("📋 检索结果:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 【融合评分: {result.fused_score:.4f}】")
            print(f"   内容: {result.content}")
            print("-" * 60)
            
            # 显示详细的评分计算过程
            self._display_scoring_details(result, i)
            
            # 显示各检索源贡献度可视化
            self._display_source_contribution(result)
            
            # 显示实体和关系信息
            self._display_entities_and_relationships(result)
            
            # 显示元数据信息
            self._display_metadata_info(result)
            
            print()
    
    def _display_fusion_statistics(self, results: List[Any]):
        """显示融合统计信息"""
        if not results:
            return
        
        print("📈 融合统计信息:")
        print("-" * 40)
        
        # 统计各检索源的贡献
        source_contributions = {"bm25": 0, "vector": 0, "graph": 0}
        total_results = len(results)
        
        for result in results:
            for source in result.contributing_sources:
                source_contributions[source.value] += 1
        
        print(f"总结果数: {total_results}")
        for source, count in source_contributions.items():
            percentage = (count / total_results) * 100 if total_results > 0 else 0
            print(f"{source.upper()}贡献: {count}个结果 ({percentage:.1f}%)")
        
        # 显示评分分布
        scores = [result.fused_score for result in results]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            print(f"评分分布: 平均{avg_score:.3f}, 最高{max_score:.3f}, 最低{min_score:.3f}")
        
        print()
    
    def _display_error_handler_info(self, error_stats: Dict[str, Any]):
        """显示错误处理器信息"""
        if error_stats.get("total_errors", 0) == 0:
            return
        
        print("⚠️  错误处理信息:")
        print("-" * 40)
        print(f"总错误数: {error_stats.get('total_errors', 0)}")
        
        if "error_types" in error_stats:
            error_types = error_stats["error_types"]
            if error_types:
                print("错误类型:")
                for error_type, count in error_types.items():
                    print(f"  {error_type}: {count}次")
        
        print()
    
    def _display_scoring_details(self, result: Any, index: int):
        """显示详细的评分计算过程"""
        print(f"🔢 评分计算详情:")
        
        # 显示融合方法
        fusion_method = getattr(result, 'fusion_method', 'RRF')
        print(f"   融合方法: {fusion_method}")
        
        # 显示各来源的详细评分
        if hasattr(result, 'source_scores') and result.source_scores:
            print(f"   各源原始评分:")
            for source, score in result.source_scores.items():
                print(f"     {source.upper()}: {score:.4f}")
        
        # 显示RRF计算过程（如果是RRF融合）
        if hasattr(result, 'metadata') and result.metadata:
            metadata = result.metadata
            if 'rrf_contributions' in metadata:
                print(f"   RRF贡献计算:")
                for source, contribution in metadata['rrf_contributions'].items():
                    print(f"     {source.upper()}: {contribution:.6f}")
        
        # 显示位置信息
        if hasattr(result, 'metadata') and result.metadata:
            position = result.metadata.get('position', 0)
            print(f"   原始位置: 第{position + 1}位")
    
    def _display_source_contribution(self, result: Any):
        """显示各检索源贡献度的可视化"""
        if not hasattr(result, 'source_scores') or not result.source_scores:
            return
        
        print(f"📊 检索源贡献度:")
        
        # 计算贡献度百分比
        total_score = sum(result.source_scores.values())
        if total_score == 0:
            return
        
        # 创建可视化条形图
        max_bar_length = 20
        for source, score in result.source_scores.items():
            percentage = (score / total_score) * 100
            bar_length = int((score / total_score) * max_bar_length)
            bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
            print(f"   {source.upper():6}: {bar} {percentage:5.1f}% ({score:.3f})")
    
    def _display_entities_and_relationships(self, result: Any):
        """显示实体和关系信息"""
        has_entities = hasattr(result, 'entities') and result.entities
        has_relationships = hasattr(result, 'relationships') and result.relationships
        
        if has_entities or has_relationships:
            print(f"🏷️  知识图谱信息:")
            
            if has_entities:
                print(f"   实体: {', '.join(result.entities)}")
            
            if has_relationships:
                print(f"   关系: {', '.join(result.relationships)}")
    
    def _display_metadata_info(self, result: Any):
        """显示元数据信息"""
        if not hasattr(result, 'metadata') or not result.metadata:
            return
        
        metadata = result.metadata
        print(f"📋 元数据信息:")
        
        # 显示匹配类型
        if 'match_type' in metadata:
            print(f"   匹配类型: {metadata['match_type']}")
        
        # 显示文档ID
        if 'doc_id' in metadata:
            print(f"   文档ID: {metadata['doc_id']}")
        
        # 显示特殊标记
        special_flags = []
        if metadata.get('fallback_score', False):
            special_flags.append("降级评分")
        if metadata.get('structured_data', False):
            special_flags.append("结构化数据")
        if metadata.get('semantic_noise', 0) > 0:
            special_flags.append(f"语义噪声({metadata['semantic_noise']:.2f})")
        
        if special_flags:
            print(f"   特殊标记: {', '.join(special_flags)}")
        
        # 显示错误信息（如果有）
        if 'error' in metadata:
            print(f"   ⚠️  错误: {metadata['error']}")
    
    def display_debug_info(self, result_data: Dict[str, Any]):
        """显示调试信息"""
        if not result_data["success"]:
            return
        
        print("\n🔧 调试信息:")
        print("=" * 50)
        
        results = result_data["results"]
        if not results:
            return
        
        # 显示每个结果的详细调试信息
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i} 调试信息:")
            print("-" * 30)
            
            # 显示完整的元数据
            if hasattr(result, 'metadata') and result.metadata:
                print("完整元数据:")
                for key, value in result.metadata.items():
                    if isinstance(value, (list, dict)):
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
            
            # 显示评分计算步骤
            if hasattr(result, 'source_scores') and result.source_scores:
                print("评分计算步骤:")
                for source, score in result.source_scores.items():
                    print(f"  {source}: {score:.6f}")
                print(f"  融合评分: {result.fused_score:.6f}")
            
            print()
    
    def display_error_statistics(self):
        """显示错误统计信息"""
        print("🚨 错误统计信息:")
        print("=" * 50)
        
        stats = self.error_handler.get_error_statistics()
        
        if stats["total_errors"] == 0:
            print("✅ 没有错误记录")
            return
        
        print(f"总错误数: {stats['total_errors']}")
        print(f"当前错误计数: {stats['error_count']}")
        
        if "error_types" in stats:
            print("\n错误类型分布:")
            for error_type, count in stats["error_types"].items():
                print(f"  {error_type}: {count}次")
        
        if "recent_errors" in stats and stats["recent_errors"]:
            print("\n最近错误:")
            for error in stats["recent_errors"][-3:]:  # 显示最近3个错误
                print(f"  {error['timestamp']}: {error['error_type']} - {error.get('error_message', 'N/A')}")
        
        # 显示错误处理器健康状态
        is_healthy = self.error_handler.is_healthy()
        health_status = "✅ 健康" if is_healthy else "⚠️ 不健康"
        print(f"\n错误处理器状态: {health_status}")
        
        # 显示查询历史中的错误统计
        failed_queries = [q for q in self.query_history if not q.get("success", True)]
        if failed_queries:
            print(f"\n查询失败统计: {len(failed_queries)}/{len(self.query_history)} 次查询失败")
            
            # 显示失败原因分布
            failure_reasons = {}
            for query in failed_queries:
                error = query.get("error", "未知错误")
                failure_reasons[error] = failure_reasons.get(error, 0) + 1
            
            print("失败原因分布:")
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count}次")
            
            print()
    
    async def show_system_stats(self):
        """显示系统统计"""
        print("📊 系统统计信息")
        print("-" * 30)
        
        # 协调器统计
        stats = self.coordinator.get_statistics()
        print(f"总查询数: {stats.get('total_queries', 0)}")
        print(f"成功查询: {stats.get('successful_queries', 0)}")
        print(f"失败查询: {stats.get('failed_queries', 0)}")
        print(f"平均响应时间: {stats.get('average_response_time', 0):.3f}秒")
        
        if 'module_usage' in stats:
            print(f"模块使用情况: {stats['module_usage']}")
        
        # 查询历史统计
        if self.query_history:
            successful_queries = [q for q in self.query_history if q['success']]
            failed_queries = [q for q in self.query_history if not q['success']]
            
            print(f"\n本次会话统计:")
            print(f"  查询总数: {len(self.query_history)}")
            print(f"  成功: {len(successful_queries)}")
            print(f"  失败: {len(failed_queries)}")
            
            if successful_queries:
                avg_time = sum(q['execution_time'] for q in successful_queries) / len(successful_queries)
                print(f"  平均响应时间: {avg_time:.3f}秒")
        
        print()
    
    async def show_health_status(self):
        """显示健康状态"""
        print("🏥 系统健康状态")
        print("-" * 30)
        
        health_status = await self.coordinator.health_check()
        
        overall_status = "✅ 正常" if health_status.overall_healthy else "❌ 异常"
        print(f"整体状态: {overall_status}")
        
        print("模块状态:")
        for module in health_status.modules:
            status = "✅ 正常" if module.is_healthy else "❌ 异常"
            print(f"  {module.module_name}: {status}")
            
            if not module.is_healthy and module.error_message:
                print(f"    错误: {module.error_message}")
        
        print()
    
    def show_help(self):
        """显示帮助信息"""
        print("📖 帮助信息")
        print("-" * 30)
        print("可用命令:")
        print("  help          - 显示此帮助信息")
        print("  stats         - 显示系统统计信息")
        print("  health        - 显示系统健康状态")
        print("  history       - 显示查询历史")
        print("  debug         - 显示最近查询的详细调试信息")
        print("  errors        - 显示错误统计信息")
        print("  scoring       - 显示评分统计和分析")
        print("  performance   - 显示性能优化报告")
        print("  cache         - 显示缓存统计信息")
        print("  optimize      - 执行性能优化")
        print("  clear         - 清屏")
        print("  quit/exit     - 退出程序")
        print()
        print("查询示例:")
        print("  肚子疼怎么办")
        print("  头痛的中医治疗方法")
        print("  人参的功效和作用")
        print("  四君子汤的组成")
        print("  感冒发烧怎么治疗")
        print("  失眠的中医调理方法")
        print()
        print("💡 提示:")
        print("  - 执行查询后，输入 'debug' 可查看详细的评分计算过程")
        print("  - 结果会显示各检索源的贡献度和评分详情")
        print("  - 支持BM25、向量检索、知识图谱三种检索方式")
        print("  - 输入 'scoring' 查看评分统计和分析")
        print("  - 输入 'performance' 查看性能优化报告")
        print("  - 输入 'cache' 查看缓存使用情况")
        print("  - 输入 'optimize' 执行性能优化操作")
        print()
    
    def show_history(self):
        """显示查询历史"""
        print("📜 查询历史")
        print("-" * 30)
        
        if not self.query_history:
            print("暂无查询历史")
            return
        
        for i, record in enumerate(self.query_history[-10:], 1):  # 显示最近10条
            status = "✅" if record['success'] else "❌"
            time_str = record['timestamp'].split('T')[1][:8]  # 只显示时间部分
            print(f"{i}. [{time_str}] {status} {record['query']}")
            
            if record['success']:
                print(f"   结果: {record.get('result_count', 0)}条, "
                      f"耗时: {record['execution_time']:.3f}秒")
            else:
                print(f"   错误: {record.get('error', '未知错误')}")
        
        print()
    
    def _collect_scoring_stats(self, results, execution_time):
        """收集评分统计信息"""
        self.scoring_stats['total_queries'] += 1
        self.scoring_stats['query_times'].append(execution_time)
        
        if results:
            # 收集融合评分
            fusion_scores = [result.fused_score for result in results]
            self.scoring_stats['fusion_scores'].extend(fusion_scores)
            
            # 收集各源评分
            for result in results:
                if hasattr(result, 'source_scores') and result.source_scores:
                    for source, score in result.source_scores.items():
                        if source.lower() == 'bm25':
                            self.scoring_stats['bm25_scores'].append(score)
                        elif source.lower() == 'vector':
                            self.scoring_stats['vector_scores'].append(score)
                        elif source.lower() == 'graph':
                            self.scoring_stats['graph_scores'].append(score)
        
        # 限制统计数据的数量，避免内存过度使用
        max_samples = 1000
        for key in ['bm25_scores', 'vector_scores', 'graph_scores', 'fusion_scores', 'query_times']:
            if len(self.scoring_stats[key]) > max_samples:
                self.scoring_stats[key] = self.scoring_stats[key][-max_samples:]
    
    def display_scoring_statistics(self):
        """显示评分统计和分析"""
        print("📊 评分统计和分析")
        print("-" * 40)
        
        if self.scoring_stats['total_queries'] == 0:
            print("暂无评分数据，请先执行查询")
            return
        
        print(f"总查询数: {self.scoring_stats['total_queries']}")
        print(f"缓存命中: {self.scoring_stats['cache_hits']}")
        print(f"缓存未命中: {self.scoring_stats['cache_misses']}")
        
        if self.scoring_stats['cache_hits'] + self.scoring_stats['cache_misses'] > 0:
            hit_rate = self.scoring_stats['cache_hits'] / (self.scoring_stats['cache_hits'] + self.scoring_stats['cache_misses'])
            print(f"缓存命中率: {hit_rate:.2%}")
        
        # 显示各检索源的评分统计
        for source, scores in [('BM25', self.scoring_stats['bm25_scores']), 
                              ('Vector', self.scoring_stats['vector_scores']),
                              ('Graph', self.scoring_stats['graph_scores']),
                              ('Fusion', self.scoring_stats['fusion_scores'])]:
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                print(f"\n{source}评分统计:")
                print(f"  平均评分: {avg_score:.4f}")
                print(f"  评分范围: {min_score:.4f} - {max_score:.4f}")
                print(f"  样本数: {len(scores)}")
        
        # 显示查询时间统计
        if self.scoring_stats['query_times']:
            avg_time = sum(self.scoring_stats['query_times']) / len(self.scoring_stats['query_times'])
            min_time = min(self.scoring_stats['query_times'])
            max_time = max(self.scoring_stats['query_times'])
            print(f"\n查询时间统计:")
            print(f"  平均时间: {avg_time:.3f}秒")
            print(f"  时间范围: {min_time:.3f} - {max_time:.3f}秒")
        
        print()
    
    def display_performance_report(self):
        """显示性能优化报告"""
        print("🚀 性能优化报告")
        print("-" * 40)
        
        try:
            # 获取性能报告
            report = get_performance_report()
            
            # 显示性能统计
            perf_stats = report.get('performance_stats', {})
            print(f"总查询数: {perf_stats.get('total_queries', 0)}")
            print(f"总执行时间: {perf_stats.get('total_time', 0):.3f}秒")
            
            if 'avg_query_time' in report:
                print(f"平均查询时间: {report['avg_query_time']:.3f}秒")
            
            if 'queries_per_second' in report:
                print(f"查询吞吐量: {report['queries_per_second']:.2f}查询/秒")
            
            # 显示缓存统计
            cache_stats = report.get('cache_stats', {})
            if cache_stats:
                print(f"\n缓存统计:")
                print(f"  缓存大小: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
                print(f"  命中率: {cache_stats.get('hit_rate', 0):.2%}")
                print(f"  总访问次数: {cache_stats.get('total_access', 0)}")
            
            # 显示内存使用情况
            memory_usage = report.get('memory_usage', {})
            if memory_usage:
                print(f"\n内存使用:")
                print(f"  物理内存: {memory_usage.get('rss_mb', 0):.1f}MB")
                print(f"  虚拟内存: {memory_usage.get('vms_mb', 0):.1f}MB")
                print(f"  内存使用率: {memory_usage.get('percent', 0):.1f}%")
            
            # 显示系统状态
            memory_status = report.get('memory_status', 'unknown')
            print(f"\n系统状态: {memory_status}")
            
        except Exception as e:
            print(f"❌ 获取性能报告失败: {e}")
        
        print()
    
    def display_cache_statistics(self):
        """显示缓存统计信息"""
        print("💾 缓存统计信息")
        print("-" * 40)
        
        try:
            # 获取缓存统计
            cache_stats = self.performance_optimizer.scoring_cache.get_stats()
            
            print(f"缓存大小: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"命中率: {cache_stats['hit_rate']:.2%}")
            print(f"总访问次数: {cache_stats['total_access']}")
            
            if cache_stats['size'] > 0:
                print(f"最旧条目: {cache_stats['oldest_entry']}")
                print(f"最新条目: {cache_stats['newest_entry']}")
            
            # 显示缓存使用建议
            if cache_stats['hit_rate'] < 0.5:
                print("\n💡 建议: 缓存命中率较低，可能需要调整缓存策略")
            elif cache_stats['hit_rate'] > 0.8:
                print("\n✅ 缓存命中率良好")
            
        except Exception as e:
            print(f"❌ 获取缓存统计失败: {e}")
        
        print()
    
    def execute_performance_optimization(self):
        """执行性能优化"""
        print("🔧 执行性能优化")
        print("-" * 40)
        
        try:
            # 执行垃圾回收
            print("🔄 执行垃圾回收...")
            gc_result = self.performance_optimizer.memory_monitor.force_garbage_collection()
            print(f"✅ 垃圾回收完成: 回收对象 {gc_result['objects_collected']} 个, 释放内存 {gc_result['memory_freed_mb']:.1f}MB")
            
            # 清理缓存
            print("🔄 清理过期缓存...")
            self.performance_optimizer.clear_cache()
            print("✅ 缓存清理完成")
            
            # 重置统计信息
            print("🔄 重置性能统计...")
            self.performance_optimizer.reset_stats()
            print("✅ 统计信息已重置")
            
            # 显示优化后的状态
            memory_usage = self.performance_optimizer.memory_monitor.get_memory_usage()
            print(f"\n📊 优化后状态:")
            print(f"  内存使用: {memory_usage['rss_mb']:.1f}MB")
            print(f"  内存使用率: {memory_usage['percent']:.1f}%")
            
        except Exception as e:
            print(f"❌ 性能优化失败: {e}")
        
        print()
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 停止性能监控
            stop_performance_monitoring()
            print("📊 性能监控已停止")
            
            # 清理性能优化器
            if hasattr(self, 'performance_optimizer'):
                self.performance_optimizer.cleanup()
                print("🔧 性能优化器已清理")
            
            # 显示最终统计信息
            if self.scoring_stats['total_queries'] > 0:
                print(f"\n📊 会话统计:")
                print(f"  总查询数: {self.scoring_stats['total_queries']}")
                print(f"  平均查询时间: {sum(self.scoring_stats['query_times'])/len(self.scoring_stats['query_times']):.3f}秒")
                
                if self.scoring_stats['cache_hits'] + self.scoring_stats['cache_misses'] > 0:
                    hit_rate = self.scoring_stats['cache_hits'] / (self.scoring_stats['cache_hits'] + self.scoring_stats['cache_misses'])
                    print(f"  缓存命中率: {hit_rate:.2%}")
            
            print("✅ 资源清理完成")
            
        except Exception as e:
            print(f"❌ 清理资源时出错: {e}")
    
    async def run_interactive_session(self):
        """运行交互式会话"""
        self.display_welcome()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("🔍 请输入查询 (输入 'help' 查看帮助): ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用！再见！")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    await self.show_system_stats()
                    continue
                
                elif user_input.lower() == 'health':
                    await self.show_health_status()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                elif user_input.lower() == 'debug':
                    # 显示最近一次查询的调试信息
                    if hasattr(self, '_last_result_data'):
                        self.display_debug_info(self._last_result_data)
                    else:
                        print("❌ 没有可用的调试信息，请先执行一次查询")
                    continue
                
                elif user_input.lower() == 'errors':
                    # 显示错误统计信息
                    self.display_error_statistics()
                    continue
                
                elif user_input.lower() == 'scoring':
                    # 显示评分统计和分析
                    self.display_scoring_statistics()
                    continue
                
                elif user_input.lower() == 'performance':
                    # 显示性能优化报告
                    self.display_performance_report()
                    continue
                
                elif user_input.lower() == 'cache':
                    # 显示缓存统计信息
                    self.display_cache_statistics()
                    continue
                
                elif user_input.lower() == 'optimize':
                    # 执行性能优化
                    self.execute_performance_optimization()
                    continue
                
                # 处理查询
                print()  # 空行分隔
                result_data = await self.process_query(user_input)
                self._last_result_data = result_data  # 保存结果用于调试
                self.display_results(result_data)
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被中断，再见！")
                break
            
            except Exception as e:
                print(f"❌ 发生错误: {str(e)}")
                self.logger.error(f"交互式会话错误: {str(e)}")
    
    async def cleanup(self):
        """清理资源"""
        if self.coordinator:
            self.coordinator.close()
        print("🧹 资源清理完成")


async def main():
    """主函数"""
    tester = InteractiveQueryTester()
    
    try:
        # 初始化系统
        await tester.initialize()
        
        # 运行交互式会话
        await tester.run_interactive_session()
        
    except Exception as e:
        print(f"❌ 程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        await tester.cleanup()


if __name__ == "__main__":
    # 运行交互式测试程序
    asyncio.run(main())
