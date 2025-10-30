"""
智能路由分类器
基于实体识别和复杂推理判断对查询进行分类，决定使用向量检索、知识图谱检索或混合检索
使用Qwen-Flash API替代本地Qwen3-1.7B模型
"""

import os
import csv
import logging
import requests
import json
from typing import Tuple, Dict, Optional, List, Set
from enum import Enum
from pathlib import Path


class RouteType(Enum):
    """路由类型枚举"""
    ENTITY_DRIVEN = "entity_driven"            # 实体主导型查询（向量检索）
    COMPLEX_REASONING = "complex_reasoning"    # 复杂推理查询（混合检索）


class IntelligentRouter:
    """
    智能路由分类器
    
    基于实体识别和复杂推理判断对查询进行分类：
    1. 实体主导型查询（向量检索）- 包含明确实体或推荐查询
    2. 复杂推理查询（混合检索）- 需要多步骤推理的复杂问题或语义模糊型查询
    """
    
    def __init__(self, 
                 entity_csv_path: Optional[str] = None,
                 qwen_api_config: Optional[Dict] = None,
                 confidence_threshold: float = 0.65):
        """
        初始化智能路由器
        
        Args:
            entity_csv_path: 实体关键词库CSV文件路径
            qwen_api_config: Qwen-Flash API配置
            confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
        
        # 初始化日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 加载实体关键词库
        self.entity_set = self._load_entity_keywords(entity_csv_path)
        
        # Qwen-Flash API配置
        self.qwen_api_config = qwen_api_config or {
            "api_key": "sk-6157e39178ac439bb00c43ba6b094501",
            "model_name": "qwen-flash",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
        
        # 复杂推理关键词
        self.complex_reasoning_keywords = [
            '步骤', '过程', '流程', '方法', '如何', '怎么',
            '首先', '然后', '接着', '最后', '分为', '包括',
            '引用', '发表', '研究', '进展', '实验室', '团队',
            '张三', '李四', '王五', '赵六'  # 示例人名，实际使用时需要更完整的实体库
        ]
        
        self.logger.info(f"智能路由器初始化完成 - 实体数量: {len(self.entity_set)}")
        self.logger.info(f"使用Qwen-Flash API: {self.qwen_api_config['model_name']}")
    
    def _load_entity_keywords(self, csv_path: Optional[str] = None) -> Set[str]:
        """
        加载实体关键词库
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            实体关键词集合
        """
        if csv_path is None:
            # 使用knowledge_graph_entities_only.csv作为实体库
            current_dir = Path(__file__).parent  # 应用协调层/middle/utils
            project_root = current_dir.parent.parent.parent  # 项目根目录
            csv_path = str(project_root / "测试与质量保障层" / "testdataset" / "knowledge_graph_entities_only.csv")
        
        entity_set = set()
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 读取实体列（假设实体在第一列）
            if len(df.columns) > 0:
                entity_column = df.columns[0]
                entities = df[entity_column].dropna().astype(str).tolist()
                
                # 过滤掉空字符串和过短的实体（放宽为1个字符）
                for entity in entities:
                    entity = entity.strip()
                    if len(entity) >= 1:  # 降低门槛：至少1个字符
                        entity_set.add(entity)
                
                self.logger.info(f"成功加载实体关键词库: {len(entity_set)} 个实体")
            else:
                self.logger.warning("CSV文件中没有找到实体列")
            
            return entity_set
            
        except Exception as e:
            self.logger.error(f"加载实体关键词库失败: {e}")
            return set()
    
    def _call_qwen_api(self, prompt: str, max_tokens: int = 10) -> str:
        """
        调用Qwen-Flash API
        
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            
        Returns:
            API响应文本
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.qwen_api_config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.qwen_api_config["model_name"],
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.qwen_api_config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                self.logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            self.logger.error(f"API调用异常: {e}")
            return ""
    
    def classify(self, query: str) -> Tuple[RouteType, float]:
        """
        对查询进行分类（优化版：更合理的分类逻辑）
        
        Args:
            query: 用户查询
            
        Returns:
            (路由类型, 置信度分数)
        """
        query = query.strip()
        
        # 步骤1: 检查是否为复杂推理查询（最严格的判断）
        if self._is_complex_reasoning(query):
            return RouteType.COMPLEX_REASONING, 0.9
        
        # 步骤2: 检查是否为简单推荐查询（向量检索）
        # 简单推荐查询：包含推荐关键词但不包含复杂推理要求
        recommendation_keywords = ['推荐', '提供', '建议', '给出']
        is_recommendation_query = any(keyword in query for keyword in recommendation_keywords)
        
        # 复杂推理标识词
        complex_reasoning_keywords = [
            "请考虑所有症状", "请输出推理过程", "推理过程", 
            "要求：", "请根据", "请分析", "请解释"
        ]
        has_complex_reasoning = any(keyword in query for keyword in complex_reasoning_keywords)
        
        # 简单推荐查询且不包含复杂推理要求 -> 向量检索
        if is_recommendation_query and not has_complex_reasoning:
            # 进一步检查是否为简单推荐（字数较少，结构简单）
            if len(query) < 50 and query.count('，') < 3:
                confidence = 0.85
                self.logger.debug(f"分类结果: ENTITY_DRIVEN (简单推荐查询) - 置信度: {confidence:.2f}")
                return RouteType.ENTITY_DRIVEN, confidence
        
        # 步骤3: 检查实体和功效查询
        entities_found = self._extract_entities(query)
        is_effect_query = self._is_effect_query(query)
        
        # 决策逻辑（优化版）
        if is_effect_query:
            # 功效/作用查询 -> 混合检索
            confidence = 0.8
            self.logger.debug(f"分类结果: COMPLEX_REASONING (功效查询) - 置信度: {confidence:.2f}")
            return RouteType.COMPLEX_REASONING, confidence
        elif len(entities_found) >= 2:
            # 包含2个及以上实体 -> 向量检索
            confidence = min(0.8 + (len(entities_found) * 0.02), 0.9)
            self.logger.debug(f"分类结果: ENTITY_DRIVEN (找到{len(entities_found)}个实体) - 置信度: {confidence:.2f}")
            return RouteType.ENTITY_DRIVEN, confidence
        elif len(entities_found) == 1:
            # 包含1个实体 -> 根据问题复杂度判断
            if len(query) > 40 or query.count('，') >= 2:
                # 复杂问题 -> 混合检索
                confidence = 0.75
                self.logger.debug(f"分类结果: COMPLEX_REASONING (1个实体但问题复杂) - 置信度: {confidence:.2f}")
                return RouteType.COMPLEX_REASONING, confidence
            else:
                # 简单问题 -> 向量检索
                confidence = 0.8
                self.logger.debug(f"分类结果: ENTITY_DRIVEN (1个实体且问题简单) - 置信度: {confidence:.2f}")
                return RouteType.ENTITY_DRIVEN, confidence
        else:
            # 不包含实体 -> 混合检索
            confidence = 0.7
            self.logger.debug(f"分类结果: COMPLEX_REASONING (未找到实体) - 置信度: {confidence:.2f}")
            return RouteType.COMPLEX_REASONING, confidence
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        从查询中提取实体，使用knowledge_graph_entities_only.csv作为实体库
        
        Args:
            query: 用户查询
            
        Returns:
            找到的实体列表
        """
        entities_found = []
        
        # 遍历实体库，检查查询中是否包含实体
        for entity in self.entity_set:
            if entity in query:
                entities_found.append(entity)
        
        # 按长度排序，优先匹配长实体（避免短实体误匹配）
        entities_found.sort(key=len, reverse=True)
        
        # 去重（避免重叠匹配）
        unique_entities = []
        for entity in entities_found:
            is_duplicate = False
            for existing in unique_entities:
                if entity in existing or existing in entity:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_entities.append(entity)
        
        # 使用Qwen模型判断实体类型，只保留有效实体
        valid_entities = []
        for entity in unique_entities:
            if self._is_valid_entity(entity):
                valid_entities.append(entity)
        
        return valid_entities
    
    def _is_valid_entity(self, entity: str) -> bool:
        """
        使用Qwen-Flash API判断实体是否为有效实体类型
        
        Args:
            entity: 实体名称
            
        Returns:
            是否为有效实体
        """
        try:
            # 构建提示词
            prompt = f"""你是一个实体分类专家，擅长判断一个词是否属于以下有效实体类型：
- 药材、药物
- 病症、疾病、病情
- 食物
- 穴位
- 身体部位
- 身体状况

只有以上类型的词才能作为有效实体。

请判断以下词是否为有效实体，只回答"是"或"否"：

词：{entity}

判断结果："""

            # 使用API进行判断
            response = self._call_qwen_api(prompt, max_tokens=5)
            
            # 判断结果
            is_valid = "是" in response
            
            self.logger.debug(f"Qwen API判断实体 '{entity}': {response} -> 有效: {is_valid}")
            return is_valid
            
        except Exception as e:
            self.logger.warning(f"Qwen API判断实体失败，使用备用方法: {e}")
            return self._is_valid_entity_fallback(entity)
    
    def _is_valid_entity_fallback(self, entity: str) -> bool:
        """
        备用实体类型判断方法
        
        Args:
            entity: 实体名称
            
        Returns:
            是否为有效实体
        """
        # 停用词列表
        stopwords = {'对', '请', '不要', '的', '了', '在', '是', '有', '和', '与', '或', '但', '而', '及', '等', '等', '什么', '怎么', '如何', '为什么', '哪里', '哪个', '哪些'}
        
        # 过滤停用词
        if entity in stopwords:
            return False
        
        # 长度过滤
        if len(entity) < 2:
            return False
        
        # 简单规则：包含常见的中医相关字符
        medical_chars = {'药', '症', '病', '穴', '脉', '气', '血', '肝', '心', '脾', '肺', '肾', '胃', '肠', '胆', '膀胱', '三焦', '经络', '穴位', '方剂', '汤', '丸', '散', '膏', '丹'}
        
        return any(char in entity for char in medical_chars)
    
    def _is_complex_reasoning(self, query: str) -> bool:
        """
        使用Qwen-Flash API判断是否为复杂推理查询
        
        Args:
            query: 用户查询
            
        Returns:
            是否为复杂推理查询
        """
        try:
            # 构建提示词（更严格的判断标准）
            prompt = f"""你是一个问题分类专家，擅长判断一个问题是否属于复杂问题。
            
复杂问题的必要标准：
1. 包含关键词："请考虑所有症状"或"请输出推理过程"
2. 包含4个以上的标点符号（逗号、句号、分号）
3. 字数超过30个自动判断为复杂推理

直接以下面我举的复杂问题为例子，字数越多判断为复杂问题的可能性越高
复杂问题的例子：
- 我感觉恶寒，但是一直没有出汗，该怎么办？请帮我推荐中药或者方剂。要求：1. 请考虑所有症状。2. 请输出推理过程。

请判断以下问题是否为复杂问题，只回答"是"或"否"：

问题：{query}

判断结果："""

            # 使用API进行判断
            response = self._call_qwen_api(prompt, max_tokens=10)
            
            # 判断结果
            is_complex = "是" in response or "复杂" in response
            
            self.logger.debug(f"Qwen API判断结果: {response} -> 复杂推理: {is_complex}")
            return is_complex
            
        except Exception as e:
            self.logger.warning(f"Qwen API判断失败，使用备用方法: {e}")
            return self._is_complex_reasoning_fallback(query)
    
    def _is_complex_reasoning_fallback(self, query: str) -> bool:
        """
        备用复杂推理判断方法
        
        Args:
            query: 用户查询
            
        Returns:
            是否为复杂推理查询
        """
        # 复杂推理判断的关键词（只有包含"请考虑所有症状"才判定为复杂推理）
        must_have_keyword = "请考虑所有症状"
        
        # 检查是否包含必要关键词
        has_must_keyword = must_have_keyword in query
        
        # 计算标点符号数量（逗号、句号、分号）
        punctuation_count = (
            query.count('，') + query.count(',') +  # 逗号
            query.count('。') + query.count('.') +  # 句号
            query.count('；') + query.count(';')    # 分号
        )
        
        # 判断是否为复杂推理：必须包含"请考虑所有症状"且标点符号≥5
        is_complex = has_must_keyword and punctuation_count >= 5
        
        if is_complex:
            self.logger.debug(f"备用方法判断为复杂推理: 包含'请考虑所有症状'且标点符号数={punctuation_count}")
        
        return is_complex
    
    def _is_effect_query(self, query: str) -> bool:
        """
        使用Qwen-Flash API判断是否为功效查询
        
        Args:
            query: 用户查询
            
        Returns:
            是否为功效查询
        """
        try:
            # 构建提示词
            prompt = f"""你是一个问题分类专家，擅长判断一个问题是否属于功效查询。

功效查询的特点：询问药材、方剂、事物的作用、功能、疗效、用途等。

功效查询的例子：
- 生姜的功效有哪些？
- 火麻仁的功效与用法
- 请介绍人参的作用
- 黄芪有什么功能？
- 这个方剂的疗效如何？

非功效查询的例子：
- 请推荐适合经常口臭的中药
- 我头痛，没有其他症状
- 怎样吃芝麻其中的营养素才会被吸收？

请判断以下问题是否为功效查询，只回答"是"或"否"：

问题：{query}

判断结果："""

            # 使用API进行判断
            response = self._call_qwen_api(prompt, max_tokens=10)
            
            # 判断结果
            is_effect = "是" in response or "功效" in response
            
            self.logger.debug(f"Qwen API判断功效查询结果: {response} -> 功效查询: {is_effect}")
            return is_effect
            
        except Exception as e:
            self.logger.warning(f"Qwen API判断功效查询失败，使用备用方法: {e}")
            return self._is_effect_query_fallback(query)
    
    def _is_effect_query_fallback(self, query: str) -> bool:
        """
        备用功效查询判断方法
        
        Args:
            query: 用户查询
            
        Returns:
            是否为功效查询
        """
        effect_keywords = ['功效','定义', '作用', '功能', '疗效', '效果', '用途', '好处', '益处', '价值', '特点', '特性','效用','效能']
        has_effect_keyword = any(keyword in query for keyword in effect_keywords)
        
        if has_effect_keyword:
            self.logger.debug(f"备用方法判断为功效查询: 包含关键词")
        
        return has_effect_keyword
    
    def get_fusion_weights(self, route_type: RouteType) -> Dict[str, float]:
        """
        根据路由类型获取融合权重
        
        Args:
            route_type: 路由类型
            
        Returns:
            融合权重字典
        """
        if route_type == RouteType.ENTITY_DRIVEN:
            # 实体主导型查询（向量检索）
            return {"vector": 1.0, "graph": 0.0}
        elif route_type == RouteType.COMPLEX_REASONING:
            # 复杂推理查询（混合检索）
            return {"vector": 0.5, "graph": 0.5}
        else:
            # 默认混合检索
            return {"vector": 0.5, "graph": 0.5}
    
    
    def explain_classification(self, query: str) -> str:
        """
        解释分类结果
        
        Args:
            query: 用户查询
            
        Returns:
            分类解释文本
        """
        route_type, confidence = self.classify(query)
        
        type_descriptions = {
            RouteType.ENTITY_DRIVEN: "实体主导型查询（向量检索）",
            RouteType.COMPLEX_REASONING: "复杂推理查询（混合检索）"
        }
        
        explanation = f"路由分类: {type_descriptions[route_type]}\n"
        explanation += f"置信度: {confidence:.2f}\n"
        
        # 添加检索策略说明
        if route_type == RouteType.ENTITY_DRIVEN:
            explanation += "检索策略: 使用纯向量检索，基于语义相似度"
        else:
            explanation += "检索策略: 使用混合检索，结合向量和图谱优势"
        
        return explanation
    
    def get_retrieval_config(self, query: str) -> Dict[str, any]:
        """
        根据查询获取推荐的检索配置
        
        Args:
            query: 用户查询
            
        Returns:
            检索配置字典
        """
        route_type, confidence = self.classify(query)
        
        config = {
            "route_type": route_type.value,
            "confidence": confidence,
            "enable_vector": False,
            "enable_graph": False,
            "weights": {}
        }
        
        if route_type == RouteType.ENTITY_DRIVEN:
            # 实体主导型查询（向量检索）
            config["enable_vector"] = True
            config["enable_graph"] = False
            config["weights"] = {"vector": 1.0, "graph": 0.0}
        
        else:  # COMPLEX_REASONING
            # 复杂推理查询（混合检索）
            config["enable_vector"] = True
            config["enable_graph"] = True
            config["weights"] = {"vector": 0.5, "graph": 0.5}
        
        return config


# 全局路由器实例
_router = None


def get_intelligent_router(entity_csv_path: Optional[str] = None,
                          qwen_api_config: Optional[Dict] = None,
                          confidence_threshold: float = 0.65) -> IntelligentRouter:
    """
    获取智能路由器单例
    
    Args:
        entity_csv_path: 实体关键词库CSV文件路径
        qwen_api_config: Qwen-Flash API配置
        confidence_threshold: 置信度阈值
        
    Returns:
        IntelligentRouter实例
    """
    global _router
    if _router is None:
        _router = IntelligentRouter(
            entity_csv_path=entity_csv_path,
            qwen_api_config=qwen_api_config,
            confidence_threshold=confidence_threshold
        )
    return _router

