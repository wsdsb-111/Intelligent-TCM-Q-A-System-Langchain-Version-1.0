"""
实体提取器 - 从自然语言查询中提取中医实体
支持三种模式：
1. 规则模式：快速，基于预定义词典（默认）
2. LLM模式：智能，使用微调模型识别（可选）
3. 知识图谱模式：基于CSV知识图谱数据（推荐）
"""

import re
import json
from typing import List, Set, Optional, Dict
import logging
from .optimized_entity_extractor import OptimizedRetrievalSystem, get_optimized_retrieval_system


class EntityExtractor:
    """简单的基于规则的实体提取器"""
    
    def __init__(self, use_cache=True, model_service=None, use_llm=False, use_kg=False, csv_file_path=None):
        """
        初始化实体提取器
        
        Args:
            use_cache: 是否使用缓存的实体列表
            model_service: 模型服务实例（用于LLM实体识别）
            use_llm: 是否优先使用LLM识别实体
            use_kg: 是否使用知识图谱模式（推荐）
            csv_file_path: 知识图谱CSV文件路径
        """
        self.use_llm = use_llm
        self.use_kg = use_kg
        self.model_service = model_service
        self.csv_file_path = csv_file_path
        self.logger = logging.getLogger(__name__)
        
        # 如果使用知识图谱模式，初始化优化检索系统
        if self.use_kg:
            self.kg_system = get_optimized_retrieval_system(csv_file_path)
            if not self.kg_system.initialize():
                self.logger.warning("知识图谱系统初始化失败，回退到规则模式")
                self.use_kg = False
        # 常见中药材（扩展到100+种）
        self.herb_entities = {
            # 补益药
            '人参', '黄芪', '党参', '太子参', '西洋参', '红参',
            '白术', '茯苓', '山药', '扁豆', '大枣', '甘草',
            '当归', '白芍', '熟地', '生地', '阿胶', '何首乌',
            '枸杞', '菊花', '麦冬', '天冬', '石斛', '玉竹',
            
            # 解表药
            '麻黄', '桂枝', '紫苏', '生姜', '葱白', '防风',
            '荆芥', '薄荷', '牛蒡子', '蝉蜕', '升麻', '柴胡',
            '金银花', '连翘', '板蓝根', '大青叶', '鱼腥草',
            
            # 清热药
            '黄连', '黄芩', '黄柏', '栀子', '龙胆草', '苦参',
            '知母', '芦根', '天花粉', '竹叶', '淡竹叶',
            
            # 活血化瘀药
            '川芎', '红花', '桃仁', '丹参', '三七', '蒲黄',
            '五灵脂', '郁金', '姜黄', '乳香', '没药',
            
            # 止咳化痰药
            '杏仁', '桔梗', '贝母', '瓜蒌', '半夏', '天南星',
            '白前', '紫菀', '款冬花', '百部', '枇杷叶',
            
            # 理气药
            '陈皮', '青皮', '枳壳', '枳实', '木香', '香附',
            '佛手', '香橼', '乌药', '沉香', '檀香',
            
            # 其他常用药
            '附子', '干姜', '肉桂', '吴茱萸', '细辛',
            '天麻', '钩藤', '石决明', '珍珠母', '牡蛎',
            '酸枣仁', '远志', '柏子仁', '龙眼肉', '茯神'
        }
        
        # 常见症状
        self.symptom_entities = {
            '头痛', '发热', '咳嗽', '失眠', '腹痛', '便秘', '腹泻', '乏力',
            '盗汗', '自汗', '眩晕', '耳鸣', '心悸', '胸闷', '气短', '呕吐',
            '恶心', '食欲不振', '口干', '口苦', '水肿', '疼痛'
        }
        
        # 常见方剂关键词
        self.formula_keywords = {
            '汤', '散', '丸', '膏', '丹', '饮'
        }
        
        # 疑问词和停用词
        self.stop_words = {
            '的', '是', '什么', '怎么', '如何', '为什么', '哪些', '吗', '呢',
            '啊', '吧', '了', '着', '过', '可以', '能', '会', '和', '或',
            '与', '及', '以及', '还有', '有', '没有', '不', '要', '想', '用'
        }
    
    def extract(self, query: str) -> List[str]:
        """
        从查询中提取实体
        
        Args:
            query: 用户查询
            
        Returns:
            提取的实体列表
        """
        # 如果启用知识图谱模式，优先使用知识图谱
        if self.use_kg and hasattr(self, 'kg_system'):
            try:
                result = self.kg_system.process_query(query)
                entities = [entity['mention'] for entity in result.get('entities', [])]
                if entities:
                    self.logger.info(f"知识图谱提取到实体: {entities}")
                    return entities
                else:
                    self.logger.debug("知识图谱未提取到实体，回退到规则模式")
            except Exception as e:
                self.logger.warning(f"知识图谱实体提取失败: {e}，回退到规则模式")
        
        # 如果启用LLM模式且模型服务可用，使用LLM
        if self.use_llm and self.model_service is not None:
            try:
                llm_entities = self._extract_with_llm(query)
                if llm_entities:
                    self.logger.info(f"LLM提取到实体: {llm_entities}")
                    return llm_entities
                else:
                    self.logger.debug("LLM未提取到实体，回退到规则模式")
            except Exception as e:
                self.logger.warning(f"LLM实体提取失败: {e}，回退到规则模式")
        
        # 规则模式（默认或回退）
        entities = []
        
        # 1. 直接匹配药材名称
        for herb in self.herb_entities:
            if herb in query:
                entities.append(herb)
        
        # 2. 直接匹配症状名称
        for symptom in self.symptom_entities:
            if symptom in query:
                entities.append(symptom)
        
        # 3. 提取方剂名称（包含特定后缀的词）
        # 例如：六味地黄丸、麻黄汤
        for keyword in self.formula_keywords:
            # 查找形如"XXX汤"的模式
            pattern = f'([\\u4e00-\\u9fa5]{{2,8}}{keyword})'
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # 4. 去重并保持顺序
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_with_llm(self, query: str) -> List[str]:
        """
        使用LLM提取实体（同步方法）
        
        Args:
            query: 用户查询
            
        Returns:
            提取的实体列表
        """
        # 构建实体提取的prompt（简洁版，模型输出更稳定）
        prompt = f"""从下面的中医问题中提取实体（中药、方剂、症状）：

问题：{query}

只返回实体列表，格式：["实体1", "实体2"]
答案："""

        try:
            # 调用模型服务
            response = self.model_service.generate(
                query=prompt,  # ModelService使用query参数
                max_new_tokens=100,
                temperature=0.1  # 低温度，更确定性
            )
            
            # 解析输出（ModelService返回的键名是'answer'）
            output = response.get('answer', '').strip()
            
            self.logger.debug(f"LLM原始输出: {output[:200]}")
            
            # 策略1: 尝试直接解析列表格式 ["人参", "当归"]
            # 这是模型最可能的输出格式
            bracket_match = re.search(r'\[([^\]]+)\]', output)
            if bracket_match:
                bracket_content = bracket_match.group(1)
                # 提取引号内的内容
                entities = re.findall(r'["\']([^"\']+)["\']', bracket_content)
                if entities:
                    # 过滤和清理
                    entities = [e.strip() for e in entities if e and len(e.strip()) <= 20 and e.strip()]
                    self.logger.info(f"从列表格式提取到实体: {entities}")
                    return entities
            
            # 策略2: 尝试解析逗号/顿号分隔 "人参、当归"
            if '、' in output or '，' in output:
                # 按中文逗号或顿号分割
                entities = re.split(r'[、，]', output)
                entities = [e.strip() for e in entities if e and len(e.strip()) <= 20]
                # 过滤停用词
                entities = [e for e in entities if e not in self.stop_words]
                if entities:
                    self.logger.info(f"从分隔符提取到实体: {entities}")
                    return entities
            
            # 策略3: 尝试提取JSON格式（如果模型输出JSON）
            json_match = re.search(r'\{[^}]*"entities"[^}]*\[[^\]]*\][^}]*\}', output, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    entities = data.get('entities', [])
                    
                    if entities:
                        entities = [e.strip() for e in entities if e and len(e.strip()) <= 20]
                        self.logger.info(f"从JSON提取到实体: {entities}")
                        return entities
                except json.JSONDecodeError as e:
                    self.logger.debug(f"JSON解析失败: {e}")
            
            # 策略4: 最后回退 - 提取中文词汇（过滤停用词）
            chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,10}', output)
            if chinese_words:
                # 过滤停用词，只保留可能是实体的词
                entities = [w for w in chinese_words if w not in self.stop_words]
                # 取前5个
                entities = entities[:5]
                if entities:
                    self.logger.info(f"从中文词汇提取到: {entities}")
                    return entities
            
            self.logger.warning(f"无法从模型输出中提取实体: {output[:100]}")
            return []
                
        except Exception as e:
            self.logger.error(f"LLM实体提取失败: {e}")
            return []
    
    def has_entities(self, query: str) -> bool:
        """检查查询是否包含实体"""
        return len(self.extract(query)) > 0


# 创建全局实例
_entity_extractor = None
_entity_extractor_with_llm = None

def get_entity_extractor(model_service=None, use_llm=False, use_kg=False, csv_file_path=None) -> EntityExtractor:
    """
    获取实体提取器单例
    
    Args:
        model_service: 模型服务实例（用于LLM模式）
        use_llm: 是否使用LLM模式
        use_kg: 是否使用知识图谱模式（用于知识图谱检索）
        csv_file_path: 知识图谱CSV文件路径
        
    Returns:
        EntityExtractor实例
    """
    global _entity_extractor, _entity_extractor_with_llm, _entity_extractor_with_kg
    
    # 如果请求知识图谱模式（用于知识图谱检索）
    if use_kg:
        if not hasattr(get_entity_extractor, '_entity_extractor_with_kg'):
            get_entity_extractor._entity_extractor_with_kg = EntityExtractor(
                use_cache=True,
                model_service=model_service,
                use_llm=use_llm,
                use_kg=True,
                csv_file_path=csv_file_path
            )
        return get_entity_extractor._entity_extractor_with_kg
    
    # 如果请求LLM模式且提供了model_service
    if use_llm and model_service is not None:
        if _entity_extractor_with_llm is None:
            _entity_extractor_with_llm = EntityExtractor(
                use_cache=True,
                model_service=model_service,
                use_llm=True,
                use_kg=False
            )
        return _entity_extractor_with_llm
    
    # 默认规则模式（用于向量检索等不需要知识图谱的场景）
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor(use_cache=True, use_llm=False, use_kg=False)
    return _entity_extractor

