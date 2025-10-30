#!/usr/bin/env python3
"""
简化版实体和关系提取器
基于测试文件的成功经验重写
"""

import logging
import time
from typing import List, Dict, Any
from openai import OpenAI
import json
from src.models import Entity, Relationship, ProcessedDocument, GraphRAGResult
from src.config import GraphRAGConfig

class SimpleExtractor:
    """简化版提取器，专注于快速、准确的实体和关系提取"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SimpleExtractor")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=config.api_key or "EMPTY",
            base_url=config.api_base or "http://localhost:8000/v1"
        )
        self.logger.info(f"简化提取器初始化成功，服务地址: {config.api_base}")
    
    def extract_entities_and_relationships(self, document: ProcessedDocument) -> GraphRAGResult:
        """提取实体和关系 - 简化版"""
        start_time = time.time()
        
        try:
            self.logger.info(f"开始提取: {document.title}")
            
            # 直接处理整个文档，不分块
            entities = self._extract_entities_from_text(document.content, document.id)
            
            if not entities:
                self.logger.warning("未提取到实体")
                return GraphRAGResult(
                    document_id=document.id,
                    entities=[],
                    relationships=[],
                    processing_time=time.time() - start_time
                )
            
            # 提取关系
            relationships = self._extract_relationships_from_entities(entities, document.content, document.id)
            
            processing_time = time.time() - start_time
            self.logger.info(f"提取完成: 实体{len(entities)}个, 关系{len(relationships)}个, 耗时{processing_time:.2f}秒")
            
            return GraphRAGResult(
                document_id=document.id,
                entities=entities,
                relationships=relationships,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"提取失败: {e}")
            return GraphRAGResult(
                document_id=document.id,
                entities=[],
                relationships=[],
                processing_time=time.time() - start_time
            )
    
    def _extract_entities_from_text(self, text: str, document_id: str) -> List[Entity]:
        """从文本中提取实体"""
        try:
            # 构建实体提取提示
            prompt = f"""请从以下中医医案文本中提取所有类型的实体，严格按照JSON格式返回：

文本：{text}

重要：必须提取以下所有类型，不要只提取中药！

1. HERB(中药)：桂枝、白芍、生姜、大枣、甘草等中药名称
2. SYNDROME(证型)：肾阴阳两虚证、脾虚气滞证等中医证型
3. SYMPTOM(症状)：全身乏力、短气、月经不调、腹胀等患者症状
4. DISEASE(疾病)：感冒、失眠、焦虑等疾病名称
5. FORMULA(方剂)：桂枝汤、四物汤等方剂名称
6. TREATMENT(治法)：温补肾阳、健脾益气等治疗方法

注意：文本中明确提到的"证型:"、"治法:"、"症状"等关键词后的内容必须提取！

只返回JSON数组，格式：
[{{"name":"实体名","type":"类型","description":"描述"}}]"""
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的中医文本分析专家。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                entities_data = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析失败: {e}")
                return []
            
            entities = []
            for entity_info in entities_data:
                entity = Entity(
                    name=entity_info.get('name', '').strip(),
                    type=entity_info.get('type', 'OTHER'),
                    description=entity_info.get('description', ''),
                    confidence=0.8,
                    source_document_id=document_id
                )
                entities.append(entity)
            
            self.logger.info(f"成功提取 {len(entities)} 个实体")
            return entities
                
        except Exception as e:
            self.logger.error(f"实体提取失败: {e}")
            return []
    
    def _extract_relationships_from_entities(self, entities: List[Entity], text: str, document_id: str) -> List[Relationship]:
        """从实体中提取关系"""
        try:
            if not entities:
                return []
            
            relationships = []
            
            # 分类实体
            herbs = [e for e in entities if e.type in ['HERB', 'CONCEPT']]
            formulas = [e for e in entities if e.type == 'FORMULA']
            syndromes = [e for e in entities if e.type == 'SYNDROME']
            symptoms = [e for e in entities if e.type in ['SYMPTOM', 'SYMPТОM']]  # 处理字符编码问题
            diseases = [e for e in entities if e.type == 'DISEASE']
            treatments = [e for e in entities if e.type == 'TREATMENT']
            
            self.logger.info(f"实体分类: 中药{len(herbs)}个, 方剂{len(formulas)}个, 证型{len(syndromes)}个, 症状{len(symptoms)}个, 疾病{len(diseases)}个, 治法{len(treatments)}个")
            
            # 提取方剂与中药的关系（包含关系）
            for formula in formulas:
                for herb in herbs:
                    if formula.name in text and herb.name in text:
                        relationship = Relationship(
                            source_entity_id=formula.id,
                            target_entity_id=herb.id,
                            relationship_type="包含",
                            description=f"{formula.name}包含{herb.name}",
                            confidence=0.8,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取方剂与疾病的关系（治疗关系）
            for formula in formulas:
                for disease in diseases:
                    if formula.name in text and disease.name in text:
                        relationship = Relationship(
                            source_entity_id=formula.id,
                            target_entity_id=disease.id,
                            relationship_type="治疗",
                            description=f"{formula.name}治疗{disease.name}",
                            confidence=0.8,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取方剂与症状的关系（治疗关系）
            for formula in formulas:
                for symptom in symptoms:
                    if formula.name in text and symptom.name in text:
                        relationship = Relationship(
                            source_entity_id=formula.id,
                            target_entity_id=symptom.id,
                            relationship_type="治疗",
                            description=f"{formula.name}治疗{symptom.name}",
                            confidence=0.7,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取中药与症状的关系（治疗关系）
            for herb in herbs:
                for symptom in symptoms:
                    if herb.name in text and symptom.name in text:
                        relationship = Relationship(
                            source_entity_id=herb.id,
                            target_entity_id=symptom.id,
                            relationship_type="治疗",
                            description=f"{herb.name}治疗{symptom.name}",
                            confidence=0.6,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取证型与症状的关系（表现为关系）
            for syndrome in syndromes:
                for symptom in symptoms:
                    if syndrome.name in text and symptom.name in text:
                        relationship = Relationship(
                            source_entity_id=syndrome.id,
                            target_entity_id=symptom.id,
                            relationship_type="表现为",
                            description=f"{syndrome.name}表现为{symptom.name}",
                            confidence=0.7,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取治法与证型的关系（治疗关系）
            for treatment in treatments:
                for syndrome in syndromes:
                    if treatment.name in text and syndrome.name in text:
                        relationship = Relationship(
                            source_entity_id=treatment.id,
                            target_entity_id=syndrome.id,
                            relationship_type="治疗",
                            description=f"{treatment.name}治疗{syndrome.name}",
                            confidence=0.8,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取治法与症状的关系（治疗关系）
            for treatment in treatments:
                for symptom in symptoms:
                    if treatment.name in text and symptom.name in text:
                        relationship = Relationship(
                            source_entity_id=treatment.id,
                            target_entity_id=symptom.id,
                            relationship_type="治疗",
                            description=f"{treatment.name}治疗{symptom.name}",
                            confidence=0.7,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取疾病与症状的关系（引起关系）
            for disease in diseases:
                for symptom in symptoms:
                    if disease.name in text and symptom.name in text:
                        relationship = Relationship(
                            source_entity_id=disease.id,
                            target_entity_id=symptom.id,
                            relationship_type="引起",
                            description=f"{disease.name}引起{symptom.name}",
                            confidence=0.6,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            # 提取中药与证型的关系（治疗关系）
            for herb in herbs:
                for syndrome in syndromes:
                    if herb.name in text and syndrome.name in text:
                        relationship = Relationship(
                            source_entity_id=herb.id,
                            target_entity_id=syndrome.id,
                            relationship_type="治疗",
                            description=f"{herb.name}治疗{syndrome.name}",
                            confidence=0.6,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
            
            self.logger.info(f"基于规则提取到 {len(relationships)} 个关系")
            return relationships
            
        except Exception as e:
            self.logger.error(f"关系提取失败: {e}")
            return []
