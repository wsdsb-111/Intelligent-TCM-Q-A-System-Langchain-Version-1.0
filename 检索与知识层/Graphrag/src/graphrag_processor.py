"""
中医知识图谱GraphRAG处理器
集成GraphRAG功能，专门用于提取中医领域的实体和关系
"""

import os
import re
import json
import uuid
import logging
import signal
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path
import tiktoken

from .models import Entity, Relationship, ProcessedDocument, GraphRAGResult, EntityType, RelationshipType
from .exceptions import GraphRAGProcessingError, EntityExtractionError, APIError, RetryableError
from .config import GraphRAGConfig


class PauseController:
    """暂停控制器，支持随时暂停和恢复处理"""
    
    def __init__(self):
        self._is_paused = False
        self._should_stop = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始状态为运行
        self._lock = threading.Lock()
        
    def pause(self):
        """暂停处理"""
        with self._lock:
            self._is_paused = True
            self._pause_event.clear()
            
    def resume(self):
        """恢复处理"""
        with self._lock:
            self._is_paused = False
            self._pause_event.set()
            
    def stop(self):
        """停止处理"""
        with self._lock:
            self._should_stop = True
            self._is_paused = False
            self._pause_event.set()
            
    def reset(self):
        """重置状态"""
        with self._lock:
            self._is_paused = False
            self._should_stop = False
            self._pause_event.set()
            
    def is_paused(self) -> bool:
        """检查是否暂停"""
        with self._lock:
            return self._is_paused
            
    def should_stop(self) -> bool:
        """检查是否应该停止"""
        with self._lock:
            return self._should_stop
            
    def wait_if_paused(self):
        """如果暂停则等待，直到恢复或停止"""
        self._pause_event.wait()
        
    def get_status(self) -> str:
        """获取当前状态"""
        with self._lock:
            if self._should_stop:
                return "停止"
            elif self._is_paused:
                return "暂停"
            else:
                return "运行"


class GraphRAGProcessor:
    """GraphRAG处理器"""
    
    def __init__(self, config: GraphRAGConfig):
        """
        初始化GraphRAG处理器
        
        Args:
            config: GraphRAG配置
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GraphRAGProcessor")
        
        # 初始化暂停控制器
        self.pause_controller = PauseController()
        
        # 初始化OpenAI客户端
        self._init_openai_client()
        
        # 实体提取提示模板
        self.entity_extraction_prompt = self._load_entity_extraction_prompt()
        
        # 关系提取提示模板
        self.relationship_extraction_prompt = self._load_relationship_extraction_prompt()
        
        # 支持的实体类型（更新为8大类别）
        self.supported_entity_types = {
            'BASIC_THEORY': '基础理论',      # 中医基础理论
            'DISEASE': '病症',              # 疾病和证候
            'HERB': '中药',                # 中药
            'FORMULA': '方剂',             # 方剂
            'THERAPY': '疗法',             # 治疗方法
            'DIAGNOSIS': '诊断',           # 诊断方法
            'LITERATURE': '文献',          # 中医文献
            'PERSON': '人物'               # 名医、人物
        }
        
        # 支持的关系类型（基于截图的10大核心关系类型）
        self.supported_relationship_types = {
            # 1. 关联关系 (Association Relationship)
            '关联经络': '关联关系',
            '对应证候': '关联关系',
            
            # 2. 治疗关系 (Treatment Relationship)
            '常用中药': '治疗关系',
            '适用疗法': '治疗关系',
            '治疗': '治疗关系',
            
            # 3. 组成关系 (Composition Relationship)
            '组成中药': '组成关系',
            '涉及穴位': '组成关系',
            '包含': '组成关系',
            
            # 4. 归属关系 (Attribution Relationship)
            '源于方剂': '归属关系',
            '归属病症': '归属关系',
            '归属': '归属关系',
            
            # 5. 功效关系 (Efficacy Relationship)
            '核心功效': '功效关系',
            '主要作用': '功效关系',
            '功效': '功效关系',
            
            # 6. 诊断关系 (Diagnosis Relationship)
            '用于诊断': '诊断关系',
            '基于症状': '诊断关系',
            '诊断': '诊断关系',
            
            # 7. 文献关系 (Literature Relationship)
            '记载实体': '文献关系',
            '著有文献': '文献关系',
            '记载': '文献关系',
            
            # 8. 禁忌关系 (Contraindication Relationship)
            '配伍禁忌': '禁忌关系',
            '禁忌疗法': '禁忌关系',
            '禁忌': '禁忌关系',
            
            # 9. 体质关系 (Constitution Relationship)
            '易患病症': '体质关系',
            '适合体质': '体质关系',
            '体质': '体质关系',
            
            # 10. 传承关系 (Inheritance Relationship)
            '创立方剂': '传承关系',
            '传承自': '传承关系',
            '传承': '传承关系',
            
            # 其他常见关系
            '相关': '相关关系',
            '相互作用': '相互作用',
            '配伍': '配伍关系',
            '相生': '相生关系',
            '相克': '相克关系'
        }
    
    def _init_openai_client(self):
        """初始化AI客户端（OpenAI或Ollama）"""
        if self.config.use_ollama:
            self._init_ollama_client()
        else:
            self._init_openai_client_only()
    
    def _init_ollama_client(self):
        """初始化Ollama客户端"""
        try:
            import requests
            
            # 测试Ollama服务是否可用
            response = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Ollama服务连接成功: {self.config.ollama_base_url}")
                self.logger.info(f"使用模型: {self.config.ollama_model}")
            else:
                raise GraphRAGProcessingError(
                    f"Ollama服务连接失败，状态码: {response.status_code}",
                    error_code="OLLAMA_CONNECTION_FAILED"
                )
                
        except ImportError:
            raise GraphRAGProcessingError(
                "缺少requests库，请安装: pip install requests",
                error_code="MISSING_DEPENDENCY"
            )
        except requests.exceptions.ConnectionError:
            raise GraphRAGProcessingError(
                f"无法连接到Ollama服务: {self.config.ollama_base_url}，请确保Ollama正在运行",
                error_code="OLLAMA_CONNECTION_ERROR"
            )
        except Exception as e:
            raise GraphRAGProcessingError(f"初始化Ollama客户端失败: {e}")
    
    def _init_openai_client_only(self):
        """初始化 vLLM 客户端"""
        try:
            from openai import OpenAI
            
            api_key = self.config.api_key or "EMPTY"
            if not self.config.api_base:
                self.config.api_base = "http://localhost:8000/v1"
            
            self.client = OpenAI(api_key=api_key, base_url=self.config.api_base)
            self.logger.info(f"vLLM 客户端初始化成功，服务地址: {self.config.api_base}")
            
        except ImportError:
            raise GraphRAGProcessingError("缺少 openai 库，请安装: pip install openai", error_code="MISSING_DEPENDENCY")
        except Exception as e:
            raise GraphRAGProcessingError(f"初始化 vLLM 客户端失败: {e}")
    
    def _load_entity_extraction_prompt(self) -> str:
        """加载中医实体提取提示模板"""
        return """请从以下中医文本中提取实体，严格按照JSON格式返回，不要添加任何其他文字：

文本：{text}

要求：
1. 提取以下8种实体类型：
   - BASIC_THEORY(基础理论)：阴阳学说、五行理论、经络学说等
   - DISEASE(病症)：疾病名称、证候、症状等
   - HERB(中药)：中药名称
   - FORMULA(方剂)：方剂名称
   - THERAPY(疗法)：治疗方法、技术手法等
   - DIAGNOSIS(诊断)：诊断方法、检查手段等
   - LITERATURE(文献)：中医典籍、文献名称
   - PERSON(人物)：名医、人物姓名

2. 只返回JSON数组，不要解释
3. 每个实体包含name、type、description字段
4. 确保提取的实体在文本中确实存在

示例格式：
[
  {{"name":"桂枝","type":"HERB","description":"温阳解表中药"}},
  {{"name":"脾虚气滞证","type":"DISEASE","description":"中医证候类型"}},
  {{"name":"阴阳学说","type":"BASIC_THEORY","description":"中医基础理论"}}
]"""
    
    def _load_relationship_extraction_prompt(self) -> str:
        """加载中医关系提取提示模板"""
        return """从文本中分析实体间的关系，返回JSON格式：

文本：{text}

实体：{entities}

要求：
1. 分析以下10大核心关系类型：
   - 关联关系：关联经络、对应证候等
   - 治疗关系：常用中药、适用疗法、治疗等
   - 组成关系：组成中药、涉及穴位、包含等
   - 归属关系：源于方剂、归属病症、归属等
   - 功效关系：核心功效、主要作用、功效等
   - 诊断关系：用于诊断、基于症状、诊断等
   - 文献关系：记载实体、著有文献、记载等
   - 禁忌关系：配伍禁忌、禁忌疗法、禁忌等
   - 体质关系：易患病症、适合体质、体质等
   - 传承关系：创立方剂、传承自、传承等

2. 只返回JSON数组，不要解释
3. 每个关系包含source、target、type、description字段

示例格式：
[
  {{"source":"桂枝","target":"脾虚气滞证","type":"治疗","description":"桂枝用于治疗脾虚气滞证"}},
  {{"source":"麻黄汤","target":"麻黄","type":"组成","description":"麻黄汤由麻黄组成"}},
  {{"source":"人参","target":"大补元气","type":"功效","description":"人参具有大补元气的功效"}}
]"""
    
    def extract_entities_and_relationships(self, document: ProcessedDocument) -> GraphRAGResult:
        """
        提取实体和关系（支持暂停功能）
        
        Args:
            document: 处理后的文档
            
        Returns:
            GraphRAGResult: 提取结果
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"开始提取实体和关系: {document.title}")
            self.logger.info(f"当前状态: {self.pause_controller.get_status()}")
            
            # 检查是否应该停止
            if self.pause_controller.should_stop():
                self.logger.info("处理被用户停止")
                return self._create_empty_result(document.id, start_time, "用户停止")
            
            # 分块处理长文档
            chunks = self._split_text_into_chunks(document.content, max_tokens=4096)
            
            # 如果文档太大，限制处理的块数
            max_chunks = 20  # 限制最大处理块数，避免API调用过多
            if len(chunks) > max_chunks:
                self.logger.warning(f"文档过大，共{len(chunks)}块，限制处理前{max_chunks}块")
                chunks = chunks[:max_chunks]
            
            all_entities = []
            all_relationships = []
            
            # 处理每个文本块
            for i, chunk in enumerate(chunks):
                # 检查暂停状态
                self.pause_controller.wait_if_paused()
                
                # 检查是否应该停止
                if self.pause_controller.should_stop():
                    self.logger.info(f"处理被用户停止，已处理 {i}/{len(chunks)} 个文本块")
                    return self._create_partial_result(
                        document.id, all_entities, all_relationships, 
                        start_time, i, len(chunks), "用户停止"
                    )
                
                self.logger.debug(f"处理文本块 {i+1}/{len(chunks)}")
                self.logger.info(f"进度: {i+1}/{len(chunks)} ({((i+1)/len(chunks)*100):.1f}%)")
                
                # 提取实体
                chunk_entities = self._extract_entities_from_chunk(chunk, document.id)
                if chunk_entities:
                    all_entities.extend(chunk_entities)
                    
                    # 检查暂停状态
                    self.pause_controller.wait_if_paused()
                    if self.pause_controller.should_stop():
                        self.logger.info(f"处理被用户停止，已处理 {i+1}/{len(chunks)} 个文本块")
                        return self._create_partial_result(
                            document.id, all_entities, all_relationships, 
                            start_time, i+1, len(chunks), "用户停止"
                        )
                    
                    # 提取关系
                    chunk_relationships = self._extract_relationships_from_chunk(
                        chunk, chunk_entities, document.id
                    )
                    if chunk_relationships:
                        all_relationships.extend(chunk_relationships)
                else:
                    self.logger.warning(f"文本块 {i+1} 未提取到实体")
                
                # 每处理完一个块，短暂休息以响应暂停信号
                time.sleep(0.1)
            
            # 检查暂停状态
            self.pause_controller.wait_if_paused()
            if self.pause_controller.should_stop():
                self.logger.info("处理被用户停止")
                return self._create_partial_result(
                    document.id, all_entities, all_relationships, 
                    start_time, len(chunks), len(chunks), "用户停止"
                )
            
            # 如果文本块级别没有提取到关系，尝试基于规则的关系提取
            if not all_relationships and len(all_entities) >= 2:
                self.logger.info(f"文本块级别未提取到关系，尝试基于规则的关系提取（实体数：{len(all_entities)}）")
                self.logger.info(f"实体列表：{[e.name + '(' + e.type + ')' for e in all_entities]}")
                
                # 检查暂停状态
                self.pause_controller.wait_if_paused()
                if self.pause_controller.should_stop():
                    self.logger.info("处理被用户停止")
                    return self._create_partial_result(
                        document.id, all_entities, all_relationships, 
                        start_time, len(chunks), len(chunks), "用户停止"
                    )
                
                rule_relationships = self._extract_relationships_by_rules(
                    document.content, all_entities, document.id
                )
                if rule_relationships:
                    all_relationships.extend(rule_relationships)
                    self.logger.info(f"基于规则提取到 {len(rule_relationships)} 个关系")
                else:
                    self.logger.info("基于规则的关系提取未找到匹配的关系")
            
            # 检查是否提取到实体
            if not all_entities:
                self.logger.warning(f"文档 {document.title} 未提取到任何实体")
                processing_time = (datetime.now() - start_time).total_seconds()
                return GraphRAGResult(
                    document_id=document.id,
                    entities=[],
                    relationships=[],
                    processing_time=processing_time,
                    metadata={
                        'chunks_processed': len(chunks),
                        'original_entity_count': 0,
                        'merged_entity_count': 0,
                        'relationship_count': 0,
                        'warning': 'No entities extracted'
                    }
                )
            
            # 检查暂停状态
            self.pause_controller.wait_if_paused()
            if self.pause_controller.should_stop():
                self.logger.info("处理被用户停止")
                return self._create_partial_result(
                    document.id, all_entities, all_relationships, 
                    start_time, len(chunks), len(chunks), "用户停止"
                )
            
            # 去重和合并相似实体
            merged_entities = self._merge_similar_entities(all_entities)
            
            # 更新关系中的实体ID
            updated_relationships = self._update_relationship_entity_ids(
                all_relationships, merged_entities
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = GraphRAGResult(
                document_id=document.id,
                entities=merged_entities,
                relationships=updated_relationships,
                processing_time=processing_time,
                metadata={
                    'chunks_processed': len(chunks),
                    'original_entity_count': len(all_entities),
                    'merged_entity_count': len(merged_entities),
                    'relationship_count': len(updated_relationships)
                }
            )
            
            self.logger.info(
                f"提取完成: {document.title}, "
                f"实体: {len(merged_entities)}, "
                f"关系: {len(updated_relationships)}, "
                f"耗时: {processing_time:.2f}秒"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"提取实体和关系失败: {document.title}, 错误: {e}")
            raise GraphRAGProcessingError(f"提取失败: {e}")
    
    def _create_empty_result(self, document_id: str, start_time: datetime, reason: str) -> GraphRAGResult:
        """创建空结果"""
        processing_time = (datetime.now() - start_time).total_seconds()
        return GraphRAGResult(
            document_id=document_id,
            entities=[],
            relationships=[],
            processing_time=processing_time,
            metadata={
                'chunks_processed': 0,
                'original_entity_count': 0,
                'merged_entity_count': 0,
                'relationship_count': 0,
                'status': reason
            }
        )
    
    def _create_partial_result(self, document_id: str, entities: List[Entity], 
                              relationships: List[Relationship], start_time: datetime,
                              processed_chunks: int, total_chunks: int, reason: str) -> GraphRAGResult:
        """创建部分处理结果"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 去重和合并相似实体
        merged_entities = self._merge_similar_entities(entities)
        
        # 更新关系中的实体ID
        updated_relationships = self._update_relationship_entity_ids(relationships, merged_entities)
        
        return GraphRAGResult(
            document_id=document_id,
            entities=merged_entities,
            relationships=updated_relationships,
            processing_time=processing_time,
            metadata={
                'chunks_processed': processed_chunks,
                'total_chunks': total_chunks,
                'progress_percentage': (processed_chunks / total_chunks * 100) if total_chunks > 0 else 0,
                'original_entity_count': len(entities),
                'merged_entity_count': len(merged_entities),
                'relationship_count': len(updated_relationships),
                'status': reason
            }
        )
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 4096) -> List[str]:
        """根据 token 数量分割文本"""
        if not text:
            return []
        
        tokenizer = tiktoken.get_encoding("cl100k_base")  # 使用适配 OpenAI 模型的 tokenizer
        tokens = tokenizer.encode(text)
        
        # 调试日志：打印总 token 数量
        self.logger.debug(f"原始文本 token 数量: {len(tokens)}")
        
        # 如果文本本身就很短，直接返回
        if len(tokens) <= max_tokens:
            self.logger.debug(f"文本长度在限制内，无需分割")
            return [text]
        
        chunks = []
        current_chunk = []
        
        for i, token in enumerate(tokens):
            if len(current_chunk) < max_tokens:
                current_chunk.append(token)
            else:
                # 当前块已满，保存并开始新块
                chunk_text = tokenizer.decode(current_chunk)
                chunks.append(chunk_text)
                self.logger.debug(f"创建文本块 {len(chunks)}: {len(current_chunk)} tokens")
                current_chunk = [token]
        
        # 添加最后一个块
        if current_chunk:
            chunk_text = tokenizer.decode(current_chunk)
            chunks.append(chunk_text)
            self.logger.debug(f"创建文本块 {len(chunks)}: {len(current_chunk)} tokens")
        
        self.logger.info(f"文本分割完成: 原始 {len(tokens)} tokens -> {len(chunks)} 个块")
        return chunks
    
    def _extract_entities_from_chunk(self, text: str, document_id: str) -> List[Entity]:
        """从文本块提取实体"""
        try:
            # 检查文本长度
            if len(text.strip()) < 10:
                self.logger.warning("文本块太短，跳过实体提取")
                return []
            
            # 验证 token 数量
            tokenizer = tiktoken.get_encoding("cl100k_base")
            text_tokens = tokenizer.encode(text)
            self.logger.debug(f"文本块 token 数量: {len(text_tokens)}")
            
            # 如果 token 数量超过限制，截断文本
            max_input_tokens = 3500  # 为提示模板和响应预留空间，Qwen2.5-7B支持4096
            if len(text_tokens) > max_input_tokens:
                self.logger.warning(f"文本块 token 数量 ({len(text_tokens)}) 超过限制 ({max_input_tokens})，进行截断")
                text = tokenizer.decode(text_tokens[:max_input_tokens])
                self.logger.debug(f"截断后文本长度: {len(text)} 字符")
            
            # 格式化提示
            prompt = self.entity_extraction_prompt.format(text=text)
            
            # 验证完整提示的 token 数量
            prompt_tokens = tokenizer.encode(prompt)
            self.logger.debug(f"完整提示 token 数量: {len(prompt_tokens)}")
            
            if len(prompt_tokens) > 4000:  # 确保不超过模型限制
                self.logger.error(f"提示 token 数量 ({len(prompt_tokens)}) 超过模型限制 (4000)")
                return []
            
            self.logger.debug(f"发送实体提取请求，文本长度: {len(text)} 字符, token: {len(text_tokens)}")
            response = self._call_ai_api(prompt)
            self.logger.info(f"收到API响应长度: {len(response)} 字符")
            self.logger.info(f"API响应内容: {response[:500]}...")
            
            if not response or len(response.strip()) == 0 or response.strip().count('\n') > len(response.strip()) * 0.8:
                self.logger.warning("API返回空响应或大量换行符，尝试基于规则提取")
                
                # 使用基于规则的方法提取实体
                rule_entities = self._extract_entities_by_rules(text)
                if rule_entities:
                    self.logger.info(f"基于规则提取到 {len(rule_entities)} 个实体")
                    return rule_entities
                else:
                    self.logger.warning("基于规则也未能提取到实体")
                    return []
            
            # 解析响应
            entities_data = self._parse_json_response(response)
            
            entities = []
            # 处理不同的响应格式
            if isinstance(entities_data, list):
                # 直接是实体列表
                entity_list = entities_data
            elif isinstance(entities_data, dict):
                # 是包含entities键的字典
                entity_list = entities_data.get('entities', [])
            else:
                self.logger.warning(f"未知的响应格式: {type(entities_data)}")
                entity_list = []
            
            for entity_info in entity_list:
                try:
                    entity = Entity(
                        name=entity_info.get('name', '').strip(),
                        type=entity_info.get('type', 'OTHER'),
                        description=entity_info.get('description', ''),
                        confidence=float(entity_info.get('confidence', 0.5)),
                        source_document_id=document_id
                    )
                    
                    # 验证实体
                    if self._validate_entity(entity):
                        entities.append(entity)
                except Exception as entity_error:
                    self.logger.warning(f"创建实体失败: {entity_error}, 数据: {entity_info}")
                    continue
            
            self.logger.debug(f"成功提取 {len(entities)} 个实体")
            return entities
            
        except KeyError as e:
            self.logger.error(f"提示模板格式化失败: {e}")
            self.logger.error(f"文本内容预览: {text[:100]}...")
            return []  # 返回空列表而不是抛出异常
        except Exception as e:
            self.logger.error(f"从文本块提取实体失败: {e}")
            self.logger.error(f"文本内容预览: {text[:100]}...")
            return []  # 返回空列表而不是抛出异常
    
    def _extract_relationships_from_chunk(self, text: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """从文本块提取关系"""
        try:
            if len(entities) < 2:
                self.logger.debug("实体数量少于2个，跳过关系提取")
                return []
            
            # 准备实体列表
            entity_names = [entity.name for entity in entities]
            entities_text = ", ".join(entity_names)
            
            prompt = self.relationship_extraction_prompt.format(
                text=text[:2000],  # 限制文本长度
                entities=entities_text
            )
            
            self.logger.debug(f"发送关系提取请求，实体数量: {len(entities)}")
            response = self._call_ai_api(prompt)
            self.logger.debug(f"收到关系提取响应: {response[:200]}...")
            
            # 解析响应
            relationships = self._parse_relationship_response(response, entities, document_id)
            
            self.logger.debug(f"成功提取 {len(relationships)} 个关系")
            return relationships
            
        except Exception as e:
            self.logger.error(f"从文本块提取关系失败: {e}")
            return []  # 关系提取失败不应该阻止整个流程
    
    def _extract_entities_by_rules(self, text: str) -> List[Entity]:
        """基于规则提取实体"""
        entities = []
        
        # 1. 基础理论词典
        theory_patterns = [
            '阴阳学说', '五行理论', '经络学说', '脏腑学说', '气血津液', '辨证论治',
            '整体观念', '天人合一', '形神合一', '经络理论', '脏象学说', '病机学说'
        ]
        
        # 2. 病症词典
        disease_patterns = [
            '感冒', '咳嗽', '头痛', '发热', '失眠', '高血压', '糖尿病', '冠心病',
            '脾虚气滞证', '肾阳虚证', '肝郁气滞证', '心脾两虚证', '肺肾阴虚证',
            '脾胃虚弱', '肝肾阴虚', '心肾不交', '脾肾阳虚', '肝火上炎',
            '痰湿内阻', '气滞血瘀', '阴虚火旺', '阳虚水泛', '气血两虚'
        ]
        
        # 3. 中药词典
        herb_patterns = [
            '桂枝', '附子', '人参', '黄芪', '当归', '白术', '茯苓', '甘草', '生姜', '大枣',
            '麻黄', '杏仁', '石膏', '知母', '黄芩', '黄连', '黄柏', '栀子', '连翘', '金银花',
            '薄荷', '菊花', '桑叶', '葛根', '柴胡', '升麻', '陈皮', '半夏', '枳实', '厚朴'
        ]
        
        # 4. 方剂词典
        formula_patterns = [
            '桂枝汤', '麻黄汤', '四君子汤', '六味地黄丸', '逍遥散', '归脾汤',
            '补中益气汤', '金匮肾气丸', '血府逐瘀汤', '温胆汤', '小柴胡汤', '白虎汤'
        ]
        
        # 5. 疗法词典
        therapy_patterns = [
            '针灸', '推拿', '拔罐', '艾灸', '刮痧', '中药内服', '中药外敷',
            '食疗', '气功', '太极拳', '八段锦', '五禽戏', '导引术'
        ]
        
        # 6. 诊断词典
        diagnosis_patterns = [
            '望诊', '闻诊', '问诊', '切诊', '脉诊', '舌诊', '面诊', '手诊',
            '辨证', '辨病', '四诊合参', '八纲辨证', '脏腑辨证', '经络辨证'
        ]
        
        # 7. 文献词典
        literature_patterns = [
            '伤寒论', '金匮要略', '黄帝内经', '神农本草经', '本草纲目',
            '温病条辨', '医宗金鉴', '景岳全书', '脾胃论', '医学衷中参西录'
        ]
        
        # 8. 人物词典
        person_patterns = [
            '张仲景', '华佗', '扁鹊', '孙思邈', '李时珍', '叶天士',
            '王叔和', '皇甫谧', '葛洪', '陶弘景', '巢元方', '钱乙'
        ]
        
        # 提取基础理论
        for pattern in theory_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="BASIC_THEORY",
                    description=f"中医基础理论：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取病症
        for pattern in disease_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="DISEASE",
                    description=f"病症：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取中药
        for pattern in herb_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="HERB",
                    description=f"中药：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取方剂
        for pattern in formula_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="FORMULA",
                    description=f"方剂：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取疗法
        for pattern in therapy_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="THERAPY",
                    description=f"疗法：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取诊断
        for pattern in diagnosis_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="DIAGNOSIS",
                    description=f"诊断方法：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取文献
        for pattern in literature_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="LITERATURE",
                    description=f"中医文献：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        # 提取人物
        for pattern in person_patterns:
            if pattern in text:
                entity = Entity(
                    name=pattern,
                    type="PERSON",
                    description=f"名医：{pattern}",
                    source_document_id="rule_extraction"
                )
                entities.append(entity)
        
        return entities
    
    def _extract_relationships_by_rules(self, text: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """基于规则提取关系（作为AI提取的备选方案）"""
        try:
            relationships = []
            
            # 分类实体
            basic_theories = [e for e in entities if e.type == 'BASIC_THEORY']
            diseases = [e for e in entities if e.type == 'DISEASE']
            herbs = [e for e in entities if e.type == 'HERB']
            formulas = [e for e in entities if e.type == 'FORMULA']
            therapies = [e for e in entities if e.type == 'THERAPY']
            diagnoses = [e for e in entities if e.type == 'DIAGNOSIS']
            literatures = [e for e in entities if e.type == 'LITERATURE']
            persons = [e for e in entities if e.type == 'PERSON']
            
            # 调试信息
            self.logger.info(f"实体分类统计: 基础理论{len(basic_theories)}个, 病症{len(diseases)}个, 中药{len(herbs)}个, 方剂{len(formulas)}个, 疗法{len(therapies)}个, 诊断{len(diagnoses)}个, 文献{len(literatures)}个, 人物{len(persons)}个")
            
            # 1. 治疗关系：中药/方剂/疗法 -> 病症
            treatment_keywords = ['治疗', '用于', '主治', '治', '医', '方剂', '中药']
            
            for herb in herbs:
                for disease in diseases:
                    if herb.name in text and disease.name in text:
                        confidence = 0.9 if any(keyword in text for keyword in treatment_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=herb.id,
                            target_entity_id=disease.id,
                            relationship_type="治疗",
                            description=f"{herb.name}用于治疗{disease.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {herb.name} -> {disease.name} (治疗)")
            
            for formula in formulas:
                for disease in diseases:
                    if formula.name in text and disease.name in text:
                        confidence = 0.9 if any(keyword in text for keyword in treatment_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=formula.id,
                            target_entity_id=disease.id,
                            relationship_type="治疗",
                            description=f"{formula.name}用于治疗{disease.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {formula.name} -> {disease.name} (治疗)")
            
            for therapy in therapies:
                for disease in diseases:
                    if therapy.name in text and disease.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in ['适用', '用于', '治疗']) else 0.6
                        relationship = Relationship(
                            source_entity_id=therapy.id,
                            target_entity_id=disease.id,
                            relationship_type="适用",
                            description=f"{therapy.name}适用于{disease.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {therapy.name} -> {disease.name} (适用)")
            
            # 2. 组成关系：方剂 -> 中药
            composition_keywords = ['成分', '包括', '含有', '组成', '由']
            
            for formula in formulas:
                for herb in herbs:
                    if formula.name in text and herb.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in composition_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=formula.id,
                            target_entity_id=herb.id,
                            relationship_type="组成",
                            description=f"{formula.name}包含{herb.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {formula.name} -> {herb.name} (组成)")
            
            # 3. 功效关系：中药 -> 功效描述
            efficacy_keywords = ['功效', '作用', '具有', '能', '可以']
            
            for herb in herbs:
                # 检查是否有功效描述
                if any(keyword in text for keyword in efficacy_keywords):
                    # 简单匹配：中药名称后面跟功效词汇
                    herb_pos = text.find(herb.name)
                    if herb_pos != -1:
                        # 查找功效相关的词汇
                        for keyword in efficacy_keywords:
                            if keyword in text[herb_pos:herb_pos+50]:  # 在中药名称后50字符内查找
                                # 创建功效实体（简化处理）
                                efficacy_desc = f"{herb.name}的功效"
                                relationship = Relationship(
                                    source_entity_id=herb.id,
                                    target_entity_id=herb.id,  # 自关联，实际应用中应该创建功效实体
                                    relationship_type="功效",
                                    description=f"{herb.name}具有功效",
                                    confidence=0.7,
                                    source_document_id=document_id
                                )
                                relationships.append(relationship)
                                break
            
            # 4. 文献关系：文献 -> 实体
            literature_keywords = ['记载', '出自', '来源', '载于']
            
            for literature in literatures:
                for herb in herbs:
                    if literature.name in text and herb.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in literature_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=literature.id,
                            target_entity_id=herb.id,
                            relationship_type="记载",
                            description=f"{literature.name}记载了{herb.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {literature.name} -> {herb.name} (记载)")
            
            # 5. 传承关系：人物 -> 方剂/理论
            inheritance_keywords = ['创', '立', '著', '写', '提出', '创立']
            
            for person in persons:
                for formula in formulas:
                    if person.name in text and formula.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in inheritance_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=person.id,
                            target_entity_id=formula.id,
                            relationship_type="创立",
                            description=f"{person.name}创立了{formula.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {person.name} -> {formula.name} (创立)")
                
                for theory in basic_theories:
                    if person.name in text and theory.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in inheritance_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=person.id,
                            target_entity_id=theory.id,
                            relationship_type="创立",
                            description=f"{person.name}创立了{theory.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {person.name} -> {theory.name} (创立)")
            
            # 6. 诊断关系：诊断方法 -> 病症
            diagnosis_keywords = ['诊断', '检查', '判断', '识别']
            
            for diagnosis in diagnoses:
                for disease in diseases:
                    if diagnosis.name in text and disease.name in text:
                        confidence = 0.8 if any(keyword in text for keyword in diagnosis_keywords) else 0.6
                        relationship = Relationship(
                            source_entity_id=diagnosis.id,
                            target_entity_id=disease.id,
                            relationship_type="诊断",
                            description=f"{diagnosis.name}用于诊断{disease.name}",
                            confidence=confidence,
                            source_document_id=document_id
                        )
                        relationships.append(relationship)
                        self.logger.debug(f"基于规则提取关系: {diagnosis.name} -> {disease.name} (诊断)")
            
            self.logger.info(f"基于规则提取到 {len(relationships)} 个关系")
            return relationships
            
        except Exception as e:
            self.logger.error(f"基于规则提取关系失败: {e}")
            return []
    
    def _call_ai_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用AI API（OpenAI或Ollama）"""
        if self.config.use_ollama:
            return self._call_ollama_api(prompt, max_retries)
        else:
            return self._call_openai_api(prompt, max_retries)
    
    def _call_ollama_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用Ollama API"""
        for attempt in range(max_retries):
            try:
                import requests
                import json
                
                # 构建请求数据
                data = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "stop": []  # 移除停止标记，让模型自由生成
                    }
                }
                
                self.logger.info(f"发送Ollama请求到: {self.config.ollama_base_url}/api/generate")
                self.logger.info(f"使用模型: {self.config.ollama_model}")
                self.logger.info(f"提示内容: {prompt[:200]}...")
                self.logger.debug(f"完整请求数据: {data}")
                
                # 发送请求
                response = requests.post(
                    f"{self.config.ollama_base_url}/api/generate",
                    json=data,
                    timeout=300  # 增加超时时间到300秒（5分钟）
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "").strip()
                    done_reason = result.get("done_reason", "")
                    
                    self.logger.info(f"Ollama API响应长度: {len(response_text)}")
                    self.logger.info(f"Ollama API响应内容: '{response_text}'")
                    self.logger.info(f"完成原因: {done_reason}")
                    self.logger.debug(f"完整响应JSON: {result}")
                    
                    if not response_text:
                        self.logger.warning("Ollama返回空响应")
                        self.logger.warning(f"完成原因: {done_reason}")
                        self.logger.warning(f"完整响应: {result}")
                        
                        # 如果是因为停止标记导致的空响应，尝试不同的提示
                        if done_reason == "stop":
                            self.logger.info("检测到停止标记导致的空响应，尝试简化提示")
                            return self._try_simplified_prompt(prompt)
                    
                    return response_text
                else:
                    self.logger.error(f"Ollama API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                    raise APIError(f"Ollama API调用失败，状态码: {response.status_code}")
                
            except Exception as e:
                self.logger.warning(f"Ollama API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                # 如果是超时错误，尝试使用更简单的提示
                if "timeout" in str(e).lower() and attempt < max_retries - 1:
                    self.logger.info("检测到超时，尝试使用简化提示")
                    try:
                        return self._try_simplified_prompt(prompt)
                    except Exception as simple_error:
                        self.logger.warning(f"简化提示也失败: {simple_error}")
                
                if attempt == max_retries - 1:
                    raise APIError(f"Ollama API调用失败: {e}")
                
                # 指数退避
                import time
                time.sleep(2 ** attempt)
    
    def _call_openai_api(self, prompt: str, max_retries: int = 3) -> str:
        """调用OpenAI API"""
        for attempt in range(max_retries):
            try:
                # 使用已初始化的客户端
                if not hasattr(self, 'client'):
                    raise GraphRAGProcessingError("OpenAI客户端未初始化")
                
                # 使用chat completions API
                messages = [
                    {"role": "system", "content": "你是一个专业的中医文本分析专家。"},
                    {"role": "user", "content": prompt}
                ]
                
                # 验证完整提示的 token 数量
                tokenizer = tiktoken.get_encoding("cl100k_base")
                full_content = "你是一个专业的中医文本分析专家。" + prompt
                prompt_tokens = tokenizer.encode(full_content)
                self.logger.debug(f"完整提示 token 数量: {len(prompt_tokens)}")
                
                # 确保 max_tokens 设置合理
                safe_max_tokens = min(self.config.max_tokens, 1000)  # 限制响应长度
                total_tokens = len(prompt_tokens) + safe_max_tokens
                
                if total_tokens > 4096:  # 确保总 token 数不超过模型限制
                    safe_max_tokens = 4096 - len(prompt_tokens) - 100  # 预留100个token的安全边界
                    if safe_max_tokens < 100:
                        self.logger.error(f"提示过长，无法生成响应。提示token: {len(prompt_tokens)}")
                        return ""
                
                self.logger.debug(f"API配置: base_url={self.config.api_base}, model={self.config.model}")
                self.logger.debug(f"Token统计: 提示={len(prompt_tokens)}, 最大响应={safe_max_tokens}, 总计={len(prompt_tokens) + safe_max_tokens}")
                
                self.logger.info(f"正在调用API: model={self.config.model}, max_tokens={safe_max_tokens}")
                
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    temperature=0.3,  # 降低温度，提高确定性
                    top_p=0.9,  # 添加top_p参数
                    frequency_penalty=0.1,  # 减少重复
                    presence_penalty=0.1,  # 鼓励多样性
                    stop=["\n\n", "<|im_end|>"],  # 添加停止标记
                    timeout=30  # 添加30秒超时
                )
                
                # 检查响应是否为空
                if not response.choices or not response.choices[0].message.content:
                    self.logger.warning("API返回空响应")
                    self.logger.debug(f"完整响应对象: {response}")
                    return ""
                
                content = response.choices[0].message.content.strip()
                self.logger.info(f"API调用成功，响应长度: {len(content)}")
                self.logger.info(f"响应内容: {content[:200]}...")
                
                return content
                
            except Exception as e:
                self.logger.warning(f"OpenAI API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    raise APIError(f"OpenAI API调用失败: {e}")
                
                # 指数退避
                import time
                time.sleep(2 ** attempt)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析响应（支持JSON和文本格式）"""
        try:
            # 清理响应文本
            response = response.strip()
            
            # 首先尝试解析为JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # 尝试提取JSON数组部分
            try:
                # 查找第一个完整的JSON数组
                start = response.find('[')
                if start != -1:
                    # 找到匹配的结束括号
                    bracket_count = 0
                    end = start
                    for i, char in enumerate(response[start:], start):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end = i + 1
                                break
                    
                    if end > start:
                        json_str = response[start:end]
                        return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # 尝试提取JSON对象部分
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # 如果JSON解析失败，尝试解析文本格式
            try:
                entities = []
                lines = response.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # 处理带|分隔符的格式
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            entity_type = parts[1].strip()
                            description = parts[2].strip() if len(parts) > 2 else f"{name} entity"
                            
                            if name and entity_type:
                                entities.append({
                                    "name": name,
                                    "type": entity_type,
                                    "description": description,
                                    "confidence": 0.8
                                })
                    # 处理简单的文本列表
                    elif line and len(line) > 1 and not line.startswith(('Example:', 'Text:', 'Types:', '从这段文字中', '这句话描述')):
                        # 处理结构化的响应（如：1. 患者症状：发热、咳嗽）
                        if '：' in line or ':' in line:
                            # 提取冒号后的内容
                            if '：' in line:
                                parts = line.split('：', 1)
                            else:
                                parts = line.split(':', 1)
                            
                            if len(parts) > 1:
                                content = parts[1].strip()
                                # 如果包含多个词，分割它们
                                if '、' in content:
                                    words = [w.strip() for w in content.split('、')]
                                elif ',' in content:
                                    words = [w.strip() for w in content.split(',')]
                                else:
                                    words = [content]
                                
                                for word in words:
                                    if word and len(word) > 1:
                                        entity_type = self._classify_entity_type(word)
                                        entities.append({
                                            "name": word,
                                            "type": entity_type,
                                            "description": f"{word} entity",
                                            "confidence": 0.7
                                        })
                        else:
                            # 简单的启发式分类
                            entity_type = self._classify_entity_type(line)
                            entities.append({
                                "name": line,
                                "type": entity_type,
                                "description": f"{line} entity",
                                "confidence": 0.7
                            })
                
                if entities:
                    self.logger.info(f"从文本格式中提取到 {len(entities)} 个实体")
                    return {"entities": entities, "relationships": []}
            except Exception as e:
                self.logger.debug(f"文本格式解析失败: {e}")
            
            # 如果无法解析，返回空结果
            self.logger.warning(f"无法解析响应，响应内容: {response[:500]}...")
            self.logger.warning(f"响应长度: {len(response)}")
            self.logger.warning(f"响应类型: {type(response)}")
            return {"entities": [], "relationships": []}
        
        except Exception as e:
            self.logger.error(f"解析响应时发生异常: {e}")
            return {"entities": [], "relationships": []}
    
    def _parse_relationship_response(self, response: str, entities: List[Entity], document_id: str) -> List[Relationship]:
        """解析关系提取响应"""
        relationships = []
        entity_name_to_id = {entity.name: entity.id for entity in entities}
        
        try:
            # 清理响应文本
            response = response.strip()
            
            # 首先尝试解析为JSON格式
            try:
                data = json.loads(response)
                # 处理不同的响应格式
                if isinstance(data, list):
                    # 直接是关系列表
                    rel_list = data
                elif isinstance(data, dict):
                    # 是包含relationships键的字典
                    rel_list = data.get('relationships', [])
                else:
                    rel_list = []
                
                for rel_info in rel_list:
                    relationship = self._create_relationship_from_dict(rel_info, entity_name_to_id, document_id)
                    if relationship:
                        relationships.append(relationship)
                return relationships
            except json.JSONDecodeError:
                pass
            
            # 尝试提取JSON部分
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    for rel_info in data.get('relationships', []):
                        relationship = self._create_relationship_from_dict(rel_info, entity_name_to_id, document_id)
                        if relationship:
                            relationships.append(relationship)
                    return relationships
            except json.JSONDecodeError:
                pass
            
            # 解析文本格式 (source|target|type)
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and len(line.split('|')) >= 3:
                    parts = line.split('|')
                    source_name = parts[0].strip()
                    target_name = parts[1].strip()
                    rel_type = parts[2].strip()
                    
                    # 查找对应的实体ID
                    source_id = entity_name_to_id.get(source_name)
                    target_id = entity_name_to_id.get(target_name)
                    
                    if source_id and target_id and source_id != target_id:
                        # 使用关系类型映射
                        normalized_rel_type = self.supported_relationship_types.get(rel_type, rel_type)
                        
                        relationship = Relationship(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relationship_type=normalized_rel_type,
                            description=f"{source_name}与{target_name}的关系",
                            weight=0.8,
                            confidence=0.7,
                            source_document_id=document_id
                        )
                        
                        if self._validate_relationship(relationship):
                            relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"解析关系响应失败: {e}")
            return []
    
    def _create_relationship_from_dict(self, rel_info: Dict[str, Any], entity_name_to_id: Dict[str, str], document_id: str) -> Optional[Relationship]:
        """从字典创建关系对象"""
        try:
            source_name = rel_info.get('source', '').strip()
            target_name = rel_info.get('target', '').strip()
            
            # 查找对应的实体ID
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)
            
            if source_id and target_id and source_id != target_id:
                # 使用关系类型映射
                raw_rel_type = rel_info.get('type', '相关')
                normalized_rel_type = self.supported_relationship_types.get(raw_rel_type, raw_rel_type)
                
                relationship = Relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=normalized_rel_type,
                    description=rel_info.get('description', ''),
                    weight=float(rel_info.get('weight', 1.0)),
                    confidence=float(rel_info.get('confidence', 0.5)),
                    source_document_id=document_id
                )
                
                if self._validate_relationship(relationship):
                    return relationship
        except Exception as e:
            self.logger.warning(f"创建关系失败: {e}, 数据: {rel_info}")
        
        return None
    
    def _try_simplified_prompt(self, original_prompt: str) -> str:
        """尝试使用更简单的提示"""
        try:
            import requests
            
            # 提取文本内容
            if "Extract key words from:" in original_prompt:
                text = original_prompt.split("Extract key words from:", 1)[1].strip()
            else:
                text = original_prompt
            
            # 使用最简单的提示
            simple_prompt = f"Keywords: {text}"
            
            self.logger.info(f"尝试简化提示: {simple_prompt}")
            
            data = {
                "model": self.config.ollama_model,
                "prompt": simple_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100
                }
            }
            
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json=data,
                timeout=180  # 简化提示的超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                self.logger.info(f"简化提示响应: '{response_text}'")
                return response_text
            else:
                self.logger.error(f"简化提示请求失败: {response.status_code}")
                return ""
                
        except Exception as e:
            self.logger.error(f"简化提示异常: {e}")
            return ""
    
    def _classify_entity_type(self, text: str) -> str:
        """简单的实体类型分类（基于8大实体类型）"""
        text_lower = text.lower()
        
        # 基础理论关键词
        theory_keywords = ['学说', '理论', '观念', '阴阳', '五行', '经络', '脏腑', '气血', '辨证']
        if any(keyword in text for keyword in theory_keywords):
            return 'BASIC_THEORY'
        
        # 病症关键词
        disease_keywords = ['感冒', '咳嗽', '头痛', '发热', '失眠', '高血压', '糖尿病', '冠心病', '证', '症', '病']
        if any(keyword in text for keyword in disease_keywords):
            return 'DISEASE'
        
        # 中药关键词
        herb_keywords = ['桂枝', '附子', '人参', '黄芪', '当归', '白术', '茯苓', '甘草', '药']
        if any(keyword in text for keyword in herb_keywords):
            return 'HERB'
        
        # 方剂关键词
        formula_keywords = ['汤', '丸', '散', '膏', '丹', '方', '桂枝汤', '麻黄汤', '四君子汤']
        if any(keyword in text for keyword in formula_keywords):
            return 'FORMULA'
        
        # 疗法关键词
        therapy_keywords = ['针灸', '推拿', '拔罐', '艾灸', '刮痧', '食疗', '气功', '太极拳']
        if any(keyword in text for keyword in therapy_keywords):
            return 'THERAPY'
        
        # 诊断关键词
        diagnosis_keywords = ['望诊', '闻诊', '问诊', '切诊', '脉诊', '舌诊', '诊断']
        if any(keyword in text for keyword in diagnosis_keywords):
            return 'DIAGNOSIS'
        
        # 文献关键词
        literature_keywords = ['论', '经', '要略', '本草', '纲目', '内经', '伤寒', '金匮']
        if any(keyword in text for keyword in literature_keywords):
            return 'LITERATURE'
        
        # 人物关键词
        person_keywords = ['张仲景', '华佗', '扁鹊', '孙思邈', '李时珍', '叶天士', '医生', '名医']
        if any(keyword in text for keyword in person_keywords):
            return 'PERSON'
        
        # 默认返回病症
        return 'DISEASE'
    
    def _validate_entity(self, entity: Entity) -> bool:
        """验证实体"""
        if not entity.name or len(entity.name.strip()) < 2:
            return False
        
        if entity.confidence < 0.3:  # 置信度太低
            return False
        
        # 过滤常见的无意义实体
        invalid_names = {'的', '了', '是', '在', '有', '和', '与', 'the', 'a', 'an', 'and', 'or'}
        if entity.name.lower() in invalid_names:
            return False
        
        return True
    
    def _validate_relationship(self, relationship: Relationship) -> bool:
        """验证关系"""
        if relationship.confidence < 0.2:  # 降低置信度阈值
            return False
        
        if not relationship.description or len(relationship.description.strip()) < 2:  # 降低描述长度要求
            return False
        
        return True
    
    def _merge_similar_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并相似实体"""
        if not entities:
            return []
        
        merged = []
        entity_groups = {}
        
        # 按名称分组（忽略大小写和空格）
        for entity in entities:
            normalized_name = entity.name.lower().strip().replace(' ', '')
            
            if normalized_name not in entity_groups:
                entity_groups[normalized_name] = []
            entity_groups[normalized_name].append(entity)
        
        # 合并每组中的实体
        for group in entity_groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 选择置信度最高的作为主实体
                primary_entity = max(group, key=lambda e: e.confidence)
                
                # 合并描述
                descriptions = [e.description for e in group if e.description]
                if descriptions:
                    primary_entity.description = ". ".join(descriptions)
                
                # 使用最高置信度
                primary_entity.confidence = max(e.confidence for e in group)
                
                merged.append(primary_entity)
        
        return merged
    
    def _update_relationship_entity_ids(self, relationships: List[Relationship], entities: List[Entity]) -> List[Relationship]:
        """更新关系中的实体ID"""
        entity_id_mapping = {}
        
        # 创建旧ID到新ID的映射
        for entity in entities:
            entity_id_mapping[entity.id] = entity.id
        
        updated_relationships = []
        for rel in relationships:
            if rel.source_entity_id in entity_id_mapping and rel.target_entity_id in entity_id_mapping:
                updated_relationships.append(rel)
        
        return updated_relationships
    
    def get_supported_entity_types(self) -> List[str]:
        """获取支持的实体类型"""
        return list(self.supported_entity_types.keys())
    
    def get_supported_relationship_types(self) -> List[str]:
        """获取支持的关系类型"""
        return list(self.supported_relationship_types.keys())
    
    def batch_extract_with_pause(self, documents: List[ProcessedDocument], 
                                progress_callback=None) -> List[GraphRAGResult]:
        """
        批量提取实体和关系（支持暂停功能）
        
        Args:
            documents: 文档列表
            progress_callback: 进度回调函数，接收 (current, total, status) 参数
            
        Returns:
            List[GraphRAGResult]: 提取结果列表
        """
        results = []
        total_docs = len(documents)
        
        self.logger.info(f"开始批量处理 {total_docs} 个文档")
        
        for i, document in enumerate(documents):
            # 检查暂停状态
            self.pause_controller.wait_if_paused()
            
            # 检查是否应该停止
            if self.pause_controller.should_stop():
                self.logger.info(f"批量处理被用户停止，已处理 {i}/{total_docs} 个文档")
                break
            
            # 调用进度回调
            if progress_callback:
                progress_callback(i, total_docs, f"正在处理: {document.title}")
            
            self.logger.info(f"处理文档 {i+1}/{total_docs}: {document.title}")
            
            try:
                # 提取实体和关系
                result = self.extract_entities_and_relationships(document)
                results.append(result)
                
                # 记录进度
                self.logger.info(
                    f"文档 {i+1}/{total_docs} 完成: "
                    f"实体{len(result.entities)}个, 关系{len(result.relationships)}个"
                )
                
            except Exception as e:
                self.logger.error(f"处理文档 {document.title} 失败: {e}")
                # 创建错误结果
                error_result = GraphRAGResult(
                    document_id=document.id,
                    entities=[],
                    relationships=[],
                    processing_time=0,
                    metadata={'error': str(e), 'status': '处理失败'}
                )
                results.append(error_result)
            
            # 短暂休息以响应暂停信号
            time.sleep(0.1)
        
        # 最终进度回调
        if progress_callback:
            status = "用户停止" if self.pause_controller.should_stop() else "完成"
            progress_callback(len(results), total_docs, status)
        
        self.logger.info(f"批量处理完成，成功处理 {len(results)}/{total_docs} 个文档")
        return results
    
    def pause_processing(self):
        """暂停处理"""
        self.pause_controller.pause()
        self.logger.info("处理已暂停")
        
    def resume_processing(self):
        """恢复处理"""
        self.pause_controller.resume()
        self.logger.info("处理已恢复")
        
    def stop_processing(self):
        """停止处理"""
        self.pause_controller.stop()
        self.logger.info("处理已停止")
        
    def reset_processing_state(self):
        """重置处理状态"""
        self.pause_controller.reset()
        self.logger.info("处理状态已重置")
        
    def get_processing_status(self) -> str:
        """获取处理状态"""
        return self.pause_controller.get_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            "use_ollama": self.config.use_ollama,
            "model": self.config.ollama_model if self.config.use_ollama else self.config.model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "supported_entity_types": self.get_supported_entity_types(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "ollama_base_url": self.config.ollama_base_url if self.config.use_ollama else None
        }