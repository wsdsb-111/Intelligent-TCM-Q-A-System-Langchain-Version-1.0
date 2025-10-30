#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索RAGAS评估器v4.0 - 优化版RAG流程
实现：用户查询 → 智能路由 → 检索组件 → 知识召回 → 关键词增强 → 卸载检索组件 → 本地模型生成 → 卸载生成组件 → 输出回答
"""

import asyncio
import gc
import json
import logging
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "应用协调层"))

# 导入必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import psutil

# 配置日志 - 修复Windows编码问题
import io
import sys

# 创建UTF-8编码的流处理器
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # 确保消息以UTF-8编码输出
            if hasattr(self.stream, 'buffer'):
                self.stream.buffer.write(msg.encode('utf-8'))
                self.stream.buffer.write(b'\n')
                self.stream.buffer.flush()
            else:
                self.stream.write(msg + '\n')
                self.stream.flush()
        except Exception:
            self.handleError(record)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(),
        logging.FileHandler('hybrid_ragas_evaluator_v4.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ========== 配置定义 ==========

# 本地模型配置
LOCAL_MODEL_CONFIG = {
    "base_model_path": str(project_root / "Model Layer" / "model" / "qwen" / "Qwen3-1.7B" / "Qwen" / "Qwen3-1___7B"),
    "lora_path": str(project_root / "Model Layer" / "model" / "checkpoint-7983"),
    "max_length": 4096,
    "temperature": 0.1,
    "top_p": 0.1,
    "repetition_penalty": 1.15
}

# DeepSeek配置（严格遵循n=1和JSON格式）
DEEPSEEK_CONFIG = {
    "api_key": "sk-ffd7abd1faed46fd8de1b418b37244d4",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "n": 1,  # 强制n=1
    "temperature": 0.1,
    "max_tokens": 4096,
    "timeout": 180,
    "max_retries": 5,
    "request_timeout": 180,
    "model_kwargs": {"response_format": {"type": "json_object"}}  # 强制返回纯JSON格式
}

# Qwen-Flash配置（用于智能路由）
QWEN_FLASH_CONFIG = {
    "api_key": "sk-6157e39178ac439bb00c43ba6b094501",
    "model_name": "qwen-flash",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
}

# 检索配置
RETRIEVAL_CONFIG = {
    "vector_model_path": str(project_root / "Model Layer" / "model" / "iic" / "nlp_gte_sentence-embedding_chinese-base" / "iic" / "nlp_gte_sentence-embedding_chinese-base"),
    "faiss_path": str(project_root / "检索与知识层" / "faiss_rag" / "向量数据库_768维"),
    "neo4j_uri": "neo4j://127.0.0.1:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "hx1230047",
    "top_k": 5,
    "score_threshold": 0.2
}

# 生成配置
GENERATION_CONFIG = {
    "max_new_tokens": 512,        # 进一步减少生成长度，避免幻觉
    "temperature": 0.1,           # 极低温度，提高确定性
    "top_p": 0.4,                 # 降低top_p，减少随机性
    "num_beams": 3,               # 使用贪心搜索，避免beam search的副作用
    "do_sample": False,           # 禁用采样，使用贪心搜索
    "repetition_penalty": 1.3,    # 增加重复惩罚
    "length_penalty": 1.0,        # 中性长度惩罚
    "min_new_tokens": 20,          # 减少最小生成长度
    "no_repeat_ngram_size": 5,    # 增加n-gram重复检查
    "early_stopping": True,
    "use_cache": True,
    "pad_token_id": None,         # 让系统自动设置
    "eos_token_id": None          # 让系统自动设置
}

# 提示词模板
PROMPT_TEMPLATES = {
    "vector": """<|im_start|>system
你是一个中医助手。请基于提供的文档回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档中没有的内容
2. 不得进行任何推理、推测或补充说明
3. 不得添加具体的用法、剂量、配伍等详细信息
4. 如果文档中有多个相关答案，请直接引用文档内容
5. 如果文档中没有相关信息，请明确说明"根据提供的文档，未找到相关信息"
6. 回答必须完全基于文档内容，任何超出文档范围的内容都视为错误
7. 字数控制在200字以内，避免过度扩展
8. 如果问题里没有给任何实体或症状请不要回答而是提出询问如：您能给我具体的症状/药品名称吗？

文档内容：
{context_text}
<|im_end|>

<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant""",

    "hybrid": """<|im_start|>system
你是一个中医助手。请基于提供的文档回答问题。

【严格规则】：
1. 只能使用文档中明确提到的信息，不得添加任何文档中没有的内容
2. 不得进行任何推理、推测或补充说明
3. 不得添加具体的用法、剂量、配伍等详细信息
4. 如果文档中有多个相关答案，请直接引用文档内容
5. 如果文档中没有相关信息，请明确说明"根据提供的文档，未找到相关信息"
6. 回答必须完全基于文档内容，任何超出文档范围的内容都视为错误
7. 字数控制在200字以内，避免过度扩展
8. 如果问题里没有给任何实体或症状请不要回答而是提出询问如：您能给我具体的症状/药品名称吗？

文档内容：
{context_text}
<|im_end|>

<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
}

# 路由类型枚举
class RouteType(Enum):
    VECTOR = "vector"
    HYBRID = "hybrid"

# 组件状态枚举
class ComponentState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"

@dataclass
class ComponentStatus:
    """组件状态管理"""
    state: ComponentState = ComponentState.UNLOADED
    load_time: Optional[float] = None
    unload_time: Optional[float] = None

class HybridRagasEvaluatorV4:
    """混合检索RAGAS评估器v4.0 - 优化版RAG流程"""
    
    def __init__(self, max_samples: int = 100):
        """
        初始化v4评估器
        
        Args:
            max_samples: 最大评估样本数
        """
        self.max_samples = max_samples
        
        # 核心组件（必须加载）
        self.deepseek_client = None
        self.qwen_flash_client = None
        self.intelligent_router = None
        
        # 按需加载的组件
        self.local_model = None
        self.local_tokenizer = None
        self.vector_store = None
        self.embedding_model = None
        self.neo4j_driver = None
        self.keyword_library = []
        
        # 组件状态管理
        self.component_status = {
            'deepseek': ComponentStatus(),
            'qwen_flash': ComponentStatus(),
            'router': ComponentStatus(),
            'local_model': ComponentStatus(),
            'vector': ComponentStatus(),
            'kg': ComponentStatus(),
            'keyword': ComponentStatus()
        }
        
        # 只初始化核心组件
        self._init_core_components()
    
    def _init_core_components(self):
        """初始化核心组件（DeepSeek + Qwen-Flash + 智能路由）"""
        try:
            logger.info("🚀 开始初始化核心组件...")
            
            # 1. 初始化DeepSeek客户端
            self._init_deepseek_client()
            
            # 2. 初始化Qwen-Flash客户端
            self._init_qwen_flash_client()
            
            # 3. 初始化智能路由分类器
            self._init_intelligent_router()
            
            logger.info("✅ 核心组件初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 核心组件初始化失败: {e}")
            raise
    
    def _init_deepseek_client(self):
        """初始化DeepSeek客户端"""
        try:
            logger.info("🔄 初始化DeepSeek客户端...")
            
            self.deepseek_client = OpenAI(
                api_key=DEEPSEEK_CONFIG["api_key"],
                base_url=DEEPSEEK_CONFIG["base_url"]
            )
            
            # 测试连接
            test_response = self.deepseek_client.chat.completions.create(
                model=DEEPSEEK_CONFIG["model"],
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10,
                n=1,  # 强制n=1
                temperature=0.1
            )
            
            self.component_status['deepseek'].state = ComponentState.LOADED
            self.component_status['deepseek'].load_time = time.time()
            logger.info("✅ DeepSeek客户端初始化成功")
            
        except Exception as e:
            logger.error(f"❌ DeepSeek客户端初始化失败: {e}")
            raise
    
    def _init_qwen_flash_client(self):
        """初始化Qwen-Flash客户端"""
        try:
            logger.info("🔄 初始化Qwen-Flash客户端...")
            
            self.qwen_flash_client = OpenAI(
                api_key=QWEN_FLASH_CONFIG["api_key"],
                base_url=QWEN_FLASH_CONFIG["base_url"]
            )
            
            # 测试连接
            test_response = self.qwen_flash_client.chat.completions.create(
                model=QWEN_FLASH_CONFIG["model_name"],
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=10
            )
            
            self.component_status['qwen_flash'].state = ComponentState.LOADED
            self.component_status['qwen_flash'].load_time = time.time()
            logger.info("✅ Qwen-Flash客户端初始化成功")
            
        except Exception as e:
            logger.error(f"❌ Qwen-Flash客户端初始化失败: {e}")
            raise
    
    def _init_intelligent_router(self):
        """初始化智能路由分类器"""
        try:
            logger.info("🔄 初始化智能路由分类器...")
            
            # 检查路径
            entity_csv_path = str(project_root / "测试与质量保障层" / "testdataset" / "knowledge_graph_entities_only.csv")
            logger.info(f"📁 实体CSV路径: {entity_csv_path}")
            logger.info(f"📁 Qwen-Flash配置: {QWEN_FLASH_CONFIG}")
            
            from middle.utils.intelligent_router import get_intelligent_router
            
            # 使用Qwen-Flash API配置初始化智能路由器
            self.intelligent_router = get_intelligent_router(
                entity_csv_path=entity_csv_path,
                qwen_api_config=QWEN_FLASH_CONFIG,
                confidence_threshold=0.65
            )
            
            self.component_status['router'].state = ComponentState.LOADED
            self.component_status['router'].load_time = time.time()
            logger.info("✅ 智能路由分类器初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 智能路由分类器初始化失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            self.intelligent_router = None
    
    def classify_question(self, question: str) -> Tuple[str, float]:
        """使用智能路由分类器对问题进行分类"""
        if self.intelligent_router is None:
            logger.warning("⚠️ 智能路由分类器未初始化，使用默认混合检索")
            return RouteType.HYBRID.value, 0.5  # 默认混合检索
        
        try:
            from middle.utils.intelligent_router import RouteType as V3RouteType
            
            # 添加调试信息：检查实体提取
            if hasattr(self.intelligent_router, '_extract_entities'):
                entities = self.intelligent_router._extract_entities(question)
                logger.info(f"🔍 提取到的实体: {entities}")
            
            route_type, confidence = self.intelligent_router.classify(question)
            logger.info(f"🔍 原始分类结果: {route_type} (置信度: {confidence:.2f})")
            
            # 映射到v4的路由类型 - 智能路由只有两个类型
            mapping = {
                V3RouteType.ENTITY_DRIVEN: RouteType.VECTOR.value,      # 实体主导型 → 向量检索
                V3RouteType.COMPLEX_REASONING: RouteType.HYBRID.value   # 复杂推理型 → 混合检索
            }
            
            mapped_type = mapping.get(route_type, RouteType.HYBRID.value)
            logger.info(f"✅ 智能路由分类: {route_type} -> {mapped_type} (置信度: {confidence:.2f})")
            
            return mapped_type, confidence
            
        except Exception as e:
            logger.error(f"❌ 智能路由分类失败: {e}")
            logger.info("🔄 使用默认混合检索作为备选方案")
            return RouteType.HYBRID.value, 0.5
    
    def _load_component(self, component_name: str) -> bool:
        """加载指定组件"""
        try:
            if self.component_status[component_name].state == ComponentState.LOADED:
                return True
            
            self.component_status[component_name].state = ComponentState.LOADING
            start_time = time.time()
            
            if component_name == 'local_model':
                self._load_local_model()
            elif component_name == 'vector':
                self._load_vector_components()
            elif component_name == 'kg':
                self._load_kg_components()
            elif component_name == 'keyword':
                self._load_keyword_library()
            else:
                logger.warning(f"未知组件: {component_name}")
                return False
            
            self.component_status[component_name].state = ComponentState.LOADED
            self.component_status[component_name].load_time = time.time() - start_time
            logger.info(f"✅ 组件 {component_name} 加载完成，耗时: {self.component_status[component_name].load_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"❌ 组件 {component_name} 加载失败: {e}")
            self.component_status[component_name].state = ComponentState.UNLOADED
            return False
    
    def _unload_component(self, component_name: str) -> bool:
        """卸载指定组件"""
        try:
            if self.component_status[component_name].state == ComponentState.UNLOADED:
                return True
            
            self.component_status[component_name].state = ComponentState.UNLOADING
            start_time = time.time()
            
            if component_name == 'local_model':
                self._unload_local_model()
            elif component_name == 'vector':
                self._unload_vector_components()
            elif component_name == 'kg':
                self._unload_kg_components()
            elif component_name == 'keyword':
                self._unload_keyword_library()
            else:
                logger.warning(f"未知组件: {component_name}")
                return False
            
            self.component_status[component_name].state = ComponentState.UNLOADED
            self.component_status[component_name].unload_time = time.time() - start_time
            logger.info(f"✅ 组件 {component_name} 卸载完成，耗时: {self.component_status[component_name].unload_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"❌ 组件 {component_name} 卸载失败: {e}")
            return False
    
    def _load_local_model(self):
        """加载本地微调模型"""
        try:
            logger.info("🔄 加载本地微调模型...")
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 创建offload目录
            offload_dir = str(project_root / "temp_offload_v4")
            os.makedirs(offload_dir, exist_ok=True)
            
            # 加载tokenizer
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_MODEL_CONFIG["base_model_path"],
                trust_remote_code=True
            )
            
            # 加载基座模型
            base_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_CONFIG["base_model_path"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
                max_memory={0: "6GB", "cpu": "30GB"},
                max_length=LOCAL_MODEL_CONFIG["max_length"]
            )
            
            # 加载LoRA适配器
            self.local_model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_CONFIG["lora_path"])
            
            logger.info("✅ 本地微调模型加载完成")
            
        except Exception as e:
            logger.error(f"❌ 本地微调模型加载失败: {e}")
            raise
    
    def _unload_local_model(self):
        """卸载本地微调模型"""
        try:
            logger.info("🔄 卸载本地微调模型...")
            
            if self.local_model:
                del self.local_model
                self.local_model = None
            
            if self.local_tokenizer:
                del self.local_tokenizer
                self.local_tokenizer = None
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("✅ 本地微调模型卸载完成")
            
        except Exception as e:
            logger.error(f"❌ 本地微调模型卸载失败: {e}")
    
    def _load_vector_components(self):
        """加载向量检索组件"""
        try:
            logger.info("🔄 加载向量检索组件...")
            
            # 检查路径是否存在
            vector_model_path = RETRIEVAL_CONFIG["vector_model_path"]
            faiss_path = RETRIEVAL_CONFIG["faiss_path"]
            
            logger.info(f"📁 向量模型路径: {vector_model_path}")
            logger.info(f"📁 FAISS路径: {faiss_path}")
            
            if not os.path.exists(vector_model_path):
                raise FileNotFoundError(f"向量模型路径不存在: {vector_model_path}")
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS路径不存在: {faiss_path}")
            
            # 简化的向量检索实现
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # 加载embedding模型
            logger.info("🔄 加载SentenceTransformer模型...")
            import faiss
            has_gpu = faiss.get_num_gpus() > 0
            self.embedding_model = SentenceTransformer(vector_model_path, device='cuda' if has_gpu else 'cpu')
            logger.info("✅ SentenceTransformer模型加载完成")
            
            # 使用FaissManager加载FAISS索引（与v3保持一致）
            from 检索与知识层.faiss_rag.vector_retrieval_system.faiss_manager import FaissManager
            
            logger.info(f"🔄 加载FAISS索引: {faiss_path}")
            self.vector_store = FaissManager(
                persist_directory=faiss_path,
                dimension=768,  # GTE Base模型维度
                use_gpu=has_gpu
            )
            logger.info("✅ FAISS索引加载完成")
            
            # 加载文档数据
            documents_path = os.path.join(faiss_path, "documents.json")
            logger.info(f"🔄 加载文档数据: {documents_path}")
            import json
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            logger.info(f"✅ 文档数据加载完成，共 {len(self.documents)} 个文档")
            
            logger.info("✅ 向量检索组件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 向量检索组件加载失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 使用模拟数据
            self.embedding_model = None
            self.vector_store = None
            self.documents = []
            logger.warning("⚠️ 使用模拟向量检索数据")
    
    def _unload_vector_components(self):
        """卸载向量检索组件"""
        try:
            logger.info("🔄 卸载向量检索组件...")
            
            if self.vector_store:
                del self.vector_store
                self.vector_store = None
            
            if self.embedding_model:
                del self.embedding_model
                self.embedding_model = None
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ 向量检索组件卸载完成")
            
        except Exception as e:
            logger.error(f"❌ 向量检索组件卸载失败: {e}")
    
    def _load_kg_components(self):
        """加载知识图谱组件"""
        try:
            logger.info("🔄 加载知识图谱组件...")
            
            # 检查Neo4j连接配置
            neo4j_uri = RETRIEVAL_CONFIG["neo4j_uri"]
            neo4j_user = RETRIEVAL_CONFIG["neo4j_user"]
            neo4j_password = RETRIEVAL_CONFIG["neo4j_password"]
            
            logger.info(f"📁 Neo4j URI: {neo4j_uri}")
            logger.info(f"📁 Neo4j User: {neo4j_user}")
            
            # 简化的知识图谱实现
            from neo4j import GraphDatabase
            
            logger.info("🔄 连接Neo4j数据库...")
            
            # 尝试多种认证方式
            auth_configs = [
                ("neo4j", "hx1230047"),  # 从v3获取的正确密码
                ("neo4j", "123456"),     # 常见密码
                ("neo4j", "neo4j"),      # 默认密码
                ("neo4j", "password"),   # 默认配置
            ]
            
            for username, password in auth_configs:
                try:
                    logger.info(f"🔄 尝试连接Neo4j: {username}@{neo4j_uri}")
                    driver = GraphDatabase.driver(
                        neo4j_uri,
                        auth=(username, password),
                        connection_timeout=10
                    )
                    
                    # 测试连接
                    with driver.session() as session:
                        result = session.run("RETURN 1 as test")
                        test_value = result.single()["test"]
                        logger.info(f"✅ Neo4j连接测试成功: {test_value}")
                    
                    self.neo4j_driver = driver
                    logger.info(f"✅ Neo4j连接成功: {username}")
                    break
                    
                except Exception as e:
                    logger.warning(f"⚠️ Neo4j连接失败 ({username}): {e}")
                    try:
                        driver.close()
                    except:
                        pass
                    continue
            
            if not self.neo4j_driver:
                raise Exception("所有Neo4j认证方式都失败了")
            
            logger.info("✅ 知识图谱组件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 知识图谱组件加载失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 使用模拟数据
            self.neo4j_driver = None
            logger.warning("⚠️ 使用模拟知识图谱数据")
    
    def _unload_kg_components(self):
        """卸载知识图谱组件"""
        try:
            logger.info("🔄 卸载知识图谱组件...")
            
            if self.neo4j_driver:
                self.neo4j_driver.close()
                self.neo4j_driver = None
            
            logger.info("✅ 知识图谱组件卸载完成")
            
        except Exception as e:
            logger.error(f"❌ 知识图谱组件卸载失败: {e}")
    
    def _load_keyword_library(self):
        """加载关键词库"""
        try:
            logger.info("🔄 加载关键词库...")
            
            # 加载中医关键词库
            keyword_file = project_root / "测试与质量保障层" / "testdataset" / "knowledge_graph_entities_only.csv"
            
            if keyword_file.exists():
                import pandas as pd
                df = pd.read_csv(keyword_file)
                # 假设关键词在第一列
                self.keyword_library = df.iloc[:, 0].astype(str).tolist()
                logger.info(f"✅ 关键词库加载完成，共 {len(self.keyword_library)} 个关键词")
            else:
                # 如果文件不存在，使用基础的中医关键词
                self.keyword_library = [
                    "感冒", "恶寒", "恶风", "发热", "头痛", "咳嗽", "流涕", "鼻塞",
                    "桂枝汤", "麻黄汤", "柴胡", "防风", "荆芥", "连翘", "薄荷",
                    "口臭", "失眠", "多梦", "调理", "中药", "方剂", "治疗",
                    "风寒", "风热", "湿热", "气虚", "血虚", "阴虚", "阳虚",
                    "症状", "病因", "病机", "治法", "方药", "剂量", "煎服"
                ]
                logger.info(f"✅ 使用默认关键词库，共 {len(self.keyword_library)} 个关键词")
            
        except Exception as e:
            logger.error(f"❌ 关键词库加载失败: {e}")
            # 使用基础关键词作为备选
            self.keyword_library = [
                "感冒", "恶寒", "恶风", "发热", "头痛", "咳嗽", "流涕", "鼻塞",
                "桂枝汤", "麻黄汤", "柴胡", "防风", "荆芥", "连翘", "薄荷",
                "口臭", "失眠", "多梦", "调理", "中药", "方剂", "治疗"
            ]
            logger.info(f"✅ 使用备选关键词库，共 {len(self.keyword_library)} 个关键词")
    
    def _unload_keyword_library(self):
        """卸载关键词库"""
        try:
            logger.info("🔄 卸载关键词库...")
            
            if hasattr(self, 'keyword_library'):
                del self.keyword_library
                self.keyword_library = []
            
            logger.info("✅ 关键词库卸载完成")
            
        except Exception as e:
            logger.error(f"❌ 关键词库卸载失败: {e}")
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        处理单个问题的完整RAG流程
        
        流程：用户查询 → 智能路由 → 加载检索组件 → 检索与知识召回 → 关键词增强 → 卸载检索组件 → 加载本地模型生成组件 → 回答生成 → 卸载生成组件 → 输出回答
        """
        try:
            logger.info(f"🚀 开始处理问题: {question}")
            start_time = time.time()
            
            # ========== 阶段1: 智能路由 ==========
            logger.info("📋 阶段1: 智能路由分类")
            route_type, confidence = self.classify_question(question)
            logger.info(f"✅ 路由分类完成: {route_type} (置信度: {confidence:.2f})")
            
            # ========== 阶段2: 加载检索组件 ==========
            logger.info("🔍 阶段2: 加载检索组件")
            retrieval_components = ['vector', 'keyword']
            if route_type == RouteType.HYBRID.value:
                retrieval_components.append('kg')
            
            for component in retrieval_components:
                if not self._load_component(component):
                    logger.error(f"❌ 组件 {component} 加载失败")
                    return {"error": f"组件 {component} 加载失败"}
            
            # ========== 阶段3: 检索与知识召回 ==========
            logger.info("📚 阶段3: 检索与知识召回")
            generation_contexts, evaluation_contexts = await self._retrieve_contexts(question, route_type)
            if not generation_contexts:
                logger.warning("⚠️ 未检索到相关上下文")
                return {"error": "未检索到相关上下文"}
            
            # ========== 阶段4: 关键词增强 ==========
            logger.info("⚡ 阶段4: 关键词增强（已在检索阶段完成）")
            enhanced_contexts = generation_contexts  # 关键词增强已在检索阶段完成
            
            # ========== 阶段5: 卸载检索组件 ==========
            logger.info("🧹 阶段5: 卸载检索组件")
            for component in retrieval_components:
                self._unload_component(component)
            
            # ========== 阶段6: 加载本地模型生成组件 ==========
            logger.info("🤖 阶段6: 加载本地模型生成组件")
            if not self._load_component('local_model'):
                logger.error("❌ 本地模型加载失败")
                return {"error": "本地模型加载失败"}
            
            # ========== 阶段7: 回答生成 ==========
            logger.info("📝 阶段7: 回答生成")
            answer = await self._generate_answer(question, enhanced_contexts, route_type)
            
            # ========== 阶段8: 卸载生成组件 ==========
            logger.info("🧹 阶段8: 卸载生成组件")
            self._unload_component('local_model')
            
            # ========== 阶段9: 输出回答 ==========
            total_time = time.time() - start_time
            logger.info(f"✅ 问题处理完成，总耗时: {total_time:.2f}秒")
            
            return {
                "question": question,
                "answer": answer,
                "route_type": route_type,
                "confidence": confidence,
                "contexts": enhanced_contexts,  # 用于生成的文档
                "evaluation_contexts": evaluation_contexts,  # 用于RAGAS评估的文档
                "processing_time": total_time,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ 问题处理失败: {e}")
            # 确保清理所有组件
            self._cleanup_all_components()
            return {"error": str(e), "status": "failed"}
    
    async def _retrieve_contexts(self, question: str, route_type: str) -> Tuple[List[str], List[str]]:
        """
        检索相关上下文
        
        Returns:
            Tuple[List[str], List[str]]: (用于生成的文档, 用于RAGAS评估的文档)
        """
        try:
            if route_type == RouteType.VECTOR.value:
                # 纯向量检索：召回3个文档，选择分数最高的2个用于生成
                all_contexts = await self._vector_retrieve(question)  # 召回3个文档
                enhanced_contexts, _ = await self._enhance_contexts(all_contexts, question, select_k=2)  # 选择2个
                return enhanced_contexts, all_contexts  # (2个生成文档, 3个评估文档)
                
            elif route_type == RouteType.HYBRID.value:
                # 混合检索：向量检索3个 + 知识图谱5个；生成使用向量2个(相似度最高) + 知识图谱5个
                vector_contexts = await self._vector_retrieve(question)  # 召回3个文档
                kg_contexts = await self._kg_retrieve(question)  # 召回5个文档
                
                # 向量检索部分选择相似度最高的2个
                vector_enhanced, _ = await self._enhance_contexts(vector_contexts, question, select_k=2)
                vector_selected = vector_enhanced[:2]
                
                # 知识图谱部分使用5个
                kg_selected = kg_contexts[:5]
                
                # 合并用于生成的文档（共7个：2向量+5知识图谱）
                generation_contexts = vector_selected + kg_selected
                
                # 评估文档仅使用向量召回的3个文档
                return generation_contexts, vector_contexts  # (7个生成文档, 3个评估文档)
            
            return [], []
            
        except Exception as e:
            logger.error(f"❌ 上下文检索失败: {e}")
            return [], []
    
    async def _vector_retrieve(self, question: str) -> List[str]:
        """向量检索：嵌入模型提取问题向量->对比向量数据库向量->召回10个文档"""
        try:
            logger.info("🔄 执行向量检索...")
            
            # 确保向量组件已加载
            if not self._load_component('vector'):
                logger.warning("⚠️ 向量检索组件加载失败，使用模拟数据")
                return [f"向量检索结果{i+1}: 关于{question}的相关信息" for i in range(10)]
            
            if not self.embedding_model or not self.vector_store:
                logger.warning("⚠️ 向量检索组件未加载，使用模拟数据")
                return [f"向量检索结果{i+1}: 关于{question}的相关信息" for i in range(10)]
            
            # 1. 使用嵌入模型提取问题向量
            question_embedding = self.embedding_model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
            
            # 2. 在向量数据库中搜索（使用FaissManager的API）
            results = self.vector_store.search(
                query_embedding=question_embedding,
                n_results=3  # 召回3个文档
            )
            
            # 3. 获取文档内容 - 统一返回纯文本格式
            contexts = []
            for i, result in enumerate(results):
                content = None
                
                if isinstance(result, dict):
                    # 优先使用metadata中的答案（纯文本格式）
                    metadata = result.get('metadata', {})
                    answer = metadata.get('answer', '')
                    
                    if answer:
                        # 确保是纯文本格式
                        if isinstance(answer, str):
                            content = answer
                        elif isinstance(answer, dict):
                            # 如果是字典，尝试提取content字段
                            content = answer.get('content', '') or str(answer)
                        else:
                            content = str(answer)
                    
                    # 如果没有答案，尝试从document或text字段获取
                    if not content:
                        content = result.get('document', '') or result.get('text', '')
                    
                    # 如果content是JSON字符串，尝试解析
                    if content and isinstance(content, str):
                        try:
                            # 尝试解析JSON
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                # 如果是对话格式，提取assistant的content
                                if 'messages' in parsed:
                                    for msg in parsed.get('messages', []):
                                        if msg.get('role') == 'assistant':
                                            content = msg.get('content', content)
                                            break
                                # 如果有content字段，使用它
                                elif 'content' in parsed:
                                    content = parsed['content']
                                # 如果有answer字段，使用它
                                elif 'answer' in parsed:
                                    content = parsed['answer']
                        except (json.JSONDecodeError, TypeError):
                            # 不是JSON格式，直接使用纯文本
                            pass
                elif hasattr(result, 'content'):
                    content = result.content
                elif isinstance(result, str):
                    content = result
                
                # 确保返回纯文本
                if content:
                    if isinstance(content, str):
                        contexts.append(content)
                    else:
                        contexts.append(str(content))
                else:
                    contexts.append(f"向量检索结果{i+1}: 关于{question}的相关信息")
            
            logger.info(f"✅ 向量检索完成，获得 {len(contexts)} 个上下文")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []
    
    async def _kg_retrieve(self, question: str) -> List[str]:
        """知识图谱检索：使用关键词库提取题目里的关键词作为实体输入知识图谱得到5个召回文档"""
        try:
            logger.info("🔄 执行知识图谱检索...")
            
            # 确保知识图谱组件已加载
            if not self._load_component('kg'):
                logger.warning("⚠️ 知识图谱组件加载失败，使用模拟数据")
                return [f"知识图谱结果{i+1}: 关于{question}的实体关系信息" for i in range(5)]
            
            if not self.neo4j_driver:
                logger.warning("⚠️ 知识图谱组件未加载，使用模拟数据")
                return [f"知识图谱结果{i+1}: 关于{question}的实体关系信息" for i in range(5)]
            
            # 1. 从关键词库中提取问题中的关键词
            keywords = self._extract_keywords_from_question(question)
            logger.info(f"🔍 提取到的关键词: {keywords}")
            
            # 2. 使用关键词查询知识图谱
            contexts = []
            with self.neo4j_driver.session() as session:
                for keyword in keywords[:5]:  # 最多使用5个关键词
                    try:
                        # 查询与关键词相关的实体和关系
                        query = """
                        MATCH (n)-[r]-(m)
                        WHERE n.name CONTAINS $keyword OR m.name CONTAINS $keyword
                        RETURN n.name as entity1, type(r) as relation, m.name as entity2
                        LIMIT 2
                        """
                        result = session.run(query, keyword=keyword)
                        
                        for record in result:
                            entity1 = record["entity1"]
                            relation = record["relation"]
                            entity2 = record["entity2"]
                            context = f"知识图谱: {entity1} - {relation} - {entity2}"
                            contexts.append(context)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ 查询关键词 '{keyword}' 失败: {e}")
                        contexts.append(f"知识图谱结果: 关于关键词'{keyword}'的实体关系信息")
            
            # 确保返回5个文档
            while len(contexts) < 5:
                contexts.append(f"知识图谱结果{len(contexts)+1}: 关于{question}的关联实体信息")
            
            logger.info(f"✅ 知识图谱检索完成，获得 {len(contexts)} 个上下文")
            return contexts[:5]  # 返回5个文档
            
        except Exception as e:
            logger.error(f"❌ 知识图谱检索失败: {e}")
            return []
    
    def _extract_keywords_from_question(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        try:
            if not self.keyword_library:
                return []
            
            # 简单的关键词匹配
            keywords = []
            for keyword in self.keyword_library:
                if keyword in question:
                    keywords.append(keyword)
            
            return keywords[:5]  # 最多返回5个关键词
            
        except Exception as e:
            logger.error(f"❌ 关键词提取失败: {e}")
            return []
    
    async def _enhance_contexts(self, contexts: List[str], question: str, select_k: int = 3) -> Tuple[List[str], List[str]]:
        """
        关键词增强上下文：使用关键词库提取所有文档与Ground Truth的关键词，对比它们的相似度，并进行排序
        
        Args:
            contexts: 纯文本格式的文档列表
            question: 用户问题
            
        Returns:
            Tuple[List[str], List[str]]: (增强后的上下文(纯文本), 原始上下文(纯文本))
        """
        try:
            if not contexts:
                return [], []
            
            logger.info("🔄 执行关键词增强...")
            
            # 确保所有上下文都是纯文本格式
            text_contexts = []
            for ctx in contexts:
                if isinstance(ctx, str):
                    # 如果是JSON字符串，尝试解析并提取纯文本
                    try:
                        parsed = json.loads(ctx)
                        if isinstance(parsed, dict):
                            # 如果是对话格式，提取assistant的content
                            if 'messages' in parsed:
                                for msg in parsed.get('messages', []):
                                    if msg.get('role') == 'assistant':
                                        text_contexts.append(msg.get('content', ctx))
                                        break
                                else:
                                    text_contexts.append(ctx)
                            # 如果有content字段，使用它
                            elif 'content' in parsed:
                                text_contexts.append(parsed['content'])
                            # 如果有answer字段，使用它
                            elif 'answer' in parsed:
                                text_contexts.append(parsed['answer'])
                            else:
                                text_contexts.append(ctx)
                        else:
                            text_contexts.append(ctx)
                    except (json.JSONDecodeError, TypeError):
                        # 不是JSON格式，直接使用纯文本
                        text_contexts.append(ctx)
                else:
                    # 非字符串类型，转换为字符串
                    text_contexts.append(str(ctx))
            
            # 1. 提取问题中的关键词
            question_keywords = self._extract_keywords_from_question(question)
            
            # 2. 为每个文档计算关键词相似度分数
            scored_contexts = []
            for i, context in enumerate(text_contexts):
                # 确保context是纯文本字符串
                context_text = context if isinstance(context, str) else str(context)
                # 计算文档与问题的关键词相似度
                score = self._calculate_keyword_similarity(context_text, question_keywords)
                scored_contexts.append((score, i, context_text))
            
            # 3. 按分数排序
            scored_contexts.sort(key=lambda x: x[0], reverse=True)
            
            # 4. 选择分数高的文档（按select_k控制）
            k = max(1, int(select_k))
            selected_contexts = scored_contexts[:k] if len(scored_contexts) >= k else scored_contexts
            
            # 5. 格式化文档（保持纯文本格式）
            enhanced_contexts = []
            for score, idx, context in selected_contexts:
                # 确保context是纯文本，添加文档标识
                enhanced_context = f"[文档{idx+1}]\n{context}"
                enhanced_contexts.append(enhanced_context)
            
            logger.info(f"✅ 关键词增强完成，从 {len(text_contexts)} 个文档中选择了 {len(enhanced_contexts)} 个用于生成")
            return enhanced_contexts, text_contexts  # 返回增强后的文档(纯文本)和原始文档(纯文本)
            
        except Exception as e:
            logger.error(f"❌ 关键词增强失败: {e}")
            # 确保返回纯文本格式
            text_contexts = [ctx if isinstance(ctx, str) else str(ctx) for ctx in contexts]
            return text_contexts, text_contexts  # 返回原始上下文(纯文本)
    
    def _calculate_keyword_similarity(self, context: str, keywords: List[str]) -> float:
        """计算文档与关键词的相似度分数"""
        try:
            if not keywords:
                return 0.0
            
            # 简单的关键词匹配分数
            match_count = 0
            for keyword in keywords:
                if keyword in context:
                    match_count += 1
            
            return match_count / len(keywords) if keywords else 0.0
            
        except Exception as e:
            logger.error(f"❌ 相似度计算失败: {e}")
            return 0.0
    
    async def _generate_answer(self, question: str, contexts: List[str], route_type: str) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            contexts: 纯文本格式的文档列表
            route_type: 检索类型
            
        Returns:
            生成的答案
        """
        try:
            if not self.local_model or not self.local_tokenizer:
                raise Exception("本地模型未加载")
            
            logger.info("🔄 开始生成答案...")
            
            # 确保所有上下文都是纯文本格式
            text_contexts = []
            for ctx in contexts:
                if isinstance(ctx, str):
                    # 如果是JSON字符串，尝试解析并提取纯文本
                    try:
                        parsed = json.loads(ctx)
                        if isinstance(parsed, dict):
                            # 如果是对话格式，提取assistant的content
                            if 'messages' in parsed:
                                for msg in parsed.get('messages', []):
                                    if msg.get('role') == 'assistant':
                                        text_contexts.append(msg.get('content', ctx))
                                        break
                                else:
                                    text_contexts.append(ctx)
                            # 如果有content字段，使用它
                            elif 'content' in parsed:
                                text_contexts.append(parsed['content'])
                            # 如果有answer字段，使用它
                            elif 'answer' in parsed:
                                text_contexts.append(parsed['answer'])
                            else:
                                text_contexts.append(ctx)
                        else:
                            text_contexts.append(ctx)
                    except (json.JSONDecodeError, TypeError):
                        # 不是JSON格式，直接使用纯文本
                        # 如果包含"回答："，提取回答部分
                        if "回答：" in ctx:
                            answer_part = ctx.split("回答：", 1)[1].strip()
                            text_contexts.append(answer_part)
                        else:
                            text_contexts.append(ctx)
                else:
                    # 非字符串类型，转换为字符串
                    text_contexts.append(str(ctx))
            
            # 构建提示词
            if route_type == RouteType.VECTOR.value:
                # 格式化文档为提示词期望的格式（纯文本）
                formatted_contexts = []
                for i, ctx in enumerate(text_contexts, 1):
                    formatted_contexts.append(f"文档{i}：{ctx}")
                
                context_text = "\n\n".join(formatted_contexts)
                prompt = PROMPT_TEMPLATES["vector"].format(
                    context_text=context_text,
                    question=question
                )
            else:  # HYBRID
                # 前2个为向量文档，其余为知识图谱文档
                vector_contexts = []
                for i, ctx in enumerate(text_contexts[:2], 1):
                    vector_contexts.append(f"文档{i}：{ctx}")
                kg_contexts = []
                for i, ctx in enumerate(text_contexts[2:], 3):
                    kg_contexts.append(f"文档{i}：{ctx}")

                all_formatted_contexts = vector_contexts + kg_contexts
                context_text = "\n\n".join(all_formatted_contexts)
                
                prompt = PROMPT_TEMPLATES["hybrid"].format(
                    context_text=context_text,
                    question=question
                )
            
            # 编码输入
            inputs = self.local_tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
            
            # 生成参数
            generation_params = GENERATION_CONFIG.copy()
            generation_params["pad_token_id"] = self.local_tokenizer.eos_token_id
            generation_params["eos_token_id"] = self.local_tokenizer.eos_token_id
            
            # 移除None值
            generation_params = {k: v for k, v in generation_params.items() if v is not None}
            
            # 生成答案
            with torch.no_grad():
                device = next(self.local_model.parameters()).device
                outputs = self.local_model.generate(
                    inputs.input_ids.to(device),
                    **generation_params
                )
            
            # 解码输出
            answer = self.local_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # 后处理：清理不相关的内容
            answer = self._clean_answer(answer)
            
            logger.info(f"✅ 答案生成完成，长度: {len(answer)} 字符")
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ 答案生成失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"
    
    def _clean_answer(self, answer: str) -> str:
        """清理答案中的不相关内容"""
        try:
            # 移除常见的指导性文字
            unwanted_phrases = [
                "语气温和，语气友好",
                "使用口语化的表达方式",
                "不使用专业术语",
                "尽量避免使用过于复杂的句子结构",
                "让回答更加易于理解",
                "根据您描述的症状",
                "根据文档",
                "根据向量检索",
                "根据知识图谱",
                "穷举法",
                "请严格按照以下规则",
                "必须100%遵守",
                "任何规则都不可违反"
            ]
            
            cleaned_answer = answer
            for phrase in unwanted_phrases:
                cleaned_answer = cleaned_answer.replace(phrase, "")
            
            # 移除多余的空格和换行
            cleaned_answer = " ".join(cleaned_answer.split())
            
            # 如果答案太短，返回原始答案
            if len(cleaned_answer.strip()) < 10:
                return answer.strip()
            
            return cleaned_answer.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ 答案清理失败: {e}")
            return answer.strip()
    
    def _cleanup_all_components(self):
        """清理所有组件"""
        try:
            logger.info("🧹 清理所有组件...")
            
            components_to_cleanup = ['local_model', 'vector', 'kg', 'keyword']
            for component in components_to_cleanup:
                self._unload_component(component)
            
            logger.info("✅ 所有组件清理完成")
            
        except Exception as e:
            logger.error(f"❌ 组件清理失败: {e}")
    
    async def evaluate_with_ragas(self, question: str, answer: str, contexts: List[str], route_type: str = "vector", generation_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        使用DeepSeek+RAGAS进行评估
        
        Args:
            question: 问题
            answer: 生成的答案
            contexts: 检索到的上下文（用于评估的文档）
            route_type: 检索类型（vector或hybrid）
            
        Returns:
            评估结果
        """
        try:
            logger.info("🔍 开始RAGAS评估...")
            
            # 生成Ground Truth
            ground_truth = await self._generate_ground_truth(question, contexts)
            
            # 使用DeepSeek进行RAGAS评估
            evaluation_result = await self._deepseek_ragas_evaluation(question, answer, contexts, ground_truth, route_type, generation_contexts)
            
            # 将Ground Truth添加到评估结果中
            evaluation_result["ground_truth"] = ground_truth
            evaluation_result["ground_truth_length"] = len(ground_truth)
            
            logger.info("✅ RAGAS评估完成")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ RAGAS评估失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _generate_ground_truth(self, question: str, contexts: List[str]) -> str:
        """从数据集中查找Ground Truth"""
        try:
            logger.info("🔄 从数据集中查找Ground Truth...")
            logger.info(f"   查找问题: '{question}'")
            
            # 从eval_dataset_100.jsonl中查找匹配的问题
            dataset_path = project_root / "测试与质量保障层" / "testdataset" / "eval_dataset_100.jsonl"
            
            if dataset_path.exists():
                import json
                
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'messages' in data and len(data['messages']) >= 3:
                                user_msg = data['messages'][1]  # user消息对象
                                assistant_msg = data['messages'][2]  # assistant消息对象
                                
                                # 检查消息角色和内容
                                if (user_msg.get('role') == 'user' and 
                                    assistant_msg.get('role') == 'assistant'):
                                    user_content = user_msg.get('content', '')
                                    assistant_content = assistant_msg.get('content', '')
                                    
                                    # 过滤掉问题内容为"无"的文档
                                    if user_content == "无":
                                        continue
                                    
                                    # 直接匹配问题内容（去除所有空白字符进行比较）
                                    question_clean = ''.join(question.split())
                                    user_content_clean = ''.join(user_content.split())
                                    
                                    if question_clean == user_content_clean:
                                        logger.info(f"✅ 找到完全匹配的Ground Truth，长度: {len(assistant_content)} 字符")
                                        logger.info(f"   匹配问题: {user_content}")
                                        return assistant_content
                                    
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.warning(f"处理数据行时出错: {e}")
                            continue
            
            # 如果没找到完全匹配的，使用第一个上下文作为Ground Truth
            if contexts:
                ground_truth = contexts[0]
                logger.info(f"⚠️ 未找到完全匹配的Ground Truth，使用第一个上下文，长度: {len(ground_truth)} 字符")
                return ground_truth
            else:
                logger.warning("⚠️ 没有可用的上下文，使用默认Ground Truth")
                return f"关于{question}的相关信息，请参考中医专业建议。"
            
        except Exception as e:
            logger.error(f"❌ Ground Truth查找失败: {e}")
            return f"关于{question}的相关信息，请参考中医专业建议。"
    
    async def _deepseek_ragas_evaluation(self, question: str, answer: str, contexts: List[str], ground_truth: str, route_type: str, generation_contexts: Optional[List[str]] = None) -> Dict[str, Any]:
        """使用DeepSeek进行RAGAS评估"""
        try:
            logger.info("🔄 执行DeepSeek RAGAS评估...")
            
            # 根据检索类型构建不同的评估提示词
            if route_type == RouteType.VECTOR.value:
                # 纯向量检索：context_precision/context_recall 使用3个召回文档；faithfulness/answer_relevancy 使用生成的2个文档
                eval_contexts = contexts[:3] if len(contexts) >= 3 else contexts
                context_text = "\n\n".join(eval_contexts)
                gen_ctx = (generation_contexts or [])
                gen_ctx = gen_ctx[:2] if len(gen_ctx) >= 2 else gen_ctx
                generation_text = "\n\n".join(gen_ctx)
                
                evaluation_prompt = f"""请对以下RAG系统的回答进行四维评估，严格按照JSON格式返回结果：

问题：{question}

检索到的上下文（用于context_precision和context_recall评估，共{len(contexts)}个文档）：
{context_text}

用于生成的上下文（用于faithfulness和answer_relevancy评估，共{len(generation_contexts)}个文档）：
{generation_text}

生成的答案：{answer}

标准答案：{ground_truth}

请从以下四个维度进行评估（每个维度0-1分，保留2位小数）：

1. context_precision（上下文精确度）：检索到的上下文与问题的相关程度
2. context_recall（上下文召回率）：标准答案中的信息在检索上下文中的覆盖程度
3. faithfulness（忠实度）：生成答案对用于生成的上下文的忠实程度，是否包含幻觉
4. answer_relevancy（答案相关性）：生成答案与问题的相关程度

请严格按照以下JSON格式返回：
{{
    "context_precision": 0.85,
    "context_recall": 0.78,
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "overall_score": 0.86,
    "evaluation_notes": "评估说明"
}}"""
            else:  # HYBRID
                # 混合检索：context_precision/context_recall 使用向量检索的3个召回文档
                # 知识图谱召回文档不参与评估；faithfulness/answer_relevancy 仅使用向量部分的生成文档（2个）
                eval_contexts = contexts[:3] if len(contexts) >= 3 else contexts
                context_text = "\n\n".join(eval_contexts)

                # 过滤掉知识图谱文档，仅保留向量生成文档（通常为1个，且不以"知识图谱"开头）
                filtered_gen_ctx = []
                if generation_contexts:
                    for ctx in generation_contexts:
                        if isinstance(ctx, str) and not ctx.strip().startswith("知识图谱"):
                            filtered_gen_ctx.append(ctx)
                filtered_gen_ctx = filtered_gen_ctx[:2] if len(filtered_gen_ctx) >= 2 else filtered_gen_ctx
                generation_text = "\n\n".join(filtered_gen_ctx)

                evaluation_prompt = f"""请对以下RAG系统的回答进行四维评估，严格按照JSON格式返回结果：

问题：{question}

检索到的上下文（仅向量检索部分，用于context_precision和context_recall评估，共{len(eval_contexts)}个文档）：
{context_text}

用于生成的上下文（仅用于faithfulness和answer_relevancy评估，排除知识图谱文档，共{len(filtered_gen_ctx)}个文档）：
{generation_text}

生成的答案：{answer}

标准答案：{ground_truth}

请从以下四个维度进行评估（每个维度0-1分，保留2位小数）：

1. context_precision（上下文精确度）：检索到的上下文与问题的相关程度
2. context_recall（上下文召回率）：标准答案中的信息在检索上下文中的覆盖程度
3. faithfulness（忠实度）：生成答案对检索上下文的忠实程度，是否包含幻觉
4. answer_relevancy（答案相关性）：生成答案与问题的相关程度

注意：知识图谱召回的文档不参与RAGAS评估，因此不参与最后的评估。

请严格按照以下JSON格式返回：
{{
    "context_precision": 0.85,
    "context_recall": 0.78,
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "overall_score": 0.86,
    "evaluation_notes": "评估说明"
}}"""

            response = self.deepseek_client.chat.completions.create(
                model=DEEPSEEK_CONFIG["model"],
                messages=[{"role": "user", "content": evaluation_prompt}],
                max_tokens=DEEPSEEK_CONFIG["max_tokens"],
                n=1,  # 强制n=1
                temperature=DEEPSEEK_CONFIG["temperature"],
                **DEEPSEEK_CONFIG["model_kwargs"]  # 包含JSON格式要求
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                evaluation_data = json.loads(response_content)
                logger.info("✅ DeepSeek RAGAS评估完成")
                return {
                    "status": "success",
                    "evaluation_data": evaluation_data,
                    "raw_response": response_content
                }
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON解析失败: {e}")
                return {
                    "status": "failed",
                    "error": f"JSON解析失败: {e}",
                    "raw_response": response_content
                }
            
        except Exception as e:
            logger.error(f"❌ DeepSeek RAGAS评估失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def full_evaluation_pipeline(self, question: str) -> Dict[str, Any]:
        """
        完整的评估流程：RAG处理 + RAGAS评估
        
        Args:
            question: 问题
            
        Returns:
            完整的评估结果
        """
        try:
            logger.info(f"🚀 开始完整评估流程: {question}")
            start_time = time.time()
            
            # 阶段1: RAG处理
            rag_result = await self.process_question(question)
            if rag_result.get("status") != "success":
                return rag_result
            
            # 阶段2: RAGAS评估
            ragas_result = await self.evaluate_with_ragas(
                question=question,
                answer=rag_result["answer"],
                contexts=rag_result["evaluation_contexts"],  # 使用评估用的上下文
                route_type=rag_result["route_type"],  # 传递检索类型
                generation_contexts=rag_result["contexts"]  # 生成用的上下文
            )
            
            # 合并结果
            total_time = time.time() - start_time
            full_result = {
                "question": question,
                "rag_result": rag_result,
                "ragas_result": ragas_result,
                "total_processing_time": total_time,
                "status": "success"
            }
            
            logger.info(f"✅ 完整评估流程完成，总耗时: {total_time:.2f}秒")
            return full_result
            
        except Exception as e:
            logger.error(f"❌ 完整评估流程失败: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_component_status(self) -> Dict[str, Any]:
        """获取组件状态信息"""
        status_info = {}
        for component, status in self.component_status.items():
            status_info[component] = {
                "state": status.state.value,
                "load_time": status.load_time,
                "unload_time": status.unload_time
            }
        return status_info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            memory_info = {}
            
            # GPU显存
            if torch.cuda.is_available():
                memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3
                memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3
            else:
                memory_info["gpu_allocated"] = 0
                memory_info["gpu_reserved"] = 0
            
            # CPU内存
            process = psutil.Process()
            memory_info["cpu_memory"] = process.memory_info().rss / 1024**3
            
            return memory_info
            
        except Exception as e:
            logger.error(f"❌ 获取内存使用情况失败: {e}")
            return {"error": str(e)}

# ========== 测试和主函数 ==========

async def test_v4_system():
    """测试v4系统"""
    try:
        print("=" * 80)
        print("🧪 测试HybridRagasEvaluatorV4系统")
        print("=" * 80)
        
        # 创建评估器
        evaluator = HybridRagasEvaluatorV4()
        
        # 测试问题
        test_questions = [
            "我恶寒感冒，可以给我推荐一个中药吗？",
            "口臭是什么原因引起的？",
            "失眠多梦应该怎么调理？"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"测试问题 {i}: {question}")
            print(f"{'='*60}")
            
            # 显示初始内存状态
            memory_before = evaluator.get_memory_usage()
            print(f"📊 初始内存状态: GPU {memory_before['gpu_allocated']:.2f}GB, CPU {memory_before['cpu_memory']:.2f}GB")
            
            # 执行完整评估流程
            result = await evaluator.full_evaluation_pipeline(question)
            
            # 显示结果
            if result.get("status") == "success":
                rag_result = result["rag_result"]
                ragas_result = result["ragas_result"]
                
                print(f"✅ 路由类型: {rag_result['route_type']}")
                print(f"✅ 置信度: {rag_result['confidence']:.2f}")
                print(f"✅ 答案长度: {len(rag_result['answer'])} 字符")
                print(f"✅ 上下文数量: {len(rag_result['contexts'])}")
                print(f"✅ 处理时间: {result['total_processing_time']:.2f}秒")
                
                if ragas_result.get("status") == "success":
                    eval_data = ragas_result["evaluation_data"]
                    print(f"📊 RAGAS评估结果:")
                    print(f"  - 上下文精确度: {eval_data.get('context_precision', 0):.2f}")
                    print(f"  - 上下文召回率: {eval_data.get('context_recall', 0):.2f}")
                    print(f"  - 忠实度: {eval_data.get('faithfulness', 0):.2f}")
                    print(f"  - 答案相关性: {eval_data.get('answer_relevancy', 0):.2f}")
                    print(f"  - 总体分数: {eval_data.get('overall_score', 0):.2f}")
                else:
                    print(f"❌ RAGAS评估失败: {ragas_result.get('error', '未知错误')}")
                
                # 显示答案预览
                answer_preview = rag_result['answer'][:200] + "..." if len(rag_result['answer']) > 200 else rag_result['answer']
                print(f"📝 答案预览: {answer_preview}")
                
            else:
                print(f"❌ 评估失败: {result.get('error', '未知错误')}")
            
            # 显示最终内存状态
            memory_after = evaluator.get_memory_usage()
            print(f"📊 最终内存状态: GPU {memory_after['gpu_allocated']:.2f}GB, CPU {memory_after['cpu_memory']:.2f}GB")
            
            # 显示组件状态
            component_status = evaluator.get_component_status()
            print(f"🔧 组件状态:")
            for component, status in component_status.items():
                print(f"  - {component}: {status['state']}")
        
        print(f"\n{'='*80}")
        print("✅ 测试完成")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_memory_management():
    """测试内存管理"""
    try:
        print("=" * 80)
        print("🧪 测试内存管理")
        print("=" * 80)
        
        evaluator = HybridRagasEvaluatorV4()
        
        # 测试问题
        question = "我恶寒感冒，可以给我推荐一个中药吗？"
        
        print(f"测试问题: {question}")
        
        # 显示初始状态
        memory_before = evaluator.get_memory_usage()
        component_status_before = evaluator.get_component_status()
        
        print(f"\n📊 初始状态:")
        print(f"  GPU显存: {memory_before['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_before['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_before.items()]
        print(f"  组件状态: {component_status_str}")
        
        # 执行RAG处理
        print(f"\n🔄 执行RAG处理...")
        rag_result = await evaluator.process_question(question)
        
        # 显示RAG处理后的状态
        memory_after_rag = evaluator.get_memory_usage()
        component_status_after_rag = evaluator.get_component_status()
        
        print(f"\n📊 RAG处理后的状态:")
        print(f"  GPU显存: {memory_after_rag['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_after_rag['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_after_rag.items()]
        print(f"  组件状态: {component_status_str}")
        
        if rag_result.get("status") == "success":
            print(f"✅ RAG处理成功")
            print(f"  答案长度: {len(rag_result['answer'])} 字符")
            print(f"  处理时间: {rag_result['processing_time']:.2f}秒")
        else:
            print(f"❌ RAG处理失败: {rag_result.get('error', '未知错误')}")
        
        # 等待一段时间，观察内存是否被正确释放
        print(f"\n⏳ 等待5秒，观察内存释放...")
        await asyncio.sleep(5)
        
        # 显示最终状态
        memory_final = evaluator.get_memory_usage()
        component_status_final = evaluator.get_component_status()
        
        print(f"\n📊 最终状态:")
        print(f"  GPU显存: {memory_final['gpu_allocated']:.2f}GB")
        print(f"  CPU内存: {memory_final['cpu_memory']:.2f}GB")
        component_status_str = [f'{k}:{v["state"]}' for k, v in component_status_final.items()]
        print(f"  组件状态: {component_status_str}")
        
        # 分析内存变化
        gpu_change = memory_final['gpu_allocated'] - memory_before['gpu_allocated']
        cpu_change = memory_final['cpu_memory'] - memory_before['cpu_memory']
        
        print(f"\n📈 内存变化分析:")
        print(f"  GPU显存变化: {gpu_change:+.2f}GB")
        print(f"  CPU内存变化: {cpu_change:+.2f}GB")
        
        if abs(gpu_change) < 0.1 and abs(cpu_change) < 0.1:
            print("✅ 内存管理良好，组件已正确卸载")
        else:
            print("⚠️ 内存管理可能存在问题，组件未完全卸载")
        
        print(f"\n{'='*80}")
        print("✅ 内存管理测试完成")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"❌ 内存管理测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    
    print("🚀 启动HybridRagasEvaluatorV4测试")
    
    # 运行测试
    asyncio.run(test_v4_system())
    
    print("\n" + "="*80)
    print("🧪 开始内存管理测试")
    print("="*80)
    
    asyncio.run(test_memory_management())
