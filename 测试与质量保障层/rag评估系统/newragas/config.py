"""
新RAGAS评估系统配置文件 - 使用官方ragas库
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 模型路径 - 用于RAG系统生成答案
MODEL_PATH = PROJECT_ROOT / "Model Layer" / "model" / "qwen" / "Qwen3-1.7B" / "Qwen" / "Qwen3-1___7B"

# 数据集路径
DATASET_PATH = PROJECT_ROOT / "测试与质量保障层" / "testdataset" / "eval_dataset_100.jsonl"

# Ground Truth标注数据路径
GROUND_TRUTH_PATH = PROJECT_ROOT / "测试与质量保障层" / "testdataset" / "nexx.json"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "results"

# RAGAS评估配置
RAGAS_CONFIG = {
    # 使用本地模型进行评估
    "llm": "qwen3-1.7b",  # 将映射到本地模型
    "embeddings": "local",  # 使用本地embedding模型
    
    # 评估指标
    "metrics": [
        "context_precision",
        "context_recall", 
        "faithfulness",
        "answer_relevancy"
    ],
    
    # 批处理大小
    "batch_size": 1,
    
    # 超时设置
    "timeout": 30,
}

# 本地模型配置（用于RAG系统生成答案）
LOCAL_MODEL_CONFIG = {
    "base_model_path": r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\qwen\Qwen3-1.7B\Qwen\Qwen3-1___7B",  # 实际模型路径（注意三个下划线）
    "lora_path": r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\checkpoint-7983",
    "device": "cuda",
    "max_length": 4096,  # 增加到4096，确保完整上下文不被截断
    "max_new_tokens": 1024,  # 增加到1024 tokens，确保答案完整详细（约500-800字）
    "temperature": 0.1,  # 低温度，提高确定性，减少幻觉
    "top_p": 0.3,        # 低top_p，减少随机性
    "do_sample": True,  # 使用采样，增加多样性
    "repetition_penalty": 1.2,  # 增加重复惩罚，避免重复模式
    "max_time": 30,
    "merge_lora": False,  # 不合并LoRA以优化性能
}

# 评估参数
EVALUATION_CONFIG = {
    "max_samples": 10,  # 先测试10个样本
    "batch_size": 1,
    "timeout": 30,
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "newragas_evaluation.log"
}

# 检索系统配置
RETRIEVAL_CONFIG = {
    "vector_model_path": r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base",
    "faiss_path": str(PROJECT_ROOT / "检索与知识层" / "faiss_rag" / "向量数据库_768维"),
    "vector_dimension": 768,  # 新增：指定向量维度
    "top_k": 3,  # 检索文档数量为3个
    "timeout": 30,
    "score_threshold": 0.2,  # 添加相似度阈值，过滤低质量结果
    "weights": {"vector": 0.5, "graph": 0.5}  # 平衡权重设置
}

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
