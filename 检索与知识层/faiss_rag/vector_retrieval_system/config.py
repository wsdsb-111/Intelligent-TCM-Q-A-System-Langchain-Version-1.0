"""
向量检索系统配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据集路径配置
DATASET_CONFIG = {
    "csv_path": PROJECT_ROOT / "dataset" / "top_500k_tcm_data_20250918_164254.csv",
    "json_path": PROJECT_ROOT / "dataset" / "top_500k_tcm_data_20250918_164300.json",
    "arrow_path": PROJECT_ROOT / "dataset" / "sylvan_l___traditional-chinese-medicine-dataset-sft" / "b3a4fa258dbe5d13"
}

# 模型配置 - 切换为 GTE Base 模型（768维）
MODEL_CONFIG = {
    # 使用本地 GTE 文本向量-中文-通用领域-base 模型
    "embedding_model": r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base",
    "model_cache_dir": None,  # 已使用本地模型，不需要缓存
    # RTX 4060 8GB 优化配置
    "max_length": 2048,   # 增加到2048，避免文档截断
    "batch_size": 16,    # 4060 8GB 减少批次以适应768维模型
    "device": "cuda",    # 使用GPU加速
    "embedding_dimension": 768  # GTE-base 维度
}

# Chroma数据库配置
CHROMA_CONFIG = {
    # 强制使用仓库内统一路径，避免 OS 特定名导致混淆
    "persist_directory": PROJECT_ROOT / "向量数据库" / "chroma",
    "collection_name": "tcm_qa_collection",
    "distance_metric": "cosine"
}


def check_persist_directory(persist_path):
    """简单检查持久化目录的存在性和可读性，返回 (ok: bool, message: str)"""
    p = Path(persist_path)
    if not p.exists():
        return False, f"persist_directory 不存在: {p}"

    sqlite_path = p / "chroma.sqlite3"
    simple_store_dir = p / "simple_store"
    has_sqlite = sqlite_path.exists()
    has_simple = simple_store_dir.exists()

    if has_sqlite or has_simple:
        return True, f"persist_directory 存在且包含数据库文件: sqlite={has_sqlite}, simple_store={has_simple}"

    try:
        children = list(p.iterdir())
        if not children:
            return False, f"persist_directory 存在但为空: {p}"
        else:
            return True, f"persist_directory 存在但未检测到标准DB文件，子项数={len(children)}"
    except Exception as e:
        return False, f"检查 persist_directory 时出错: {e}"

# 文本处理配置
TEXT_CONFIG = {
    # 更大的 chunk_size 用于减少向量数量同时保留上下文
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "min_chunk_size": 120
}

# HNSW向量索引配置（RTX 5090优化参数）
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    # 通过更大的 M 和 construction_ef 提升索引质量（会占用更多内存/构建时间）
    "hnsw:M": 256,
    "hnsw:construction_ef": 800,
    # search_ef 用于查询阶段，较大值提高召回
    "hnsw:search_ef": 400
}

# 检索配置
RETRIEVAL_CONFIG = {
    "top_k": 50,             # 扩大候选集以便后续精排
    "score_threshold": 0.0,  # 先不依赖 Chroma 的分数，在本地重排时应用阈值
    "rerank": True,
    "final_top_k": 10,       # 精排后返回的结果数量
    "final_score_threshold": 0.25  # 精排的余弦相似度阈值（可调）
}
