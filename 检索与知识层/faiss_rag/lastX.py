#!/usr/bin/env python3
"""
lastX_full_run.py

用途：
 - test 模式：只写入前 100 条，验证流程是否可靠（默认）
 - full 模式：写入全部数据（小心，可能需要较长时间）

特点：
 - 只走手动 embeddings（不绑定 embedding_function）
 - 每批写入后即时验证 sample embeddings
 - 写入失败自动重试，连续失败则中止
 - 优化：移除已弃用 persist()，HNSW 参数修正
"""

import argparse
import logging
import time
import shutil
from pathlib import Path
import gc
import numpy as np
import torch
import json
import signal
import sys
import os
import sqlite3  # 新增：用于索引重建查询

import chromadb
from chromadb.config import Settings

# 确保根据你的项目结构调整 sys.path（若需要）
import sys

sys.path.append(str(Path(__file__).parent))

from vector_retrieval_system.data_loader import DataLoader
from vector_retrieval_system.embedding_service import EmbeddingService
from vector_retrieval_system.config import MODEL_CONFIG, CHROMA_CONFIG

# ------------- 配置 -------------
# 使用 config 中的 persist_directory，保证所有组件使用相同的持久化路径
PERSIST_DIR = Path(str(CHROMA_CONFIG.get("persist_directory")))
COLLECTION_NAME = CHROMA_CONFIG.get("collection_name", "tcm_qa_collection")

# 默认参数（可通过 args 覆盖）- RTX 5090优化
DEFAULT_CHUNK_SIZE = 400  # 每批写入多少条（RTX 5090支持更大批次）
DEFAULT_ENCODE_BATCH = 128  # encode 时传给 model.encode 的 batch_size（RTX 5090优化）
TEST_LIMIT = 5000  # test 模式写入条数（RTX 5090支持更多测试数据）
RETRY_TIMES = 3  # 每批写入失败后重试次数
SAMPLE_VERIFY = 15  # 每批从前 SAMPLE_VERIFY 个 ids 验证（增加验证样本）
SLEEP_AFTER_ADD = 0.2  # add 后睡眠，给 I/O 留时间（秒）（RTX 5090更快）

# HNSW 优化参数（RTX 5090优化配置）
HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 128,  # 节点连接数（RTX 5090支持更大参数）
    "hnsw:construction_ef": 400,  # 构建时候选数（提升索引质量）
    "hnsw:search_ef": 200  # 搜索时候选数（提升搜索精度）
}

# 断点续传相关
CHECKPOINT_FILE = "lastX_checkpoint.json"  # 检查点文件
# ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("lastX_full_run")

# 版本日志
logger.info(f"ChromaDB 版本: {chromadb.__version__}")

# 全局变量用于信号处理
pause_requested = False


def signal_handler(signum, frame):
    """处理Ctrl+C信号，请求暂停"""
    global pause_requested
    pause_requested = True
    logger.info("\n🛑 收到暂停信号，将在当前批次完成后安全暂停...")


def save_checkpoint(mode, chunk_size, encode_batch, processed_count, total_chunks, failed_batches, start_time):
    """保存检查点"""
    checkpoint_data = {
        "mode": mode,
        "chunk_size": chunk_size,
        "encode_batch": encode_batch,
        "processed_count": processed_count,
        "total_chunks": total_chunks,
        "failed_batches": failed_batches,
        "start_time": start_time,
        "timestamp": time.time(),
        "checkpoint_file": CHECKPOINT_FILE
    }

    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 检查点已保存: {CHECKPOINT_FILE}")
        return True
    except Exception as e:
        logger.error(f"保存检查点失败: {e}")
        return False


def load_checkpoint():
    """加载检查点"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        logger.info(f"📂 找到检查点: {CHECKPOINT_FILE}")
        logger.info(f"   模式: {checkpoint_data['mode']}")
        logger.info(f"   已处理: {checkpoint_data['processed_count']}/{checkpoint_data['total_chunks']}")
        logger.info(f"   失败批次: {checkpoint_data['failed_batches']}")
        return checkpoint_data
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return None


def cleanup_checkpoint():
    """清理检查点文件"""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info(f"🗑️ 检查点文件已清理: {CHECKPOINT_FILE}")
    except Exception as e:
        logger.warning(f"清理检查点文件失败: {e}")


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def encode_texts(texts, model, device="cuda", encode_batch=DEFAULT_ENCODE_BATCH):
    """调用 embedding model，返回 numpy.ndarray (N, dim)"""
    try:
        # 隐藏 tqdm 输出
        import os
        os.environ['TQDM_DISABLE'] = '1'
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
            embs = model.encode(
                texts,
                batch_size=encode_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                device=device
            )
        if isinstance(embs, list):
            embs = np.array(embs)
        return embs
    finally:
        if 'TQDM_DISABLE' in globals() or 'TQDM_DISABLE' in locals():
            try:
                del os.environ['TQDM_DISABLE']
            except Exception:
                pass


def batch_add_and_verify(collection, ids, docs, embs, retry_times=RETRY_TIMES, sample_verify=SAMPLE_VERIFY):
    """
    将单个批次添加到 collection，并对前 sample_verify 个 id 做即时验证。
    优化：移除已弃用 persist()，依赖自动持久化。
    返回 True/False 表示该批次是否确认写入 embeddings。
    """
    embs_list = embs.tolist()
    attempt = 0
    while attempt <= retry_times:
        try:
            collection.add(ids=ids, documents=docs, embeddings=embs_list)
        except Exception as e:
            logger.error(f"collection.add 出错（尝试 {attempt + 1}/{retry_times + 1}）: {e}")
            attempt += 1
            time.sleep(1.0)
            continue

        # 等待短暂时间，给磁盘/后台写入留时间
        time.sleep(SLEEP_AFTER_ADD)
        
        # 额外等待时间，确保RTX 5090高速写入完成
        time.sleep(0.5)

        # 即时验证 sample
        sample_ids = ids[:min(sample_verify, len(ids))]
        try:
            res = collection.get(ids=sample_ids, include=["embeddings"])
            emb_list = res.get("embeddings")
            
            # 详细调试信息
            logger.info(f"🔍 验证调试: 请求IDs数量={len(sample_ids)}, 返回embeddings数量={0 if emb_list is None else len(emb_list)}")
            logger.info(f"🔍 详细调试: emb_list is None = {emb_list is None}")
            if emb_list is not None:
                logger.info(f"🔍 详细调试: len(emb_list) = {len(emb_list)}, len(sample_ids) = {len(sample_ids)}")
                logger.info(f"🔍 详细调试: len(emb_list) == len(sample_ids) = {len(emb_list) == len(sample_ids)}")
            
            if emb_list is not None and len(emb_list) >= len(sample_ids):
                # 进一步校验每个向量维度（GTE 是512维）
                dims_list = []
                dims_ok = True
                
                for i, e in enumerate(emb_list):
                    if hasattr(e, "shape"):
                        dim = e.shape[0]
                        dims_list.append(dim)
                        if dim != 512:
                            dims_ok = False
                            logger.warning(f"❌ 第{i}个embedding维度错误: {dim} (期望512)")
                    elif isinstance(e, list):
                        dim = len(e)
                        dims_list.append(dim)
                        if dim != 512:
                            dims_ok = False
                            logger.warning(f"❌ 第{i}个embedding维度错误: {dim} (期望512)")
                    else:
                        dims_ok = False
                        logger.warning(f"❌ 第{i}个embedding格式错误: {type(e)}")
                        dims_list.append("unknown")
                
                if dims_ok:
                    # 只显示前几个embedding的维度，避免日志过长
                    if len(dims_list) > 5:
                        logger.info(f"✅ 验证通过: 前5个embeddings维度 = {dims_list[:5]} (总共{len(emb_list)}个)")
                    else:
                        logger.info(f"✅ 验证通过: embeddings维度 = {dims_list}")
                    return True
                else:
                    logger.warning(f"❌ 向量维度验证失败: {dims_list}")
            else:
                logger.warning(
                    f"⚠️ 本次写入后 sample 验证未通过（返回 embeddings 数量 {0 if emb_list is None else len(emb_list)}，期望 {len(sample_ids)}），尝试重试...")
        except Exception as e:
            logger.error(f"验证时出错（尝试 {attempt + 1}/{retry_times + 1}）: {e}")

        attempt += 1
        time.sleep(1.0)

    return False


def rebuild_hnsw_index(persist_dir: Path, collection_name: str):
    """
    新增：重建 HNSW 索引以修复加载失败。
    通过删除二进制索引目录，触发从 WAL 重建。
    """
    db_path = persist_dir / "chroma.sqlite3"
    if not db_path.exists():
        logger.error(f"数据库文件不存在: {db_path}")
        return False

    try:
        # 查询向量段 UUID
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, c.name FROM segments s 
            JOIN collections c ON s.collection = c.id 
            WHERE s.scope = 'VECTOR' AND c.name = ?
        """, (collection_name,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            logger.warning(f"未找到 {collection_name} 的向量段")
            return False

        uuid_dir = persist_dir / result[0]
        if uuid_dir.exists():
            shutil.rmtree(uuid_dir)
            logger.info(f"已删除索引目录: {uuid_dir}，将触发重建")
        else:
            logger.warning(f"索引目录不存在: {uuid_dir}")

        # 触发重建：简单 get 操作
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_collection(collection_name)
        _ = collection.get(limit=1, include=["embeddings"])  # 触发 WAL 重建
        logger.info("HNSW 索引重建触发成功。请重试查询。")
        return True
    except Exception as e:
        logger.error(f"索引重建失败: {e}")
        return False


def main(mode="test", chunk_size=DEFAULT_CHUNK_SIZE, encode_batch=DEFAULT_ENCODE_BATCH, resume=False, rebuild=False):
    logger.info("=== lastX 全流程写入（手动 embeddings + 断点续传 + HNSW 优化） ===")
    logger.info(
        f"模式: {mode}, chunk_size={chunk_size}, encode_batch={encode_batch}, resume={resume}, rebuild={rebuild}")

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"设备: {device} | GPU Enabled: {torch.cuda.is_available()}")

    # 若指定重建，先执行重建
    if rebuild:
        logger.info("🔧 执行 HNSW 索引重建...")
        if rebuild_hnsw_index(PERSIST_DIR, COLLECTION_NAME):
            logger.info("重建完成。建议测试查询。")
        else:
            logger.error("重建失败。请检查持久目录。")
        return

    # 检查是否有检查点
    checkpoint = None
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            logger.info("🔄 从检查点恢复处理...")
            mode = checkpoint["mode"]
            chunk_size = checkpoint["chunk_size"]
            encode_batch = checkpoint["encode_batch"]
        else:
            logger.info("ℹ️ 未找到检查点，开始全新处理...")

    # 初始化组件
    data_loader = DataLoader()
    
    # 优先使用 config 中的模型路径与缓存目录（允许本地绝对路径）
    FORCED_MODEL_NAME = MODEL_CONFIG.get("embedding_model", r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base")
    FORCED_CACHE_DIR = MODEL_CONFIG.get("model_cache_dir", None)

    # 调试：打印配置信息
    logger.info(f"🔍 调试信息:")
    logger.info(f"  配置文件模型名称: {MODEL_CONFIG.get('embedding_model')}")
    logger.info(f"  配置文件缓存目录: {MODEL_CONFIG.get('model_cache_dir')}")
    logger.info(f"  选定模型名称: {FORCED_MODEL_NAME}")
    logger.info(f"  选定缓存目录: {FORCED_CACHE_DIR}")

    # 如果配置中指定了本地绝对路径但不存在，则回退到远程 GTE 模型
    if isinstance(FORCED_MODEL_NAME, str) and (FORCED_MODEL_NAME.startswith("/") or FORCED_MODEL_NAME.startswith("C:\\") or FORCED_MODEL_NAME.startswith("E:\\")) and not Path(FORCED_MODEL_NAME).exists():
        logger.warning(f"配置中指定的本地模型路径不存在: {FORCED_MODEL_NAME}，将回退到远程 GTE 模型")
        FORCED_MODEL_NAME = "Alibaba-NLP/gte-base-zh"
        FORCED_CACHE_DIR = None

    # 验证模型名称基本合法性（简单检查）
    if isinstance(FORCED_MODEL_NAME, str) and ":" in FORCED_MODEL_NAME and not FORCED_MODEL_NAME.startswith("/"):
        logger.error(f"❌ 模型名称包含非法字符 ':': {FORCED_MODEL_NAME}")
        logger.info("🔧 请检查模型名称格式")
        return

    logger.info(f"🚀 开始初始化嵌入服务，使用模型: {FORCED_MODEL_NAME}")
    logger.info(f"🔍 传递给EmbeddingService的参数:")
    logger.info(f"  model_path: {FORCED_MODEL_NAME}")
    logger.info(f"  cache_dir: {FORCED_CACHE_DIR}")

    embedding_service = EmbeddingService(
        model_path=FORCED_MODEL_NAME,
        cache_dir=FORCED_CACHE_DIR
    )

    # 处理持久目录
    if not resume or not checkpoint:
        # 全新开始或没有检查点，清空目录
        if PERSIST_DIR.exists():
            logger.info(f"删除旧持久化目录: {PERSIST_DIR}")
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(PERSIST_DIR),
                                           settings=Settings(anonymized_telemetry=False, allow_reset=True))
        collection = client.create_collection(name=COLLECTION_NAME, metadata=HNSW_METADATA)
        logger.info(f"已创建 collection: {COLLECTION_NAME} (HNSW 优化参数已应用)")
    else:
        # 从检查点恢复，使用现有数据库
        client = chromadb.PersistentClient(path=str(PERSIST_DIR),
                                           settings=Settings(anonymized_telemetry=False, allow_reset=True))
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            # 优化：恢复时验证/更新元数据一致性（若不支持更新，则日志警告）
            try:
                current_meta = collection.metadata
                if current_meta != HNSW_METADATA:
                    logger.warning(f"恢复集合元数据不匹配: 当前 {current_meta}，预期 {HNSW_METADATA}。建议删除目录重建。")
            except Exception as meta_e:
                logger.warning(f"无法验证元数据: {meta_e}")
            logger.info(f"已连接到现有 collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"无法连接到现有 collection: {e}")
            return

    # 载入数据
    csv_all = data_loader.load_csv_data()
    total_docs = len(csv_all)
    logger.info(f"载入 CSV 数据，总条数: {total_docs}")

    if mode == "test":
        csv_all = csv_all[:TEST_LIMIT]
        logger.info(f"test 模式：仅使用前 {TEST_LIMIT} 条")

    processed = data_loader.prepare_for_embedding(csv_all)
    total_chunks = len(processed)
    logger.info(f"预处理完成，文本块数: {total_chunks}")

    # 确定起始位置
    if checkpoint:
        start_index = checkpoint["processed_count"]
        failed_batches = checkpoint["failed_batches"]
        start_time = checkpoint["start_time"]
        logger.info(f"从第 {start_index} 个文档开始恢复处理...")
    else:
        start_index = 0
        failed_batches = 0
        start_time = time.time()

    processed_count = start_index

    # 从起始位置开始处理
    for i in range(start_index, total_chunks, chunk_size):
        # 检查暂停请求
        if pause_requested:
            logger.info("⏸️ 用户请求暂停，保存检查点...")
            save_checkpoint(mode, chunk_size, encode_batch, processed_count, total_chunks, failed_batches, start_time)
            logger.info("✅ 已安全暂停，检查点已保存。使用 --resume 参数恢复处理。")
            return

        batch_items = processed[i: i + chunk_size]
        texts = [x["text"] for x in batch_items]
        ids = [f"doc_{i + idx}" for idx in range(len(batch_items))]
        metadatas = [{"source": item.get("source", "unknown"), "chunk_id": i + idx} for idx, item in
                     enumerate(batch_items)]

        cleanup_gpu()
        batch_num = i // chunk_size + 1
        total_batches = (total_chunks + chunk_size - 1) // chunk_size
        logger.info(f"向量化 批次 {batch_num} / {total_batches} (items={len(texts)}) ...")

        embs = encode_texts(texts, embedding_service.model, device=device, encode_batch=encode_batch)

        # 向量化后再次检查暂停请求
        if pause_requested:
            logger.info("⏸️ 用户请求暂停，保存检查点...")
            save_checkpoint(mode, chunk_size, encode_batch, processed_count, total_chunks, failed_batches, start_time)
            logger.info("✅ 已安全暂停，检查点已保存。使用 --resume 参数恢复处理。")
            return

        if embs is None:
            logger.error("向量化失败，跳过该批次")
            failed_batches += 1
            continue

        # 验证向量维度
        if not (embs.ndim == 2 and embs.shape[1] >= 1):
            logger.error(f"向量维度异常: {embs.shape}, 跳过")
            failed_batches += 1
            continue

        ok = batch_add_and_verify(collection, ids, texts, embs, retry_times=RETRY_TIMES, sample_verify=SAMPLE_VERIFY)
        processed_count += len(texts)
        if not ok:
            logger.error(f"批次 {batch_num} 写入后验证失败，已中止。请检查磁盘/权限/Chroma 版本。")
            failed_batches += 1
            break

        # 每 10 批保存一次检查点
        if batch_num % 10 == 0:
            save_checkpoint(mode, chunk_size, encode_batch, processed_count, total_chunks, failed_batches, start_time)
            try:
                tot = collection.count()
                logger.info(f"当前 collection.count() = {tot}")
            except Exception as e:
                logger.warning(f"获取 count 出错: {e}")

        # 小间隔避免 I/O 峰值
        time.sleep(0.2)

    duration = time.time() - start_time
    logger.info("----- 完成摘要 -----")
    logger.info(f"模式: {mode} | 已处理文档: {processed_count} | 失败批次: {failed_batches} | 用时: {duration:.1f}s")

    # 最终验证：检查数据库中的embeddings状态
    logger.info("----- 最终验证 -----")
    try:
        final_count = collection.count()
        logger.info(f"数据库总文档数: {final_count}")

        if final_count > 0:
            # 随机抽取5个文档验证embeddings
            sample_ids = [f"doc_{i}" for i in range(min(5, final_count))]
            res = collection.get(ids=sample_ids, include=["embeddings"])
            emb_list = res.get("embeddings")

            if emb_list is not None and len(emb_list) > 0:
                lens = [len(e) if isinstance(e, list) else e.shape[0] for e in emb_list]
                logger.info(f"验证结果: embeddings长度={lens}")
                logger.info("✅ 向量存储验证成功！")

                # 处理完成，清理检查点
                cleanup_checkpoint()
            else:
                logger.warning("验证结果: 没有 embeddings")
                logger.warning("❌ 向量存储可能有问题")
        else:
            logger.warning("数据库为空，无法验证")

    except Exception as e:
        logger.error(f"最终验证出错: {e}")

    logger.info("💡 若查询 HNSW 加载失败，请运行: python lastX.py --rebuild")
    logger.info("建议：若全部成功，可考虑继续写入剩余数据（full 模式）。若问题持续，迁移至 WSL/Linux。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lastX.py - 向量数据库构建工具（支持断点续传）")
    parser.add_argument("--mode", choices=["test", "full"], default="test", help="test (100 条) 或 full (全部)")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="每批写入条数")
    parser.add_argument("--encode_batch", type=int, default=DEFAULT_ENCODE_BATCH, help="encode 时的 batch_size")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复处理")
    parser.add_argument("--rebuild", action="store_true", help="重建 HNSW 索引（修复加载失败）")
    args = parser.parse_args()

    logger.info("🚀 启动 lastX.py 向量数据库构建工具")
    logger.info("💡 提示：使用 Ctrl+C 可以安全暂停处理")
    logger.info("💡 提示：使用 --resume 参数可以从上次暂停点恢复")
    logger.info("💡 提示：使用 --rebuild 参数重建索引")

    main(mode=args.mode, chunk_size=args.chunk_size, encode_batch=args.encode_batch, resume=args.resume,
         rebuild=args.rebuild)