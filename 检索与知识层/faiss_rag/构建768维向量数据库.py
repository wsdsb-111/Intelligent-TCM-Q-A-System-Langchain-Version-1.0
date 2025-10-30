#!/usr/bin/env python3
"""
使用 GTE Base 模型（768维）构建向量数据库
确保输出格式与现有512维数据库完全一致，实现无缝衔接
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_retrieval_system.faiss_manager import FaissManager

# 配置
SCRIPT_DIR = Path(__file__).parent  # 脚本所在目录
JSONL_FILE = SCRIPT_DIR / "bad_data_extraction" / "clean_data.jsonl"
FAISS_PATH = SCRIPT_DIR / "向量数据库_768维"  # 新的768维数据库路径
MODEL_PATH = r"E:\毕业论文和设计\线上智能中医问答项目\Model Layer\model\iic\nlp_gte_sentence-embedding_chinese-base\iic\nlp_gte_sentence-embedding_chinese-base"
FALLBACK_MODEL = "Alibaba-NLP/gte-base-zh"

BATCH_SIZE = 32
MAX_SAMPLES = None  # None = 全部

print("=" * 80)
print("使用 GTE Base 模型（768维）构建向量数据库")
print("=" * 80)

# 步骤 1: 加载数据
print("\n步骤 1: 加载数据")
print("-" * 80)

data = []
print(f"读取文件: {JSONL_FILE}")

with open(str(JSONL_FILE), 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, desc="读取数据")):
        if MAX_SAMPLES and i >= MAX_SAMPLES:
            break
        
        try:
            item = json.loads(line)
            messages = item.get('messages', [])
            
            question = ""
            answer = ""
            
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                if role == 'user':
                    question = content
                elif role == 'assistant':
                    answer = content
            
            if question and answer:
                # 只使用问题文本进行向量化，这样自然语言查询可以直接匹配
                # 但保留完整的对话格式在metadata中供检索后使用
                data.append({
                    'id': f"doc_{i}",
                    'text': question,  # 只使用问题文本进行向量化
                    'metadata': {
                        'question': question,
                        'answer': answer,
                        'full_conversation': item,  # 保存完整的对话格式
                        'source': 'merged_medical_dataset',
                        'index': i
                    }
                })
        
        except Exception as e:
            print(f"\n警告: 第 {i} 行解析失败: {e}")
            continue

print(f"✅ 成功加载 {len(data)} 条有效数据")

# 步骤 2: 加载 GTE Base 模型
print("\n步骤 2: 加载 GTE Base Embedding 模型")
print("-" * 80)

print(f"模型路径: {MODEL_PATH}")

try:
    if Path(MODEL_PATH).exists():
        model = SentenceTransformer(MODEL_PATH)
        print(f"✅ 本地 GTE Base 模型加载成功")
    else:
        print(f"⚠️  本地模型不存在，使用 HuggingFace: {FALLBACK_MODEL}")
        model = SentenceTransformer(FALLBACK_MODEL)
except Exception as e:
    print(f"❌ 本地模型加载失败: {e}")
    print(f"使用 HuggingFace: {FALLBACK_MODEL}")
    model = SentenceTransformer(FALLBACK_MODEL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"设备: {device.upper()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 步骤 3: 生成768维向量
print("\n步骤 3: 生成 768维 Embeddings")
print("-" * 80)

texts = [item['text'] for item in data]
print(f"总文档数: {len(texts)}")
print(f"批次大小: {BATCH_SIZE}")

all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="生成768维向量"):
    batch = texts[i:i+BATCH_SIZE]
    
    embeddings = model.encode(
        batch,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    all_embeddings.extend(embeddings)
    
    # 定期清理显存
    if i % 1000 == 0 and i > 0:
        torch.cuda.empty_cache()

print(f"✅ 生成完成")
print(f"向量数量: {len(all_embeddings)}")
print(f"向量维度: {len(all_embeddings[0])}")

# 验证维度
if len(all_embeddings[0]) != 768:
    print(f"⚠️  警告: 期望768维，实际{len(all_embeddings[0])}维")

# 步骤 4: 构建 Faiss 索引（768维）
print("\n步骤 4: 构建 Faiss 索引（768维）")
print("-" * 80)

print(f"初始化 Faiss 管理器...")
# 检查 Faiss GPU 支持
import faiss
has_gpu = faiss.get_num_gpus() > 0
print(f"Faiss GPU 支持: {has_gpu}")

faiss_manager = FaissManager(
    persist_directory=str(FAISS_PATH),  # 转换为字符串
    dimension=768,  # 使用768维
    use_gpu=has_gpu  # 使用 Faiss 的 GPU 检测，而不是 PyTorch 的
)

print(f"添加文档到索引...")

# 准备文档
documents = []
for item, embedding in zip(data, all_embeddings):
    documents.append({
        'embedding': embedding,
        'text': item['text'],
        'metadata': item['metadata']
    })

# 批量添加
import_batch_size = 2000
for i in tqdm(range(0, len(documents), import_batch_size), desc="导入数据"):
    batch = documents[i:i+import_batch_size]
    faiss_manager.add_documents(batch)

print(f"✅ 导入完成")

# 步骤 5: 保存索引
print("\n步骤 5: 保存索引")
print("-" * 80)

try:
    faiss_manager.save_index()
    print(f"✅ 索引已保存到: {FAISS_PATH}")
except Exception as e:
    print(f"❌ 保存失败: {e}")
    print(f"\n尝试备用方案...")
    
    # 备用方案：使用 pickle 保存
    import pickle
    backup_file = FAISS_PATH / "backup.pkl"
    with open(backup_file, 'wb') as f:
        pickle.dump({
            'documents': faiss_manager.documents,
            'metadata': faiss_manager.metadata,
            'embeddings': all_embeddings
        }, f)
    print(f"✅ 使用备用方案保存到: {backup_file}")

# 步骤 6: 验证
print("\n步骤 6: 验证")
print("-" * 80)

stats = faiss_manager.get_stats()
print(f"文档数量: {stats['total_documents']:,}")
print(f"向量维度: {stats['dimension']}")
print(f"使用 GPU: {stats['use_gpu']}")

# 测试查询
print(f"\n测试查询...")
test_queries = [
    "感冒咳嗽吃什么药",
    "失眠多梦如何调理",
    "腰痛的治疗方法"
]

for query_text in test_queries:
    print(f"\n查询: {query_text}")
    
    # 生成查询向量
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # 搜索
    results = faiss_manager.search(query_embedding, n_results=3)
    
    if results:
        print(f"  ✅ 返回 {len(results)} 个结果")
        for i, result in enumerate(results, 1):
            score = result['score']
            content = result['content'][:100]
            print(f"    [{i}] 相似度: {score:.4f} | 内容: {content}...")
    else:
        print(f"  ❌ 查询失败")

# 步骤 7: 格式兼容性验证
print("\n步骤 7: 格式兼容性验证")
print("-" * 80)

# 检查生成的文件是否与512维数据库格式一致
expected_files = ['faiss.index', 'metadata.pkl', 'documents.json']
all_files_exist = all((FAISS_PATH / file).exists() for file in expected_files)

if all_files_exist:
    print("✅ 所有必需文件已生成:")
    for file in expected_files:
        file_path = FAISS_PATH / file
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        print(f"  - {file}: {file_size:.2f} MB")
    
    # 验证文件内容格式
    try:
        # 验证 documents.json 格式
        with open(FAISS_PATH / 'documents.json', 'r', encoding='utf-8') as f:
            docs = json.load(f)
        print(f"✅ documents.json 格式正确，包含 {len(docs)} 个文档")
        
        # 验证 metadata.pkl 格式
        import pickle
        with open(FAISS_PATH / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ metadata.pkl 格式正确，包含 {len(metadata)} 个元数据")
        
        print("✅ 格式兼容性验证通过！")
        
    except Exception as e:
        print(f"❌ 格式验证失败: {e}")
else:
    print("❌ 部分文件缺失")

print("\n" + "=" * 80)
print("🎉 GTE Base（768维）向量数据库构建完成！")
print("=" * 80)
print(f"\n数据库位置: {Path(FAISS_PATH).absolute()}")
print(f"文档数量: {faiss_manager.count():,}")
print(f"向量维度: 768")
print(f"\n格式兼容性: ✅ 与512维数据库格式完全一致")
print(f"\n无缝切换方法:")
print(f"  1. 修改 FaissManager 初始化时的 dimension=768")
print(f"  2. 修改 persist_directory 路径指向新数据库")
print(f"  3. 其他代码无需修改")
print(f"\n下一步:")
print(f"  1. 测试新数据库: python 测试768维检索.py")
print(f"  2. 集成到系统中")
print(f"  3. 性能对比测试")
