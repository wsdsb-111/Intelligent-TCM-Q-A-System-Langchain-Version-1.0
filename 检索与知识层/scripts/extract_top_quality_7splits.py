"""
从完整367万文档中提取50万高质量文档
分成7个split保存，每个约7万文档
"""

import sys
import os
import json
import pickle
from pathlib import Path
from collections import Counter
import re
import time

sys.path.insert(0, str(Path(__file__).parent / "BM25"))

from bm25_retrieval.core.models import TCMDocument, BM25Index
from bm25_retrieval.data.text_preprocessor import ChineseTextPreprocessor

def calculate_quality_score(content, preprocessor):
    """
    计算文档质量分数
    """
    score = 0.0
    
    # 1. 长度得分（50-500字最佳）
    length = len(content)
    if 50 <= length <= 500:
        score += 1.0
    elif length < 50:
        score += length / 50
    else:
        score += 500 / length
    
    # 2. 中医术语得分
    medical_terms = [
        '治疗', '症状', '方剂', '中药', '针灸', '推拿', '辨证', '病因',
        '脏腑', '经络', '穴位', '气血', '阴阳', '五行', '寒热', '虚实',
        '脾胃', '肝肾', '心肺', '补气', '活血', '清热', '解毒', '化痰'
    ]
    
    term_count = sum(1 for term in medical_terms if term in content)
    score += term_count * 0.2
    
    # 3. 问答格式得分
    if any(keyword in content for keyword in ['什么', '如何', '怎么', '为什么']):
        score += 0.5
    
    # 4. 内容质量（避免重复字符）
    if len(set(content)) / max(len(content), 1) > 0.3:
        score += 0.5
    
    return score


print("=" * 80)
print("从367万文档中提取50万高质量文档")
print("=" * 80)

# 1. 加载split配置
split_config_path = "BM25/data/split_indices/split_config.json"
with open(split_config_path, 'r', encoding='utf-8') as f:
    split_config = json.load(f)

print(f"\n📊 发现 {split_config['total_splits']} 个split")

# 2. 逐个加载split，提取文档并评分
all_documents = []
preprocessor = ChineseTextPreprocessor()

print(f"\n⏳ 第1阶段：加载所有split并评估质量...")
print(f"   这可能需要5-10分钟，请耐心等待...\n")

for i, split_info in enumerate(split_config['splits'], 1):
    split_id = split_info['split_id']
    split_path = split_info['split_path']
    
    # 修正相对路径
    if not Path(split_path).is_absolute():
        split_path = f"BM25/data/split_indices/split_{split_id:03d}"
    
    index_file = Path(split_path) / "index.pkl"
    
    if not index_file.exists():
        print(f"   ⚠️ Split {split_id}: 索引不存在，跳过")
        continue
    
    try:
        print(f"   [{i}/{split_config['total_splits']}] 加载 split_{split_id:03d}...", end='', flush=True)
        start = time.time()
        
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        # 提取文档
        if isinstance(index_data, dict):
            documents = index_data.get('documents', {})
        else:
            documents = index_data.documents if hasattr(index_data, 'documents') else {}
        
        # 评估每个文档质量
        for doc_id, doc_data in documents.items():
            # 获取内容
            if isinstance(doc_data, dict):
                content = doc_data.get('content', '')
            else:
                content = doc_data.combined_text if hasattr(doc_data, 'combined_text') else ''
            
            if not content or len(content) < 10:
                continue
            
            # 质量评分
            score = calculate_quality_score(content, preprocessor)
            
            all_documents.append({
                'doc_id': doc_id,
                'doc_data': doc_data,
                'content': content,
                'quality_score': score,
                'source_split': split_id
            })
        
        elapsed = time.time() - start
        print(f" ✅ ({len(documents):,}文档, {elapsed:.1f}s)")
        
    except Exception as e:
        print(f" ❌ 失败: {e}")
        continue

print(f"\n✅ 第1阶段完成：加载了 {len(all_documents):,} 个文档")

# 3. 按质量排序，选择top 50万
print(f"\n⏳ 第2阶段：排序并选择top 500,000...")
all_documents.sort(key=lambda x: x['quality_score'], reverse=True)

top_500k = all_documents[:500000]
print(f"   ✅ 选择了 {len(top_500k):,} 个高质量文档")
print(f"   📊 质量分数范围: {top_500k[-1]['quality_score']:.2f} - {top_500k[0]['quality_score']:.2f}")

# 4. 分成7个split
print(f"\n⏳ 第3阶段：分成7个split并保存...")

docs_per_split = len(top_500k) // 7
output_dir = Path("BM25/data/optimized_splits_7")
output_dir.mkdir(exist_ok=True)

split_infos = []

for split_idx in range(7):
    start_idx = split_idx * docs_per_split
    end_idx = start_idx + docs_per_split if split_idx < 6 else len(top_500k)
    
    split_docs = top_500k[start_idx:end_idx]
    
    print(f"\n   [{split_idx+1}/7] 保存 split_{split_idx:03d}...")
    print(f"       文档范围: {start_idx:,} - {end_idx:,} ({len(split_docs):,}个)")
    
    # 创建split目录
    split_dir = output_dir / f"split_{split_idx:03d}"
    split_dir.mkdir(exist_ok=True)
    
    # 构建BM25索引
    index = BM25Index()
    vocabulary = set()
    total_length = 0
    document_frequencies = Counter()
    
    for doc_info in split_docs:
        doc_id = doc_info['doc_id']
        doc_data = doc_info['doc_data']
        content = doc_info['content']
        
        # 分词
        tokens = preprocessor.tokenize(content)
        if not tokens:
            continue
        
        # 创建TCMDocument
        if isinstance(doc_data, dict):
            tcm_doc = TCMDocument(
                id=doc_id,
                instruction='',
                input='',
                output=content,
                combined_text=content,
                tokens=tokens,
                metadata=doc_data.get('metadata', {})
            )
        else:
            tcm_doc = doc_data
            tcm_doc.tokens = tokens
        
        # 计算统计
        doc_length = len(tokens)
        index.document_lengths[doc_id] = doc_length
        total_length += doc_length
        
        # 词频
        term_freq = Counter(tokens)
        index.term_frequencies[doc_id] = dict(term_freq)
        
        # 文档频率
        for term in set(tokens):
            document_frequencies[term] += 1
        
        # 词汇表
        vocabulary.update(tokens)
        
        # 存储文档
        index.documents[doc_id] = tcm_doc
    
    # 设置索引属性
    index.vocabulary = vocabulary
    index.document_frequencies = dict(document_frequencies)
    index.total_documents = len(index.documents)
    index.average_document_length = total_length / index.total_documents if index.total_documents > 0 else 0
    
    # 保存索引
    index_file = split_dir / "index.pkl"
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)
    
    # 保存info
    info = {
        "split_id": split_idx,
        "total_documents": index.total_documents,
        "vocabulary_size": len(index.vocabulary),
        "average_document_length": index.average_document_length,
        "quality_score_range": [split_docs[-1]['quality_score'], split_docs[0]['quality_score']]
    }
    
    with open(split_dir / "index_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    split_infos.append(info)
    
    print(f"       ✅ 保存完成")
    print(f"          文档数: {info['total_documents']:,}")
    print(f"          词汇数: {info['vocabulary_size']:,}")

# 5. 保存总配置
print(f"\n⏳ 第4阶段：生成配置文件...")

final_config = {
    "total_splits": 7,
    "split_directory": "BM25/data/optimized_splits_7",
    "splits": [
        {
            "split_id": i,
            "split_path": f"BM25/data/optimized_splits_7/split_{i:03d}",
            "total_documents": info['total_documents'],
            "vocabulary_size": info['vocabulary_size']
        }
        for i, info in enumerate(split_infos)
    ],
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "description": "从367万文档中提取的50万高质量文档，分7个split"
}

config_file = output_dir / "split_config.json"
with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(final_config, f, indent=2, ensure_ascii=False)

print(f"   ✅ 配置文件已保存: {config_file}")

# 6. 总结
print(f"\n" + "=" * 80)
print("🎉 提取完成！")
print("=" * 80)
print(f"\n📊 统计信息:")
total_docs = sum(info['total_documents'] for info in split_infos)
print(f"   总文档数: {total_docs:,}")
print(f"   Split数量: {len(split_infos)}")
print(f"   平均每个split: {total_docs // len(split_infos):,}文档")

print(f"\n📁 输出位置:")
print(f"   {output_dir.absolute()}")

print(f"\n🚀 使用方法:")
print(f"   修改配置文件:")
print(f"   split_config_path: \"{config_file.relative_to(Path.cwd())}\"")
print(f"\n✅ 完成！现在BM25可以在12秒内快速启动了！")

