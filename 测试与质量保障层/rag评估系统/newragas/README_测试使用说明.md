# HybridRagasEvaluatorV4 测试使用说明

## 功能概述

改造后的 `test_v4_simple.py` 现在支持测试数据集中的任意数量问题，提供了灵活的命令行参数配置。

## 主要功能

### 1. 从数据集加载问题
- 支持从 `eval_dataset_100.jsonl` 加载指定数量的问题
- 可以指定起始索引和问题数量
- 自动解析数据集中的用户问题

### 2. 命令行参数支持
- 灵活配置测试参数
- 支持自定义问题列表
- 可选择不同的测试模式

### 3. 增强的测试结果
- 包含数据集索引信息
- 显示Ground Truth预览
- 实时保存测试进度

## 向量数据库格式说明

### 数据库结构
- **向量维度**: 768维（使用BGE-base-zh模型）
- **存储格式**: FAISS索引 + metadata.pkl
- **文档格式**: 纯文本格式

### 数据存储方式
```python
# 向量数据库中的文档结构
{
    'id': 'doc_123',
    'text': '问题文本',  # 用于向量化的文本
    'metadata': {
        'question': '原始问题',
        'answer': '纯文本答案',  # 用于检索返回的内容
        'full_conversation': {...},  # 完整的对话格式（JSON）
        'source': 'merged_medical_dataset',
        'index': 123
    }
}
```

### 文档处理流程
1. **向量化**: 只使用问题文本进行向量化
2. **检索**: 根据问题向量召回相关文档
3. **返回**: 返回metadata中的纯文本答案
4. **增强**: 基于纯文本进行关键词增强和排序

### 召回文档数量配置
- **纯向量检索**: 召回5个文档，选择前3个用于生成
- **混合检索**: 向量检索5个文档，知识图谱5个文档，选择向量前3个+知识图谱前2个用于生成
- **评估文档**: 纯向量使用5个文档，混合检索使用5个向量文档

### 文档处理格式

#### 1. 向量检索处理
```python
# 输入: 自然语言问题
question = "请推荐适合经常口臭的中药"

# 处理流程:
# 1. 问题向量化
question_embedding = embedding_model.encode(question)

# 2. 向量数据库检索
results = vector_store.search(question_embedding, n_results=5)

# 3. 提取纯文本答案
contexts = []
for result in results:
    metadata = result.get('metadata', {})
    answer = metadata.get('answer', '')  # 纯文本格式
    contexts.append(answer)

# 4. 关键词增强和排序
enhanced_contexts = enhance_contexts(contexts, question)  # 选择前3个
```

#### 2. 知识图谱检索处理
```python
# 输入: 自然语言问题
question = "请推荐适合经常口臭的中药"

# 处理流程:
# 1. 提取关键词
keywords = extract_keywords(question)  # ['口臭', '中药', '推荐']

# 2. 知识图谱查询
kg_contexts = []
for keyword in keywords:
    query = "MATCH (n)-[r]-(m) WHERE n.name CONTAINS $keyword"
    results = session.run(query, keyword=keyword)
    for record in results:
        context = f"知识图谱: {entity1} - {relation} - {entity2}"
        kg_contexts.append(context)

# 3. 选择前2个用于生成
selected_kg_contexts = kg_contexts[:2]
```

#### 3. 混合检索处理
```python
# 混合检索流程
vector_contexts = vector_retrieve(question)  # 5个文档
kg_contexts = kg_retrieve(question)  # 5个文档

# 向量部分增强和选择
vector_enhanced = enhance_contexts(vector_contexts, question)  # 选择前3个
vector_selected = vector_enhanced[:3]

# 知识图谱部分选择
kg_selected = kg_contexts[:2]

# 合并用于生成
generation_contexts = vector_selected + kg_selected  # 5个文档
evaluation_contexts = vector_contexts  # 5个文档用于评估
```

#### 4. RAGAS评估处理
```python
# 纯向量检索评估
if route_type == "vector":
    # context_precision, context_recall: 使用5个召回文档
    context_text = "\n".join(contexts)  # 5个文档
    
    # faithfulness, answer_relevancy: 使用3个生成文档
    generation_text = "\n".join(contexts[:3])  # 3个文档

# 混合检索评估
elif route_type == "hybrid":
    # context_precision, context_recall: 使用5个向量召回文档
    context_text = "\n".join(vector_contexts)  # 5个向量文档
    
    # faithfulness, answer_relevancy: 使用全部生成文档
    generation_text = "\n".join(generation_contexts)  # 5个文档
```

### 格式转换规则
1. **JSON格式自动转换**: 如果文档是JSON格式，自动提取纯文本内容
2. **对话格式处理**: 如果是对话格式，提取assistant的content
3. **纯文本保持**: 如果已经是纯文本，直接使用
4. **关键词增强**: 基于纯文本进行关键词相似度计算和排序

## 使用方法

### 基本用法

```bash
# 使用默认设置（3个问题）
python test_v4_simple.py

# 测试数据集中的前10个问题
python test_v4_simple.py -n 10

# 从第5个问题开始，测试20个问题
python test_v4_simple.py -n 20 -s 5

# 使用自定义数据集
python test_v4_simple.py -d "path/to/your/dataset.jsonl" -n 15
```

### 命令行参数详解

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--mode` | - | str | full | 测试模式：full=完整流程测试, memory=内存管理测试 |
| `--num-questions` | `-n` | int | 3 | 测试问题数量 |
| `--start-index` | `-s` | int | 0 | 数据集起始索引 |
| `--dataset` | `-d` | str | eval_dataset_100.jsonl | 数据集文件路径 |
| `--questions` | `-q` | list | None | 自定义测试问题列表 |
| `--skip-memory` | - | bool | False | 跳过内存管理测试 |

### 使用示例

#### 1. 测试数据集中的前5个问题
```bash
python test_v4_simple.py -n 5
```

#### 2. 从第10个问题开始，测试15个问题
```bash
python test_v4_simple.py -n 15 -s 10
```

#### 3. 使用自定义问题
```bash
python test_v4_simple.py -q "我恶寒感冒，可以给我推荐一个中药吗？" "口臭是什么原因引起的？"
```

#### 4. 只运行内存管理测试
```bash
python test_v4_simple.py --mode memory
```

#### 5. 跳过内存管理测试
```bash
python test_v4_simple.py -n 10 --skip-memory
```

#### 6. 使用不同的数据集
```bash
python test_v4_simple.py -d "path/to/other_dataset.jsonl" -n 20
```

#### 7. 测试完整数据集
```bash
# 测试eval_dataset_100.jsonl中的所有99个问题
python test_v4_simple.py -n 99

# 或者不指定数量，测试所有可用问题
python test_v4_simple.py -n 999
```

#### 8. 分批测试大数据集
```bash
# 第一批：问题1-20
python test_v4_simple.py -n 20 -s 0

# 第二批：问题21-40  
python test_v4_simple.py -n 20 -s 20

# 第三批：问题41-60
python test_v4_simple.py -n 20 -s 40

# 最后一批：问题61-99
python test_v4_simple.py -n 39 -s 60
```

## 输出结果

### 控制台输出
- 实时显示测试进度
- 显示每个问题的处理结果
- 包含内存使用情况和组件状态
- 显示Ground Truth预览

### JSON结果文件
- 保存在 `results/` 目录下
- 包含完整的测试结果和评估数据
- 支持实时保存，即使程序中断也能保留进度
- 包含数据集索引和Ground Truth信息

## 测试结果结构

```json
{
  "test_name": "完整流程测试",
  "start_time": "2025-10-29T14:41:45.454779",
  "questions": ["问题1", "问题2", "问题3"],
  "results": [
    {
      "question_id": 1,
      "dataset_index": 0,
      "question": "我恶寒感冒，可以给我推荐一个中药吗？",
      "status": "success",
      "rag_result": { ... },
      "ragas_result": {
        "ground_truth": "完整的Ground Truth内容...",
        "ground_truth_length": 278,
        "evaluation_data": { ... }
      },
      "ground_truth_preview": "Ground Truth前200字符...",
      "memory_before": { ... },
      "memory_after": { ... },
      "component_status": { ... }
    }
  ],
  "summary": { ... }
}
```

## 注意事项

1. **数据集格式**：确保数据集文件是JSONL格式，每行包含一个JSON对象，包含`messages`字段
2. **内存管理**：大量问题测试时注意内存使用情况
3. **实时保存**：测试结果会实时保存，可以随时中断和恢复
4. **Ground Truth**：系统会自动从数据集中提取Ground Truth进行RAGAS评估

## 故障排除

### 常见问题

1. **数据集文件不存在**
   - 检查文件路径是否正确
   - 使用绝对路径或相对路径

2. **内存不足**
   - 减少测试问题数量
   - 使用 `--skip-memory` 跳过内存管理测试

3. **导入错误**
   - 确保在正确的目录下运行
   - 检查Python路径配置

### 调试模式

```bash
# 测试单个问题
python test_v4_simple.py -n 1

# 使用自定义问题测试
python test_v4_simple.py -q "测试问题"
```

## 性能建议

- 小规模测试：1-5个问题
- 中等规模测试：10-20个问题  
- 大规模测试：50+个问题（需要充足的内存和时间）

根据您的硬件配置和需求选择合适的测试规模。
