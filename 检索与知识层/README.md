# 智能中医问答系统 - 检索与知识层

基于LangChain的混合检索增强生成(RAG)系统，整合BM25、向量检索和知识图谱，使用Qwen1.5-1.8B微调模型生成专业中医问答。

## ✨ 核心特性

- ✅ **三源混合检索**: BM25(500K文档) + 向量(语义理解) + 知识图谱(217K实体)
- ✅ **智能加权融合**: 根据查询类型自动调整三源权重
- ✅ **LLM实体提取**: 使用微调模型智能识别中医实体
- ✅ **自适应策略**: 查询分类器自动优化检索策略
- ✅ **高性能**: 总响应时间4-13秒，成功率100%

## 📁 项目结构

```
检索与知识层/
├── scripts/                     # 启动和工具脚本
│   ├── start_langchain_service.py   # 服务启动脚本
│   ├── start.py                     # 旧版启动脚本
│   └── extract_top_quality_7splits.py  # 数据处理脚本
│
├── tests/                       # 测试文件
│   ├── test_system.py           # 快速测试（推荐）
│   ├── test_final_hybrid_rag.py # 完整测试
│   └── ...
│
├── docs/                        # 文档文件
│   ├── README_LangChain中间层.md    # 详细技术文档
│   ├── 快速使用指南.md
│   └── ...
│
├── langchain/                   # LangChain核心代码
│   ├── api/                     # FastAPI服务层
│   ├── core/                    # 核心逻辑（协调器、融合）
│   ├── adapters/                # 检索适配器
│   ├── services/                # 服务层（模型、RAG）
│   ├── models/                  # 数据模型
│   ├── utils/                   # 工具类（分类器、实体提取）
│   └── config/                  # 配置文件
│
├── BM25/                        # BM25数据（已删除）
├── faiss_rag/                   # Faiss向量数据库（155,902条文档）
├── Graphrag/                    # 知识图谱（Neo4j dump）
│
├── 启动服务.py                  # 启动快捷方式
├── 测试系统.py                  # 测试快捷方式
└── README.md                    # 本文件
```

## 🚀 快速开始

### 方式1: 使用快捷方式（最简单）

```bash
# 启动服务
python 启动服务.py

# 测试系统（新开终端）
python 测试系统.py
```

### 方式2: 直接调用脚本

```bash
# 启动服务
python scripts/start_langchain_service.py

# 快速测试
python tests/test_system.py

# 完整测试
python tests/test_final_hybrid_rag.py
```

### 访问服务

启动后访问：
- **API服务**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/v1/health

## 💡 使用示例

### Python API调用

```python
import requests

# 问答接口
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "人参的功效",
        "top_k": 5,
        "max_new_tokens": 200
    }
)

result = response.json()
print(f"答案: {result['answer']}")
print(f"检索时间: {result['metadata']['retrieval_time']}s")
print(f"来源分布: {result['metadata']['sources_used']}")
```

### curl调用

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"头痛怎么治疗"}'
```

## ⚙️ 配置文件

主配置文件: `langchain/config/service_config.yaml`

```yaml
# 检索配置
retrieval:
  fusion_method: "weighted"  # 加权融合
  weights:
    bm25: 0.5      # BM25权重
    vector: 0.3    # 向量权重
    graph: 0.2     # 图谱权重
  
  enable_bm25: true
  enable_vector: true
  enable_graph: true
  
  # 图检索
  graph:
    use_llm_entity_extraction: true  # LLM智能实体提取
```

## 📊 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **文档规模** | 500,000 | BM25索引 |
| **图谱规模** | 217K节点，1.6M关系 | Neo4j知识图谱 |
| **检索时间** | 2-3秒 | 三源并行 |
| **生成时间** | 2-10秒 | Qwen模型 |
| **总响应** | 4-13秒 | 端到端 |
| **成功率** | 100% | 稳定可靠 |

## 🎯 核心模块

### 1. 查询分类器
自动识别查询类型并推荐最佳权重：
- 实体查询 → BM25=60%, Vector=20%, Graph=20%
- 治疗查询 → BM25=40%, Vector=40%, Graph=20%
- 关系查询 → BM25=30%, Vector=30%, Graph=40%

### 2. LLM实体提取
使用微调模型提取中医实体：
- ✅ 识别完整方剂名（天麻钩藤饮）
- ✅ 识别症状组合（感冒发烧）
- ✅ 不限于预定义词典
- ✅ 有规则模式回退

### 3. 加权融合
确保三源都能参与融合：
- 归一化评分到[0,1]
- 按权重组合
- 避免结果丢失

## 📚 文档

- **详细文档**: `docs/README_LangChain中间层.md`
- **快速指南**: `docs/快速使用指南.md`
- **API文档**: http://localhost:8000/docs （启动服务后访问）

## 🐛 故障排除

常见问题请查看: `docs/快速使用指南.md`中的"常见问题"章节

---

**版本**: v3.0 - LangChain中间层  
**更新时间**: 2025-10-11  
**状态**: 生产就绪 ✅

**技术栈**: LangChain + FastAPI + Qwen1.5-1.8B + Neo4j + Chroma