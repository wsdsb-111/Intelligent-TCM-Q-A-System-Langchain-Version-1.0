# 基于智能路由混合检索增强生成的智能中医问答系统

[![Version](https://img.shields.io/badge/version-v4.4-blue.svg)](docs/项目介绍.md)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## 📖 项目简介

本项目是一个基于**智能路由混合检索增强生成(Intelligent Routing Hybrid RAG)**技术的智能中医问答系统，旨在为用户提供专业、准确、全面的中医知识问答服务。

### 🎯 核心特点

- 🔍 **二元智能路由**：基于Qwen-Flash API的智能分类器，自动选择向量检索或混合检索
- 📚 **二元检索策略**：
  - **纯向量检索**：实体主导型查询，直接使用Faiss向量搜索
  - **混合检索**：复杂推理查询，向量+知识图谱并行检索后融合
- 🧠 **大规模知识库**：46,697个中医实体 + Neo4j知识图谱 + 15万条向量化文档
- ⚡ **高性能响应**：平均28.66秒端到端响应，支持并发处理
- 🎨 **结构化组件管理**：参照评估系统实现组件状态跟踪和生命周期管理
- 🔧 **生产级架构**：五层分布式架构，从数据层到API层的完整技术栈

### 🏆 创新亮点

1. **五层分布式架构**：文档层、检索与知识层、应用协调层、部署与基础设施层、测试与质量保障层
2. **二元智能路由**：基于Qwen-Flash API的实体识别和复杂推理判断，实现精准路由
3. **混合检索融合**：向量检索(Faiss+GTE模型) + 知识图谱检索(Neo4j)并行执行后智能融合
4. **结构化组件管理**：参照评估系统实现组件状态跟踪、预热机制和错误处理
5. **端到端测试体系**：完整的生产流程测试、RAGAS评估框架和性能监控

## 🛠️ 技术栈

### 核心技术栈
- **后端框架**: FastAPI + Uvicorn
- **检索增强**: LangChain + Faiss + Neo4j
- **大语言模型**: Qwen3-1.7B (Qwen-Flash API)
- **向量嵌入**: GTE (General Text Embeddings)
- **重排序模型**: BGE-Reranker-Base
- **查询扩展**: Text2Vec-Base-Chinese-Paraphrase

### 数据存储
- **向量数据库**: Faiss (15万条向量化文档)
- **知识图谱**: Neo4j (46,697个中医实体)
- **配置管理**: YAML
- **缓存系统**: 本地文件缓存

### 测试与评估
- **单元测试**: pytest
- **RAG评估**: RAGAS (DeepSeek API)
- **性能监控**: 内置性能统计
- **集成测试**: 端到端流程测试

## 🚀 快速开始

### 环境要求

- Python 3.9+
- GPU (推荐NVIDIA显卡，8GB+显存)
- Neo4j数据库
- 16GB+ 内存
- 50GB+ 磁盘空间

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd 线上智能中医问答项目
```

#### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装应用协调层依赖
pip install -r 应用协调层/middle/requirements.txt

# 安装检索与知识层依赖
pip install -r 检索与知识层/requirements.txt
```

#### 3. 配置数据库

**Neo4j知识图谱**
```bash
# 启动Neo4j服务
neo4j start

# 默认连接信息
# URI: bolt://localhost:7687
# Username: neo4j
# Password: your_password
```

**Faiss向量数据库**
- 已预构建，位于：`检索与知识层/faiss_rag/向量数据库_简单查询`
- 包含155,902条文档，查询速度8.76ms

#### 4. 配置系统

编辑 `应用协调层/middle/config/service_config.yaml`：

```yaml
model:
  base_model_path: "Model Layer/model/qwen/Qwen3-1.7B/Qwen/Qwen3-1___7B"
  adapter_path: "Model Layer/model/checkpoint-7983"

retrieval:
  vector:
    persist_directory: "检索与知识层/faiss_rag/向量数据库_简单查询"
  
  graph:
    neo4j_uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "your_password"
```

#### 5. 启动服务（FastAPI）

推荐使用以下任一方式启动服务：

```bash
# 方式一：中文快捷启动（推荐，在项目根目录执行）
python 部署与基础设施层/启动服务.py

# 方式二：通用脚本启动（在项目根目录执行，支持 --reload 等参数）
python 部署与基础设施层/scripts/start_langchain_service.py --reload

# 方式三：直接使用 uvicorn（在项目根目录执行）
uvicorn middle.api.main_app:app --host 0.0.0.0 --port 8000 --log-level info

# 方式四：在部署与基础设施层目录执行
cd 部署与基础设施层
python scripts/start_langchain_service.py
```

说明：
- 应用入口为 `middle.api.main_app:app`（对应 `应用协调层/middle/api/main_app.py`）
- 配置文件位于 `应用协调层/middle/config/service_config.yaml`，启动脚本会自动加载
- 建议在**项目根目录**执行启动命令

#### 6. 访问服务

- **API文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/api/v1/health
- **根路径**：http://localhost:8000/

## 📝 使用指南

### API接口调用

#### 1. 问答接口

```python
import requests

# 问答请求
response = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "请推荐适合经常口臭的中药",
    "top_k": 5,
    "temperature": 0.5
})

result = response.json()
print("答案:", result["answer"])
print("路由决策:", result["metadata"]["routing_decision"])
print("置信度:", result["metadata"]["routing_confidence"])
```

#### 2. 纯检索接口

```python
# 检索请求
response = requests.post("http://localhost:8000/api/v1/retrieve", json={
    "query": "生姜的功效有哪些？",
    "top_k": 3
})

results = response.json()["results"]
for i, result in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {result['content']}")
    print(f"来源: {result['source']}")
```

#### 3. 健康检查

```python
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())
```

### 智能路由示例

系统会根据查询复杂度自动选择最优检索策略：

| 查询类型 | 示例 | 路由策略 | 召回规则 | 生成使用规则 |
|---------|------|---------|---------|------------|
| **实体主导型** | "请推荐适合经常口臭的中药" | vector_only | 召回3个向量文档 | 使用3个文档 |
| **复杂推理型** | "如何治疗失眠多梦？" | hybrid | 召回5向量+5图谱（10个） | 使用3向量+5图谱（8个） |
| **复杂推理型** | "人参和黄芪的配伍关系是什么？" | hybrid | 召回5向量+5图谱（10个） | 使用3向量+5图谱（8个） |

### 完整RAG流程

```
1. 启动FastAPI服务并加载组件
   ├── 加载向量适配器（Faiss + GTE，已移除关键词增强）
   ├── 加载图谱适配器（Neo4j）
   ├── 加载智能路由器（Qwen-Flash API）
   ├── 加载模型服务（Qwen3-1.7B + LoRA）
   ├── 加载查询扩展模型（text2vec-base-chinese-paraphrase）
   ├── 加载重排序模型（bge-reranker-base）
   └── 预热检索组件

2. 用户输入问题
   └── 接收用户查询请求

3. 智能路由分类
   ├── 使用Qwen-Flash API进行路由判断
   ├── 返回路由决策：vector_only 或 hybrid
   └── 返回路由置信度

4. 查询扩展（可选，已集成）
   ├── 使用text2vec-base-chinese-paraphrase模型
   └── 扩展原始查询以提升召回

5. 向量检索/混合检索召回文档
   ├── vector_only模式：
   │   ├── 向量检索：召回3个文档
   │   └── 生成使用：3个文档
   └── hybrid模式：
       ├── 向量检索：召回5个文档
       ├── 图谱检索：召回5个文档
       └── 总召回：10个文档（5向量+5图谱）

6. 重排序（可选，已集成）
   ├── 使用bge-reranker-base模型
   └── 对召回文档按相关性重新排序

7. 选出生成回答的文档
   ├── vector_only模式：使用所有3个文档
   └── hybrid模式：使用3个向量文档 + 5个图谱文档（共8个）

8. 模型接受文档生成回答
   ├── 根据路由决策选择提示词模板
   ├── 使用统一生成参数（与评估系统一致）
   └── Qwen3-1.7B模型生成最终答案

9. 输出结果
   ├── 返回答案和检索结果
   ├── 检索结果包含source字段标记（vector/graph）
   └── metadata包含selected_for_generation字段
```

### 高级配置

#### 自定义检索配置

```python
response = requests.post("http://localhost:8000/api/v1/query", json={
    "query": "头痛怎么治疗",
    "top_k": 5,
    "temperature": 0.7,
    "max_new_tokens": 768,
    "enable_vector": True,
    "enable_graph": True
})
```

#### 查询参数说明

- `query`: 用户问题（必需）
- `top_k`: 返回结果数量（默认5）
- `temperature`: 生成温度（默认0.5）
- `max_new_tokens`: 最大生成token数（默认512）
- `enable_vector`: 启用向量检索（默认True）
- `enable_graph`: 启用知识图谱检索（默认True）

## 🏗️ 系统架构

### 五层分布式架构

```
┌────────────────────────────────────────┐
│   第一层：文档层                        │
│  - README.md                            │
│  - 项目结构文档                          │
│  - API文档                              │
│  - 部署指南                              │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│   第二层：检索与知识层                   │
│  - Faiss向量数据库                      │
│  - Neo4j知识图谱                        │
│  - 嵌入模型(GTE)                        │
│  - 检索系统                             │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│   第三层：应用协调层                     │
│  - 智能路由分类器                        │
│  - 检索协调器                            │
│  - RAG链路                              │
│  - FastAPI服务                          │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│   第四层：部署与基础设施层               │
│  - Docker配置                           │
│  - 启动脚本                             │
│  - 环境配置                             │
│  - 监控脚本                             │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│   第五层：测试与质量保障层               │
│  - 单元测试                              │
│  - 集成测试                              │
│  - RAGAS评估                            │
│  - 性能测试                              │
└────────────────────────────────────────┘
```

### 核心组件

#### 1. 智能路由分类器 (IntelligentRouter)

```python
from middle.utils.intelligent_router import get_intelligent_router

router = get_intelligent_router()
route_type, confidence = router.classify("生姜的功效有哪些？")
```

**二元路由策略**：
- `vector_only`: 实体主导型查询（纯向量检索）
- `hybrid`: 复杂推理型查询（混合检索：向量50% + 图谱50%）

**路由决策依据**：
- 基于Qwen-Flash API的实体识别
- 基于查询复杂度分析（关键词数、句子长度等）
- 基于语义推理判断

#### 2. 检索协调器 (HybridRetrievalCoordinator)

```python
from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.models.data_models import RetrievalConfig

coordinator = HybridRetrievalCoordinator(
    vector_adapter=vector_adapter,
    graph_adapter=graph_adapter,
    use_intelligent_routing=True
)

config = RetrievalConfig(
    enable_vector=True,
    enable_graph=True,
    top_k=5
)

results = await coordinator.retrieve("头痛治疗", config)
```

#### 3. RAG链路 (RAGChain)

```python
from middle.services.rag_chain import RAGChain

rag_chain = RAGChain(
    retrieval_coordinator=coordinator,
    max_context_tokens=1500
)

result = await rag_chain.query("请推荐治疗失眠的中药")
```

## 📊 性能指标

| 指标类别 | 具体指标 | 数值 |
|---------|---------|------|
| **数据规模** | 图谱节点 | 47,335个 |
| | 图谱关系 | 402,184个 |
| | 向量文档数 | 155,902条 |
| **响应性能** | 检索时间 | 1.5-2.5秒 |
| | 向量查询速度 | 8.76ms |
| | 向量QPS | 114 |
| | 总响应时间 | 3.5-10.5秒 |
| **智能路由** | 路由准确率 | >85% |
| | 分类置信度 | 0.8-0.95 |
| **资源消耗** | GPU显存 | ~4GB |
| | 内存需求 | 16GB+ |

## 🔧 配置说明

### 主要配置文件

#### service_config.yaml

```yaml
# 模型配置
model:
  base_model_path: "Model Layer/model/qwen/Qwen3-1.7B/Qwen/Qwen3-1___7B"
  adapter_path: "Model Layer/model/checkpoint-7983"

# 检索配置
retrieval:
  vector:
    persist_directory: "检索与知识层/faiss_rag/向量数据库_简单查询"
    collection_name: "tcm_qa_collection"
    model_path: "BAAI/bge-base-zh"
  
  graph:
    neo4j_uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "your_password"

# RAG配置
rag:
  max_context_tokens: 2000
  max_retrieval_results: 10
  default_temperature: 0.5
```

详细配置说明请参考：[配置说明.md](docs/配置说明.md)

## 🧪 测试

### 运行测试

```bash
# 单元测试
cd 测试与质量保障层
python tests/test_system.py

# 完整测试
python main.py

# Windows批处理
run_test.bat
```

### 评估系统

```bash
cd 测试与质量保障层/rag评估系统/newragas
python hybrid_ragas_evaluator_v3.py
```

评估报告保存在 `results/` 目录。

## 📁 项目结构

```
线上智能中医问答项目/
├── Model Layer/              # 模型层
│   └── model/
│       ├── qwen/            # Qwen3-1.7B基座模型
│       ├── checkpoint-7983/  # LoRA微调模型
│       └── bert-base-chinese/ # BERT路由模型
│
├── 应用协调层/              # 应用协调层
│   └── middle/
│       ├── api/             # FastAPI接口
│       ├── core/            # 核心模块
│       ├── services/        # 服务层
│       ├── adapters/        # 适配器
│       ├── utils/           # 工具层
│       └── config/          # 配置文件
│
├── 检索与知识层/            # 检索与知识层
│   ├── faiss_rag/          # Faiss向量检索
│   ├── Graphrag/           # Neo4j知识图谱
│   └── keyword/            # 关键词库
│
├── 部署与基础设施层/        # 部署层
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── scripts/
│
├── 测试与质量保障层/        # 测试层
│   ├── tests/
│   └── rag评估系统/
│
└── 文档层/                  # 文档层
    └── docs/
        ├── 项目介绍.md
        ├── 详细项目结构.txt
        └── README.md (本文件)
```

详细结构请参考：[详细项目结构.txt](docs/详细项目结构.txt)

## 🛠️ 故障排除

### 常见问题

#### 1. 模型加载失败

**问题**：出现 `FileNotFoundError` 或模型路径错误

**解决方案**：
```bash
# 检查模型路径
ls Model\ Layer/model/qwen/Qwen3-1.7B/Qwen/Qwen3-1___7B

# 确保配置文件中的路径正确
vim 应用协调层/middle/config/service_config.yaml
```

#### 2. Neo4j连接失败

**问题**：`Neo4j connection failed`

**解决方案**：
```bash
# 检查Neo4j服务状态
neo4j status

# 启动Neo4j服务
neo4j start

# 验证连接
neo4j console
```

#### 3. 显存不足

**问题**：`CUDA out of memory`

**解决方案**：
- 降低 `max_context_tokens` 和 `max_retrieval_results`
- 使用CPU模式（在配置中设置 `device: "cpu"`）
- 使用量化模型

#### 4. Faiss查询失败

**问题**：`Faiss query error`

**解决方案**：
```bash
# 检查向量数据库路径
ls 检索与知识层/faiss_rag/向量数据库_简单查询

# 重新构建向量数据库
cd 检索与知识层/faiss_rag
python 构建向量数据库_Faiss.py
```

### 日志查看

```bash
# 应用日志
tail -f 应用协调层/middle/logs/langchain_service.log

# 评估日志
tail -f 测试与质量保障层/rag评估系统/newragas/results/hybrid_ragas_evaluation_v3.log
```

## 📚 文档导航

- [项目介绍](docs/项目介绍.md) - 完整的项目介绍和技术细节
- [详细项目结构](docs/详细项目结构.txt) - 项目结构详细说明
- [配置说明](docs/配置说明.md) - 配置文件详细说明
- [快速启动指南](docs/快速启动指南-重组后.md) - 快速上手指南
- [部署文档](docs/DEPLOYMENT.md) - 部署和运维文档

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

- 代码：遵循 Apache License 2.0（Apache-2.0）。详见 [LICENSE](LICENSE)。
- 文档：遵循 Creative Commons Attribution 4.0 International（CC BY 4.0）。详见 [LICENSE-docs](LICENSE-docs)。

Copyright (c) 2025 项目作者

## 👥 作者

智能中医问答系统开发团队

## 🙏 致谢

- Qwen团队提供优秀的基础模型
- LangChain项目提供RAG框架
- Faiss和Neo4j提供数据库支持
- 中医药学界提供专业知识支撑

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- Issue: [GitHub Issues](https://github.com/your-repo/issues)
- Email: your-email@example.com

---

**当前版本**: v4.4  
**最后更新**: 2025-12-XX  
**架构状态**: 优化RAG流程（Faiss向量 + Neo4j知识图谱 + Qwen-Flash路由）  
**模型状态**: Qwen3-1.7B微调模型（checkpoint-7983）  
**检索召回规则**:
- 纯向量检索（vector_only）：召回3个，使用3个
- 混合检索（hybrid）：召回5向量+5图谱（10个），生成用3向量+5图谱（8个）
