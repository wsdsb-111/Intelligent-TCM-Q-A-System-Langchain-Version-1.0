# 中医知识图谱构建系统

## 项目简介

这是一个专门用于构建中医知识图谱的系统，支持从中医医案文本中自动提取实体和关系，构建结构化的知识图谱。系统使用 vLLM 服务（Qwen2.5-7B-Instruct）进行智能实体识别和关系抽取，生成的知识图谱数据导入 Neo4j 后可为整个中医问答系统提供知识图谱检索能力。

## 模块定位与功能

本模块是**线上智能中医问答项目**的**知识层**核心组件之一，主要功能包括：

1. **实体与关系提取**：从中医医案文本中提取症状、疾病、证型、中药、方剂等实体及其关系
2. **知识图谱构建**：将提取的实体和关系结构化，生成 Neo4j 兼容的 CSV 文件
3. **知识库支撑**：为应用协调层提供知识图谱检索基础（217K 节点，1.6M 关系）
4. **智能推理**：支持基于知识图谱的关系推理和路径查询

## 主要特性

- 🏥 **中医专业支持**: 专门针对中医领域优化的实体和关系提取
- 🤖 **AI 驱动提取**: 基于 Qwen2.5-7B-Instruct 模型的智能实体识别
- 📊 **灵活导出**: 生成 Neo4j 兼容的 CSV 文件和导入脚本
- ⚡ **高效处理**: 支持 JSON 格式大数据集批量处理
- 🔄 **双重策略**: AI 提取 + 规则匹配的混合抽取策略
- 📈 **完整流程**: 从文本处理到图数据库导入的完整工作流
- 🆕 **神农数据集**: 支持处理大规模中医问答数据集（112K 条记录）
- ⏸️ **暂停控制**: 支持暂停、恢复、停止的交互式处理控制
- 🧵 **线程安全**: 线程安全的状态管理，支持长时间运行任务
- 📊 **实时监控**: 提供详细的处理进度和统计信息

## 项目结构

```
Graphrag/
├── src/                          # 源代码目录
│   ├── __init__.py               # 包初始化
│   ├── config.py                 # 配置管理（支持 vLLM 和 Ollama）
│   ├── graphrag_processor.py     # GraphRAG 处理器（完整版，支持暂停控制）
│   ├── simple_extractor.py       # 简化版提取器（推荐使用）
│   ├── models.py                 # 数据模型（Entity、Relationship 等）
│   ├── csv_exporter.py           # CSV 导出器
│   ├── document_processor.py     # 文档处理器
│   ├── exceptions.py             # 异常处理
│   └── logger.py                 # 日志处理
├── config/                       # 配置文件目录
│   └── config.yaml               # 主配置文件（vLLM/Ollama 配置）
├── dataset/                      # 数据集目录
│   ├── complete_extracted_dataset_20250919_200117.csv   # CSV格式数据集
│   ├── complete_extracted_dataset_20250919_200117.json  # JSON格式数据集
│   └── shennong/                 # 神农数据集目录
│       ├── ChatMed_TCM-v0.2.json    # 原始JSONL格式（112K记录）
│       └── shennong_simple.csv      # 转换后的简单CSV格式
├── Knowledge_Graph/              # 知识图谱数据库
│   └── neo4j.dump                # Neo4j 数据库备份文件（217K节点，1.6M关系）
├── neo4j_batches/                # Neo4j 导入脚本目录
│   ├── 01_import_entities.cypher            # 实体导入脚本
│   ├── 02_import_relationships_*.cypher     # 关系导入脚本（分批）
│   └── 03_final_validation_correct.cypher   # 验证脚本
├── logs/                         # 日志目录
│   └── app.log                   # 应用日志
├── output/                       # 输出目录（自动生成）
│   ├── *_entities.csv            # 提取的实体CSV文件
│   ├── *_relationships.csv       # 提取的关系CSV文件
│   └── *_neo4j_import_guide.txt  # Neo4j导入指导文件
├── convert_shennong_simple.py    # 神农数据集格式转换工具
├── extract_from_shennong_csv.py  # 从神农CSV提取实体关系（支持暂停）
├── pause_control_example.py      # 暂停控制功能示例
├── neo4j_config.py               # Neo4j 连接配置
└── requirements.txt              # Python 依赖包
```

## 实际使用示例

### 完整示例：从文本到知识图谱

以下是一个完整的工作流程示例：

```python
#!/usr/bin/env python3
"""
完整的知识图谱构建示例
从中医医案文本提取实体和关系，导出为 Neo4j 格式
"""

from src.simple_extractor import SimpleExtractor
from src.csv_exporter import CSVExporter
from src.config import SimpleConfigManager
from src.models import ProcessedDocument

def main():
    # 1. 加载配置
    print("加载配置...")
    config_manager = SimpleConfigManager()
    config = config_manager.load_config()
    
    # 2. 创建提取器和导出器
    print("初始化提取器...")
    extractor = SimpleExtractor(config.graphrag)
    exporter = CSVExporter(output_dir="output")
    
    # 3. 准备中医医案文本
    medical_case = """
    患者症状：全身乏力，短气，月经不调。
    证型：肾阴阳两虚证，脾虚气滞证。
    治法：温补肾阳，健脾益气。
    方药：桂枝15.0、白芍15.0、生姜9.0、大枣4.0、甘草6.0。
    """
    
    # 4. 创建文档对象
    document = ProcessedDocument(
        title="示例医案",
        content=medical_case
    )
    
    # 5. 提取实体和关系
    print(f"\n开始提取实体和关系...")
    result = extractor.extract_entities_and_relationships(document)
    
    print(f"✅ 提取完成:")
    print(f"   - 实体: {len(result.entities)} 个")
    print(f"   - 关系: {len(result.relationships)} 个")
    print(f"   - 耗时: {result.processing_time:.2f} 秒")
    
    # 6. 导出 CSV 文件
    print(f"\n导出 CSV 文件...")
    entities_csv, relationships_csv = exporter.export_graphrag_result(
        result, 
        source_file="example_medical_case"
    )
    
    print(f"✅ 导出完成:")
    print(f"   - 实体文件: {entities_csv}")
    print(f"   - 关系文件: {relationships_csv}")
    
    # 7. 生成 Neo4j 导入指导
    guide_path = exporter.generate_neo4j_import_guide(
        entities_csv, 
        relationships_csv
    )
    print(f"   - 导入指导: {guide_path}")
    
    # 8. 打印实体样例
    print(f"\n实体样例（前5个）:")
    for entity in result.entities[:5]:
        print(f"   - {entity.name} ({entity.type}): {entity.description}")
    
    # 9. 打印关系样例
    print(f"\n关系样例（前5个）:")
    entity_dict = {e.id: e.name for e in result.entities}
    for rel in result.relationships[:5]:
        source_name = entity_dict.get(rel.source_entity_id, "未知")
        target_name = entity_dict.get(rel.target_entity_id, "未知")
        print(f"   - {source_name} --[{rel.relationship_type}]--> {target_name}")

if __name__ == "__main__":
    main()
```

### 批量处理数据集

如果需要处理大量医案数据：

```python
import json
from pathlib import Path

def batch_process_dataset(dataset_path: str, max_records: int = 100):
    """批量处理 JSON 数据集"""
    
    # 加载配置
    config_manager = SimpleConfigManager()
    config = config_manager.load_config()
    
    extractor = SimpleExtractor(config.graphrag)
    exporter = CSVExporter()
    
    # 读取数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 限制处理数量
    records = dataset[:max_records]
    
    all_entities = []
    all_relationships = []
    
    # 处理每条记录
    for i, record in enumerate(records):
        print(f"处理记录 {i+1}/{len(records)}...")
        
        document = ProcessedDocument(
            title=f"record_{i}",
            content=record.get('content', '')
        )
        
        result = extractor.extract_entities_and_relationships(document)
        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)
    
    # 导出结果
    print(f"\n总计: {len(all_entities)} 个实体, {len(all_relationships)} 个关系")
    
    # 生成 CSV 文件...
    # 参考上面的完整示例
```

## 快速开始

### 环境准备

#### 1. 硬件要求

**推荐配置（RTX 4090 24GB）**：
- **GPU**: NVIDIA RTX 4090 24GB VRAM
- **CPU**: 16+ 核心
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间（模型 + 数据集）
- **CUDA**: 12.1+ (推荐 12.4)

**最低配置**：
- **GPU**: NVIDIA GPU with 16GB+ VRAM (如 RTX 3090, 4080)
- **CPU**: 8+ 核心
- **内存**: 16GB+ RAM

#### 2. 软件环境

**基础环境**：
```bash
# Python 版本
Python 3.8 - 3.11 (推荐 3.10)

# CUDA 工具包
CUDA 12.1 或更高版本

# 驱动版本
NVIDIA Driver 525.60.13 或更高版本
```

#### 3. 安装步骤

##### 步骤 1：创建虚拟环境（推荐）

```bash
# 使用 conda 创建环境（推荐）
conda create -n graphrag python=3.10
conda activate graphrag

# 或使用 venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

##### 步骤 2：安装 GraphRAG 依赖

```bash
# 进入 Graphrag 目录
cd 检索与知识层/Graphrag

# 安装项目依赖
pip install -r requirements.txt
```

##### 步骤 3：安装 vLLM 服务端（单独环境）

vLLM 需要在单独的环境中安装，避免依赖冲突：

```bash
# 创建 vLLM 专用环境
conda create -n vllm python=3.10
conda activate vllm

# 安装 vLLM（包含 torch、transformers 等）
pip install vllm==0.5.0 或更高版本

# 安装额外依赖
pip install ray  # 分布式支持（可选）
```

**验证 vLLM 安装**：
```bash
python -c "import vllm; print(vllm.__version__)"
```

##### 步骤 4：下载模型

```bash
# 方式 1：使用 Hugging Face CLI（推荐）
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 方式 2：使用 Python 下载
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# 模型默认保存位置：~/.cache/huggingface/hub/
```

#### 4. 启动 vLLM 服务

##### 方式 1：使用启动脚本（推荐）

**Windows**：
```bash
# 编辑 start_vllm_service.bat，修改 MODEL_PATH
# 然后双击运行或在命令行执行：
start_vllm_service.bat
```

**Linux/Mac**：
```bash
# 添加执行权限
chmod +x start_vllm_service.sh

# 运行脚本
./start_vllm_service.sh
```

##### 方式 2：手动启动

```bash
# 激活 vLLM 环境
conda activate vllm

# 启动 vLLM API 服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype half \
    --trust-remote-code
```

**启动参数说明**：
- `--model`: 模型路径或 Hugging Face 模型名称
- `--served-model-name`: API 调用时使用的模型名称
- `--host`: 服务监听地址（0.0.0.0 表示允许外部访问）
- `--port`: 服务端口（默认 8000）
- `--max-model-len`: 最大上下文长度（8192 对 7B 模型合适）
- `--gpu-memory-utilization`: GPU 显存利用率（0.85 = 85%）
- `--dtype`: 数据类型（half = FP16，节省显存）
- `--trust-remote-code`: 信任模型代码（Qwen 模型需要）

**性能优化参数**（可选）：
```bash
# 添加以下参数可进一步优化性能
--tensor-parallel-size 1        # 单卡推理
--enforce-eager                  # 快速启动（开发环境）
--disable-log-requests           # 减少日志输出
```

#### 5. 验证 vLLM 服务

**检查服务状态**：
```bash
# 检查服务是否启动
curl http://localhost:8000/v1/models

# 预期输出：
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "qwen2.5-7b-instruct",
#       "object": "model",
#       ...
#     }
#   ]
# }
```

**简单测试**：
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct",
    "prompt": "中医治疗感冒的常用方剂有：",
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

#### 6. 配置 GraphRAG

编辑 `config/config.yaml`：

```yaml
graphrag:
  # vLLM 服务配置
  use_ollama: false
  api_key: "EMPTY"                          # vLLM 不需要真实 API key
  api_base: "http://localhost:8000/v1"     # vLLM 服务地址
  model: "qwen2.5-7b-instruct"             # 与启动时的模型名称一致
  
  # 提取参数
  max_tokens: 4000
  temperature: 0.0      # 0.0 = 确定性输出
  chunk_size: 1200      # 文本分块大小
  chunk_overlap: 100    # 分块重叠大小
```

#### 7. 常见问题排查

**问题 1：vLLM 启动失败 - CUDA Out of Memory**
```bash
# 解决方案：降低显存使用率
--gpu-memory-utilization 0.75  # 从 0.85 降到 0.75
--max-model-len 4096           # 从 8192 降到 4096
```

**问题 2：导入错误 - No module named 'vllm'**
```bash
# 确认已激活 vLLM 环境
conda activate vllm
# 重新安装 vLLM
pip install vllm --upgrade
```

**问题 3：连接失败 - Connection refused**
```bash
# 检查防火墙设置
# Windows: 允许端口 8000
# Linux: sudo ufw allow 8000

# 检查服务是否在运行
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows
```

**问题 4：推理速度慢**
```bash
# 优化建议：
1. 使用 --enforce-eager 加快启动（开发环境）
2. 移除 --enforce-eager 使用 CUDA Graph（生产环境）
3. 确保使用 Flash Attention：
   export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 8. GPU 监控

```bash
# 实时监控 GPU 使用情况
watch -n 1 nvidia-smi  # Linux
# Windows: 在 PowerShell 中运行
while($true) { nvidia-smi; sleep 1; cls }
```

### 🚀 完整启动流程（新用户必读）

#### 快速启动清单

**第一次使用**，请按以下顺序完成所有步骤：

```bash
# ✅ 步骤 1：创建并激活 GraphRAG 环境
conda create -n graphrag python=3.10
conda activate graphrag
cd 检索与知识层/Graphrag
pip install -r requirements.txt

# ✅ 步骤 2：创建并激活 vLLM 环境（新终端）
conda create -n vllm python=3.10
conda activate vllm
pip install vllm>=0.5.0

# ✅ 步骤 3：下载 Qwen2.5-7B-Instruct 模型
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# ✅ 步骤 4：启动 vLLM 服务（保持运行）
# 编辑 start_vllm_service.bat 修改 MODEL_PATH
# Windows: start_vllm_service.bat
# Linux: ./start_vllm_service.sh

# ✅ 步骤 5：验证 vLLM 服务（新终端）
curl http://localhost:8000/v1/models

# ✅ 步骤 6：配置 GraphRAG（编辑 config/config.yaml）
# 确保 api_base 和 model 名称正确

# ✅ 步骤 7：运行神农数据集处理
conda activate graphrag
python convert_shennong_simple.py        # 转换数据格式
python extract_from_shennong_csv.py      # 提取实体关系
```

**后续使用**，只需：

```bash
# 终端 1：启动 vLLM 服务
conda activate vllm
start_vllm_service.bat  # 或 ./start_vllm_service.sh

# 终端 2：运行提取任务
conda activate graphrag
python extract_from_shennong_csv.py
```

### 🆕 神农数据集快速处理（推荐）

环境配置完成后，使用神农数据集快速体验系统功能：

#### 数据转换

```bash
# 激活 GraphRAG 环境
conda activate graphrag

# 转换数据格式（仅需运行一次）
python convert_shennong_simple.py
```

**输入**: `dataset/shennong/ChatMed_TCM-v0.2.json` (112K 条中医问答)  
**输出**: `dataset/shennong/shennong_simple.csv` (id, query, response 格式)

#### 实体关系提取

```bash
# 确保 vLLM 服务正在运行
# 然后执行提取脚本
python extract_from_shennong_csv.py
```

**交互式控制命令**：
```
p      - 暂停处理
r      - 恢复处理
s      - 停止处理
status - 查看当前状态
stats  - 查看统计信息
q      - 退出控制面板
```

**配置参数**（在 `extract_from_shennong_csv.py` 的 `main()` 函数中）：
```python
max_records = 50      # 处理记录数（None=全部，建议先测试 50 条）
batch_size = 5        # 每批处理数量（根据 GPU 性能调整）
```

**输出文件**：
- `output_shennong_extraction/entities_batch_*.csv` - 实体文件
- `output_shennong_extraction/relationships_batch_*.csv` - 关系文件
- `output_shennong_extraction/stats_batch_*.json` - 批次统计
- `output_shennong_extraction/final_extraction_stats.json` - 最终统计
- 详细的处理日志和进度报告

**特点**：
- ✅ 使用真实的中医问答数据（112K 条记录）
- ✅ 支持暂停、恢复、停止控制
- ✅ 自动保存处理进度
- ✅ 详细的统计和错误报告
- ✅ 支持断点续传（停止后可继续处理）

**性能参考**（RTX 4090）：
- 处理速度：约 5-10 条/分钟（取决于文本长度）
- 预计时间：50 条约 5-10 分钟
- 完整数据集：约 200-400 小时（建议分批处理）

### 配置说明

**主配置文件** (`config/config.yaml`):
```yaml
graphrag:
  # 使用 vLLM 服务（推荐）
  use_ollama: false
  api_key: "EMPTY"  # vLLM 不需要真实 API key
  api_base: "http://localhost:8000/v1"  # vLLM 服务地址
  model: "qwen2.5-7b-instruct"  # 使用的模型
  
  # Ollama 本地模型配置（可选）
  use_ollama: false
  ollama_base_url: "http://localhost:11434"
  ollama_model: "yi:6b"
  
  # 通用配置
  max_tokens: 4000
  temperature: 0.0
  chunk_size: 1200
  chunk_overlap: 100
  
csv_export:
  output_directory: "output"
  encoding: "utf-8"
  generate_import_guide: true
```

### 使用方法

#### 方式一：使用简化版程序（推荐）

简化版提取器 (`src/simple_extractor.py`) 是基于测试验证的稳定版本，处理流程清晰，易于调试。

```bash
# 启动 vLLM 服务（需要在模型层先启动）
# 然后运行简化版程序处理数据集
python -c "
from src.simple_extractor import SimpleExtractor
from src.config import SimpleConfigManager
from src.models import ProcessedDocument

# 加载配置
config_manager = SimpleConfigManager()
config = config_manager.load_config()

# 创建提取器
extractor = SimpleExtractor(config.graphrag)

# 处理文档（示例）
document = ProcessedDocument(
    title='test_document',
    content='患者症状：全身乏力。证型：脾虚气滞证。治法：温补肾阳。'
)

result = extractor.extract_entities_and_relationships(document)
print(f'提取实体：{len(result.entities)}个')
print(f'提取关系：{len(result.relationships)}个')
"
```

#### 方式二：使用完整版处理器

完整版处理器 (`src/graphrag_processor.py`) 支持更复杂的文本处理和分块策略。

```bash
# 运行完整版处理器
python -c "
from src.graphrag_processor import GraphRAGProcessor
from src.config import SimpleConfigManager

config_manager = SimpleConfigManager()
config = config_manager.load_config()

processor = GraphRAGProcessor(config.graphrag)
# 使用 processor 处理文档...
"
```

## 神农数据集处理（新功能）

### 数据集转换

系统现已支持处理神农 ChatMed_TCM-v0.2 数据集（112K 条中医问答记录）。

#### 步骤 1：转换数据格式

将 JSONL 格式转换为简单的 CSV 格式：

```bash
python convert_shennong_simple.py
```

**输入**：`dataset/shennong/ChatMed_TCM-v0.2.json`（JSONL格式）  
**输出**：`dataset/shennong/shennong_simple.csv`（包含 id, query, response 三列）

转换后的 CSV 格式：
```csv
"id","query","response"
"SHENNONG_000001","我腹痛，没有其他症状...","首先需要确定腹痛的性质..."
"SHENNONG_000002","阴疮无其他症状...","很抱歉，阴疮是一个模糊..."
```

#### 步骤 2：提取实体和关系

使用提取系统从 CSV 文件中提取实体和关系：

```bash
python extract_from_shennong_csv.py
```

**主要功能**：
- ✅ 自动读取 `shennong_simple.csv` 文件
- ✅ 批量提取实体和关系（支持配置批次大小）
- ✅ 支持暂停、恢复、停止功能
- ✅ 自动保存提取结果为 CSV 格式
- ✅ 生成详细的统计报告

**配置参数**（在脚本 `main()` 函数中）：
```python
max_records = 50      # 处理记录数（None=处理全部）
batch_size = 5        # 每批处理5条记录
```

**输出文件**：
- `output_shennong_extraction/entities_batch_*.csv` - 实体文件
- `output_shennong_extraction/relationships_batch_*.csv` - 关系文件  
- `output_shennong_extraction/stats_batch_*.json` - 批次统计
- `output_shennong_extraction/final_extraction_stats.json` - 最终统计

### 交互式控制

运行提取脚本时，可以使用交互式命令控制处理过程：

```
命令:
  p - 暂停处理
  r - 恢复处理
  s - 停止处理
  status - 查看当前状态
  stats - 查看统计信息
  q - 退出控制面板
```

**使用场景**：
- 需要临时暂停处理查看日志
- 需要调整配置后继续处理
- 发现问题需要立即停止

### 暂停控制功能详解

`GraphRAGProcessor` 现在支持完整的暂停控制功能：

```python
from src.graphrag_processor import GraphRAGProcessor
from src.config import SimpleConfigManager

# 初始化
config_manager = SimpleConfigManager()
config = config_manager.load_config()
processor = GraphRAGProcessor(config.graphrag)

# 控制方法
processor.pause_processing()     # 暂停
processor.resume_processing()    # 恢复
processor.stop_processing()      # 停止
processor.reset_processing_state()  # 重置状态
processor.get_processing_status()   # 获取状态（"运行"/"暂停"/"停止"）
```

**暂停控制特点**：
- 🔄 **线程安全**：使用线程锁确保状态一致性
- ⏸️ **响应迅速**：在每个文本块处理前检查暂停状态
- 💾 **保存进度**：停止时保存已处理的部分结果
- 📊 **进度追踪**：实时显示处理进度和统计信息

**暂停控制示例**：

查看 `pause_control_example.py` 了解完整使用示例：

```bash
python pause_control_example.py
```

## 支持的实体类型

系统支持以下中医专业实体类型（最新更新为 8 大类别）：

### 最新实体类型（2024更新）
- **BASIC_THEORY**: 基础理论（阴阳学说、五行理论、经络学说等）
- **DISEASE**: 病症（疾病名称、证候、症状等）
- **HERB**: 中药（中药名称）
- **FORMULA**: 方剂（方剂名称）
- **THERAPY**: 疗法（治疗方法、技术手法等）
- **DIAGNOSIS**: 诊断（诊断方法、检查手段等）
- **LITERATURE**: 文献（中医典籍、文献名称）
- **PERSON**: 人物（名医、人物姓名）

### 传统实体类型（向后兼容）
- **PERSON**: 人物（医生、患者、历史名医等）
- **ORGANIZATION**: 组织机构（医院、医学院、药房等）
- **GEO**: 地理位置（产地、医院位置等）
- **EVENT**: 事件（诊疗过程、学术会议等）
- **CONCEPT**: 概念（理论、学说等）
- **OBJECT**: 物品（医疗器械、工具等）
- **TIME**: 时间（朝代、时期等）

### 中医特有实体类型（传统）
- **SYMPTOM**: 症状（发热、头痛、咳嗽、气喘等）
- **DISEASE**: 疾病（感冒、高血压、糖尿病等）
- **SYNDROME**: 证候/证型（脾虚气滞证、肾阳虚证、肝郁气滞证等）
- **HERB**: 中药（桂枝、附子、人参、黄芪、当归等）
- **FORMULA**: 方剂（桂枝汤、四君子汤、六味地黄丸等）
- **ACUPOINT**: 穴位（足三里、百会、合谷等）
- **MERIDIAN**: 经络（手太阴肺经、足阳明胃经等）
- **ORGAN**: 脏腑（心、肝、脾、肺、肾等）
- **THEORY**: 理论（阴阳学说、五行理论等）
- **METHOD**: 治法（温补肾阳、清热解毒、活血化瘀等）
- **TECHNIQUE**: 技术手法（针灸、推拿、拔罐等）

## 支持的关系类型

### 最新关系类型（10大核心关系，2025更新）

系统现支持以下 10 大核心关系类型及其子类型：

#### 1. 关联关系
- **关联经络**: 实体与经络的关联
- **对应证候**: 症状对应的证候

#### 2. 治疗关系
- **常用中药**: 疾病的常用治疗药物
- **适用疗法**: 适用的治疗方法
- **治疗**: 一般治疗关系

#### 3. 组成关系
- **组成中药**: 方剂的组成药物
- **涉及穴位**: 疗法涉及的穴位
- **包含**: 一般包含关系

#### 4. 归属关系
- **源于方剂**: 药物或疗法来源的方剂
- **归属病症**: 症状归属的疾病
- **归属**: 一般归属关系

#### 5. 功效关系
- **核心功效**: 药物或方剂的核心功效
- **主要作用**: 主要治疗作用
- **功效**: 一般功效关系

#### 6. 诊断关系
- **用于诊断**: 诊断方法的应用
- **基于症状**: 诊断依据的症状
- **诊断**: 一般诊断关系

#### 7. 文献关系
- **记载实体**: 文献中记载的实体
- **著有文献**: 人物著作的文献
- **记载**: 一般记载关系

#### 8. 禁忌关系
- **配伍禁忌**: 药物配伍禁忌
- **禁忌疗法**: 不适合的疗法
- **禁忌**: 一般禁忌关系

#### 9. 体质关系
- **易患病症**: 某体质易患的疾病
- **适合体质**: 药物或疗法适合的体质
- **体质**: 一般体质关系

#### 10. 传承关系
- **创立方剂**: 人物创立的方剂
- **传承自**: 学术传承关系
- **传承**: 一般传承关系

### 传统关系类型（向后兼容）

## 传统支持的关系类型

系统支持以下关系类型（定义于 `src/models.py`，支持中英文）：

### 治疗与诊断关系
- **治疗/TREATS**: 药物/方法治疗疾病、症状或证型
- **诊断/DIAGNOSES**: 诊断疾病或证型
- **开方/PRESCRIBES**: 开具方剂
- **表现为**: 证型表现为症状
- **引起/CAUSES**: 疾病引起症状

### 组成与配伍关系
- **包含/CONTAINS**: 方剂包含中药
- **配伍**: 中药配伍关系
- **相生**: 中药相生关系
- **相克**: 中药相克关系
- **配伍禁忌**: 中药配伍禁忌

### 药物属性关系
- **归经**: 药物归入经络
- **功效**: 药物功效
- **主治**: 药物主治疾病
- **性味**: 药物性味属性
- **炮制**: 中药炮制方法
- **产地**: 中药产地

### 其他关系
- **适应症/INDICATES**: 药物适用疾病
- **禁忌症/CONTRAINDICATES**: 药物禁忌情况
- **相互作用/INTERACTS_WITH**: 药物相互作用
- **用法**: 药物用法
- **用量**: 药物用量
- **相关/RELATED_TO**: 一般关联关系

## 系统要求

### 本地环境
- **Python**: 3.8+
- **内存**: 8GB+（使用Yi:6B模型）
- **存储**: 10GB+（模型和输出文件）

### AutoDL RTX 4090云端环境
- **GPU**: RTX 4090 24GB VRAM
- **CPU**: 16核心
- **内存**: 120GB
- **环境**: Ubuntu 22.04 + Python 3.12 + PyTorch 2.8.0 + CUDA 12.8

## 核心工作流程

### 1. 实体与关系提取

系统采用 **AI 提取 + 规则匹配** 的混合策略：

#### AI 驱动提取（主要方式）
```python
# 使用 vLLM API 调用 Qwen2.5-7B-Instruct 模型
# 1. 实体提取：识别中药、症状、证型、方剂等
# 2. 关系提取：识别实体间的语义关系
```

#### 规则匹配提取（备选方式）
```python
# 当 AI 提取失败或结果为空时，使用规则匹配
# 基于中医领域知识库和关键词匹配
# 提取常见的中药、证型、症状等实体
```

### 2. CSV 导出

提取的实体和关系会被导出为 Neo4j 兼容的 CSV 格式：

- `*_entities.csv`：包含实体 ID、名称、类型、描述、置信度等
- `*_relationships.csv`：包含源实体 ID、目标实体 ID、关系类型、描述、置信度等
- `*_neo4j_import_guide.txt`：Neo4j 导入指导文档，包含完整的 Cypher 脚本

### 3. Neo4j 导入

#### 自动导入（使用生成的脚本）
```bash
# 1. 将 CSV 文件复制到 Neo4j import 目录
cp output/*.csv /path/to/neo4j/import/

# 2. 使用 neo4j_batches 中的 Cypher 脚本导入
# 在 Neo4j Browser 或 cypher-shell 中执行
cypher-shell < neo4j_batches/01_import_entities.cypher
cypher-shell < neo4j_batches/02_import_relationships_correct.cypher
cypher-shell < neo4j_batches/03_final_validation_correct.cypher
```

#### 手动导入（使用导入指导）
```bash
# 按照 output/*_neo4j_import_guide.txt 中的步骤操作
# 1. 复制 CSV 文件
# 2. 执行实体导入 Cypher 语句
# 3. 执行关系导入 Cypher 语句
# 4. 验证导入结果
```

### 4. 与应用协调层集成

导入 Neo4j 后，知识图谱会被应用协调层的 `GraphRetrievalAdapter` 使用：

```python
# 应用协调层/langchain/adapters/graph_adapter.py
from neo4j import GraphDatabase

class GraphRetrievalAdapter:
    """知识图谱检索适配器"""
    
    def __init__(self, neo4j_uri, username, password, database):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
        self.database = database
    
    def search_entities(self, entity_name):
        """搜索实体"""
        query = "MATCH (e:Entity {name: $name}) RETURN e"
        # 执行查询...
    
    def find_relationships(self, source_id, target_id):
        """查找实体间关系"""
        query = """
        MATCH (s:Entity {id: $source_id})-[r]->(t:Entity {id: $target_id})
        RETURN r
        """
        # 执行查询...
```

### 完整处理流程图

```
[中医医案文本] 
    ↓
[实体提取] → AI提取 (Qwen2.5-7B) / 规则匹配
    ↓
[关系提取] → 基于实体的关系推理
    ↓
[CSV导出] → entities.csv + relationships.csv
    ↓
[Neo4j导入] → 使用Cypher脚本批量导入
    ↓
[知识图谱数据库] → 217K节点 + 1.6M关系
    ↓
[应用协调层] → GraphRetrievalAdapter 提供检索服务
    ↓
[混合检索] → BM25 + 向量检索 + 知识图谱检索
```

## 技术特点

### 混合提取策略

系统采用双重提取策略，确保高召回率和准确率：

1. **AI 驱动提取**（主要方式）
   - 使用 Qwen2.5-7B-Instruct 模型
   - 通过 vLLM 服务提供高性能推理
   - 支持复杂的语义理解和关系推理
   - 自动识别多样化的实体类型

2. **规则匹配提取**（备选方式）
   - 基于中医领域知识库
   - 包含常见中药、证型、症状词典
   - 在 AI 提取失败时作为兜底方案
   - 保证基础实体的识别准确率

### 关键代码模块

#### 1. 简化版提取器（推荐）
```python
# src/simple_extractor.py
class SimpleExtractor:
    """简化版提取器，专注于快速、准确的实体和关系提取"""
    
    def extract_entities_and_relationships(self, document):
        # 1. 提取实体
        entities = self._extract_entities_from_text(document.content, document.id)
        
        # 2. 基于实体提取关系
        relationships = self._extract_relationships_from_entities(
            entities, document.content, document.id
        )
        
        return GraphRAGResult(
            document_id=document.id,
            entities=entities,
            relationships=relationships
        )
```

#### 2. 完整版处理器
```python
# src/graphrag_processor.py
class GraphRAGProcessor:
    """完整版 GraphRAG 处理器，支持文本分块和批处理"""
    
    def extract_entities_and_relationships(self, document):
        # 1. 分块处理长文档
        chunks = self._split_text_into_chunks(document.content)
        
        # 2. 对每个块提取实体和关系
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            # AI 提取
            entities = self._extract_entities_from_chunk(chunk, document.id)
            relationships = self._extract_relationships_from_chunk(chunk, entities)
            
            # 如果提取失败，使用规则匹配
            if not entities:
                entities = self._extract_entities_by_rules(chunk)
            if not relationships:
                relationships = self._extract_relationships_by_rules(chunk, entities)
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # 3. 合并相似实体
        merged_entities = self._merge_similar_entities(all_entities)
        
        return GraphRAGResult(
            document_id=document.id,
            entities=merged_entities,
            relationships=all_relationships
        )
```

#### 3. CSV 导出器
```python
# src/csv_exporter.py
class CSVExporter:
    """CSV 导出器，生成 Neo4j 兼容的 CSV 文件"""
    
    def export_entities(self, entities, filename_prefix):
        """导出实体到 CSV"""
        # 生成实体 CSV 文件
        
    def export_relationships(self, relationships, entities, filename_prefix):
        """导出关系到 CSV"""
        # 生成关系 CSV 文件
        
    def generate_neo4j_import_guide(self, entities_csv, relationships_csv):
        """生成 Neo4j 导入指导文档"""
        # 生成包含 Cypher 脚本的导入指导
```

### 数据模型

系统使用以下核心数据模型（定义于 `src/models.py`）：

```python
@dataclass
class Entity:
    """实体数据模型"""
    id: str                      # 唯一标识符
    name: str                    # 实体名称
    type: str                    # 实体类型（HERB, SYMPTOM, etc.）
    description: str             # 实体描述
    confidence: float            # 置信度 (0.0-1.0)
    source_document_id: str      # 来源文档ID

@dataclass
class Relationship:
    """关系数据模型"""
    id: str                      # 唯一标识符
    source_entity_id: str        # 源实体ID
    target_entity_id: str        # 目标实体ID
    relationship_type: str       # 关系类型（治疗、包含、etc.）
    description: str             # 关系描述
    weight: float                # 关系权重
    confidence: float            # 置信度
    source_document_id: str      # 来源文档ID

@dataclass
class GraphRAGResult:
    """GraphRAG 处理结果"""
    document_id: str             # 文档ID
    entities: List[Entity]       # 提取的实体列表
    relationships: List[Relationship]  # 提取的关系列表
    processing_time: float       # 处理耗时
    metadata: Dict[str, Any]     # 元数据
```

## 核心文件说明

### 源代码
- 📦 `src/models.py`: 核心数据模型（Entity, Relationship, GraphRAGResult）
- 🔧 `src/simple_extractor.py`: 简化版提取器（推荐使用）
- 🔄 `src/graphrag_processor.py`: 完整版处理器（支持文本分块）
- 📤 `src/csv_exporter.py`: CSV 导出器
- ⚙️ `src/config.py`: 配置管理（支持 vLLM 和 Ollama）
- 🗂️ `src/document_processor.py`: 文档处理器
- ⚠️ `src/exceptions.py`: 异常定义
- 📝 `src/logger.py`: 日志配置

### 配置文件
- 🔧 `config/config.yaml`: 主配置文件（vLLM/Ollama 设置）
- 🔌 `neo4j_config.py`: Neo4j 数据库连接配置

### 数据文件
- 📊 `dataset/*.json`: JSON 格式数据集
- 💾 `Knowledge_Graph/neo4j.dump`: Neo4j 数据库备份（217K 节点）
- 📜 `neo4j_batches/*.cypher`: Neo4j 导入脚本

### 输出文件
- 📁 `output/*_entities.csv`: 提取的实体数据
- 📁 `output/*_relationships.csv`: 提取的关系数据
- 📋 `output/*_neo4j_import_guide.txt`: Neo4j 导入指导

## Neo4j 知识图谱查询示例

导入数据后，可以在 Neo4j Browser 中执行以下查询：

### 基础查询

```cypher
// 1. 查看所有实体数量
MATCH (e:Entity) 
RETURN count(e) as entity_count

// 2. 查看所有关系数量
MATCH ()-[r]->() 
RETURN count(r) as relationship_count

// 3. 查看实体类型分布
MATCH (e:Entity) 
RETURN e.type, count(e) as count 
ORDER BY count DESC

// 4. 查看关系类型分布
MATCH ()-[r]->() 
RETURN type(r), count(r) as count 
ORDER BY count DESC
```

### 中医专业查询

```cypher
// 1. 查找特定症状的治疗方法
MATCH (h:Entity {type: 'HERB'})-[r:治疗]->(s:Entity {type: 'SYMPTOM'})
WHERE s.name CONTAINS '乏力'
RETURN h.name, s.name, r.description
LIMIT 10

// 2. 查找方剂的组成成分
MATCH (f:Entity {type: 'FORMULA'})-[r:包含]->(h:Entity {type: 'HERB'})
WHERE f.name = '桂枝汤'
RETURN f.name, h.name

// 3. 查找证型的典型症状
MATCH (syn:Entity {type: 'SYNDROME'})-[r:表现为]->(sym:Entity {type: 'SYMPTOM'})
WHERE syn.name CONTAINS '脾虚'
RETURN syn.name, collect(sym.name) as symptoms

// 4. 查找中药的治疗范围
MATCH (h:Entity {type: 'HERB'})-[r]->(target:Entity)
WHERE h.name = '桂枝' AND r.relationship_type IN ['治疗', '主治']
RETURN target.type, target.name, r.relationship_type
```

### 关系路径查询

```cypher
// 1. 查找症状到治疗方法的路径
MATCH path = (symptom:Entity {type: 'SYMPTOM'})-[*1..3]->(treatment:Entity {type: 'METHOD'})
WHERE symptom.name = '头痛'
RETURN path
LIMIT 5

// 2. 查找实体间的最短路径
MATCH path = shortestPath(
  (a:Entity {name: '桂枝'})-[*1..5]-(b:Entity {name: '感冒'})
)
RETURN path

// 3. 查找相关实体网络
MATCH (center:Entity {name: '脾虚气滞证'})-[r*1..2]-(related:Entity)
RETURN center, r, related
LIMIT 50
```

## 与应用协调层的集成

本模块生成的知识图谱数据会被应用协调层使用：

### 集成点

1. **配置文件**：`应用协调层/langchain/config/service_config.yaml`
```yaml
retrieval:
  modules:
    graph:
      neo4j_uri: "bolt://localhost:7687"
      username: "neo4j"
      password: "your_password"
      database: "neo4j"
      dump_path: "../../../检索与知识层/Graphrag/Knowledge_Graph/neo4j.dump"
  enable_graph: true  # 启用知识图谱检索
```

2. **检索适配器**：`应用协调层/langchain/adapters/graph_adapter.py`
```python
class GraphRetrievalAdapter:
    """知识图谱检索适配器"""
    
    def retrieve(self, query: str, top_k: int = 5):
        """执行知识图谱检索"""
        # 1. 实体识别
        entities = self.extract_entities(query)
        
        # 2. 图谱查询
        results = []
        for entity in entities:
            # 查询实体及其关系
            cypher = """
            MATCH (e:Entity {name: $entity_name})-[r*1..2]-(related:Entity)
            RETURN e, r, related
            LIMIT $limit
            """
            # 执行查询并处理结果...
        
        return results
```

3. **混合检索**：`应用协调层/langchain/core/retrieval_coordinator.py`
```python
class RetrievalCoordinator:
    """检索协调器，整合 BM25、向量检索和知识图谱"""
    
    async def hybrid_search(self, query: str):
        # 并行执行三种检索
        bm25_results = await self.bm25_adapter.retrieve(query)
        vector_results = await self.vector_adapter.retrieve(query)
        graph_results = await self.graph_adapter.retrieve(query)  # 使用知识图谱
        
        # 融合结果
        merged_results = self.merge_results(bm25_results, vector_results, graph_results)
        return merged_results
```

## 技术支持

### 常见问题

#### 1. vLLM 服务连接失败
```bash
# 检查 vLLM 服务是否启动
curl http://localhost:8000/v1/models

# 检查配置文件中的 api_base 是否正确
```

#### 2. Neo4j 导入失败
```bash
# 确认 CSV 文件在 Neo4j import 目录中
# 确认 Neo4j 数据库已启动
# 检查 Cypher 脚本中的文件名是否与实际文件名匹配
```

#### 3. 实体提取结果为空
```bash
# 检查日志文件：logs/app.log
# 确认 vLLM 服务正常运行
# 尝试使用规则匹配模式
```

### 日志和调试

```bash
# 查看应用日志
tail -f logs/app.log

# 查看详细的提取过程
# 在代码中设置日志级别为 DEBUG
```

### 联系方式

如果遇到问题，请：

1. 查看日志文件：`logs/app.log`
2. 检查配置文件：`config/config.yaml`
3. 查看 Neo4j 连接配置：`neo4j_config.py`
4. 参考应用协调层文档：`应用协调层/langchain/README.md`

## 依赖说明

主要依赖包（详见 `requirements.txt`）：

### 核心依赖
- `pandas>=1.5.0`：数据处理
- `numpy>=1.21.0`：数值计算
- `pyyaml>=6.0`：配置文件解析
- `python-dotenv>=1.0.0`：环境变量管理

### AI 模型相关
- `torch>=2.8.0`：深度学习框架
- `transformers>=4.40.0`：模型加载
- `openai>=1.0.0`：OpenAI API 客户端（用于 vLLM）
- `tiktoken>=0.5.0`：Token 计数

### Neo4j 相关
- `neo4j>=5.0.0`：Neo4j Python 驱动
- `py2neo>=2021.2.3`：Neo4j 高级接口

### 其他工具
- `networkx>=3.0`：图算法
- `scikit-learn>=1.3.0`：机器学习工具

## 更新记录

### 2025-10-16 重大更新

#### 🆕 新增功能
- **神农数据集支持**：支持处理 ChatMed_TCM-v0.2 数据集（112K 条记录）
- **数据格式转换**：提供 `convert_shennong_simple.py` 转换工具
- **批量提取脚本**：新增 `extract_from_shennong_csv.py` 专用提取脚本
- **暂停控制功能**：实现完整的暂停、恢复、停止控制机制
- **交互式控制面板**：支持实时进度监控和状态管理

#### 🔄 核心优化
- **实体关系类型升级**：更新为 8 大核心实体类型 + 10 大核心关系类型
- **AI 提取优化**：改进 prompt 模板，提升提取准确性
- **规则匹配增强**：扩展规则库，覆盖更多中医专业术语
- **线程安全控制**：实现线程安全的暂停控制机制

#### 📚 文档完善
- **README.md 全面重构**：添加神农数据集处理和暂停控制章节
- **使用示例更新**：提供完整的数据处理和提取流程示例
- **配置指南优化**：详细说明新功能的配置和使用方法

### 2025-10-16 早期更新
- 📝 更新 README.md，反映实际使用情况
- 🎯 明确模块定位：知识层核心组件
- 📊 补充 Neo4j 查询示例和应用协调层集成说明
- 🔧 更新配置说明，强调 vLLM 服务使用

### 2025-09-21
- 🚀 新增简化版提取器 (`src/simple_extractor.py`)
- ✅ 优化实体和关系提取算法，支持多样化实体类型
- 📋 自动生成 Neo4j 导入指导文档
- 🔄 使用 Qwen2.5-7B-Instruct 模型（通过 vLLM）
- 🧹 清理项目结构，移除冗余文件
- 🔍 新增质量检测工具（基础、高级、快速）
- 📊 支持质量可视化分析和评分

### 2025-09-14
- 🎯 专门针对中医领域优化实体和关系提取
- 🏥 定义中医专业实体类型和关系类型
- 🔧 集成 Ollama 支持（可选）

## 项目贡献

本模块是**线上智能中医问答项目**的重要组成部分：

- **知识构建**：从中医医案文本中提取结构化知识
- **图谱支撑**：为混合检索系统提供知识图谱基础（217K 节点，1.6M 关系）
- **智能推理**：支持基于图谱的关系推理和路径查询
- **检索增强**：与 BM25、向量检索协同工作，提供更全面的检索结果

## 相关文档

- **应用协调层文档**：`应用协调层/langchain/README.md`
- **混合检索配置**：`应用协调层/langchain/config/hybrid_retrieval.yaml`
- **图检索适配器**：`应用协调层/langchain/adapters/graph_adapter.py`
- **Neo4j 配置**：`neo4j_config.py`

## 许可证

本项目采用开源许可证，详见 LICENSE 文件。
