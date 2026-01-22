# Dify工作流模型生成参数配置指南

## 概述

本文档基于**线上智能中医问答项目**的实际模型配置（`应用协调层/middle/config/service_config.yaml`），为在Dify工作流中部署本地微调模型（Qwen3-1.7B + LoRA适配器 checkpoint-7983）提供准确的参数配置建议。

### 参考配置来源

1. **主配置文件**：`应用协调层/middle/config/service_config.yaml`
2. **评估系统配置**：`测试与质量保障层/rag评估系统/newragas/config.py`
3. **模型基础配置**：`Model Layer/model/qwen/Qwen3-1.7B/Qwen/Qwen3-1___7B/config.json`

### 模型信息

- **基础模型**：Qwen3-1.7B
- **微调适配器**：checkpoint-7983 (LoRA)
- **模型层数**：28层
- **上下文长度**：最大支持 40,960 tokens，实际使用 4,096 tokens
- **设备**：CUDA (GPU推理)

---

## 第一部分：生成参数配置（第一张图片）

### 1. 温度 (Temperature)
- **当前截图值**：`0.7` ✅ 已启用
- **项目实际配置**：`0.1`（主配置）、`0.1`（评估配置）
- **推荐值**：`0.1` - `0.3`
- **说明**：
  - 项目配置使用**极低温度（0.1）**以最大化确定性，减少中医问答中的幻觉风险
  - 由于系统已通过混合检索提供准确上下文，低温度可确保答案严格基于检索内容
  - 建议：**启用，设置为 `0.1`**

### 2. Top P
- **当前截图值**：`0.9` ❌ 未启用
- **项目实际配置**：`0.4`（主配置）、`0.3`（评估配置）
- **推荐值**：`0.3` - `0.4`（与项目保持一致）
- **说明**：
  - 项目使用较低top_p值以配合低温度，进一步减少随机性
  - 建议：**启用，设置为 `0.4`**（与主配置一致）

### 3. Top K
- **当前截图值**：`1` ❌ 未启用
- **模型默认配置**：`20`（generation_config.json）
- **推荐值**：`0`（禁用）或 `20`（如启用）
- **说明**：
  - 项目主配置使用`num_beams=3`（束搜索），不使用Top K采样
  - 如果Dify不支持束搜索，可启用Top K并设为`20`（模型默认值）
  - 建议：**保持禁用（0）**，优先使用Top P控制

### 4. Repeat Penalty（重复惩罚）
- **当前截图值**：`-2` ❌ 未启用（⚠️ 错误值）
- **项目实际配置**：`1.3`（主配置）、`1.2`（评估配置）
- **推荐值**：`1.2` - `1.3`（✅ **必须启用**）
- **说明**：
  - 截图中的`-2`值会**鼓励重复**，这是错误的配置
  - 项目配置使用`1.3`以有效抑制重复，提升答案质量
  - 建议：**启用，设置为 `1.3`**（与主配置一致）

### 5. 最大令牌数预测 (Max Tokens)
- **当前截图值**：`512` ❌ 未启用
- **项目实际配置**：`512`（主配置）、`1024`（评估配置）
- **推荐值**：`512`（标准答案长度）或 `768` - `1024`（详细回答）
- **说明**：
  - 主配置使用`512` tokens（约250-400字），适合标准问答
  - 评估配置使用`1024`以生成更详细答案（约500-800字）
  - 建议：**启用，设置为 `512`**（与主配置一致），如需更详细答案可调至`768`或`1024`

### 6. Mirostat 采样 (Mirostat Sampling)
- **当前截图值**：`0` ❌ 未启用
- **推荐值**：`0`（禁用）
- **说明**：
  - 项目不使用Mirostat采样，使用传统的Temperature + Top P组合
  - 建议：**保持禁用（0）**

### 7. 学习率 (Learning Rate)
- **当前截图值**：`0` ❌ 未启用
- **推荐值**：`0`（始终禁用）
- **说明**：
  - 学习率是**训练参数**，与模型推理（生成）无关
  - 建议：**保持禁用（0）**

---

## 第二部分：模型基础配置（第二张图片）

### 1. 文本连贯度 (Text Coherence)
- **当前值**：`0`
- **推荐值**：`0`（保持默认）
- **说明**：项目未使用此参数，保持默认即可

### 2. 上下文窗口大小 (Context Window Size)
- **当前值**：`2048`
- **项目实际配置**：`4096`（评估系统）
- **模型最大支持**：`40960`
- **推荐值**：`4096`
- **说明**：
  - 评估系统使用`4096`作为上下文窗口，确保完整上下文不被截断
  - 项目中的检索结果通常包含3-8个文档，加上提示词，4096足够
  - 建议：**修改为 `4096`**

### 3. GPU 层数 (GPU Layers)
- **当前值**：`1`
- **模型层数**：28层
- **推荐值**：`28`（全部加载到GPU）或根据显存调整
- **说明**：
  - 项目配置使用`device: "cuda"`，模型全部加载到GPU
  - 如果显存充足（≥8GB），建议设置为`28`（全部层）
  - 如果显存不足，可减少GPU层数，剩余层使用CPU
  - 建议：**根据显存情况设置，推荐 `28`（全部）**

### 4. 线程数 (Number of Threads)
- **当前值**：`1`
- **推荐值**：`4` - `8`
- **说明**：
  - 项目未明确指定线程数，但单线程会限制CPU推理性能
  - 对于CPU卸载层或多线程推理，建议设置为`4-8`
  - 建议：**设置为 `4`**

### 5. 回溯内容 (Rollback Content)
- **当前值**：`-1`
- **推荐值**：`-1`（保持默认，表示禁用）
- **说明**：项目未使用此参数，保持默认即可

### 6. 减少标记影响 (Reduce Tag Influence)
- **当前值**：`0`
- **推荐值**：`0`（保持默认）
- **说明**：项目未使用此参数，保持默认即可

### 7. 随机数种子 (Random Seed)
- **当前值**：`0`
- **推荐值**：`0`（用于测试）或固定值如`42`（用于生产）
- **说明**：
  - 设置为固定值（如`42`）可确保结果可复现，便于调试
  - 设置为`0`表示使用随机种子
  - 建议：**测试时使用 `0`，生产环境建议使用固定值如 `42`**

### 8. 模型存活时间 (Model Lifetime)
- **当前值**：（空）
- **推荐值**：`300`（5分钟）或 `600`（10分钟）
- **说明**：
  - 如果Dify支持模型缓存，设置存活时间可优化性能
  - 建议：**设置为 `300`（5分钟）或根据使用频率调整**

### 9. 返回格式 (Return Format)
- **当前值**：`请选择`
- **推荐值**：`json`（JSON格式）
- **说明**：
  - **注意**：Dify工作流在此处仅支持JSON格式返回
  - 虽然项目原始配置返回纯文本，但在Dify中需要适配为JSON格式
  - 建议：**选择 `json`** 或根据下拉菜单选择JSON相关选项

### 10. JSON Schema
- **当前值**：`json`（已填写）
- **推荐值**：使用项目实际的JSON Schema结构
- **说明**：
  - Dify要求返回JSON格式，需要定义JSON Schema规范输出结构
  - **项目实际使用的JSON Schema**（基于`应用协调层/middle/api/schemas.py`）：
    ```json
    {
      "type": "object",
      "properties": {
        "success": {
          "type": "boolean",
          "description": "是否成功"
        },
        "query": {
          "type": "string",
          "description": "用户问题"
        },
        "answer": {
          "type": "string",
          "description": "生成的答案"
        },
        "retrieval_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "content": {"type": "string", "description": "文档内容"},
              "fused_score": {"type": "number", "description": "融合后评分"},
              "source": {"type": "string", "description": "检索源类型（vector/graph）"},
              "source_scores": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "各源评分"
              },
              "contributing_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "贡献的来源"
              },
              "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "实体列表"
              },
              "relationships": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关系列表"
              }
            },
            "required": ["content", "fused_score"]
          },
          "description": "检索结果列表"
        },
        "metadata": {
          "type": "object",
          "properties": {
            "retrieval_time": {"type": "number", "description": "检索耗时（秒）"},
            "generation_time": {"type": "number", "description": "生成耗时（秒）"},
            "total_time": {"type": "number", "description": "总耗时（秒）"},
            "num_retrieval_results": {"type": "integer", "description": "检索结果数量"},
            "model": {"type": "string", "description": "使用的模型"},
            "temperature": {"type": "number", "description": "生成温度"},
            "routing_decision": {"type": "string", "enum": ["vector_only", "hybrid"], "description": "路由决策"},
            "routing_confidence": {"type": "number", "description": "路由置信度"},
            "tokens_generated": {"type": "integer", "description": "生成的token数"},
            "tokens_per_second": {"type": "number", "description": "生成速度"},
            "gpu_memory_used": {"type": "string", "description": "GPU显存占用"}
          },
          "description": "元数据信息"
        },
        "error": {
          "type": "string",
          "description": "错误信息"
        }
      },
      "required": ["success", "query"]
    }
    ```
  - 建议：**使用上述完整的JSON Schema结构**，确保与项目API响应格式一致

---

## 第三部分：停止序列与思考模式（第三张图片）

### 1. 思考模式 (Thinking Mode)
- **当前状态**：✅ 已启用（True）
- **Qwen3模型特性**：支持思考模式（`enable_thinking=True`）
- **推荐值**：根据需求选择
- **说明**：
  - **启用（True）**：适合复杂推理问题，模型会生成推理过程
  - **禁用（False）**：直接生成答案，响应更快
  - 对于中医问答，**建议启用**，有助于展示推理过程，提升可解释性
  - 建议：**保持启用（True）**

### 2. 停止序列 (Stop Sequence)
- **当前值**：空
- **项目实际配置**：
  - 评估系统：`["\n\n", "<|im_end|>"]`
  - Qwen3默认：`[151645, 151643]`（EOS token IDs）
- **推荐值**：`<|im_end|>` 或 `</think>`
- **说明**：
  - 如果启用思考模式，建议添加`</think>`作为停止序列
  - 标准停止序列：`<|im_end|>`
  - 建议：**在左侧输入框添加 `<|im_end|>`**，在右侧输入框添加 `</think>`（如果启用思考模式）

---

## 总结：推荐配置表

### 生成参数（第一部分）
| 参数 | 推荐值 | 是否启用 | 说明 |
|------|--------|---------|------|
| 温度 (Temperature) | `0.1` | ✅ 启用 | 与项目配置一致，最大化确定性 |
| Top P | `0.4` | ✅ 启用 | 与主配置一致 |
| Top K | `0` | ❌ 禁用 | 项目使用束搜索，不使用Top K |
| Repeat Penalty | `1.3` | ✅ **必须启用** | 修正错误的-2值，抑制重复 |
| 最大令牌数 (Max Tokens) | `512` | ✅ 启用 | 与主配置一致，标准答案长度 |
| Mirostat 采样 | `0` | ❌ 禁用 | 项目不使用 |
| 学习率 | `0` | ❌ 禁用 | 推理不使用学习率 |

### 模型基础配置（第二部分）
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 文本连贯度 | `0` | 保持默认 |
| 上下文窗口大小 | `4096` | 与评估系统一致 |
| GPU 层数 | `28` | 全部层加载到GPU（显存充足时） |
| 线程数 | `4` | 优化CPU推理性能 |
| 回溯内容 | `-1` | 保持默认 |
| 减少标记影响 | `0` | 保持默认 |
| 随机数种子 | `0` 或 `42` | 测试用0，生产用固定值 |
| 模型存活时间 | `300` | 5分钟缓存 |
| 返回格式 | `json` | JSON格式（Dify要求） |
| JSON Schema | 定义Schema | 定义包含answer等字段的JSON结构 |

### 停止序列与思考模式（第三部分）
| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 思考模式 | ✅ **True** | 启用，提升可解释性 |
| 停止序列（左） | `<|im_end|>` | 标准EOS标记 |
| 停止序列（右） | `</think>` | 思考模式结束标记 |

---

## 重要说明：JSON格式返回

**⚠️ Dify工作流限制**：在Dify平台中，模型输出必须为JSON格式，不支持纯文本返回。

### 项目实际JSON Schema结构

基于项目源代码（`应用协调层/middle/api/schemas.py`），系统使用的完整JSON响应格式如下：

```json
{
  "success": true,
  "query": "用户问题",
  "answer": "模型生成的答案内容",
  "retrieval_results": [
    {
      "content": "检索到的文档内容",
      "fused_score": 0.85,
      "source": "vector",
      "source_scores": {},
      "contributing_sources": ["vector"],
      "entities": ["实体1", "实体2"],
      "relationships": ["关系1"]
    }
  ],
  "metadata": {
    "retrieval_time": 0.15,
    "generation_time": 1.2,
    "total_time": 1.35,
    "num_retrieval_results": 5,
    "model": "qwen3-1.7b-tcm",
    "temperature": 0.1,
    "routing_decision": "hybrid",
    "routing_confidence": 0.92,
    "tokens_generated": 256,
    "tokens_per_second": 213.3,
    "gpu_memory_used": "5.2GB"
  },
  "error": null
}
```

### 最小化JSON Schema（仅核心字段）

如果Dify仅需要模型生成答案部分，可以使用简化版本：

```json
{
  "type": "object",
  "properties": {
    "success": {"type": "boolean"},
    "query": {"type": "string"},
    "answer": {"type": "string"},
    "metadata": {
      "type": "object",
      "properties": {
        "routing_decision": {"type": "string", "enum": ["vector_only", "hybrid"]},
        "routing_confidence": {"type": "number"}
      }
    }
  },
  "required": ["success", "query", "answer"]
}
```

### 提示词调整建议

在Dify工作流的提示词中，需要明确要求模型以JSON格式输出，格式需匹配上述Schema：

```
请以JSON格式返回答案，必须包含以下字段：
{
  "success": true,
  "query": "用户的问题",
  "answer": "您的答案内容",
  "metadata": {
    "routing_decision": "hybrid 或 vector_only",
    "routing_confidence": 0.0-1.0之间的数值
  }
}

请确保返回的是有效的JSON格式，可以直接解析。
```

---

## 配置验证

配置完成后，建议进行以下测试：

1. **准确性测试**：使用标准问题测试答案准确性
2. **JSON格式验证**：确认返回的是有效JSON格式，能够正常解析
3. **重复检查**：确认答案无重复内容（Repeat Penalty生效）
4. **长度检查**：确认答案长度在预期范围内（Max Tokens控制）
5. **推理检查**：如果启用思考模式，确认推理过程正常显示

---

## 参考文档

- 项目主配置：`应用协调层/middle/config/service_config.yaml`
- 评估系统配置：`测试与质量保障层/rag评估系统/newragas/config.py`
- Qwen3官方文档：https://qwen.readthedocs.io/en/latest/

---

**最后更新**：2025-12-XX  
**配置版本**：基于项目 v4.4
