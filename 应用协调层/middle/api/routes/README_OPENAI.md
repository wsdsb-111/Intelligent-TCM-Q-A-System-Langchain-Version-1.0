# OpenAI兼容API使用说明

## 概述

本API提供OpenAI兼容的接口，用于在Dify工作流中集成本地微调的Qwen3-1.7B+LoRA模型。

## API配置信息

### API Base
```
http://localhost:8000/v1/chat/completions
```

### API Key
默认API Key（可通过环境变量`OPENAI_API_KEY`自定义）：
```
sk-qwen3-1.7b-local-dev-key-12345
```

### 设置自定义API Key

在启动FastAPI服务前，设置环境变量：
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-custom-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-custom-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-custom-api-key-here"
```

## API端点

### 1. 聊天完成接口
```
POST /v1/chat/completions
```

**请求头**（两种方式任选其一）：
- `x-api-key: sk-qwen3-1.7b-local-dev-key-12345`
- `Authorization: Bearer sk-qwen3-1.7b-local-dev-key-12345`

**请求体示例**：
```json
{
  "model": "qwen3-1.7b-finetuned",
  "messages": [
    {
      "role": "system",
      "content": "你是中医助手..."
    },
    {
      "role": "user",
      "content": "请推荐适合经常口臭的中药"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

**响应示例**：
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen3-1.7b-finetuned",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "根据您的情况，推荐以下中药..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

### 2. 列出模型接口
```
GET /v1/models
```

**请求头**：
- `x-api-key: sk-qwen3-1.7b-local-dev-key-12345`

**响应示例**：
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-1.7b-finetuned",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local"
    }
  ]
}
```

## 在Dify中配置

### 步骤1：进入模型供应商配置
1. 打开Dify
2. 进入 **设置 → 模型供应商**
3. 选择支持"Custom API"的供应商（如"OpenAI-API-compatible"或"Custom Model"）

### 步骤2：添加模型
点击"添加模型"，配置以下参数：

#### 1. 模型名称
```
qwen3-1.7b-finetuned
```
（这是显示名称，可自定义）

#### 2. API endpoint URL（必填）
```
http://localhost:8000/v1/chat/completions
```
⚠️ **注意**：这里填写**完整的chat/completions端点URL**，不是base URL
- 如果Dify和FastAPI服务在同一台机器：`http://localhost:8000/v1/chat/completions`
- 如果Dify和FastAPI服务在不同机器：`http://<FastAPI服务器IP>:8000/v1/chat/completions`

#### 3. API Key
```
sk-qwen3-1.7b-local-dev-key-12345
```
（或您设置的自定义API Key，从环境变量`OPENAI_API_KEY`或配置文件`dify.api_key`读取）

#### 4. API endpoint中的模型名称
```
qwen3-1.7b-finetuned
```
（这是请求中使用的模型标识符，必须与代码中定义的model名称一致）

#### 请求格式
Dify会自动处理，或使用以下模板：
```json
{
  "model": "qwen3-1.7b-finetuned",
  "messages": {{ messages }},
  "temperature": {{ temperature }},
  "max_tokens": {{ max_tokens }},
  "stream": false
}
```

#### 响应解析
- **非流式响应**：`choices[0].message.content`
- **流式响应**：勾选"支持流式响应"，解析`choices[0].delta.content`

### 步骤3：测试连接并保存
1. 点击"测试连接"
2. 确认返回正常响应
3. 保存配置

## 重要说明

1. **提示词配置**：提示词内容由Dify工作流节点配置，本API只负责转发
2. **生成参数**：温度、max_tokens等参数可在Dify工作流中配置，或使用API请求中的参数
3. **流式输出**：当前实现为模拟流式输出，如需真正的流式生成，需要修改`model_service`支持流式生成

## 模型路径

本地模型路径（已在`service_config.yaml`中配置）：
- **基础模型**：`Model Layer/model/qwen/Qwen3-1.7B/Qwen/Qwen3-1___7B`
- **LoRA适配器**：`Model Layer/model/checkpoint-7983`

## 故障排查

### API Key验证失败
- 检查请求头是否正确设置`x-api-key`或`Authorization`
- 确认API Key与服务器配置一致（检查环境变量`OPENAI_API_KEY`）

### 模型服务未初始化
- 确认FastAPI服务已完全启动
- 检查模型路径是否正确
- 查看日志确认模型加载成功

### 连接超时
- 确认FastAPI服务运行在`localhost:8000`
- 检查防火墙设置
- 如果在不同机器，确认IP地址和端口可访问

