# OpenAI兼容API故障排查指南

## Internal Server Error (500) 错误排查

如果遇到 "Internal Server Error"，请按以下步骤排查：

### 1. 检查FastAPI服务日志

查看服务启动时的日志，确认：
- ✅ 模型是否成功加载
- ✅ API Key是否正确配置
- ✅ 服务是否正常启动

### 2. 常见错误原因

#### 错误1：模型服务未初始化
**错误信息**：`Model service not initialized`

**解决方法**：
1. 确认模型路径正确（检查 `service_config.yaml`）
2. 检查模型是否成功加载（查看启动日志）
3. 等待模型加载完成后再测试

#### 错误2：API Key验证失败
**错误信息**：`Invalid API Key`

**解决方法**：
1. 确认Dify中配置的API Key与服务端一致
2. 检查环境变量 `OPENAI_API_KEY` 或配置文件 `dify.api_key`
3. 确认请求头格式正确（`x-api-key` 或 `Authorization: Bearer`）

#### 错误3：请求格式错误
**错误信息**：`No user message found in messages`

**解决方法**：
1. 确认Dify工作流配置了用户消息
2. 检查 `messages` 数组是否包含 `role: "user"` 的消息

#### 错误4：模型生成失败
**错误信息**：`Model generation failed`

**解决方法**：
1. 检查GPU显存是否充足
2. 确认模型文件完整
3. 查看详细错误日志定位问题

### 3. 测试API连接

使用curl测试API连接：

```bash
# 测试连接
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-qwen3-1.7b-local-dev-key-12345" \
  -d '{
    "model": "qwen3-1.7b-finetuned",
    "messages": [
      {
        "role": "user",
        "content": "你好"
      }
    ],
    "max_tokens": 100
  }'
```

### 4. 查看详细日志

在 `service_config.yaml` 中设置日志级别为 `DEBUG`：

```yaml
api:
  log_level: "DEBUG"
```

重启服务后，日志会显示更详细的错误信息。

### 5. 检查Dify配置

确认Dify中的配置：
- ✅ **API endpoint URL**: `http://localhost:8000/v1/chat/completions`
- ✅ **API Key**: `sk-qwen3-1.7b-local-dev-key-12345`
- ✅ **API endpoint中的模型名称**: `qwen3-1.7b-finetuned`

### 6. 常见配置错误

#### 错误：API endpoint URL填写错误
❌ 错误：`http://localhost:8000/v1` （缺少 `/chat/completions`）
✅ 正确：`http://localhost:8000/v1/chat/completions`

#### 错误：模型名称不一致
❌ Dify中填写：`qwen`（不匹配）
✅ 正确：`qwen3-1.7b-finetuned`（与代码中一致）

### 7. 获取错误详情

如果问题仍然存在，请提供：
1. FastAPI服务的完整错误日志
2. Dify中的具体错误信息
3. 请求的完整内容（可在Dify的调试模式中查看）

这样可以更准确地定位问题。

