# RTX 4090 + vLLM 快速启动指南

> 本指南专门针对使用 RTX 4090 24GB 和 vLLM 进行中医知识图谱提取的用户

## 📋 准备清单

在开始之前，请确保您有：

- ✅ NVIDIA RTX 4090 24GB GPU
- ✅ CUDA 12.1+ 和最新的 NVIDIA 驱动
- ✅ Python 3.10（推荐使用 conda）
- ✅ 至少 32GB RAM
- ✅ 50GB+ 可用磁盘空间
- ✅ 稳定的网络连接（用于下载模型）

## 🚀 5 分钟快速启动

### 步骤 1：创建环境（5 分钟）

```bash
# 创建 GraphRAG 环境
conda create -n graphrag python=3.10 -y
conda activate graphrag
cd 检索与知识层/Graphrag
pip install -r requirements.txt

# 创建 vLLM 环境（新终端）
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm>=0.5.0
```

### 步骤 2：下载模型（10-30 分钟，取决于网速）

```bash
# 在 vLLM 环境中
conda activate vllm
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### 步骤 3：启动 vLLM 服务（2 分钟）

**Windows**：
```bash
# 编辑 start_vllm_service.bat，确保 MODEL_PATH 正确
# 默认使用 Hugging Face 缓存路径，无需修改
start_vllm_service.bat
```

**Linux**：
```bash
chmod +x start_vllm_service.sh
./start_vllm_service.sh
```

**预期输出**：
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 步骤 4：验证服务（1 分钟）

```bash
# 新终端
curl http://localhost:8000/v1/models

# 应该看到：
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "qwen2.5-7b-instruct",
#       ...
#     }
#   ]
# }
```

### 步骤 5：运行提取（开始工作！）

```bash
# 激活 GraphRAG 环境
conda activate graphrag

# 第一次运行：转换数据格式
python convert_shennong_simple.py

# 开始提取（建议先测试 50 条）
python extract_from_shennong_csv.py
```

## 📊 预期性能（RTX 4090）

| 指标 | 数值 |
|------|------|
| GPU 显存占用 | ~18-20 GB |
| 处理速度 | 5-10 条/分钟 |
| 50 条测试数据 | 5-10 分钟 |
| 1000 条数据 | 2-3 小时 |
| 完整数据集 (112K) | 200-400 小时 |

## 🎮 交互式控制

运行提取脚本时，您可以使用以下命令：

```
p      - 暂停处理（查看日志、调整参数）
r      - 恢复处理
s      - 停止处理（保存当前进度）
status - 查看当前状态
stats  - 查看统计信息
q      - 退出控制面板
```

## ⚙️ 性能优化建议

### 1. 显存不足时

编辑 `start_vllm_service.bat` 或 `.sh`：

```bash
# 降低显存利用率
set GPU_MEMORY_UTIL=0.75  # 从 0.85 降到 0.75

# 减少最大上下文长度
set MAX_MODEL_LEN=4096    # 从 8192 降到 4096
```

### 2. 提升处理速度

在 `extract_from_shennong_csv.py` 中调整：

```python
# 增大批次大小（如果显存充足）
batch_size = 10  # 从 5 增加到 10

# 并行处理（实验性功能）
# 在配置中启用多线程
```

### 3. 减少 API 调用延迟

在 `config/config.yaml` 中：

```yaml
graphrag:
  temperature: 0.0        # 确定性输出，减少变异
  max_tokens: 3000        # 减少 token 数量
  chunk_size: 800         # 减小分块大小
```

## 🔍 故障排查

### 问题 1：vLLM 启动时显存不足

**错误**：`CUDA out of memory`

**解决**：
```bash
# 降低显存利用率
--gpu-memory-utilization 0.7

# 或减少上下文长度
--max-model-len 4096
```

### 问题 2：提取脚本连接失败

**错误**：`Connection refused` 或 `Connection timeout`

**检查**：
```bash
# 1. 确认 vLLM 服务正在运行
netstat -an | findstr 8000  # Windows
netstat -an | grep 8000     # Linux

# 2. 测试连接
curl http://localhost:8000/v1/models

# 3. 检查防火墙设置
# Windows: 控制面板 -> 防火墙 -> 允许端口 8000
```

### 问题 3：提取速度很慢

**可能原因**：
1. GPU 未充分利用
2. 文本过长导致处理时间增加
3. 网络 I/O 瓶颈

**解决**：
```bash
# 1. 监控 GPU 使用率
nvidia-smi -l 1

# 2. 如果 GPU 利用率低，增加批次大小
batch_size = 10

# 3. 确保使用 Flash Attention
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

### 问题 4：提取结果为空

**检查步骤**：
1. 查看日志：`logs/app.log`
2. 验证 vLLM 服务响应：
   ```bash
   curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5-7b-instruct","prompt":"测试","max_tokens":10}'
   ```
3. 检查配置文件：`config/config.yaml`

## 📈 监控和日志

### 实时监控 GPU

**Windows PowerShell**：
```powershell
while($true) { 
    nvidia-smi
    sleep 2
    cls
}
```

**Linux**：
```bash
watch -n 2 nvidia-smi
```

### 查看处理日志

```bash
# 实时查看日志
tail -f logs/app.log  # Linux
Get-Content logs/app.log -Wait  # Windows PowerShell

# 查看最近的错误
grep ERROR logs/app.log  # Linux
Select-String -Path logs/app.log -Pattern "ERROR"  # Windows
```

### 查看提取统计

```bash
# 查看最终统计
cat output_shennong_extraction/final_extraction_stats.json

# 查看批次统计
cat output_shennong_extraction/stats_batch_001.json
```

## 🎯 最佳实践

### 1. 分批处理大数据集

```python
# 在 extract_from_shennong_csv.py 中
max_records = 1000  # 每次处理 1000 条

# 运行多次，逐步处理完整数据集
# 第一次：records 0-1000
# 第二次：修改脚本跳过前 1000 条
# ...
```

### 2. 定期保存检查点

系统会自动保存批次结果，确保：
- 每批处理后自动保存 CSV
- 保留详细的统计信息
- 错误日志完整记录

### 3. 监控系统资源

```bash
# 监控 GPU 温度和功耗
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv -l 5

# 监控 CPU 和内存
htop  # Linux
任务管理器  # Windows
```

## 📦 输出文件说明

提取完成后，您将获得：

```
output_shennong_extraction/
├── entities_batch_001.csv          # 第 1 批实体
├── entities_batch_002.csv          # 第 2 批实体
├── relationships_batch_001.csv     # 第 1 批关系
├── relationships_batch_002.csv     # 第 2 批关系
├── stats_batch_001.json            # 第 1 批统计
├── stats_batch_002.json            # 第 2 批统计
└── final_extraction_stats.json     # 最终汇总统计
```

每个 CSV 文件可直接导入 Neo4j 或用于后续分析。

## 🔗 相关资源

- **完整文档**：`README.md`
- **配置文件**：`config/config.yaml`
- **vLLM 启动脚本**：`start_vllm_service.bat` / `.sh`
- **提取脚本**：`extract_from_shennong_csv.py`
- **暂停控制示例**：`pause_control_example.py`

## 💡 技巧和提示

1. **首次运行建议测试 50 条数据**，确认系统正常工作
2. **使用交互式控制**，随时暂停查看中间结果
3. **定期备份输出文件**，防止意外数据丢失
4. **监控 GPU 温度**，确保散热良好
5. **准备长时间运行**，完整数据集需要数天时间

## 📞 获取帮助

如果遇到问题：

1. 查看 `logs/app.log` 日志文件
2. 检查 `README.md` 的故障排查部分
3. 验证环境配置是否正确
4. 确认 vLLM 服务正常运行

---

**祝您使用愉快！🎉**

如果一切正常，您应该能看到实体和关系被持续提取，进度条稳步推进。

