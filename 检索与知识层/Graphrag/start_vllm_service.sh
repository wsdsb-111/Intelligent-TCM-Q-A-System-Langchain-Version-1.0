#!/bin/bash
# vLLM 服务启动脚本
# 适用于 RTX 4090 24GB + Qwen2.5-7B-Instruct
# 更新日期：2025-10-16

echo "=========================================="
echo "启动 vLLM 服务 - Qwen2.5-7B-Instruct"
echo "GPU: RTX 4090 24GB"
echo "=========================================="

# 检查 GPU 状态
echo ""
echo "检查 GPU 状态..."
nvidia-smi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 使用 Flash Attention

# 模型路径（请根据实际情况修改）
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# 如果是本地路径，使用类似: MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"

# vLLM 服务配置
HOST="0.0.0.0"
PORT=8000
MAX_MODEL_LEN=8192        # 最大上下文长度
GPU_MEMORY_UTIL=0.85      # GPU 显存利用率（85%，为系统预留空间）
TENSOR_PARALLEL_SIZE=1    # 单卡推理
DTYPE="half"              # 使用 FP16 精度

echo ""
echo "启动参数："
echo "  模型: $MODEL_PATH"
echo "  端口: $PORT"
echo "  最大长度: $MAX_MODEL_LEN"
echo "  GPU 利用率: ${GPU_MEMORY_UTIL}"
echo ""

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name qwen2.5-7b-instruct \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --dtype $DTYPE \
    --trust-remote-code \
    --enforce-eager

# 注意事项：
# 1. 如果遇到显存不足，降低 --gpu-memory-utilization 到 0.8 或 0.75
# 2. 如果需要更长的上下文，可以增加 --max-model-len（注意显存限制）
# 3. --enforce-eager 可以减少编译时间，适合开发测试
# 4. 移除 --enforce-eager 可以使用 CUDA Graph 优化，适合生产环境

