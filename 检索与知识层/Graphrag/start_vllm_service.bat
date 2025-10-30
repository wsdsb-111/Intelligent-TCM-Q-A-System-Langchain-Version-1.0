@echo off
REM vLLM 服务启动脚本 (Windows)
REM 适用于 RTX 4090 24GB + Qwen2.5-7B-Instruct
REM 更新日期：2025-10-16

echo ==========================================
echo 启动 vLLM 服务 - Qwen2.5-7B-Instruct
echo GPU: RTX 4090 24GB
echo ==========================================

REM 检查 GPU 状态
echo.
echo 检查 GPU 状态...
nvidia-smi

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0
set VLLM_ATTENTION_BACKEND=FLASH_ATTN

REM 模型路径（请根据实际情况修改）
set MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
REM 如果是本地路径，使用类似: set MODEL_PATH=E:\models\Qwen2.5-7B-Instruct

REM vLLM 服务配置
set HOST=0.0.0.0
set PORT=8000
set MAX_MODEL_LEN=8192
set GPU_MEMORY_UTIL=0.85
set TENSOR_PARALLEL_SIZE=1
set DTYPE=half

echo.
echo 启动参数：
echo   模型: %MODEL_PATH%
echo   端口: %PORT%
echo   最大长度: %MAX_MODEL_LEN%
echo   GPU 利用率: %GPU_MEMORY_UTIL%
echo.

REM 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server ^
    --model %MODEL_PATH% ^
    --served-model-name qwen2.5-7b-instruct ^
    --host %HOST% ^
    --port %PORT% ^
    --max-model-len %MAX_MODEL_LEN% ^
    --gpu-memory-utilization %GPU_MEMORY_UTIL% ^
    --tensor-parallel-size %TENSOR_PARALLEL_SIZE% ^
    --dtype %DTYPE% ^
    --trust-remote-code ^
    --enforce-eager

REM 注意事项：
REM 1. 如果遇到显存不足，降低 --gpu-memory-utilization 到 0.8 或 0.75
REM 2. 如果需要更长的上下文，可以增加 --max-model-len（注意显存限制）
REM 3. --enforce-eager 可以减少编译时间，适合开发测试
REM 4. 移除 --enforce-eager 可以使用 CUDA Graph 优化，适合生产环境

pause

