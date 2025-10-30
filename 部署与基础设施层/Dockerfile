# 混合检索系统 Dockerfile
# 基于Python 3.11官方镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量 - 添加应用协调层到Python路径
ENV PYTHONPATH=/app:/app/应用协调层
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件（从项目根目录）
COPY ../检索与知识层/requirements.txt ./retrieval_requirements.txt
COPY ../应用协调层/langchain/requirements.txt ./langchain_requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r retrieval_requirements.txt && \
    pip install --no-cache-dir -r langchain_requirements.txt

# 复制项目各层文件
COPY ../应用协调层 ./应用协调层
COPY ../检索与知识层 ./检索与知识层
COPY ../模型层 ./模型层
COPY . ./部署与基础设施层

# 创建必要的目录
RUN mkdir -p /app/logs \
    /app/data \
    /app/cache \
    /app/benchmark_results \
    /app/stress_test_results \
    /app/integration_test_results

# 设置权限
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# 暴露端口
EXPOSE 8000 8001 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动命令（从应用协调层启动）
WORKDIR /app/应用协调层
CMD ["python", "-m", "langchain.api.fastapi_server"]
