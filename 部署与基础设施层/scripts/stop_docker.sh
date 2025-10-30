#!/bin/bash

# 混合检索系统Docker停止脚本

set -e

echo "=========================================="
echo "混合检索系统 Docker 停止脚本"
echo "=========================================="

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装"
    exit 1
fi

# 停止所有服务
echo "🛑 停止所有服务..."
docker-compose down

# 清理未使用的镜像和容器
echo "🧹 清理未使用的资源..."
docker system prune -f

# 显示清理结果
echo "📊 系统资源使用情况："
docker system df

echo "=========================================="
echo "✅ 混合检索系统已停止"
echo "=========================================="
