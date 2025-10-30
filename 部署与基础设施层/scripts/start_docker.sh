#!/bin/bash

# 混合检索系统Docker启动脚本

set -e

echo "=========================================="
echo "混合检索系统 Docker 启动脚本"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data logs cache config
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p nginx/ssl

# 设置权限
echo "🔐 设置目录权限..."
chmod 755 data logs cache config
chmod 755 monitoring/grafana/dashboards
chmod 755 monitoring/grafana/datasources
chmod 755 nginx/ssl

# 检查环境配置文件
if [ ! -f .env ]; then
    echo "⚠️ 未找到.env文件，从env.example创建..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "✅ 已创建.env文件，请根据需要修改配置"
    else
        echo "❌ 未找到env.example文件"
        exit 1
    fi
fi

# 停止现有容器
echo "🛑 停止现有容器..."
docker-compose down --remove-orphans

# 构建镜像
echo "🔨 构建Docker镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 运行健康检查
echo "🏥 运行健康检查..."
docker-compose exec hybrid-retrieval-api python scripts/health_check.py

echo "=========================================="
echo "🎉 混合检索系统启动完成！"
echo "=========================================="
echo "服务访问地址："
echo "  - API文档: http://localhost/docs"
echo "  - 健康检查: http://localhost/health"
echo "  - Neo4j浏览器: http://localhost/neo4j"
echo "  - Grafana监控: http://localhost/grafana"
echo "  - Prometheus: http://localhost/prometheus"
echo "=========================================="
