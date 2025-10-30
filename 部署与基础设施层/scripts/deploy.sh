#!/bin/bash

# 混合检索系统自动化部署脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查内存
    total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ $total_mem -lt 8000 ]; then
        log_warning "系统内存不足8GB，可能影响性能"
    fi
    
    # 检查磁盘空间
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $available_space -lt 20 ]; then
        log_warning "磁盘空间不足20GB，可能影响数据存储"
    fi
    
    log_success "系统要求检查完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    mkdir -p data/{chroma,neo4j,logs}  # BM25已移除
    mkdir -p cache
    mkdir -p config
    mkdir -p monitoring/grafana/{dashboards,datasources}
    mkdir -p nginx/ssl
    
    # 设置权限
    chmod 755 data cache config
    chmod 755 monitoring/grafana/dashboards
    chmod 755 monitoring/grafana/datasources
    chmod 755 nginx/ssl
    
    log_success "目录创建完成"
}

# 配置环境变量
setup_environment() {
    log_info "配置环境变量..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            log_success "已创建.env文件"
        else
            log_error "未找到env.example文件"
            exit 1
        fi
    else
        log_warning ".env文件已存在，跳过创建"
    fi
}

# 构建和启动服务
deploy_services() {
    log_info "构建和启动服务..."
    
    # 停止现有服务
    log_info "停止现有服务..."
    docker-compose down --remove-orphans
    
    # 构建镜像
    log_info "构建Docker镜像..."
    docker-compose build --no-cache
    
    # 启动服务
    log_info "启动服务..."
    docker-compose up -d
    
    log_success "服务启动完成"
}

# 等待服务就绪
wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待API服务
    log_info "等待API服务启动..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "API服务已就绪"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "API服务启动超时"
            exit 1
        fi
        sleep 2
    done
    
    # 等待Neo4j
    log_info "等待Neo4j服务启动..."
    for i in {1..60}; do
        if curl -f http://localhost:7474 &> /dev/null; then
            log_success "Neo4j服务已就绪"
            break
        fi
        if [ $i -eq 60 ]; then
            log_warning "Neo4j服务启动超时"
        fi
        sleep 2
    done
    
    # 等待Chroma
    log_info "等待Chroma服务启动..."
    for i in {1..30}; do
        if curl -f http://localhost:8003/api/v1/heartbeat &> /dev/null; then
            log_success "Chroma服务已就绪"
            break
        fi
        if [ $i -eq 30 ]; then
            log_warning "Chroma服务启动超时"
        fi
        sleep 2
    done
}

# 运行健康检查
run_health_check() {
    log_info "运行健康检查..."
    
    if docker-compose exec -T hybrid-retrieval-api python scripts/health_check.py; then
        log_success "健康检查通过"
    else
        log_error "健康检查失败"
        exit 1
    fi
}

# 显示部署信息
show_deployment_info() {
    log_success "部署完成！"
    echo ""
    echo "=========================================="
    echo "🎉 混合检索系统部署成功！"
    echo "=========================================="
    echo "服务访问地址："
    echo "  - API文档: http://localhost/docs"
    echo "  - 健康检查: http://localhost/health"
    echo "  - Neo4j浏览器: http://localhost/neo4j"
    echo "  - Grafana监控: http://localhost/grafana"
    echo "  - Prometheus: http://localhost/prometheus"
    echo "=========================================="
    echo ""
    echo "管理命令："
    echo "  - 查看日志: docker-compose logs"
    echo "  - 停止服务: docker-compose down"
    echo "  - 重启服务: docker-compose restart"
    echo "  - 健康检查: docker-compose exec hybrid-retrieval-api python scripts/health_check.py"
    echo "=========================================="
}

# 主函数
main() {
    echo "=========================================="
    echo "混合检索系统自动化部署脚本"
    echo "=========================================="
    
    check_requirements
    create_directories
    setup_environment
    deploy_services
    wait_for_services
    run_health_check
    show_deployment_info
}

# 运行主函数
main "$@"
