# 混合检索API接口

## 概述

混合检索API提供了完整的RESTful接口，支持智能中医混合检索系统的所有功能。基于FastAPI构建，提供高性能、可扩展的API服务。

## 🚀 快速开始

### 安装依赖

```bash
pip install fastapi uvicorn pydantic psutil aiohttp
```

### 启动服务

```bash
# 开发模式
python langchain/api/server.py --dev

# 生产模式  
python langchain/api/server.py --prod

# 自定义配置
python langchain/api/server.py --host 0.0.0.0 --port 8080 --workers 4
```

### 访问文档

- API文档: http://localhost:8000/docs
- ReDoc文档: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/api/v1/health

## 📋 API端点

### 1. 基础检索

#### POST /api/v1/retrieve

执行单个查询的混合检索。

**请求体:**
```json
{
    "query": "人参的功效与作用",
    "retrieval_type": "hybrid",
    "fusion_method": "smart", 
    "top_k": 10,
    "weights": {
        "bm25": 0.4,
        "vector": 0.4,
        "graph": 0.2
    },
    "timeout": 30
}
```

**响应:**
```json
{
    "success": true,
    "query": "人参的功效与作用",
    "retrieval_type": "hybrid",
    "fusion_method": "smart",
    "total_results": 5,
    "response_time": 0.234,
    "results": [
        {
            "content": "人参大补元气，主治气虚欲脱...",
            "score": 0.95,
            "source_scores": {
                "bm25": 0.4,
                "vector": 0.4,
                "graph": 0.15
            },
            "fusion_method": "smart",
            "contributing_sources": ["bm25", "vector"],
            "metadata": {
                "type": "herb",
                "category": "补气药"
            },
            "entities": ["人参", "元气"],
            "relationships": ["治疗", "功效"],
            "timestamp": "2024-01-01T12:00:00"
        }
    ],
    "query_analysis": {
        "query_type": "专业术语查询",
        "detected_weights": {
            "bm25": 0.6,
            "vector": 0.25,
            "graph": 0.15
        }
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### 2. 批量检索

#### POST /api/v1/batch_retrieve

执行多个查询的批量检索。

**请求体:**
```json
{
    "queries": ["人参功效", "黄芪作用", "当归用法"],
    "retrieval_type": "hybrid",
    "fusion_method": "smart",
    "top_k": 5,
    "timeout": 60
}
```

**响应:**
```json
{
    "success": true,
    "total_queries": 3,
    "successful_queries": 3,
    "failed_queries": 0,
    "total_response_time": 0.456,
    "results": {
        "人参功效": {
            "success": true,
            "query": "人参功效",
            "total_results": 5,
            "results": [...]
        },
        "黄芪作用": {
            "success": true,
            "query": "黄芪作用", 
            "total_results": 4,
            "results": [...]
        }
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### 3. 上下文检索

#### POST /api/v1/contextual_retrieve

基于对话上下文执行检索。

**请求体:**
```json
{
    "query": "这个药怎么用？",
    "context": [
        {
            "role": "user",
            "content": "人参有什么功效？"
        },
        {
            "role": "assistant", 
            "content": "人参具有大补元气的功效..."
        }
    ],
    "retrieval_type": "hybrid",
    "top_k": 5,
    "use_context": true
}
```

### 4. 健康检查

#### GET /api/v1/health

获取系统健康状态。

**响应:**
```json
{
    "status": "healthy",
    "overall_healthy": true,
    "modules": [
        {
            "name": "bm25",
            "healthy": true,
            "last_check": "2024-01-01T12:00:00",
            "error_message": null,
            "response_time": 0.001
        }
    ],
    "system_info": {
        "platform": "Windows-10",
        "python_version": "3.11.0",
        "cpu_count": 8,
        "memory_total": 17179869184,
        "memory_available": 8589934592,
        "memory_percent": 50.0
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

#### GET /api/v1/health/quick

快速健康检查（用于负载均衡器）。

**响应:**
```json
{
    "status": "ok",
    "timestamp": "2024-01-01T12:00:00",
    "service": "hybrid-retrieval-api"
}
```

### 5. 指标监控

#### GET /api/v1/metrics

获取系统指标。

**响应:**
```json
{
    "service_metrics": {
        "uptime_seconds": 3600,
        "total_requests": 1000,
        "successful_requests": 950,
        "failed_requests": 50,
        "success_rate": 0.95,
        "requests_per_second": 0.278,
        "average_response_time": 0.156
    },
    "retrieval_metrics": {
        "retriever": {
            "total_queries": 800,
            "successful_queries": 780,
            "failed_queries": 20,
            "average_response_time": 0.145
        },
        "coordinator": {
            "module_usage": {
                "bm25": 300,
                "vector": 280,
                "graph": 200
            },
            "fusion_method_usage": {
                "smart": 400,
                "rrf": 200,
                "weighted": 200
            }
        }
    },
    "performance_metrics": {
        "current_cpu_percent": 25.5,
        "current_memory_percent": 45.2,
        "peak_cpu_percent": 80.0,
        "peak_memory_percent": 75.0,
        "available_memory_gb": 4.2
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

#### GET /api/v1/statistics

获取详细统计信息。

#### GET /api/v1/metrics/performance

获取实时性能指标。

#### GET /api/v1/metrics/endpoints

获取各端点的调用统计。

#### POST /api/v1/metrics/reset

重置所有指标统计。

## 🔧 配置参数

### 检索类型 (retrieval_type)

- **bm25**: BM25关键词精确匹配检索
- **vector**: 向量语义相似度检索
- **graph**: 知识图谱关系推理检索
- **hybrid**: 混合检索（推荐）

### 融合方法 (fusion_method)

- **rrf**: 倒数排名融合，自动平衡各来源
- **weighted**: 加权融合，可自定义权重
- **rank_based**: 基于排名的融合
- **smart**: 智能融合，自动选择最优策略（推荐）

### 权重配置 (weights)

```json
{
    "bm25": 0.4,    // BM25检索权重
    "vector": 0.4,  // 向量检索权重  
    "graph": 0.2    // 图检索权重
}
```

## 📊 错误处理

### 错误响应格式

```json
{
    "success": false,
    "error_code": "RETRIEVAL_ERROR",
    "error_message": "检索失败: 连接超时",
    "error_details": {
        "query": "测试查询",
        "response_time": 30.0
    },
    "timestamp": "2024-01-01T12:00:00",
    "request_id": "uuid-string"
}
```

### 常见错误码

- **RETRIEVAL_ERROR**: 检索执行失败
- **BATCH_RETRIEVAL_ERROR**: 批量检索失败
- **CONTEXTUAL_RETRIEVAL_ERROR**: 上下文检索失败
- **VALIDATION_ERROR**: 请求参数验证失败
- **TIMEOUT_ERROR**: 请求超时
- **INTERNAL_SERVER_ERROR**: 服务器内部错误

## 🐳 Docker部署

### 构建镜像

```bash
cd langchain/api
docker build -t hybrid-retrieval-api .
```

### 使用Docker Compose

```bash
docker-compose up -d
```

### 环境变量

```bash
# 服务配置
PYTHONPATH=/app
LOG_LEVEL=INFO
WORKERS=4

# 数据路径
BM25_DATA_PATH=/app/data/bm25
CHROMA_DATA_PATH=/app/data/chroma
GRAPH_DATA_PATH=/app/data/graph

# Neo4j配置
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## 🔍 使用示例

### Python客户端

```python
import requests

# 基础检索
response = requests.post("http://localhost:8000/api/v1/retrieve", json={
    "query": "人参的功效与作用",
    "retrieval_type": "hybrid",
    "fusion_method": "smart",
    "top_k": 10
})

result = response.json()
if result["success"]:
    for doc in result["results"]:
        print(f"评分: {doc['score']:.3f}")
        print(f"内容: {doc['content'][:100]}...")
        print(f"来源: {doc['contributing_sources']}")
        print("-" * 50)
```

### JavaScript客户端

```javascript
// 批量检索
const response = await fetch("http://localhost:8000/api/v1/batch_retrieve", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        queries: ["人参功效", "黄芪作用", "当归用法"],
        retrieval_type: "hybrid",
        top_k: 5
    })
});

const result = await response.json();
if (result.success) {
    console.log(`批量检索完成: ${result.successful_queries}/${result.total_queries}`);
    
    for (const [query, queryResult] of Object.entries(result.results)) {
        console.log(`查询: ${query}`);
        console.log(`结果数: ${queryResult.total_results}`);
    }
}
```

### cURL示例

```bash
# 健康检查
curl -X GET "http://localhost:8000/api/v1/health"

# 基础检索
curl -X POST "http://localhost:8000/api/v1/retrieve" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "人参的功效",
       "retrieval_type": "hybrid",
       "fusion_method": "smart",
       "top_k": 5
     }'

# 获取指标
curl -X GET "http://localhost:8000/api/v1/metrics"
```

## 🚀 性能优化

### 并发处理

- 支持异步处理，提高并发性能
- 使用连接池管理数据库连接
- 实现请求队列和限流机制

### 缓存策略

- 查询结果缓存（可选）
- 模型加载缓存
- 连接复用

### 监控告警

- 实时性能监控
- 健康状态检查
- 错误率告警
- 资源使用监控

## 🔒 安全配置

### 生产环境建议

```python
# 生产环境配置
config = APIConfig(
    host="0.0.0.0",
    port=8000,
    workers=4,
    cors_origins=["https://yourdomain.com"],  # 限制CORS
    enable_docs=False,  # 关闭API文档
    rate_limit={"requests": 1000, "window": 60}  # 速率限制
)
```

### 安全特性

- CORS配置
- 请求频率限制
- 输入验证和过滤
- 错误信息脱敏
- 访问日志记录

## 📈 扩展性

### 水平扩展

- 支持多实例部署
- 负载均衡配置
- 无状态设计

### 功能扩展

- 插件式架构
- 自定义检索模块
- 自定义融合算法
- 中间件支持

## 🛠️ 开发指南

### 添加新端点

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["custom"])

@router.post("/custom_endpoint")
async def custom_function(request: CustomRequest):
    # 实现自定义逻辑
    return {"result": "success"}

# 在app.py中注册路由
app.include_router(router)
```

### 自定义中间件

```python
@app.middleware("http")
async def custom_middleware(request: Request, call_next):
    # 请求前处理
    response = await call_next(request)
    # 响应后处理
    return response
```

## 📚 相关文档

- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [Pydantic数据验证](https://pydantic-docs.helpmanual.io/)
- [Uvicorn ASGI服务器](https://www.uvicorn.org/)
- [Docker部署指南](https://docs.docker.com/)

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 编写测试用例
4. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证。