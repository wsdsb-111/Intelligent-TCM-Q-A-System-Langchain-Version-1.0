"""
FastAPI应用主文件
创建和配置混合检索API服务
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

try:
    from fastapi import FastAPI, Request, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.openapi.utils import get_openapi
    FASTAPI_AVAILABLE = True
except ImportError:
    # 如果FastAPI不可用，创建基础类
    class FastAPI:
        def __init__(self, **kwargs):
            self.routes = []
            self.middleware = []
        
        def include_router(self, router, **kwargs):
            pass
        
        def add_middleware(self, middleware_class, **kwargs):
            pass
        
        def middleware(self, middleware_type):
            def decorator(func):
                return func
            return decorator
        
        def exception_handler(self, exc_class):
            def decorator(func):
                return func
            return decorator
    
    class Request:
        def __init__(self):
            self.client = None
            self.url = None
    
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
    
    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code
    
    def asynccontextmanager(func):
        return func
    
    FASTAPI_AVAILABLE = False

from .models import APIConfig, ErrorResponse, create_error_response
from .routes import retrieval_router, health_router, metrics_router
from .routes.metrics import update_service_metrics
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from middle.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 全局配置
_api_config = APIConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🚀 混合检索API服务启动中...")
    
    # 初始化检索器（预热）
    try:
        from ..integrations.hybrid_retriever import create_hybrid_retriever_async
        retriever = await create_hybrid_retriever_async()
        logger.info("✅ 检索器预热完成")
        
        # 执行健康检查
        health = await retriever.health_check()
        if health.get("overall_healthy"):
            logger.info("✅ 系统健康检查通过")
        else:
            logger.warning("⚠️ 系统健康检查发现问题")
        
    except Exception as e:
        logger.error(f"❌ 检索器初始化失败: {str(e)}")
    
    logger.info("🎉 混合检索API服务启动完成")
    
    yield
    
    # 关闭时执行
    logger.info("🛑 混合检索API服务关闭中...")
    
    # 清理资源
    try:
        # 这里可以添加资源清理逻辑
        logger.info("✅ 资源清理完成")
    except Exception as e:
        logger.error(f"❌ 资源清理失败: {str(e)}")
    
    logger.info("👋 混合检索API服务已关闭")


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """
    创建FastAPI应用实例
    
    Args:
        config: API配置
        
    Returns:
        配置好的FastAPI应用实例
    """
    global _api_config
    if config:
        _api_config = config
    
    # 创建FastAPI应用
    app = FastAPI(
        title="混合检索API",
        description="""
        智能中医混合检索系统API
        
        ## 功能特点
        
        - 🔍 **多模态检索**: 支持BM25关键词检索、向量语义检索、知识图谱检索
        - 🤖 **智能融合**: 自动识别查询类型，智能选择最优融合策略
        - ⚡ **高性能**: 并行检索处理，毫秒级响应时间
        - 📊 **完整监控**: 实时健康检查、性能指标、统计分析
        - 🛡️ **稳定可靠**: 多层降级策略，确保服务高可用性
        
        ## 检索类型
        
        - **bm25**: BM25关键词精确匹配检索
        - **vector**: 向量语义相似度检索
        - **graph**: 知识图谱关系推理检索
        - **hybrid**: 混合检索（推荐）
        
        ## 融合方法
        
        - **rrf**: 倒数排名融合，自动平衡各来源
        - **weighted**: 加权融合，可自定义权重
        - **rank_based**: 基于排名的融合
        - **smart**: 智能融合，自动选择最优策略（推荐）
        
        ## 使用示例
        
        ```python
        import requests
        
        # 基础检索
        response = requests.post("/api/v1/retrieve", json={
            "query": "人参的功效与作用",
            "retrieval_type": "hybrid",
            "fusion_method": "smart",
            "top_k": 10
        })
        
        # 批量检索
        response = requests.post("/api/v1/batch_retrieve", json={
            "queries": ["人参功效", "黄芪作用", "当归用法"],
            "retrieval_type": "hybrid",
            "top_k": 5
        })
        ```
        """,
        version="1.0.0",
        docs_url="/docs" if _api_config.enable_docs else None,
        redoc_url="/redoc" if _api_config.enable_docs else None,
        openapi_url="/openapi.json" if _api_config.enable_docs else None,
        lifespan=lifespan
    )
    
    # 配置CORS
    if _api_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_api_config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
    
    # 配置受信任主机（生产环境建议启用）
    # app.add_middleware(
    #     TrustedHostMiddleware,
    #     allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
    # )
    
    # 请求处理中间件
    @app.middleware("http")
    async def process_request_middleware(request: Request, call_next):
        """请求处理中间件"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # 记录请求开始
        logger.info(f"[{request_id}] {request.method} {request.url.path} - 开始处理")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # 记录请求完成
            logger.info(f"[{request_id}] {request.method} {request.url.path} - "
                       f"完成 {response.status_code} ({process_time:.3f}s)")
            
            # 更新指标
            endpoint = request.url.path
            success = 200 <= response.status_code < 400
            update_service_metrics(endpoint, process_time, success)
            
            return response
        
        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录错误
            logger.error(f"[{request_id}] {request.method} {request.url.path} - "
                        f"错误: {str(e)} ({process_time:.3f}s)")
            
            # 更新错误指标
            endpoint = request.url.path
            update_service_metrics(endpoint, process_time, False, "middleware_error")
            
            # 返回错误响应
            error_response = create_error_response(
                error_code="INTERNAL_SERVER_ERROR",
                error_message=f"服务器内部错误: {str(e)}",
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={"X-Request-ID": request_id}
            )
    
    # 异常处理器
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """HTTP异常处理器"""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.warning(f"[{request_id}] HTTP异常: {exc.status_code} - {exc.detail}")
        
        error_response = create_error_response(
            error_code=f"HTTP_{exc.status_code}",
            error_message=exc.detail,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理器"""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.error(f"[{request_id}] 未处理异常: {str(exc)}")
        
        error_response = create_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="服务器内部错误",
            error_details={"exception_type": type(exc).__name__},
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    # 注册路由
    app.include_router(retrieval_router)
    app.include_router(health_router)
    app.include_router(metrics_router)
    
    # 根路径
    @app.get("/", tags=["root"])
    async def root():
        """根路径，返回API信息"""
        return {
            "service": "混合检索API",
            "version": "1.0.0",
            "description": "智能中医混合检索系统",
            "docs_url": "/docs" if _api_config.enable_docs else None,
            "health_url": "/api/v1/health",
            "metrics_url": "/api/v1/metrics",
            "timestamp": time.time()
        }
    
    # 自定义OpenAPI文档
    if _api_config.enable_docs:
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            openapi_schema = get_openapi(
                title="混合检索API",
                version="1.0.0",
                description="智能中医混合检索系统API文档",
                routes=app.routes,
            )
            
            # 添加自定义信息
            openapi_schema["info"]["x-logo"] = {
                "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
            }
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
    
    logger.info("✅ FastAPI应用创建完成")
    return app


def get_app_config() -> APIConfig:
    """获取当前应用配置"""
    return _api_config


def update_app_config(config: APIConfig):
    """更新应用配置"""
    global _api_config
    _api_config = config
    logger.info("应用配置已更新")


# 便捷函数
def create_development_app() -> FastAPI:
    """创建开发环境应用"""
    config = APIConfig(
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="DEBUG",
        enable_docs=True,
        enable_metrics=True
    )
    return create_app(config)


def create_production_app() -> FastAPI:
    """创建生产环境应用"""
    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        log_level="INFO",
        cors_origins=["https://yourdomain.com"],  # 生产环境应该限制CORS
        enable_docs=False,  # 生产环境可以关闭文档
        enable_metrics=True,
        rate_limit={"requests": 1000, "window": 60}  # 更严格的速率限制
    )
    return create_app(config)


# 如果直接运行此文件
if __name__ == "__main__":
    import uvicorn
    
    app = create_development_app()
    
    uvicorn.run(
        app,
        host=_api_config.host,
        port=_api_config.port,
        reload=_api_config.reload,
        log_level=_api_config.log_level.lower()
    )