"""
LangChain中间层主应用
集成RAG问答服务的FastAPI应用
参照评估系统的组件加载方式进行优化
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import sys
import os
import yaml
from enum import Enum
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 添加项目根目录到路径
# main_app.py在: 应用协调层/middle/api/
# 向上两级到达: 应用协调层/
middle_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
application_layer = os.path.dirname(middle_dir)  # 应用协调层
# 向上一级到达项目根目录
project_root = os.path.dirname(application_layer)
# 配置文件所在目录
config_dir = os.path.join(middle_dir, "config")
sys.path.insert(0, middle_dir)
sys.path.insert(0, project_root)

from middle.utils.logging_utils import get_logger
from middle.api.v1_routes import router as v1_router, init_routes
from middle.api.routes.dify_nodes import router as dify_router, init_dify_routes
from middle.api.routes.openai_compatible import router as openai_router
from middle.services.model_service import get_model_service
from middle.services.rag_chain import RAGChain
from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
# BM25适配器已移除
from middle.adapters.simple_vector_adapter import SimpleVectorAdapter
from middle.adapters.graph_adapter import GraphRetrievalAdapter

logger = get_logger(__name__)

# ========== 参照评估系统的组件状态管理 ==========

class ComponentState(Enum):
    """组件状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"

@dataclass
class ComponentStatus:
    """组件状态管理"""
    state: ComponentState = ComponentState.UNLOADED
    load_time: Optional[float] = None
    unload_time: Optional[float] = None
    last_error: Optional[str] = None

# 全局变量 - 保持同步加载方式
_app_start_time = time.time()
_model_service = None
_rag_chain = None
_retrieval_coordinator = None

# 组件状态跟踪（V4.0: 移除model_service状态跟踪）
_component_status = {
    'retrieval_coordinator': ComponentStatus(),
    'vector_adapter': ComponentStatus(),
    'graph_adapter': ComponentStatus(),
    'rag_chain': ComponentStatus()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 参照评估系统的结构化组件初始化"""
    global _model_service, _rag_chain, _retrieval_coordinator

    # 启动时执行
    logger.info("=" * 80)
    logger.info("🚀 检索文档转发服务启动中（V4.0）...")
    logger.info("=" * 80)

    try:
        # 0. 初始化组件状态
        _init_component_status()

        # 1. 加载配置
        config = _load_config()
        if not config:
            raise RuntimeError("配置文件加载失败")

        # 2. 初始化所有组件（参照评估系统的结构化方式）
        # 辅助函数：解析相对路径
        def resolve_path(relative_path: str) -> str:
            """解析相对路径为绝对路径"""
            if not relative_path or os.path.isabs(relative_path):
                return relative_path
            return os.path.abspath(os.path.join(config_dir, relative_path))

        # 1. 初始化向量适配器
        logger.info("[1/4] 初始化向量适配器...")
        _component_status['vector_adapter'].state = ComponentState.LOADING
        start_time = time.time()

        vector_config = config.get('retrieval', {}).get('vector', {})
        vector_persist_dir = vector_config.get('persist_directory', "faiss_rag/向量数据库_768维")

        # 处理model_path：
        # - 绝对路径：直接使用
        # - HF模型标识（如 "iic/nlp_gte_sentence-embedding_chinese-base"）：保持原样，不做路径解析
        # - 其他相对路径：解析为绝对路径
        model_path = vector_config.get('model_path')
        def _looks_like_hf_repo_id(p: str) -> bool:
            # 经验判断：包含单斜杠、不是绝对路径、本地不存在该路径
            try:
                return (
                    isinstance(p, str)
                    and '/' in p
                    and '\\' not in p
                    and not os.path.isabs(p)
                    and not os.path.exists(p)
                )
            except Exception:
                return False

        if model_path:
            if os.path.isabs(model_path):
                pass
            elif _looks_like_hf_repo_id(model_path):
                logger.info(f"检测到HF模型标识，按名称加载: {model_path}")
            else:
                model_path = resolve_path(model_path)

        # 如果模型路径不存在或不完整，回退到适配器内置默认模型路径（与评估器一致）
        try:
            if model_path and not os.path.exists(model_path):
                logger.warning(f"向量模型路径不存在，回退到默认: {model_path}")
                model_path = None
            else:
                # 若缺少核心文件（如config.json），也回退
                if model_path:
                    cfg_file = os.path.join(model_path, 'config.json')
                    if not os.path.exists(cfg_file):
                        logger.warning(f"模型目录缺少config.json，回退到默认: {model_path}")
                        model_path = None
        except Exception as _:
            model_path = None

        resolved_persist_dir = resolve_path(vector_persist_dir)
        if not os.path.exists(resolved_persist_dir):
            logger.warning(f"向量数据库目录不存在：{resolved_persist_dir}，请确认已构建FAISS索引与documents.json")

        # 关键词库路径优先复用智能路由实体CSV，提升召回（与评估器一致）
        intelligent_router_cfg = config.get('retrieval', {}).get('intelligent_router', {})
        keyword_csv_path = intelligent_router_cfg.get('entity_csv_path')
        if keyword_csv_path and not os.path.isabs(keyword_csv_path):
            keyword_csv_path = resolve_path(keyword_csv_path)

        vector_adapter = SimpleVectorAdapter(
            persist_directory=resolved_persist_dir,
            model_path=model_path,
            timeout=vector_config.get('timeout', 60),
            score_threshold=vector_config.get('score_threshold', 0.0),
            enable_keyword_enhancement=False,  # 已移除关键词增强功能
            keyword_csv_path=keyword_csv_path  # 保留参数以避免初始化错误，但不会使用
        )

        _component_status['vector_adapter'].state = ComponentState.LOADED
        _component_status['vector_adapter'].load_time = time.time() - start_time
        logger.info(f"   ✓ 向量适配器初始化完成，耗时: {_component_status['vector_adapter'].load_time:.2f}秒")

        # 2. 初始化图检索适配器（V4.0: 不再加载生成模型，生成由Ollama服务处理）
        logger.info("[2/4] 初始化图检索适配器...")
        _component_status['graph_adapter'].state = ComponentState.LOADING
        start_time = time.time()

        graph_config = config.get('retrieval', {}).get('graph', {})
        # V4.0: 图检索适配器不再需要model_service（如果配置了use_llm_entity_extraction，可能需要LLM）
        # 但为了简化，图检索适配器可以接受None作为model_service
        graph_adapter = GraphRetrievalAdapter(
            neo4j_uri=graph_config.get('neo4j_uri', "neo4j://127.0.0.1:7687"),
            username=graph_config.get('username', "neo4j"),
            password=graph_config.get('password', "hx1230047"),
            database=graph_config.get('database', "neo4j"),
            timeout=graph_config.get('timeout', 20),
            model_service=None,  # V4.0: 不再加载生成模型，图检索适配器不使用LLM实体提取
            use_llm_entity_extraction=False  # V4.0: 禁用LLM实体提取，避免需要模型服务
        )

        _component_status['graph_adapter'].state = ComponentState.LOADED
        _component_status['graph_adapter'].load_time = time.time() - start_time
        logger.info(f"   ✓ 图检索适配器初始化完成，耗时: {_component_status['graph_adapter'].load_time:.2f}秒")

        # 3. 初始化检索协调器
        logger.info("[3/4] 初始化检索协调器...")
        _component_status['retrieval_coordinator'].state = ComponentState.LOADING
        start_time = time.time()

        # 获取智能路由器配置
        intelligent_router_config = config.get('retrieval', {}).get('intelligent_router', {})

        _retrieval_coordinator = HybridRetrievalCoordinator(
            vector_adapter=vector_adapter,
            graph_adapter=graph_adapter,
            use_intelligent_routing=True,  # 启用智能路由
            intelligent_router_config=intelligent_router_config
        )

        _component_status['retrieval_coordinator'].state = ComponentState.LOADED
        _component_status['retrieval_coordinator'].load_time = time.time() - start_time
        logger.info(f"   ✓ 检索协调器初始化完成，耗时: {_component_status['retrieval_coordinator'].load_time:.2f}秒")

        # 4. 初始化RAG链路（V4.0: 保留RAG链路用于v1接口，但不再用于生成）
        logger.info("[4/4] 初始化RAG链路...")
        _component_status['rag_chain'].state = ComponentState.LOADING
        start_time = time.time()

        # V4.0: RAG链路仍然需要模型服务用于生成，但我们可以传入None或者保留旧的逻辑
        # 为了兼容性，我们保留RAG链路的初始化，但生成功能已迁移到Ollama
        # 如果RAGChain需要model_service，我们需要处理
        try:
            _rag_chain = RAGChain(
                retrieval_coordinator=_retrieval_coordinator,
                max_context_tokens=1500,
                max_retrieval_results=5
            )
            # 初始化路由依赖（v1接口仍然可以使用，但生成功能已迁移）
            init_routes(_rag_chain, _retrieval_coordinator)
        except Exception as e:
            logger.warning(f"RAG链路初始化失败（不影响检索功能）: {e}")
            _rag_chain = None
        
        # 初始化Dify节点路由依赖（只需要检索协调器）
        init_dify_routes(None, _retrieval_coordinator)  # V4.0: Dify节点不再需要RAG链路

        _component_status['rag_chain'].state = ComponentState.LOADED
        _component_status['rag_chain'].load_time = time.time() - start_time
        logger.info(f"   ✓ RAG链路初始化完成，耗时: {_component_status['rag_chain'].load_time:.2f}秒")

        # 5. 预热检索模块（参照评估系统的预热方式）
        logger.info("[5/5] 预热检索模块...")
        try:
            from middle.models.data_models import RetrievalConfig

            # 预热向量检索
            logger.info("   预热向量检索...")
            warmup_config_vector = RetrievalConfig(
                enable_vector=True,
                enable_graph=False,
                top_k=10,  # 参考评估系统，使用更大的top_k
                timeout=60
            )
            warmup_results = await _retrieval_coordinator.retrieve("头痛", warmup_config_vector)
            logger.info(f"   ✓ 向量检索预热完成（返回{len(warmup_results)}个结果）")

            # 预热图检索
            logger.info("   预热图检索...")
            warmup_config_graph = RetrievalConfig(
                enable_vector=False,
                enable_graph=True,
                top_k=10,  # 参考评估系统，使用更大的top_k
                timeout=30
            )
            warmup_results = await _retrieval_coordinator.retrieve("人参", warmup_config_graph)
            logger.info(f"   ✓ 图检索预热完成（返回{len(warmup_results)}个结果）")

            logger.info("   ✓ 检索模块预热完成")
        except Exception as e:
            logger.warning(f"检索模块预热失败（不影响服务运行）: {e}")

        logger.info("=" * 80)
        logger.info("✅ 所有组件初始化完成，服务启动成功")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise

    yield

    # 关闭时执行
    logger.info("🔄 服务关闭中...")
    _cleanup_components()
    logger.info("✅ 服务已关闭")


def _init_component_status():
    """初始化组件状态"""
    global _component_status
    for component_name in _component_status:
        _component_status[component_name].state = ComponentState.UNLOADED
        _component_status[component_name].last_error = None


def _load_config() -> Dict[str, Any]:
    """加载配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "service_config.yaml")
        logger.info(f"🔄 加载配置文件: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info("✅ 配置文件加载成功")
        return config

    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        return {}





def _cleanup_components():
    """清理组件"""
    global _model_service, _rag_chain, _retrieval_coordinator

    try:
        # 清理组件（V4.0: 不再清理模型服务）
        if _rag_chain:
            _rag_chain = None

        if _retrieval_coordinator:
            _retrieval_coordinator = None

        # V4.0: 不再清理_model_service，因为不再加载

        # 更新组件状态
        for component_name in _component_status:
            _component_status[component_name].state = ComponentState.UNLOADED
            _component_status[component_name].unload_time = time.time()

        # 清理其他资源
        logger.info("   ✓ 资源清理完成")

    except Exception as e:
        logger.error(f"❌ 组件清理失败: {e}", exc_info=True)
    
    logger.info("=" * 80)
    logger.info("👋 检索文档转发服务已关闭")
    logger.info("=" * 80)


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    
    # 创建应用（V4.0: 专注于检索文档转发）
    app = FastAPI(
        title="检索文档转发服务API",
        description="""
        # 智能中医问答检索文档转发服务（V4.0）
        
        ## 功能特点
        
        - 🔍 **混合检索**: 集成向量检索、知识图谱两种检索方式
        - 📄 **文档转发**: 将检索与知识层召回的文档转发给Dify工作流
        - 🔄 **查询扩展**: text2vec查询扩展，提升检索效果
        - 📊 **重排序**: bge-reranker重排序，优化文档相关性
        - 🔌 **Dify集成**: 为Dify工作流提供检索文档转发服务
        
        ## 核心接口
        
        - **POST /api/dify/retrieve_documents**: Dify检索与知识召回节点，返回召回文档
        - **POST /api/dify/expand_and_rerank**: Dify查询扩展与重排序节点
        - **POST /api/v1/retrieve**: 纯检索接口，仅返回检索结果
        - **GET /api/v1/health**: 健康检查，查看系统状态
        
        ## 架构说明（V4.0）
        
        - **检索组件**: 启动时全量加载（向量适配器、图谱适配器、查询扩展模型、重排序模型）
        - **生成服务**: 由Dify工作流通过Ollama服务独立处理，本服务不提供生成功能
        - **职责分离**: 专注于检索文档转发，生成由Dify+Ollama处理
        
        ## 使用示例
        
        ```python
        import requests
        
        # Dify检索节点调用
        response = requests.post("http://localhost:8000/api/dify/retrieve_documents", json={
            "query": "头痛怎么治疗",
            "router_type": "hybrid"
        })
        
        result = response.json()
        print(result["documents"])  # 返回召回文档
        ```
        
        ## 技术栈
        
        - **检索**: ChromaDB + Faiss + Neo4j
        - **查询扩展**: text2vec-base-chinese-paraphrase
        - **重排序**: bge-reranker-base
        - **框架**: FastAPI
        - **融合**: 加权融合
        """,
        version="4.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 请求处理中间件
    @app.middleware("http")
    async def process_request(request: Request, call_next):
        """请求处理和日志中间件"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # 记录请求
        logger.info(f"[{request_id}] {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # 添加响应头
            process_time = time.time() - start_time
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            logger.info(f"[{request_id}] 完成 {response.status_code} ({process_time:.3f}s)")
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"[{request_id}] 错误: {e} ({process_time:.3f}s)", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"服务器内部错误: {str(e)}",
                    "request_id": request_id
                },
                headers={"X-Request-ID": request_id}
            )
    
    # 异常处理
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """HTTP异常处理"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """请求验证异常处理"""
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "请求参数验证失败",
                "detail": exc.errors()
            }
        )
    
    # 注册路由
    app.include_router(v1_router)
    
    # 注册Dify节点路由
    app.include_router(dify_router)
    logger.info("✅ Dify节点路由已注册: /api/dify/*")
    
    # 注册OpenAI兼容路由
    app.include_router(openai_router)
    logger.info("✅ OpenAI兼容API已注册: /v1/chat/completions")
    logger.info(f"✅ API Key: {os.getenv('OPENAI_API_KEY', 'sk-qwen3-1.7b-local-dev-key-12345')}")
    logger.info("✅ API Base: http://localhost:8000/v1/chat/completions")
    
    # 根路径
    @app.get("/", tags=["root"])
    async def root():
        """根路径"""
        uptime = int(time.time() - _app_start_time)
        return {
            "service": "检索文档转发服务API",
            "version": "4.0.0",
            "status": "running",
            "uptime_seconds": uptime,
            "docs": "/docs",
            "health": "/api/v1/health",
            "architecture": "V4.0 - 检索文档转发架构",
            "note": "生成功能由Dify工作流通过Ollama服务处理",
            "timestamp": time.time()
        }
    
    logger.info("✅ FastAPI应用创建完成")
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境设为False
        log_level="info"
    )

