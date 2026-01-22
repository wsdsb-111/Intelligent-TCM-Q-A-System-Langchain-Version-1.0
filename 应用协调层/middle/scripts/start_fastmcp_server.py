"""基于 FastMCP 的网页搜索服务入口

该脚本使用 fastmcp 提供符合 MCP `streamable-http` 规范的工具接口，
直接对接项目现有的 `HybridRetrievalMCPTool`，可被 Dify v1.6+ 原生识别。
"""

import os
import sys
from typing import Optional, List
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# FastMCP 服务器
try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "未安装 fastmcp，请先执行 `pip install fastmcp` 后再启动服务"
    ) from exc

# 项目依赖
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT))

from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.integrations.mcp_tools import HybridRetrievalMCPTool
from middle.utils.logging_utils import get_logger


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

load_dotenv()

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "8062"))
SERVER_NAME = os.getenv("MCP_NAME", "WebSearchMCP")


# ---------------------------------------------------------------------------
# Pydantic 模型定义（用于 FastMCP 自动生成工具 schema）
# ---------------------------------------------------------------------------


class WebSearchRequest(BaseModel):
    """网页搜索工具请求参数"""

    query: str = Field(..., description="搜索关键词")
    limit: int = Field(5, ge=1, le=20, description="返回结果条数，1-20")
    include: Optional[str] = Field(
        None,
        description="限定搜索域名，多个用 '|' 分隔，如 'zhihu.com|baidu.com'",
    )
    exclude: Optional[str] = Field(
        None,
        description="排除域名，多个用 '|' 分隔",
    )
    freshness: Optional[str] = Field(
        None,
        description="时间范围：noLimit/oneDay/oneWeek/oneMonth/oneYear 或 日期",
    )
    summary: Optional[bool] = Field(
        None,
        description="是否返回摘要（true/false）",
    )
    count: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="原始接口的 count 参数，若未指定则与 limit 相同",
    )


class HybridRetrievalRequest(BaseModel):
    query: str = Field(..., description="检索查询内容")
    retrieval_type: str = Field(
        "hybrid",
        description="检索类型：hybrid/vector/graph",
    )
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")
    fusion_method: str = Field(
        "rrf",
        description="融合方法：rrf 或 weighted",
    )


class BatchRetrievalRequest(BaseModel):
    queries: List[str] = Field(..., description="查询列表")
    retrieval_type: str = Field("hybrid", description="检索类型")
    top_k: int = Field(3, ge=1, le=20, description="每个查询返回结果数")


# ---------------------------------------------------------------------------
# 初始化基础组件
# ---------------------------------------------------------------------------

logger = get_logger(__name__)
coordinator = HybridRetrievalCoordinator()
mcp_tool = HybridRetrievalMCPTool(coordinator)

server = FastMCP(
    SERVER_NAME,
    host=HOST,
    port=PORT,
    transport="streamable-http",
    stateless_http=True,
)


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------


@server.tool(description="调用博查 API 进行网页搜索", parameters=WebSearchRequest)
async def web_search(request: WebSearchRequest):
    """网页搜索 MCP 工具"""

    return await mcp_tool.web_search(**request.dict())


@server.tool(description="混合检索（向量+知识图谱）", parameters=HybridRetrievalRequest)
async def hybrid_retrieval(request: HybridRetrievalRequest):
    return await mcp_tool.hybrid_retrieval(**request.dict())


@server.tool(description="批量混合检索", parameters=BatchRetrievalRequest)
async def batch_retrieval(request: BatchRetrievalRequest):
    return await mcp_tool.batch_retrieval(**request.dict())


@server.tool(description="获取系统健康状态")
async def health_check():
    return await mcp_tool.health_check()


@server.tool(description="获取检索统计信息")
async def get_statistics(include_details: bool = False):
    return await mcp_tool.get_statistics(include_details=include_details)


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

def main():
    logger.info(
        "[FastMCP] %s 启动，监听 %s:%s/mcp (transport=streamable-http)",
        SERVER_NAME,
        HOST,
        PORT,
    )
    server.run()


if __name__ == "__main__":
    main()

