import os
from typing import Any, Dict

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from app.search_client import BochaSearchClient, BochaSearchRequest


load_dotenv()

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8062"))
MCP_NAME = os.getenv("MCP_NAME", "WebSearchMCP")

server = FastMCP(
    MCP_NAME,
    host=MCP_HOST,
    port=MCP_PORT,
)


@server.tool(name="web_search", description="调用博查搜索 API 获取实时网页搜索结果")
async def web_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = BochaSearchRequest(**payload)
    results = BochaSearchClient.search(schema)
    return {
        "results": [item.dict() for item in results],
        "total": len(results),
    }


def main() -> None:
    print(
        f"[{MCP_NAME}] 启动中：http://{MCP_HOST}:{MCP_PORT}/mcp "
        "(transport=streamable-http)"
    )
    server.run()


if __name__ == "__main__":
    main()

