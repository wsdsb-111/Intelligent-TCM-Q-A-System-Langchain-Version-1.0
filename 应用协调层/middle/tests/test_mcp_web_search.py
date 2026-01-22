"""MCP网页搜索功能测试

此测试用于验证MCP工具中的网页搜索功能是否能够正常返回搜索结果。
运行方式：
    pytest 应用协调层/middle/tests/test_mcp_web_search.py
或直接执行：
    python 应用协调层/middle/tests/test_mcp_web_search.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from middle.integrations.mcp_tools import HybridRetrievalMCPTool


TEST_QUERY = os.getenv("MCP_TEST_QUERY", "中医治感冒的常用方剂")


def _assert_result_structure(result: Dict[str, Any]) -> None:
    assert "title" in result, "返回结果缺少title字段"
    assert "url" in result, "返回结果缺少url字段"
    assert result["title"], "返回结果title为空"
    assert result["url"], "返回结果url为空"


def _run_sync(coro):
    return asyncio.run(coro)


def test_mcp_web_search_returns_results():
    tool = HybridRetrievalMCPTool()
    response = _run_sync(tool.web_search(query=TEST_QUERY, limit=2))

    assert response.get("success"), f"MCP搜索调用失败: {response.get('error')}"

    results = response.get("results", [])
    assert results, "MCP搜索未返回任何结果"

    _assert_result_structure(results[0])


if __name__ == "__main__":
    try:
        test_mcp_web_search_returns_results()
        print("✅ MCP网页搜索测试通过")
    except AssertionError as exc:
        print(f"❌ MCP网页搜索测试失败: {exc}")

