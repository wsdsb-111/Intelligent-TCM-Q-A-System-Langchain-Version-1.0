"""
检索文档转发服务启动快捷方式（V4.0）
从根目录启动服务
专注于检索文档转发，生成功能由Dify通过Ollama处理
"""

import sys
import subprocess
from pathlib import Path
import time
import atexit

# 调用scripts中的启动脚本
script_path = Path(__file__).parent / "scripts" / "start_langchain_service.py"

# FastMCP 启动脚本
fastmcp_script = Path(__file__).parent.parent / "应用协调层" / "middle" / "scripts" / "start_fastmcp_server.py"

print("=" * 80)
print("启动检索文档转发服务（V4.0）")
print("=" * 80)
print(f"调用: {script_path}\n")

project_root = Path(__file__).parent.parent.resolve()

# 启动MCP服务（后台），输出重定向到日志文件
mcp_stdout_path = Path(__file__).parent / "mcp_log.txt"
mcp_stderr_path = Path(__file__).parent / "mcp_error.txt"
mcp_stdout_file = open(mcp_stdout_path, "w", encoding="utf-8")
mcp_stderr_file = open(mcp_stderr_path, "w", encoding="utf-8")

mcp_process = subprocess.Popen(
    [
        sys.executable,
        str(fastmcp_script),
    ],
    stdout=mcp_stdout_file,
    stderr=mcp_stderr_file,
    cwd=project_root,
)


def _cleanup_mcp():
    if mcp_process.poll() is None:
        try:
            print("正在关闭MCP服务进程...")
            mcp_process.terminate()
            mcp_process.wait(timeout=5)
        except Exception:
            pass
    if not mcp_stdout_file.closed:
        mcp_stdout_file.close()
    if not mcp_stderr_file.closed:
        mcp_stderr_file.close()


aexit_registered = atexit.register(_cleanup_mcp)

# 等待MCP初始化
time.sleep(5)
mcp_running = mcp_process.poll() is None

if mcp_running:
    print("FastMCP服务启动成功！")
    print("- MCP端点: http://0.0.0.0:8062/mcp")
    print("- MCP工具发现: http://0.0.0.0:8062/mcp/tools")
    print("- MCP健康检查: http://0.0.0.0:8062/health")
    print(f"- MCP日志: {mcp_stdout_path}")
else:
    print("⚠️ MCP服务启动失败！请检查错误日志。")
    print(f"- 错误日志: {mcp_stderr_path}")

# 主服务URL信息
print("主FastAPI服务URL: http://0.0.0.0:8000")
print("- API文档: http://0.0.0.0:8000/docs")
print("- 健康检查: http://0.0.0.0:8000/health")

# 启动主服务
try:
    exit_code = subprocess.call([sys.executable, str(script_path)] + sys.argv[1:])
finally:
    _cleanup_mcp()
    if aexit_registered:
        atexit.unregister(_cleanup_mcp)

sys.exit(exit_code)

