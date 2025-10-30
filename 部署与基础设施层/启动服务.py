"""
LangChain中间层服务启动快捷方式
从根目录启动服务
"""

import sys
import subprocess
from pathlib import Path

# 调用scripts中的启动脚本
script_path = Path(__file__).parent / "scripts" / "start_langchain_service.py"

print("=" * 80)
print("启动LangChain中间层服务")
print("=" * 80)
print(f"调用: {script_path}\n")

# 直接执行
sys.exit(subprocess.call([sys.executable, str(script_path)] + sys.argv[1:]))

