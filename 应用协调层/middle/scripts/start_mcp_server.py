#!/usr/bin/env python3
"""
MCP工具服务器启动脚本
启动混合检索系统的MCP工具服务器
"""

import asyncio
import json
import sys
import os
import argparse
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from middle.integrations.mcp_tools import HybridRetrievalMCPTool, get_mcp_tool
from middle.core.retrieval_coordinator import HybridRetrievalCoordinator
from middle.models.data_models import RetrievalConfig
from middle.utils.logging_utils import get_logger, setup_logging


class MCPToolServer:
    """MCP工具服务器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化MCP工具服务器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = get_logger(__name__)
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化MCP工具
        self.mcp_tool = None
        self.coordinator = None
        
        # 服务器状态
        self.is_running = False
        self.stats = {
            "start_time": None,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = project_root / "langchain" / "config" / "mcp_tools_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "mcp_tools": {
                "server": {
                    "name": "hybrid-retrieval-mcp-server",
                    "version": "1.0.0"
                },
                "performance": {
                    "max_concurrent_requests": 10,
                    "timeout": {
                        "hybrid_retrieval": 30,
                        "batch_retrieval": 60,
                        "contextual_retrieval": 30,
                        "health_check": 10,
                        "get_statistics": 5
                    }
                },
                "logging": {
                    "level": "INFO"
                }
            }
        }
    
    async def initialize(self):
        """初始化服务器组件"""
        try:
            self.logger.info("初始化MCP工具服务器...")
            
            # 初始化协调器
            self.coordinator = HybridRetrievalCoordinator()
            
            # 初始化MCP工具
            self.mcp_tool = HybridRetrievalMCPTool(self.coordinator)
            
            # 执行健康检查
            health_status = await self.coordinator.health_check()
            self.logger.info(f"系统健康状态: {health_status}")
            
            self.logger.info("MCP工具服务器初始化完成")
            
        except Exception as e:
            self.logger.error(f"服务器初始化失败: {e}")
            raise
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理MCP请求
        
        Args:
            request: MCP请求
            
        Returns:
            MCP响应
        """
        try:
            self.stats["total_requests"] += 1
            
            # 解析请求
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            self.logger.info(f"处理MCP请求: method={method}, id={request_id}")
            
            # 路由请求
            if method == "tools/list":
                response = await self._handle_list_tools()
            elif method == "tools/call":
                response = await self._handle_call_tool(params)
            else:
                response = {
                    "error": {
                        "code": -32601,
                        "message": f"未知方法: {method}"
                    }
                }
            
            # 添加请求ID
            if request_id:
                response["id"] = request_id
            
            self.stats["successful_requests"] += 1
            return response
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"处理MCP请求失败: {e}")
            
            return {
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"内部错误: {str(e)}"
                }
            }
    
    async def _handle_list_tools(self) -> Dict[str, Any]:
        """处理工具列表请求"""
        try:
            tools = self.mcp_tool.get_tool_definitions()
            
            # 转换为MCP格式
            mcp_tools = []
            for tool in tools:
                mcp_tool = {
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool["parameters"]
                }
                mcp_tools.append(mcp_tool)
            
            return {
                "result": {
                    "tools": mcp_tools
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取工具列表失败: {e}")
            raise
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理工具调用请求"""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return {
                    "error": {
                        "code": -32602,
                        "message": "缺少工具名称"
                    }
                }
            
            # 调用工具
            result = await self.mcp_tool.call_tool(tool_name, arguments)
            
            # 转换为MCP格式
            if result.get("success"):
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False, indent=2)
                            }
                        ]
                    }
                }
            else:
                return {
                    "error": {
                        "code": -32603,
                        "message": result.get("error", "工具调用失败")
                    }
                }
                
        except Exception as e:
            self.logger.error(f"工具调用失败: {e}")
            raise
    
    async def start_server(self, host: str = "localhost", port: int = 8001):
        """
        启动MCP工具服务器
        
        Args:
            host: 服务器主机
            port: 服务器端口
        """
        try:
            from datetime import datetime
            
            self.stats["start_time"] = datetime.now()
            self.is_running = True
            
            self.logger.info(f"MCP工具服务器启动: {host}:{port}")
            
            # 这里应该实现实际的MCP服务器逻辑
            # 由于MCP协议的具体实现比较复杂，这里提供一个简化的演示
            
            print(f"MCP工具服务器已启动")
            print(f"服务器信息:")
            print(f"  - 主机: {host}")
            print(f"  - 端口: {port}")
            print(f"  - 可用工具数: {len(self.mcp_tool.get_tool_definitions())}")
            
            # 显示可用工具
            tools = self.mcp_tool.get_tool_definitions()
            print(f"\n可用工具:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
            
            # 保持服务器运行
            print(f"\n服务器正在运行... (按 Ctrl+C 停止)")
            
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在关闭服务器...")
            await self.stop_server()
        except Exception as e:
            self.logger.error(f"服务器运行错误: {e}")
            raise
    
    async def stop_server(self):
        """停止MCP工具服务器"""
        self.is_running = False
        self.logger.info("MCP工具服务器已停止")
        
        # 显示统计信息
        print(f"\n服务器统计信息:")
        print(f"  - 总请求数: {self.stats['total_requests']}")
        print(f"  - 成功请求数: {self.stats['successful_requests']}")
        print(f"  - 失败请求数: {self.stats['failed_requests']}")
        
        if self.stats["start_time"]:
            from datetime import datetime
            uptime = datetime.now() - self.stats["start_time"]
            print(f"  - 运行时间: {uptime}")
    
    async def run_interactive_demo(self):
        """运行交互式演示"""
        print("=== MCP工具交互式演示 ===")
        print("输入 'help' 查看可用命令，输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    await self._show_help()
                elif user_input.lower() == 'tools':
                    await self._show_tools()
                elif user_input.lower() == 'health':
                    await self._demo_health_check()
                elif user_input.lower() == 'stats':
                    await self._demo_statistics()
                elif user_input.startswith('search '):
                    query = user_input[7:]  # 去掉 'search ' 前缀
                    await self._demo_search(query)
                else:
                    print("未知命令。输入 'help' 查看可用命令。")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"执行命令时出错: {e}")
        
        print("演示结束")
    
    async def _show_help(self):
        """显示帮助信息"""
        print("\n可用命令:")
        print("  help    - 显示此帮助信息")
        print("  tools   - 显示可用工具列表")
        print("  health  - 执行健康检查")
        print("  stats   - 显示统计信息")
        print("  search <查询> - 执行检索（例: search 感冒治疗）")
        print("  quit    - 退出演示")
    
    async def _show_tools(self):
        """显示可用工具"""
        tools = self.mcp_tool.get_tool_definitions()
        print(f"\n可用工具 ({len(tools)} 个):")
        for i, tool in enumerate(tools, 1):
            print(f"  {i}. {tool['name']}")
            print(f"     描述: {tool['description']}")
    
    async def _demo_health_check(self):
        """演示健康检查"""
        print("\n执行健康检查...")
        result = await self.mcp_tool.health_check()
        
        if result["success"]:
            print(f"系统整体健康: {'✓ 正常' if result['overall_healthy'] else '✗ 异常'}")
            print("各模块状态:")
            for module, status in result["modules"].items():
                status_text = "✓ 正常" if status else "✗ 异常"
                print(f"  {module}: {status_text}")
        else:
            print(f"健康检查失败: {result['error']}")
    
    async def _demo_statistics(self):
        """演示统计信息"""
        print("\n获取统计信息...")
        result = await self.mcp_tool.get_statistics()
        
        if result["success"]:
            stats = result["statistics"]
            print("系统统计信息:")
            print(f"  总查询数: {stats.get('total_queries', 0)}")
            print(f"  成功查询数: {stats.get('successful_queries', 0)}")
            print(f"  失败查询数: {stats.get('failed_queries', 0)}")
            print(f"  平均响应时间: {stats.get('average_response_time', 0.0):.3f}秒")
        else:
            print(f"获取统计信息失败: {result['error']}")
    
    async def _demo_search(self, query: str):
        """演示检索功能"""
        if not query:
            print("请提供检索查询")
            return
        
        print(f"\n检索查询: {query}")
        result = await self.mcp_tool.hybrid_retrieval(
            query=query,
            retrieval_type="hybrid",
            top_k=3
        )
        
        if result["success"]:
            print(f"检索成功，找到 {result['total_results']} 个结果:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n  结果 {i}:")
                print(f"    评分: {res['score']:.3f}")
                print(f"    来源: {', '.join(res['sources'])}")
                print(f"    内容: {res['content'][:150]}...")
                if res.get('entities'):
                    print(f"    实体: {', '.join(res['entities'][:5])}")
        else:
            print(f"检索失败: {result['error']}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MCP工具服务器")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--host", default="localhost", help="服务器主机")
    parser.add_argument("--port", type=int, default=8001, help="服务器端口")
    parser.add_argument("--demo", action="store_true", help="运行交互式演示")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    
    # 创建服务器
    server = MCPToolServer(args.config)
    
    try:
        # 初始化服务器
        await server.initialize()
        
        if args.demo:
            # 运行交互式演示
            await server.run_interactive_demo()
        else:
            # 启动服务器
            await server.start_server(args.host, args.port)
            
    except Exception as e:
        print(f"服务器启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())