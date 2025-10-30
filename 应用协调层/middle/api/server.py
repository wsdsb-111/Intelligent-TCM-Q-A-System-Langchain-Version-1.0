#!/usr/bin/env python3
"""
API服务启动脚本
提供命令行接口启动混合检索API服务
"""

import argparse
import sys
import os
from typing import Optional

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    print("❌ 错误: 未安装uvicorn，请运行: pip install uvicorn")
    UVICORN_AVAILABLE = False
    sys.exit(1)

from langchain.api.app import create_app, create_development_app, create_production_app
from langchain.api.models import APIConfig
from middle.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="混合检索API服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 开发模式启动
  python server.py --dev
  
  # 生产模式启动
  python server.py --prod --host 0.0.0.0 --port 8000 --workers 4
  
  # 自定义配置启动
  python server.py --host 127.0.0.1 --port 8080 --log-level DEBUG
  
  # 启用自动重载（开发时使用）
  python server.py --reload --log-level DEBUG
        """
    )
    
    # 预设模式
    parser.add_argument(
        "--dev", "--development",
        action="store_true",
        help="开发模式（等同于 --host 127.0.0.1 --port 8000 --reload --log-level DEBUG）"
    )
    
    parser.add_argument(
        "--prod", "--production",
        action="store_true",
        help="生产模式（等同于 --host 0.0.0.0 --port 8000 --workers 4 --log-level INFO）"
    )
    
    # 服务器配置
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="服务器主机地址 (默认: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (默认: 1)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用自动重载（开发时使用）"
    )
    
    # 日志配置
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    # 功能开关
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="禁用API文档"
    )
    
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="禁用指标收集"
    )
    
    # CORS配置
    parser.add_argument(
        "--cors-origins",
        nargs="*",
        default=["*"],
        help="CORS允许的源 (默认: *)"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> APIConfig:
    """根据命令行参数创建配置"""
    return APIConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        cors_origins=args.cors_origins,
        enable_docs=not args.no_docs,
        enable_metrics=not args.no_metrics
    )


def main():
    """主函数"""
    print("🚀 混合检索API服务启动器")
    print("=" * 50)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查预设模式
    if args.dev and args.prod:
        print("❌ 错误: 不能同时指定开发模式和生产模式")
        sys.exit(1)
    
    # 创建应用
    if args.dev:
        print("🔧 使用开发模式配置")
        app = create_development_app()
        config = APIConfig(
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="DEBUG",
            enable_docs=True,
            enable_metrics=True
        )
    elif args.prod:
        print("🏭 使用生产模式配置")
        app = create_production_app()
        config = APIConfig(
            host="0.0.0.0",
            port=8000,
            workers=4,
            reload=False,
            log_level="INFO",
            enable_docs=False,
            enable_metrics=True
        )
    else:
        print("⚙️ 使用自定义配置")
        config = create_config_from_args(args)
        app = create_app(config)
    
    # 显示配置信息
    print(f"📍 服务地址: http://{config.host}:{config.port}")
    print(f"👥 工作进程: {config.workers}")
    print(f"🔄 自动重载: {'启用' if config.reload else '禁用'}")
    print(f"📝 日志级别: {config.log_level}")
    print(f"📚 API文档: {'启用' if config.enable_docs else '禁用'}")
    print(f"📊 指标收集: {'启用' if config.enable_metrics else '禁用'}")
    
    if config.enable_docs:
        print(f"📖 文档地址: http://{config.host}:{config.port}/docs")
        print(f"📋 ReDoc地址: http://{config.host}:{config.port}/redoc")
    
    if config.enable_metrics:
        print(f"💚 健康检查: http://{config.host}:{config.port}/api/v1/health")
        print(f"📈 指标接口: http://{config.host}:{config.port}/api/v1/metrics")
    
    print("=" * 50)
    
    try:
        # 启动服务器
        if config.workers > 1 and not config.reload:
            # 多进程模式
            print(f"🚀 启动多进程服务器 ({config.workers} 个工作进程)...")
            uvicorn.run(
                "langchain.api.app:create_production_app",
                factory=True,
                host=config.host,
                port=config.port,
                workers=config.workers,
                log_level=config.log_level.lower(),
                access_log=True
            )
        else:
            # 单进程模式
            print("🚀 启动单进程服务器...")
            uvicorn.run(
                app,
                host=config.host,
                port=config.port,
                reload=config.reload,
                log_level=config.log_level.lower(),
                access_log=True
            )
    
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 服务器启动失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()