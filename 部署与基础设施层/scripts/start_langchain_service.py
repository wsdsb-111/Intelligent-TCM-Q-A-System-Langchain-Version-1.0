"""
LangChain中间层服务启动脚本
提供简单的命令行启动接口
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到路径
# 脚本在部署与基础设施层/scripts/中，需要回到项目根目录
project_root = Path(__file__).parent.parent.parent  # 回到项目根目录
deployment_root = Path(__file__).parent.parent  # 部署层根目录
application_layer = project_root / "应用协调层"  # 应用协调层目录
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(application_layer))  # 添加应用协调层到路径


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = project_root / "应用协调层" / "middle" / "config" / "service_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}")
        print("使用默认配置...")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return {}


def check_environment():
    """检查运行环境"""
    print("=" * 80)
    print("🔍 环境检查")
    print("=" * 80)
    
    # 检查Python版本
    import platform
    python_version = platform.python_version()
    print(f"Python版本: {python_version}")
    
    # 检查关键依赖
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
    ]
    
    missing_deps = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}: 已安装")
        except ImportError:
            print(f"❌ {name}: 未安装")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\n⚠️  缺少依赖: {', '.join(missing_deps)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  GPU: 不可用，将使用CPU")
    except:
        print("⚠️  无法检测GPU状态")
    
    print("=" * 80)
    return True


def check_data_files(config: dict):
    """检查数据文件"""
    print("=" * 80)
    print("📁 数据文件检查")
    print("=" * 80)
    
    checks = []
    
    # 配置文件所在目录（应用协调层/middle/config/）
    config_dir = project_root / "应用协调层" / "middle" / "config"
    
    def resolve_path(relative_path: str) -> Path:
        """解析相对路径为绝对路径"""
        if os.path.isabs(relative_path):
            return Path(relative_path)
        # 相对于配置文件目录解析
        return (config_dir / relative_path).resolve()
    
    # 检查模型路径
    if 'model' in config:
        base_model_path = config['model'].get('base_model_path')
        adapter_path = config['model'].get('adapter_path')
        
        if base_model_path:
            full_path = resolve_path(base_model_path)
            exists = os.path.exists(full_path)
            checks.append(('基础模型', full_path, exists))
        
        if adapter_path:
            full_path = resolve_path(adapter_path)
            exists = os.path.exists(full_path)
            checks.append(('LoRA适配器', full_path, exists))
    
    # 检查BM25索引
    if 'retrieval' in config and 'bm25' in config['retrieval']:
        bm25_config = config['retrieval']['bm25']
        
        # 如果使用分布式引擎，检查split配置
        if bm25_config.get('use_distributed') and bm25_config.get('split_config_path'):
            split_config_path = bm25_config.get('split_config_path')
            full_path = resolve_path(split_config_path)
            exists = os.path.exists(full_path)
            checks.append(('BM25 Split配置', full_path, exists))
        else:
            # 否则检查单索引文件
            index_path = bm25_config.get('index_path')
            if index_path:
                full_path = resolve_path(index_path)
                exists = os.path.exists(full_path)
                checks.append(('BM25索引', full_path, exists))
    
    # 检查向量数据库
    if 'retrieval' in config and 'vector' in config['retrieval']:
        db_path = config['retrieval']['vector'].get('persist_directory')
        if db_path:
            full_path = resolve_path(db_path)
            exists = os.path.exists(full_path)
            checks.append(('向量数据库', full_path, exists))
    
    # 显示检查结果
    all_ok = True
    for name, path, exists in checks:
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_ok = False
    
    print("=" * 80)
    return all_ok


def start_service(config: dict, host: str = None, port: int = None, reload: bool = False):
    """
    启动服务
    
    Args:
        config: 配置字典
        host: 主机地址（覆盖配置）
        port: 端口号（覆盖配置）
        reload: 是否启用热重载
    """
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn未安装，请运行: pip install uvicorn")
        sys.exit(1)
    
    # 获取API配置
    api_config = config.get('api', {})
    
    # 确定启动参数
    _host = host or api_config.get('host', '0.0.0.0')
    _port = port or api_config.get('port', 8000)
    _reload = reload or api_config.get('reload', False)
    _log_level = api_config.get('log_level', 'info').lower()
    
    print("=" * 80)
    print("🚀 启动LangChain中间层服务")
    print("=" * 80)
    print(f"地址: http://{_host}:{_port}")
    print(f"API文档: http://{_host}:{_port}/docs")
    print(f"健康检查: http://{_host}:{_port}/api/v1/health")
    print(f"热重载: {'启用' if _reload else '禁用'}")
    print(f"日志级别: {_log_level.upper()}")
    print("=" * 80)
    print("\n按 Ctrl+C 停止服务\n")
    
    # 启动服务
    # 注意：main_app.py在应用协调层/middle/api/main_app.py
    uvicorn.run(
        "middle.api.main_app:app",
        host=_host,
        port=_port,
        reload=_reload,
        log_level=_log_level
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LangChain中间层服务启动脚本')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认: langchain/config/service_config.yaml）'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='服务器地址（默认: 0.0.0.0）'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='服务器端口（默认: 8000）'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='启用热重载（开发模式）'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='跳过环境和数据检查'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🏥 智能中医问答RAG系统 - LangChain中间层")
    print("=" * 80 + "\n")
    
    # 加载配置
    config = load_config(args.config)
    
    if not args.skip_checks:
        # 环境检查
        if not check_environment():
            print("\n❌ 环境检查未通过，请安装缺少的依赖")
            sys.exit(1)
        
        # 数据文件检查
        if not check_data_files(config):
            print("\n⚠️  部分数据文件不存在，服务可能无法正常工作")
            response = input("是否继续启动？(y/N): ")
            if response.lower() != 'y':
                print("已取消启动")
                sys.exit(0)
    
    # 启动服务
    try:
        start_service(config, args.host, args.port, args.reload)
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

