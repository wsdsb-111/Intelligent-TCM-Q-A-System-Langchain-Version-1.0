#!/usr/bin/env python3
"""
环境兼容性检查脚本
检查 Python 3.12 + CUDA 12.4 + PyTorch 2.5.1 环境的依赖兼容性
"""

import sys
import subprocess
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    version = sys.version_info
    if version.major == 3 and version.minor == 12:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    elif version.major == 3 and version.minor >= 10:
        return True, f"⚠️  Python {version.major}.{version.minor}.{version.micro} (推荐 3.12)"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (需要 3.10+)"

def check_package(package_name: str, min_version: str = None) -> Tuple[bool, str]:
    """检查包是否安装及版本"""
    try:
        if package_name == "torch":
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "N/A"
            
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                return True, f"✅ {package_name} {version} (CUDA {cuda_version}, GPU: {device_name})"
            else:
                return False, f"⚠️  {package_name} {version} (CUDA 不可用)"
                
        elif package_name == "numpy":
            import numpy as np
            version = np.__version__
            major_version = int(version.split('.')[0])
            if major_version >= 2:
                return True, f"⚠️  {package_name} {version} (建议使用 1.26.x 避免兼容性问题)"
            return True, f"✅ {package_name} {version}"
            
        elif package_name == "pandas":
            import pandas as pd
            return True, f"✅ {package_name} {pd.__version__}"
            
        elif package_name == "openai":
            import openai
            return True, f"✅ {package_name} {openai.__version__}"
            
        elif package_name == "tiktoken":
            import tiktoken
            return True, f"✅ {package_name} {tiktoken.__version__}"
            
        elif package_name == "neo4j":
            import neo4j
            return True, f"✅ {package_name} {neo4j.__version__}"
            
        elif package_name == "networkx":
            import networkx as nx
            return True, f"✅ {package_name} {nx.__version__}"
            
        elif package_name == "sklearn":
            import sklearn
            return True, f"✅ scikit-learn {sklearn.__version__}"
            
        elif package_name == "transformers":
            import transformers
            return True, f"✅ {package_name} {transformers.__version__}"
            
        elif package_name == "vllm":
            import vllm
            return True, f"✅ {package_name} {vllm.__version__}"
            
        elif package_name == "flash_attn":
            import flash_attn
            return True, f"✅ flash-attn {flash_attn.__version__}"
            
        else:
            # 通用检查
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            return True, f"✅ {package_name} {version}"
            
    except ImportError:
        return False, f"❌ {package_name} 未安装"
    except Exception as e:
        return False, f"⚠️  {package_name} 检查出错: {str(e)}"

def check_cuda_environment() -> Tuple[bool, str]:
    """检查 CUDA 环境"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # 解析 nvidia-smi 输出
            lines = result.stdout.split('\n')
            cuda_version = "Unknown"
            driver_version = "Unknown"
            
            for line in lines:
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                if "Driver Version:" in line:
                    driver_version = line.split("Driver Version:")[1].strip().split()[0]
            
            return True, f"✅ NVIDIA Driver {driver_version}, CUDA {cuda_version}"
        else:
            return False, "❌ nvidia-smi 命令失败"
    except FileNotFoundError:
        return False, "❌ nvidia-smi 未找到（NVIDIA 驱动未安装）"
    except Exception as e:
        return False, f"⚠️  CUDA 检查出错: {str(e)}"

def main():
    """主检查函数"""
    print("=" * 70)
    print("GraphRAG 环境兼容性检查")
    print("Python 3.12 + CUDA 12.4 + PyTorch 2.5.1")
    print("=" * 70)
    print()
    
    # 检查 Python 版本
    print("【Python 环境】")
    success, msg = check_python_version()
    print(f"  {msg}")
    print()
    
    # 检查 CUDA 环境
    print("【CUDA 环境】")
    success, msg = check_cuda_environment()
    print(f"  {msg}")
    print()
    
    # 检查核心依赖
    print("【核心依赖】")
    core_packages = [
        ("numpy", "1.26.0"),
        ("pandas", "2.2.0"),
        ("openai", "1.12.0"),
        ("tiktoken", "0.6.0"),
    ]
    
    for package, min_version in core_packages:
        success, msg = check_package(package, min_version)
        print(f"  {msg}")
    print()
    
    # 检查 GraphRAG 依赖
    print("【GraphRAG 依赖】")
    graphrag_packages = [
        "neo4j",
        "networkx",
        "sklearn",
    ]
    
    for package in graphrag_packages:
        success, msg = check_package(package)
        print(f"  {msg}")
    print()
    
    # 检查可选依赖（PyTorch/vLLM）
    print("【可选依赖 - PyTorch/vLLM】")
    optional_packages = [
        "torch",
        "transformers",
        "vllm",
        "flash_attn",
    ]
    
    for package in optional_packages:
        success, msg = check_package(package)
        print(f"  {msg}")
    print()
    
    # 检查已知冲突
    print("【兼容性检查】")
    issues = []
    
    # NumPy 版本检查
    try:
        import numpy as np
        major_version = int(np.__version__.split('.')[0])
        if major_version >= 2:
            issues.append("⚠️  NumPy 2.x 可能与某些包不兼容，建议降级到 1.26.x")
    except ImportError:
        pass
    
    # PyTorch 与 transformers 版本匹配
    try:
        import torch
        import transformers
        torch_version = torch.__version__
        trans_version = transformers.__version__
        
        if torch_version.startswith("2.5"):
            trans_major = int(trans_version.split('.')[0])
            trans_minor = int(trans_version.split('.')[1])
            if trans_major < 4 or (trans_major == 4 and trans_minor < 46):
                issues.append("⚠️  PyTorch 2.5.x 需要 transformers>=4.46.0")
    except ImportError:
        pass
    
    # vLLM 与 PyTorch 版本匹配
    try:
        import vllm
        import torch
        vllm_version = vllm.__version__
        torch_version = torch.__version__
        
        if torch_version.startswith("2.5"):
            vllm_major_minor = '.'.join(vllm_version.split('.')[:2])
            if float(vllm_major_minor) < 0.6:
                issues.append("⚠️  PyTorch 2.5.x 需要 vLLM>=0.6.0")
    except ImportError:
        pass
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ 未发现兼容性问题")
    print()
    
    # 总结
    print("【检查总结】")
    print("  如果所有核心依赖都显示 ✅，则环境配置正确")
    print("  如果看到 ⚠️  警告，建议按照提示进行调整")
    print("  如果看到 ❌ 错误，需要安装相应的包")
    print()
    print("【推荐操作】")
    print("  1. 如果 NumPy 是 2.x，执行：")
    print("     pip install 'numpy>=1.26.0,<2.1.0' --force-reinstall")
    print()
    print("  2. 如果 PyTorch 未安装或版本不对，执行：")
    print("     pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \\")
    print("         --index-url https://download.pytorch.org/whl/cu124")
    print()
    print("  3. 如果 vLLM 未安装（在单独环境中），执行：")
    print("     pip install vllm==0.6.3.post1")
    print()
    print("  4. 如果 Flash Attention 未安装（可选但推荐），执行：")
    print("     pip install flash-attn==2.7.0.post2 --no-build-isolation")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()

