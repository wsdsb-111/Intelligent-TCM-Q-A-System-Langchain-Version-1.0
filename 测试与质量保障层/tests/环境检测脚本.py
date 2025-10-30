#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检测脚本 - 线上智能中医问答项目
=====================================

功能：
1. 分析项目中所有需要的包
2. 对比新环境包列表（1.txt）
3. 找出缺失的包
4. 生成安装建议（仅显示，不自动下载）

注意：此脚本只进行检测和报告，不会自动安装任何包

作者：AI Assistant
日期：2025-10-22
版本：v1.1
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import importlib.util

class EnvironmentDetector:
    """环境检测器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.new_env_packages = self._load_new_env_packages()
        self.project_packages = set()
        self.missing_packages = set()
        self.version_conflicts = {}
        
    def _load_new_env_packages(self) -> Dict[str, str]:
        """加载新环境包列表（从1.txt）"""
        packages = {}
        env_file = self.project_root / "1.txt"
        
        if not env_file.exists():
            print("❌ 未找到1.txt文件，请确保新环境包列表存在")
            return packages
            
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    # 解析格式：package_name : version
                    parts = line.split(':')
                    if len(parts) >= 2:
                        package_name = parts[0].strip()
                        version = parts[1].strip()
                        packages[package_name] = version
                        
        except Exception as e:
            print(f"❌ 读取1.txt文件失败: {e}")
            
        print(f"✅ 已加载新环境包列表，共 {len(packages)} 个包")
        return packages
    
    def _extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """从Python文件中提取导入的包"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 匹配 import 语句
            import_patterns = [
                r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # import package
                r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # from package import
            ]
            
            for line in content.split('\n'):
                line = line.strip()
                for pattern in import_patterns:
                    match = re.match(pattern, line)
                    if match:
                        package = match.group(1)
                        # 过滤掉标准库和相对导入
                        if not package.startswith('.') and not self._is_stdlib(package):
                            imports.add(package)
                            
        except Exception as e:
            print(f"⚠️  读取文件 {file_path} 失败: {e}")
            
        return imports
    
    def _is_stdlib(self, package: str) -> bool:
        """判断是否为Python标准库"""
        stdlib_modules = {
            'os', 'sys', 'json', 're', 'pathlib', 'typing', 'datetime', 'logging',
            'asyncio', 'subprocess', 'importlib', 'collections', 'math', 'time',
            'warnings', 'shutil', 'traceback', 'enum', 'abc', 'functools',
            'itertools', 'operator', 'copy', 'pickle', 'hashlib', 'base64',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing', 'queue',
            'concurrent', 'contextlib', 'weakref', 'gc', 'inspect', 'ast',
            'tokenize', 'keyword', 'builtins', 'types', 'sysconfig', 'platform',
            'site', 'pkgutil', 'importlib', 'runpy', 'pdb', 'profile', 'cProfile',
            'timeit', 'trace', 'faulthandler', 'signal', 'atexit', 'tempfile',
            'glob', 'fnmatch', 'linecache', 'fileinput', 'stat', 'filecmp',
            'shutil', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'zlib',
            'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib', 'calendar',
            'collections', 'heapq', 'bisect', 'array', 'weakref', 'types',
            'copy', 'pprint', 'reprlib', 'enum', 'numbers', 'math', 'cmath',
            'decimal', 'fractions', 'random', 'statistics', 'itertools',
            'functools', 'operator', 'pathlib', 'os', 'io', 'time', 'argparse',
            'getopt', 'logging', 'getpass', 'curses', 'platform', 'errno',
            'ctypes', 'struct', 'codecs', 'unicodedata', 'stringprep', 'readline',
            'rlcompleter', 'sqlite3', 'zlib', 'gzip', 'bz2', 'lzma', 'zipfile',
            'tarfile', 'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib',
            'hashlib', 'hmac', 'secrets', 'uuid', 'socket', 'ssl', 'select',
            'selectors', 'asyncio', 'signal', 'mmap', 'email', 'json', 'mailcap',
            'mailbox', 'mimetypes', 'base64', 'binhex', 'binascii', 'quopri',
            'uu', 'html', 'xml', 'webbrowser', 'cgi', 'cgitb', 'wsgiref',
            'urllib', 'http', 'ftplib', 'poplib', 'imaplib', 'nntplib',
            'smtplib', 'smtpd', 'telnetlib', 'uuid', 'socketserver', 'http',
            'wsgiref', 'urllib', 'xmlrpc', 'ipaddress', 'audioop', 'aifc',
            'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr',
            'ossaudiodev', 'gettext', 'locale', 'calendar', 'cmd', 'shlex',
            'tkinter', 'turtle', 'pdb', 'profile', 'pstats', 'timeit', 'trace',
            'faulthandler', 'tracemalloc', 'distutils', 'ensurepip', 'venv',
            'zipapp', 'runpy', 'modulefinder', 'pkgutil', 'importlib'
        }
        return package in stdlib_modules
    
    def _extract_packages_from_requirements(self, req_file: Path) -> Set[str]:
        """从requirements.txt文件中提取包名"""
        packages = set()
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                # 跳过注释和空行
                if line.startswith('#') or not line:
                    continue
                    
                # 提取包名（去掉版本号）
                if '==' in line:
                    package = line.split('==')[0]
                elif '>=' in line:
                    package = line.split('>=')[0]
                elif '<=' in line:
                    package = line.split('<=')[0]
                elif '>' in line:
                    package = line.split('>')[0]
                elif '<' in line:
                    package = line.split('<')[0]
                else:
                    package = line
                    
                packages.add(package.strip())
                
        except Exception as e:
            print(f"⚠️  读取requirements文件 {req_file} 失败: {e}")
            
        return packages
    
    def scan_project_packages(self):
        """扫描项目中所有需要的包"""
        print("🔍 开始扫描项目包依赖...")
        
        # 1. 扫描所有Python文件
        python_files = list(self.project_root.rglob("*.py"))
        print(f"📁 找到 {len(python_files)} 个Python文件")
        
        for py_file in python_files:
            imports = self._extract_imports_from_file(py_file)
            self.project_packages.update(imports)
        
        # 2. 扫描所有requirements.txt文件
        req_files = list(self.project_root.rglob("requirements.txt"))
        print(f"📋 找到 {len(req_files)} 个requirements.txt文件")
        
        for req_file in req_files:
            packages = self._extract_packages_from_requirements(req_file)
            self.project_packages.update(packages)
        
        # 3. 添加一些常见的项目特定包
        project_specific_packages = {
            'chromadb', 'neo4j', 'faiss-cpu', 'faiss-gpu', 'onnxruntime',
            'fastapi', 'uvicorn', 'pydantic', 'pydantic-settings',
            'langchain', 'langchain-core', 'langchain-community', 'langchain-openai',
            'langchain-text-splitters', 'langgraph', 'langgraph-checkpoint',
            'langgraph-prebuilt', 'langgraph-sdk', 'langsmith',
            'ragas', 'datasets', 'evaluate', 'rouge-score', 'bert-score', 'sacrebleu',
            'sentence-transformers', 'transformers', 'huggingface-hub', 'tokenizers',
            'safetensors', 'accelerate', 'torch', 'torchvision', 'torchaudio',
            'numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-network',
            'openai', 'anthropic', 'cohere', 'tiktoken',
            'httpx', 'httpcore', 'httpx-sse', 'aiohttp', 'aiofiles', 'requests',
            'tqdm', 'rich', 'colorama', 'python-dotenv', 'click', 'typer',
            'nltk', 'regex', 'orjson', 'ormsgpack', 'jsonpatch', 'jsonpointer',
            'pyyaml', 'tenacity', 'nest-asyncio', 'anyio', 'pyarrow', 'dill',
            'multiprocess', 'instructor', 'gitpython', 'pillow', 'networkx',
            'diskcache', 'appdirs', 'matplotlib', 'contourpy', 'fonttools',
            'kiwisolver', 'cycler', 'certifi', 'charset-normalizer', 'filelock',
            'fsspec', 'idna', 'packaging', 'typing-extensions', 'urllib3',
            'wheel', 'xxhash', 'yarl', 'zstandard', 'pywin32', 'dataclasses-json',
            'marshmallow', 'attrs', 'annotated-types', 'python-dateutil', 'pytz',
            'tzdata', 'jinja2', 'markdown-it-py', 'mdurl', 'pygments', 'tabulate',
            'h11', 'multidict', 'frozenlist', 'psutil', 'shellingham', 'portalocker',
            'setuptools', 'pip', 'sympy', 'mpmath', 'lxml', 'docstring-parser',
            'typing-inspect', 'typing-inspection', 'mypy-extensions', 'propcache',
            'sniffio', 'distro', 'jiter', 'gitdb', 'smmap', 'six', 'markupsafe',
            'pyparsing', 'threadpoolctl', 'joblib', 'greenlet', 'sqlalchemy',
            'redis', 'boto3', 'azure-storage-blob', 'google-cloud-storage',
            'pytest', 'pytest-asyncio', 'pytest-cov', 'black', 'isort', 'flake8',
            'coverage', 'sentencepiece', 'opentelemetry-api', 'opentelemetry-sdk',
            'opentelemetry-exporter-otlp-proto-grpc', 'opentelemetry-instrumentation',
            'opentelemetry-instrumentation-asgi', 'opentelemetry-instrumentation-fastapi',
            'msgpack', 'jsonschema', 'pybase64', 'bcrypt', 'grpcio', 'pypika',
            'mmh3', 'overrides', 'posthog', 'loguru', 'jieba', 'chardet',
            'watchdog', 'cryptography', 'cffi', 'pycparser', 'oauthlib',
            'backports.tarfile', 'durationpy', 'protobuf', 'proto-plus',
            'flatbuffers', 'fastavro', 'isodate', 'pulsar-client', 's3transfer',
            'botocore', 'jmespath', 'google-resumable-media', 'googleapis-common-protos',
            'coloredlogs', 'humanfriendly', 'absl-py', 'peft'
        }
        
        self.project_packages.update(project_specific_packages)
        
        print(f"✅ 项目扫描完成，共识别 {len(self.project_packages)} 个包")
    
    def compare_environments(self):
        """对比新环境和项目需求"""
        print("🔍 开始对比环境...")
        
        # 找出缺失的包
        for package in self.project_packages:
            if package not in self.new_env_packages:
                self.missing_packages.add(package)
        
        # 检查版本冲突
        for package in self.project_packages:
            if package in self.new_env_packages:
                # 这里可以添加版本比较逻辑
                pass
        
        print(f"✅ 环境对比完成")
        print(f"📊 新环境包数量: {len(self.new_env_packages)}")
        print(f"📊 项目需要包数量: {len(self.project_packages)}")
        print(f"❌ 缺失包数量: {len(self.missing_packages)}")
    
    def generate_report(self):
        """生成检测报告"""
        print("\n" + "="*60)
        print("🔍 环境检测报告")
        print("="*60)
        
        print(f"\n📊 统计信息:")
        print(f"  新环境包数量: {len(self.new_env_packages)}")
        print(f"  项目需要包数量: {len(self.project_packages)}")
        print(f"  缺失包数量: {len(self.missing_packages)}")
        print(f"  覆盖率: {((len(self.project_packages) - len(self.missing_packages)) / len(self.project_packages) * 100):.1f}%")
        
        if self.missing_packages:
            print(f"\n❌ 缺失的包 ({len(self.missing_packages)} 个):")
            print("-" * 40)
            
            # 按类别分组
            categories = {
                '数据库相关': ['chromadb', 'neo4j', 'faiss-cpu', 'faiss-gpu', 'onnxruntime', 'redis', 'sqlalchemy'],
                'LangChain生态': ['langchain', 'langchain-core', 'langchain-community', 'langchain-openai', 'langchain-text-splitters', 'langgraph', 'langgraph-checkpoint', 'langgraph-prebuilt', 'langgraph-sdk', 'langsmith'],
                'RAG评估': ['ragas', 'datasets', 'evaluate', 'rouge-score', 'bert-score', 'sacrebleu'],
                'AI模型': ['sentence-transformers', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'accelerate', 'peft'],
                'PyTorch': ['torch', 'torchvision', 'torchaudio'],
                'API框架': ['fastapi', 'uvicorn', 'starlette', 'pydantic', 'pydantic-settings'],
                'AI服务': ['openai', 'anthropic', 'cohere', 'tiktoken'],
                'HTTP客户端': ['httpx', 'httpcore', 'httpx-sse', 'aiohttp', 'aiofiles', 'requests'],
                '数据处理': ['numpy', 'pandas', 'scipy', 'scikit-learn', 'scikit-network', 'pyarrow'],
                '工具库': ['tqdm', 'rich', 'colorama', 'python-dotenv', 'click', 'typer', 'nltk', 'regex'],
                '序列化': ['orjson', 'ormsgpack', 'jsonpatch', 'jsonpointer', 'pyyaml', 'msgpack'],
                '异步支持': ['tenacity', 'nest-asyncio', 'anyio'],
                '系统工具': ['psutil', 'shellingham', 'portalocker', 'setuptools', 'pip'],
                '数学计算': ['sympy', 'mpmath'],
                '文档处理': ['lxml', 'docstring-parser', 'jinja2', 'markdown-it-py', 'mdurl', 'pygments'],
                '类型检查': ['typing-inspect', 'typing-inspection', 'mypy-extensions'],
                '测试工具': ['pytest', 'pytest-asyncio', 'pytest-cov', 'black', 'isort', 'flake8', 'coverage'],
                '监控': ['opentelemetry-api', 'opentelemetry-sdk', 'opentelemetry-exporter-otlp-proto-grpc'],
                '其他': []
            }
            
            categorized_missing = {cat: [] for cat in categories}
            
            for package in sorted(self.missing_packages):
                categorized = False
                for cat, keywords in categories.items():
                    if cat == '其他':
                        continue
                    if any(keyword in package.lower() for keyword in keywords):
                        categorized_missing[cat].append(package)
                        categorized = True
                        break
                
                if not categorized:
                    categorized_missing['其他'].append(package)
            
            for cat, packages in categorized_missing.items():
                if packages:
                    print(f"\n  {cat}:")
                    for package in packages:
                        print(f"    - {package}")
            
            print(f"\n💡 安装建议（请手动执行以下命令）:")
            print("-" * 50)
            print("⚠️  注意：以下命令需要手动执行，脚本不会自动安装")
            print("-" * 50)
            
            print("\n1. 安装缺失的核心包:")
            core_packages = ['chromadb', 'neo4j', 'faiss-cpu', 'fastapi', 'uvicorn', 'pydantic']
            missing_core = [p for p in core_packages if p in self.missing_packages]
            if missing_core:
                print(f"   pip install {' '.join(missing_core)}")
            
            print("\n2. 安装LangChain生态:")
            langchain_packages = [p for p in self.missing_packages if 'langchain' in p.lower()]
            if langchain_packages:
                print(f"   pip install {' '.join(langchain_packages)}")
            
            print("\n3. 安装RAG评估工具:")
            rag_packages = [p for p in self.missing_packages if p in ['ragas', 'datasets', 'evaluate', 'rouge-score', 'bert-score', 'sacrebleu']]
            if rag_packages:
                print(f"   pip install {' '.join(rag_packages)}")
            
            print("\n4. 安装AI模型相关:")
            ai_packages = [p for p in self.missing_packages if p in ['sentence-transformers', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'accelerate']]
            if ai_packages:
                print(f"   pip install {' '.join(ai_packages)}")
            
            print("\n5. 安装PyTorch (CUDA 12.8):")
            if 'torch' in self.missing_packages:
                print("   pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128")
            
            print("\n6. 安装其他缺失包:")
            other_packages = [p for p in sorted(self.missing_packages) if p not in core_packages + langchain_packages + rag_packages + ai_packages and p != 'torch']
            if other_packages:
                print(f"   pip install {' '.join(other_packages)}")
            
            print("\n" + "="*50)
            print("📋 一键安装所有缺失包（可选）:")
            all_missing = sorted(self.missing_packages)
            if all_missing:
                print(f"   pip install {' '.join(all_missing)}")
            print("="*50)
        
        else:
            print(f"\n✅ 所有需要的包都已在新环境中安装！")
        
        # 保存报告到文件
        self._save_report()
    
    def _save_report(self):
        """保存报告到文件"""
        report_file = self.project_root / "环境检测报告.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("环境检测报告\n")
                f.write("="*60 + "\n")
                f.write(f"生成时间: {subprocess.check_output(['date'], shell=True).decode().strip()}\n")
                f.write(f"项目路径: {self.project_root}\n\n")
                
                f.write("统计信息:\n")
                f.write(f"  新环境包数量: {len(self.new_env_packages)}\n")
                f.write(f"  项目需要包数量: {len(self.project_packages)}\n")
                f.write(f"  缺失包数量: {len(self.missing_packages)}\n")
                f.write(f"  覆盖率: {((len(self.project_packages) - len(self.missing_packages)) / len(self.project_packages) * 100):.1f}%\n\n")
                
                if self.missing_packages:
                    f.write("缺失的包:\n")
                    f.write("-" * 40 + "\n")
                    for package in sorted(self.missing_packages):
                        f.write(f"  - {package}\n")
                else:
                    f.write("✅ 所有需要的包都已在新环境中安装！\n")
            
            print(f"\n📄 报告已保存到: {report_file}")
            
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")

def main():
    """主函数"""
    print("🔍 环境检测脚本 - 线上智能中医问答项目")
    print("="*60)
    print("⚠️  注意：此脚本只进行检测和报告，不会自动安装任何包")
    print("="*60)
    
    # 创建检测器
    detector = EnvironmentDetector()
    
    # 执行检测
    detector.scan_project_packages()
    detector.compare_environments()
    detector.generate_report()
    
    print("\n🎯 检测完成！")
    print("💡 如需安装缺失的包，请手动执行上述建议的pip install命令")

if __name__ == "__main__":
    main()